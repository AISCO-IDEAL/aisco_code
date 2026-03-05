import torch
import torch.nn as nn  # noqa: PLR0402
import torch.nn.functional as F  # noqa: N812
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb  # noqa: F401

from .mlp import ConditionedNorm


class GroupQueryFlashAttention(nn.Module):
    def __init__(self, attn_config: dict):
        super().__init__()
        self.attn_config = attn_config

        assert attn_config["hidden_size"] % attn_config["num_heads"] == 0, (
            f"hidden_size {attn_config['hidden_size']} must be divisible by num_heads {attn_config['num_heads']}"
        )
        assert attn_config["num_heads"] % attn_config["num_kv_heads"] == 0, (
            f"num_heads {attn_config['num_heads']} must be divisible by num_kv_heads {attn_config['num_kv_heads']}"
        )
        self.num_heads = attn_config["num_heads"]
        self.num_kv_heads = attn_config["num_kv_heads"]
        self.num_repeat = self.num_heads // self.num_kv_heads
        self.head_dim = attn_config["hidden_size"] // self.num_heads
        self.atten_dropout = attn_config.get("atten_dropout", 0.0)

        kv_hidden_size = self.head_dim * self.num_kv_heads

        self.q_proj = nn.Linear(attn_config["input_size"], attn_config["hidden_size"], bias=False)
        self.k_proj = nn.Linear(attn_config["input_size"], kv_hidden_size, bias=False)
        self.v_proj = nn.Linear(attn_config["input_size"], kv_hidden_size, bias=False)
        self.o_proj = nn.Linear(attn_config["hidden_size"], attn_config["output_size"], bias=False)

        self.correction: ConditionedNorm | None
        if attn_config["use_conditional_norm"]:
            self.correction = ConditionedNorm(1, attn_config["output_size"], attn_config["cond_norm_hidden_size"])
        else:
            self.correction = None
        self.attn_dtype = torch.float16
        if attn_config["positional_embedding"] == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

    def forward(self, x, condition: float | None = None, relative_positions: torch.Tensor | None = None):
        if self.correction is not None:
            x = self.correction(c=condition, x=x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size, seq_len, _ = q.size()

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_repeat, dim=1)
            v = v.repeat_interleave(self.num_repeat, dim=1)

        if relative_positions is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        dp = self.atten_dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dp)

        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = self.o_proj(x)

        return x


class FFN(nn.Module):
    def __init__(self, ffn_config: dict):
        super().__init__()
        self.ffn_config = ffn_config

        self.w1 = nn.Linear(ffn_config["input_size"], ffn_config["hidden_size"], bias=False)
        self.w2 = nn.Linear(ffn_config["hidden_size"], ffn_config["output_size"], bias=False)
        self.w3 = nn.Linear(ffn_config["input_size"], ffn_config["hidden_size"], bias=False)

        self.correction: ConditionedNorm | None
        if ffn_config["use_conditional_norm"]:
            self.correction = ConditionedNorm(1, ffn_config["output_size"], ffn_config["cond_norm_hidden_size"])
        else:
            self.correction = None

    def forward(self, x, condition: float | None = None):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))

        if self.correction is not None:
            x = self.correction(c=condition, x=x)

        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, block_config: dict):
        super().__init__()
        self.block_config = block_config
        self.skip_connection = block_config.get("skip_connection", False)

        # Build full attn_config: inject input_size / output_size
        full_attn_config = dict(block_config["attn_config"])
        full_attn_config["input_size"] = block_config["input_size"]
        full_attn_config["output_size"] = block_config["attn_config"]["hidden_size"]

        # Build full ffn_config: inject input_size / output_size
        full_ffn_config = dict(block_config["ffn_config"])
        full_ffn_config["input_size"] = block_config["attn_config"]["hidden_size"]
        full_ffn_config["output_size"] = block_config["output_size"]

        self.attn = GroupQueryFlashAttention(attn_config=full_attn_config)
        self.ffn = FFN(ffn_config=full_ffn_config)

        self.attn_norm = (
            RMSNorm(block_config["input_size"], eps=block_config["norm_eps"]) if block_config["use_attn_norm"] else None
        )
        self.ffn_norm = (
            RMSNorm(block_config["attn_config"]["hidden_size"], eps=block_config["norm_eps"])
            if block_config["use_ffn_norm"]
            else None
        )

        if self.skip_connection:
            self.skip_proj = nn.Linear(block_config["input_size"] + block_config["output_size"], block_config["input_size"])

    def forward(
        self,
        x: torch.Tensor,
        condition: float | None = None,
        relative_positions: torch.Tensor | None = None,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.skip_connection and skip is not None:
            x = torch.cat([x, skip], dim=-1)
            x = self.skip_proj(x)

        h = x if self.attn_norm is None else self.attn_norm(x)
        h = x + self.attn(h, condition=condition, relative_positions=relative_positions)
        h = h if self.ffn_norm is None else self.ffn_norm(h)
        out = h + self.ffn(h, condition=condition)
        return out


class Transformer(nn.Module):
    def __init__(self, transformer_config: dict):
        super().__init__()
        self.transformer_config = transformer_config
        self.use_long_range_skip = transformer_config["use_long_range_skip"]

        # Propagate positional_embedding into attn_config
        attn_config = dict(transformer_config["attn_config"])
        attn_config.setdefault("positional_embedding", transformer_config.get("positional_embedding", "absolute"))

        if transformer_config["input_size"] != transformer_config["hidden_size"]:
            self.input_proj = nn.Linear(transformer_config["input_size"], transformer_config["hidden_size"])
        else:
            self.input_proj = nn.Identity()

        if transformer_config["hidden_size"] != transformer_config["output_size"]:
            self.output_proj = nn.Linear(transformer_config["hidden_size"], transformer_config["output_size"])
        else:
            self.output_proj = nn.Identity()

        num_encoder_layers = transformer_config["num_layers"] // 2
        num_decoder_layers = transformer_config["num_layers"] // 2
        middle_layer_exists = transformer_config["num_layers"] % 2 == 1

        def _make_block_config(skip: bool) -> dict:
            return {
                "input_size": transformer_config["hidden_size"],
                "output_size": transformer_config["hidden_size"],
                "use_attn_norm": transformer_config["use_attn_norm"],
                "use_ffn_norm": transformer_config["use_ffn_norm"],
                "norm_eps": transformer_config["norm_eps"],
                "attn_config": attn_config,
                "ffn_config": transformer_config["ffn_config"],
                "skip_connection": skip,
            }

        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(block_config=_make_block_config(skip=False)) for _ in range(num_encoder_layers)]
        )

        self.middle_layer = TransformerBlock(block_config=_make_block_config(skip=False)) if middle_layer_exists else None

        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(block_config=_make_block_config(skip=True)) for _ in range(num_decoder_layers)]
        )

    def forward(
        self, x: torch.Tensor, condition: float | None = None, relative_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.input_proj(x)
        skips = []

        for layer in self.encoder_layers:
            x = layer(x, condition=condition, relative_positions=relative_positions)
            skips.append(x)

        if self.middle_layer is not None:
            x = self.middle_layer(x, condition=condition, relative_positions=relative_positions)

        for layer in self.decoder_layers:
            skip = skips.pop() if self.use_long_range_skip else None
            x = layer(x, condition=condition, relative_positions=relative_positions, skip=skip)

        x = self.output_proj(x)
        return x
