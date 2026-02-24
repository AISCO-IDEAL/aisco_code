import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from omegaconf import OmegaConf

from .gaot_3d import GAOT3D
from .gaot_3d_attention import GAOT3D as GAOT3DAttention


def init_model(input_size: int, output_size: int, model: str, config: dataclass = None):

    supported_models = [
        "gaot3d",
        "gaot_3d_attention",
    ]

    if model.lower() == "gaot_3d":
        return GAOT3D(
            input_size=input_size,
            output_size=output_size,
            magno_config=config.magno,
            attn_config=config.transformer,
            latent_tokens=config.latent_tokens,
        )

    elif model.lower() == "gaot_3d_attention":
        return GAOT3DAttention(
            input_size=input_size,
            output_size=output_size,
            magno_config=config.magno,
            attn_config=config.transformer,
            latent_tokens=config.latent_tokens,
        )

    else:
        raise ValueError(f"model {model} not supported currently!")
