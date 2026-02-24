from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from omegaconf import OmegaConf

from ..optimizers import OptimizerargsConfig


def merge_config(default_config_class, user_config):
    default_config_struct = OmegaConf.structured(default_config_class)
    merged_config = OmegaConf.merge(default_config_struct, user_config)
    return OmegaConf.to_object(merged_config)


# ── Default dicts matching config.yaml structure ─────────────────────────


def _default_magno() -> Dict[str, Any]:
    return {
        "use_gno": True,
        "gno_coord_dim": 3,
        "gno_radius": 0.15,
        "lifting_channels": 64,
        "encoder_feature_attr": "x",
        "in_gno_channel_mlp_hidden_layers": [128, 128],
        "in_gno_transform_type": "linear",
        "projection_channels": 256,
        "out_gno_channel_mlp_hidden_layers": [64, 64],
        "out_gno_transform_type": "linear",
        "mlp_type": "channel",
        "scales": [1.0],
        "use_scale_weights": False,
        "use_attn": None,
        "attention_type": "cosine",
        "use_geoembed": [True, True],
        "embedding_method": "statistical",
        "pooling": "max",
        "sampling_strategy": None,
        "max_neighbors": None,
        "sample_ratio": None,
        "neighbor_strategy": "radius",
        "k_neighbors": 1,
        "precompute_edges": True,
        "asynchronous_graph_building": False,
    }


def _default_transformer() -> Dict[str, Any]:
    return {
        "patch_size": 4,
        "hidden_size": 256,
        "use_attn_norm": True,
        "use_ffn_norm": True,
        "norm_eps": 1e-6,
        "num_layers": 10,
        "positional_embedding": "rope",
        "use_long_range_skip": True,
        "attn_config": {
            "hidden_size": 256,
            "num_heads": 8,
            "num_kv_heads": 8,
            "use_conditional_norm": False,
            "cond_norm_hidden_size": 4,
            "atten_dropout": 0.1,
            "positional_embedding": "rope",
        },
        "ffn_config": {
            "hidden_size": 1024,
            "use_conditional_norm": False,
            "cond_norm_hidden_size": 4,
        },
    }


def _default_decoder() -> Dict[str, Any]:
    return {
        "hidden_features": 256,
        "num_shared_layers": 3,
        "output_heads": {
            "force": 1,
            "cd": 1,
        },
    }


@dataclass
class SetUpConfig:
    seed: int = 42  # Random seed for reproducibility
    device: str = "cuda:0"  # Computation device, (e.g., "cuda:0", "cpu")
    dtype: str = "torch.float32"  # Data type for computation (e.g., "torch.float32", "torch.float64")
    trainer_name: str = "static3d"  # Type of trainer to use, support [static3d]
    train: bool = True  # Whether to run the training phase
    test: bool = False  # Whether to run the testing phase
    ckpt: bool = False  # Whether to load the checkpoint
    use_variance_test: bool = False  # TODO Whether to use variance testing
    # Parameters for distributed mode
    distributed: bool = False  # Whether to enable distributed training
    world_size: int = 1  # Total number of processes in distributed training
    rank: int = 0  # Rank of the current process in distributed training
    local_rank: int = 0  # Local rank of the current process in distributed training
    backend: str = "nccl"  # Backend for distributed training, e.g., 'nccl' (NVIDIA Collective Communications Library)

    # Weights & Biases logging
    wandb: bool = False  # Enable Weights & Biases logging
    wandb_project: Optional[str] = None  # W&B project name
    wandb_entity: Optional[str] = None  # W&B entity (team/user)
    wandb_run_name: Optional[str] = None  # Run name
    wandb_group: Optional[str] = None  # Group name for runs
    wandb_mode: str = "online"  # 'online' | 'offline' | 'disabled'
    wandb_notes: Optional[str] = None  # Notes for the run
    wandb_tags: List[str] = field(default_factory=list)  # Tags for the run
    wandb_watch_model: bool = False  # Watch model gradients/parameters
    wandb_log_interval: int = 1  # Log interval in epochs (usually align with eval_every_eps)

    # Model checkpoint management with wandb artifacts
    resume_from_artifact: Optional[str] = None  # Load model from wandb artifact (e.g., "model-name:best")
    test_during_training: bool = False  # Run test during training when checkpoints are saved


@dataclass
class ModelArgsConfig:
    input_size: int = 6  # Input feature size
    output_size: int = 1  # Output feature size
    latent_tokens: Tuple[int, int, int] = (64, 64, 64)  # Size (D, H, W) of latent tokens
    norm_domain: List[List[float]] = field(default_factory=lambda: [[-1, -1, -1], [1, 1, 1]])
    magno: Dict[str, Any] = field(default_factory=_default_magno)  # Configuration for MAGNO (plain dict)
    transformer: Dict[str, Any] = field(default_factory=_default_transformer)  # Configuration for Transformer (plain dict)
    decoder: Dict[str, Any] = field(default_factory=_default_decoder)  # Configuration for decoder (plain dict)


@dataclass
class ModelConfig:
    name: str = "gaot_3d"  # Name of the model, support ['gaot_3d']
    use_conditional_norm: bool = False  # Whether to use time-conditional normalization (not supported in this repo)
    args: ModelArgsConfig = field(default_factory=ModelArgsConfig)  # Configuration for the model's components


@dataclass
class DatasetConfig:
    name: str = "drivaernet_fullpressure"  # Name of the dataset
    metaname: str = "gaot-unstructured/drivaernet_pressure"  # Metaname (identifier) for the dataset, used for loading the dataset
    base_path: str = "/cluster/work/math/camlab-data/graphnpde/drivaernet/"  # Base path where the dataset is stored
    processed_folder: str = "processed_pyg"  # Folder name of the saved/processed .pt files
    active_variables: Optional[List[int]] = None  # List of active variables to use, if None, use all variables
    force_recompute_stats: bool = False  # Whether to force recomputing the statistics
    use_metadata_stats: bool = False  # Whether to use metadata statistics
    sample_rate: float = 0.1  # Sampling rate of the dataset
    train_size: int = 5817  # Number of training samples
    val_size: int = 1148  # Number of validation samples
    test_size: int = 1154  # Number of test samples
    rand_dataset: bool = False  # Whether to randomize the sequence of loaded dataset
    batch_size: int = 64  # Batch size for training
    num_workers: int = 4  # Number of workers for loading the dataset
    shuffle: bool = True  # Whether to shuffle the dataset
    metric_suite: str = "drivaernet"  # Literal["poseidon", "general"]
    # for graph building
    update_pt_files_with_edges: bool = False  # Flag to trigger edge computation/saving
    use_rescale_new: bool = False  # Whether to use new rescale methods for coordinates
    # for neural field training strategy
    training_strategy: str = "full_grid"  # Training strategy: ['neural_field', 'full_grid']
    neural_field_input_nodes: Optional[int] = 40000  # Num nodes for fixed encoder input in neural_field strategy
    neural_field_query_nodes_train: Optional[int] = 40000  # Num nodes for random decoder query in neural_field train
    # for multi-head / force-only training
    order_file_name: Optional[str] = (
        None  # Custom order file name (e.g. "order_force_only.txt"), default uses order_{processed_folder}.txt
    )
    active_heads: Optional[List[str]] = None  # List of active heads for loss (e.g. ["force"]), None = all heads


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    args: OptimizerargsConfig = field(default_factory=OptimizerargsConfig)


@dataclass
class PathConfig:
    ckpt_path: str = ".ckpt/test/test.pt"  # Path to save the checkpoint
    loss_path: str = ".loss/test/test.png"  # Path to save the loss plot
    result_path: str = ".result/test/test.png"  # Path to save the result plot
    database_path: str = ".database/test/test.csv"  # Path to save the database
