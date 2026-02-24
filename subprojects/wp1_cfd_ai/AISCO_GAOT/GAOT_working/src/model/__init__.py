import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from omegaconf import OmegaConf

from .gaot_3d import GAOT3D


def init_model(input_size: int, output_size: int, model: str, config: dataclass = None):

    supported_models = [
        "gaot_3d",
    ]

    if model.lower() == "gaot_3d":
        field_output_size = getattr(config, "field_output_size", 0)
        return GAOT3D(
            input_size=input_size,
            output_size=output_size,
            magno_config=config.magno,
            attn_config=config.transformer,
            latent_tokens=config.latent_tokens,
            field_output_size=field_output_size,
        )

    else:
        raise ValueError(f"model {model} not supported currently!")
