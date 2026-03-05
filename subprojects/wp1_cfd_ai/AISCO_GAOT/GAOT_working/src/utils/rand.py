import random

import numpy as np
import torch


def manual_seed(seed: int):
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    random.seed(seed)
