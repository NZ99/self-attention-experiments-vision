from typing import Any

from flax import struct
from jax import numpy as jnp

ModuleDef = Any


@struct.dataclass
class CeiTConfig:
    """CeiT-T config"""

    dropout_rate: float = 0.
    attn_dropout_rate: float = 0.
    dtype: jnp.dtype = jnp.float32

    embed_dim: int = 192
    num_heads: int = 3
    num_layers: int = 12

    # I2T params
    conv_kernel_size: int = 7
    conv_stride: int = 2
    pool_window_size: int = 3
    pool_stride: int = 2
    num_ch: int = 32

    # LeFF settings
    expand_ratio = 4
    dw_conv_kernel_size: int = 3

    # LCA
    lca_num_layers: int = 1
