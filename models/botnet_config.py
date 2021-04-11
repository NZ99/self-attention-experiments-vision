from typing import Callable, Any, Sequence

import jax
from jax import numpy as jnp
from jax.nn import initializers

from flax import linen as nn
from flax import struct

ModuleDef = Any


@struct.dataclass
class BoTNetConfig:
    """BoTNet configuration"""
    stage_sizes: Sequence[int] = [3, 4, 6, 6]
    num_classes: int = 1000
    stride_one: bool = True
    se_ratio: float = 0.0625
    activation_fn: ModuleDef = nn.swish
    num_heads: int = 4
    head_dim: int = 128
    initial_filters: int = 64
    projection_factor: int = 4
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    precision: Any = jax.lax.Precision.DEFAULT
    kernel_init: Callable = initializers.he_uniform()
    bias_init: Callable = initializers.normal(stddev=1e-6)
    posemb_init: Callable = initializers.normal(stddev=head_dim**-0.5)