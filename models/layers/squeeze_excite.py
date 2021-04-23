from typing import Callable, Any

from flax import linen as nn

from jax.lax import Precision
from jax import numpy as jnp

from functools import partial

Dtype = Any


class SqueezeExciteBlock(nn.Module):

    se_ratio: float = None
    hidden_ch: int = None
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = nn.initializers.kaiming_uniform
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        in_ch = inputs.shape[-1]
        if self.se_ratio is None:
            if self.hidden_ch is None:
                raise ValueError('Must provide one of se_ratio or hidden_ch')
        else:
            self.hidden_ch = max(1, int(in_ch * self.se_ratio))

        dense = partial(nn.Dense,
                        use_bias=True,
                        dtype=self.dtype,
                        precision=self.precision,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init)

        x = jnp.mean(inputs, axis=(1, 2), dtype=self.dtype,
                     keepdims=True)(inputs)
        x = dense(features=self.hidden_ch)(x)
        x = self.activation_fn(x)
        x = dense(features=in_ch)(x)
        output = nn.sigmoid(x) * inputs
        return output
