from typing import Callable, Tuple

from flax.linen import initializers
from flax import linen as nn

from jax.lax import Precision
from jax import numpy as jnp

from einops import rearrange


class PatchEmbedBlock(nn.Module):

    patch_shape: Tuple[int, int]
    embed_dim: int
    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        ph, pw = self.patch_shape

        x = rearrange(inputs,
                      'b (h ph) (w pw) c -> b (h w) (ph pw c)',
                      ph=ph,
                      pw=pw)
        output = nn.Dense(features=self.embed_dim,
                          use_bias=self.use_bias,
                          dtype=self.dtype,
                          precision=self.precision,
                          kernel_init=self.kernel_init,
                          bias_init=self.bias_init)(x)
        return output