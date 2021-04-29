from typing import Callable, Tuple

from flax.linen import initializers
from flax import linen as nn

from jax.lax import Precision
from jax import numpy as jnp

from einops import rearrange

class CeiTImage2TokenPatchEmbedBlock(nn.Module):
    patch_shape: Tuple[int, int]
    num_ch: int
    conv_kernel_size: int
    conv_stride: int
    pool_window_size: int
    pool_stride: int
    embed_dim: int

    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs,):
        # inputs.shape = [b, c, h, w]
        h = nn.Conv(
            features=self.num_ch,
            kernel_size=(self.conv_kernel_size, self.conv_kernel_size),
            strides=(self.conv_stride, self.conv_stride),
            padding=[(self.patch_shape[0],)*2, (self.patch_shape[1],)*2],
        )(inputs)

        h = nn.BatchNorm(use_running_average=True)(h)
        h = nn.max_pool(
            inputs=h,
            window_shape=(self.pool_window_size,) * 2,
            strides=(self.pool_stride,) * 2,
        )

        h_hat = rearrange(
            h,
            'b (h ph) (w pw) c -> b (h w) (ph pw c)',
            ph=self.patch_shape[0],
            pw=self.patch_shape[1],
        )

        out = nn.Dense(
            features=self.embed_dim,
            use_bias=self.use_bias,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(h_hat)

        return out

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
