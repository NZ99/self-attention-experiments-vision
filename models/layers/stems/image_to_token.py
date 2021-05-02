from typing import Callable, Tuple

from flax.linen import initializers
from flax import linen as nn

from jax.lax import Precision
from jax import numpy as jnp

from einops import rearrange


class Image2TokenBlock(nn.Module):
    patch_shape: Tuple[int, int]
    num_ch: int
    conv_kernel_size: int
    conv_stride: int
    pool_window_size: int
    pool_stride: int
    embed_dim: int
    use_bias: bool = False
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):

        x = nn.Conv(features=self.num_ch,
                    use_bias=self.use_bias,
                    kernel_size=(self.conv_kernel_size, self.conv_kernel_size),
                    strides=(self.conv_stride, self.conv_stride),
                    padding=[(self.patch_shape[0],) * 2,
                             (self.patch_shape[1],) * 2])(inputs)
        x = nn.BatchNorm(use_running_average=not is_training,
                         momentum=self.bn_momentum,
                         epsilon=self.bn_epsilon,
                         dtype=self.dtype)(x)
        x = nn.max_pool(
            inputs=x,
            window_shape=(self.pool_window_size,) * 2,
            strides=(self.pool_stride,) * 2,
        )
        x = rearrange(
            x,
            'b (h ph) (w pw) c -> b (h w) (ph pw c)',
            ph=self.patch_shape[0],
            pw=self.patch_shape[1],
        )

        output = nn.Dense(features=self.embed_dim,
                          use_bias=self.use_bias,
                          dtype=self.dtype,
                          precision=self.precision,
                          kernel_init=self.kernel_init,
                          bias_init=self.bias_init)(x)
        return output
