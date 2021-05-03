from typing import Callable, Sequence

from jax import numpy as jnp
from jax.nn import initializers
from jax.lax import Precision

from flax import linen as nn

from einops import rearrange

from models.layers import CvTSelfAttentionBlock, FFBlock


class ConvTokenEmbedBlock(nn.Module):
    out_ch: int
    kernel_size: int
    strides: int
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, **unused_kwargs):
        assert inputs.ndim == 4
        x = nn.Conv(features=self.out_ch,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    strides=(self.strides, self.strides),
                    padding='SAME',
                    dtype=self.dtype,
                    precision=self.precision,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init)(inputs)
        x = rearrange(x, 'b H W c -> b (H W) c')
        output = nn.LayerNorm(dtype=self.dtype)(x)
        return output


class StageBlock(nn.Module):
    num_heads: int
    embed_dim: int
    kernel_size: int = 3
    use_bias: bool = False
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    expand_ratio: int = 4
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        assert inputs.ndim == 4
        assert self.embed_dim % self.num_heads == 0
        head_ch = int(self.embed_dim / self.num_heads)

        x = CvTSelfAttentionBlock(num_heads=self.num_heads,
                                  head_ch=head_ch,
                                  out_ch=self.embed_dim,
                                  kernel_size=self.kernel_size,
                                  use_bias=self.use_bias,
                                  bn_momentum=self.bn_momentum,
                                  bn_epsilon=self.bn_epsilon,
                                  dtype=self.dtype,
                                  precision=self.precision,
                                  kernel_init=self.kernel_init,
                                  bias_init=self.bias_init)(
                                      inputs, is_training=is_training)
        x = x + inputs

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = FFBlock(expand_ratio=self.expand_ratio,
                    dtype=self.dtype,
                    precision=self.precision,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init)(y, is_training=is_training)

        output = x + y
        return output


class Stage(nn.Module):
    size: int
    num_heads: int
    embed_dim: int
    embed_kernel_size: int
    embed_strides: int
    sa_kernel_size: int = 3
    use_bias: bool = False
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    expand_ratio: int = 4
    insert_cls: bool = False
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):

        x = ConvTokenEmbedBlock(out_ch=self.embed_dim,
                                kernel_size=self.embed_kernel_size,
                                strides=self.embed_strides,
                                dtype=self.dtype,
                                precision=self.precision,
                                kernel_init=self.kernel_init,
                                bias_init=self.bias_init)(
                                    inputs, is_training=is_training)

        if self.insert_cls:
            b = x.shape[0]
            cls_shape = (1, 1, self.embed_dim)
            cls_token = self.param('cls', initializers.zeros, cls_shape)
            cls_token = jnp.tile(cls_token, [b, 1, 1])
            x = jnp.concatenate([cls_token, x], axis=1)

        for i in range(self.size):
            x = StageBlock(num_heads=self.num_heads,
                           embed_dim=self.embed_dim,
                           kernel_size=self.sa_kernel_size,
                           use_bias=self.use_bias,
                           bn_momentum=self.bn_momentum,
                           bn_epsilon=self.bn_epsilon,
                           expand_ratio=self.expand_ratio,
                           dtype=self.dtype,
                           precision=self.precision,
                           kernel_init=self.kernel_init,
                           bias_init=self.bias_init)(x, is_training=is_training)
        output = x
        return output


class CvT(nn.Module):
    num_classes: int
    stage_sizes: Sequence[int]
    num_heads: Sequence[int]
    embed_dim: Sequence[int]
    embed_kernel_size: Sequence[int] = [7, 3, 3]
    embed_strides: Sequence[int] = [4, 2, 2]
    sa_kernel_size: Sequence[int] = [3, 3, 3]
    use_bias: bool = False
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    expand_ratio: int = 4
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = inputs
        for i in range(len(self.size) - 1):
            x = Stage(size=self.size[i],
                      num_heads=self.num_heads[i],
                      embed_dim=self.embed_dim[i],
                      embed_kernel_size=self.embed_kernel_size[i],
                      embed_strides=self.embed_strides[i],
                      sa_kernel_size=self.sa_kernel_size[i],
                      use_bias=self.use_bias,
                      bn_momentum=self.bn_momentum,
                      bn_epsilon=self.bn_epsilon,
                      expand_ratio=self.expand_ratio,
                      dtype=self.dtype,
                      precision=self.precision,
                      kernel_init=self.kernel_init,
                      bias_init=self.bias_init)(x, is_training=is_training)

            l = x.shape[1]
            spatial_ch = int(jnp.sqrt(l))
            x = rearrange(x, 'b (H W) c -> b H W c', H=spatial_ch)

        x = Stage(size=self.size[i],
                  num_heads=self.num_heads[i],
                  embed_dim=self.embed_dim[i],
                  embed_kernel_size=self.embed_kernel_size[i],
                  embed_strides=self.embed_strides[i],
                  sa_kernel_size=self.sa_kernel_size[i],
                  use_bias=self.use_bias,
                  bn_momentum=self.bn_momentum,
                  bn_epsilon=self.bn_epsilon,
                  expand_ratio=self.expand_ratio,
                  insert_cls=True,
                  dtype=self.dtype,
                  precision=self.precision,
                  kernel_init=self.kernel_init,
                  bias_init=self.bias_init)(x, is_training=is_training)

        cls_token = x[:, 0]
        output = nn.Dense(features=self.num_classes,
                          use_bias=True,
                          dtype=self.dtype,
                          precision=self.precision,
                          kernel_init=self.kernel_init,
                          bias_init=initializers.zeros)(cls_token)
        return output