from typing import Callable, Tuple

from jax import numpy as jnp
from flax import linen as nn
from einops import rearrange

from models.layers import CvTSelfAttentionBlock, FFBlock


def zero_pad_and_reshape(inputs):
    assert inputs.ndim == 3
    _, l, in_ch = inputs.shape
    spatial_ch = int(jnp.ceil(jnp.sqrt(l)))
    inputs = jnp.pad(inputs, ((0, 0), (0, spatial_ch**2 - l), (0, 0)))
    inputs = rearrange(inputs, 'b (H W) c -> b H W c', W=spatial_ch)
    return inputs


class ConvTokenEmbedBlock(nn.Module):
    out_ch: int
    kernel_size: int
    strides: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, **unused_kwargs):
        assert inputs.ndim == 4
        x = nn.Conv(features=self.out_ch,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    strides=(self.strides, self.strides),
                    padding='SAME',
                    dtype=self.dtype)(inputs)
        x = rearrange(x, 'b H W c -> b (H W) c')
        output = nn.LayerNorm(dtype=self.dtype)(x)
        return output


class StageBlock(nn.Module):
    num_heads: int
    embed_dim: int
    kernel_size: int = 3
    use_bias: bool = False
    activation_fn: Callable = nn.activation.gelu
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    expand_ratio: float = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        inputs = zero_pad_and_reshape(inputs)

        x = CvTSelfAttentionBlock(num_heads=self.num_heads,
                                  kernel_size=self.kernel_size,
                                  use_bias=self.use_bias,
                                  bn_momentum=self.bn_momentum,
                                  bn_epsilon=self.bn_epsilon,
                                  dtype=self.dtype)(inputs,
                                                    is_training=is_training)

        x = x + rearrange(inputs, 'b h w d -> b (h w) d')

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = FFBlock(expand_ratio=self.expand_ratio,
                    activation_fn=self.activation_fn,
                    dtype=self.dtype)(y, is_training=is_training)
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
    activation_fn: Callable = nn.activation.gelu
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    expand_ratio: float = 4
    insert_cls: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):

        x = ConvTokenEmbedBlock(out_ch=self.embed_dim,
                                kernel_size=self.embed_kernel_size,
                                strides=self.embed_strides,
                                dtype=self.dtype)(inputs,
                                                  is_training=is_training)

        if self.insert_cls:
            b = x.shape[0]
            cls_shape = (1, 1, self.embed_dim)
            cls_token = self.param('cls', nn.initializers.zeros, cls_shape)
            cls_token = jnp.tile(cls_token, [b, 1, 1])
            x = jnp.concatenate([cls_token, x], axis=1)

        for i in range(self.size):
            x = StageBlock(num_heads=self.num_heads,
                           embed_dim=self.embed_dim,
                           kernel_size=self.sa_kernel_size,
                           use_bias=self.use_bias,
                           activation_fn=self.activation_fn,
                           bn_momentum=self.bn_momentum,
                           bn_epsilon=self.bn_epsilon,
                           expand_ratio=self.expand_ratio,
                           dtype=self.dtype)(x, is_training=is_training)
        output = x
        return output


class CvT(nn.Module):
    num_classes: int
    stage_sizes: Tuple[int]
    num_heads: Tuple[int]
    embed_dim: Tuple[int]
    embed_kernel_size: Tuple[int] = (7, 3, 3)
    embed_strides: Tuple[int] = (4, 2, 2)
    sa_kernel_size: Tuple[int] = (3, 3, 3)
    use_bias: bool = False
    expand_ratio: float = 4
    activation_fn: Callable = nn.activation.gelu
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = inputs
        for i in range(len(self.stage_sizes) - 1):
            x = Stage(size=self.stage_sizes[i],
                      num_heads=self.num_heads[i],
                      embed_dim=self.embed_dim[i],
                      embed_kernel_size=self.embed_kernel_size[i],
                      embed_strides=self.embed_strides[i],
                      sa_kernel_size=self.sa_kernel_size[i],
                      use_bias=self.use_bias,
                      activation_fn=self.activation_fn,
                      bn_momentum=self.bn_momentum,
                      bn_epsilon=self.bn_epsilon,
                      expand_ratio=self.expand_ratio,
                      dtype=self.dtype)(x, is_training=is_training)

            l = x.shape[1]
            spatial_ch = int(jnp.sqrt(l))
            x = rearrange(x, 'b (H W) c -> b H W c', H=spatial_ch)

        x = Stage(size=self.stage_sizes[-1],
                  num_heads=self.num_heads[-1],
                  embed_dim=self.embed_dim[-1],
                  embed_kernel_size=self.embed_kernel_size[-1],
                  embed_strides=self.embed_strides[-1],
                  sa_kernel_size=self.sa_kernel_size[-1],
                  use_bias=self.use_bias,
                  activation_fn=self.activation_fn,
                  bn_momentum=self.bn_momentum,
                  bn_epsilon=self.bn_epsilon,
                  expand_ratio=self.expand_ratio,
                  insert_cls=True,
                  dtype=self.dtype)(x, is_training=is_training)

        cls_token = x[:, 0]
        output = nn.Dense(features=self.num_classes,
                          use_bias=True,
                          dtype=self.dtype,
                          kernel_init=nn.initializers.zeros)(cls_token)
        return output
