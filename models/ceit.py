from functools import partial
from typing import Tuple, Callable, Optional

from flax import linen as nn
from flax.linen import initializers
from jax import numpy as jnp

from models.layers import Image2TokenBlock, AttentionBlock, SelfAttentionBlock, LeFFBlock, FFBlock


class LCSelfAttentionBlock(AttentionBlock):

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        inputs_q = jnp.expand_dims(inputs[:, -1, :], axis=1)
        return super().__call__(inputs_q, inputs, is_training=is_training)


class EncoderBlock(nn.Module):
    num_heads: int
    expand_ratio: float = 4
    leff_kernel_size: Optional[int] = 3
    activation_fn: Callable = nn.activation.gelu
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = SelfAttentionBlock(num_heads=self.num_heads,
                               dtype=self.dtype)(inputs,
                                                 is_training=is_training)
        x += inputs
        x = nn.LayerNorm(dtype=self.dtype)(x)

        y = LeFFBlock(expand_ratio=self.expand_ratio,
                      kernel_size=self.leff_kernel_size,
                      activation_fn=self.activation_fn,
                      bn_momentum=self.bn_momentum,
                      bn_epsilon=self.bn_epsilon,
                      dtype=self.dtype)(x, is_training=is_training)
        y = x + y
        output = nn.LayerNorm(dtype=self.dtype)(y)
        return output


class Encoder(nn.Module):
    num_layers: int
    num_heads: int
    expand_ratio: float = 4
    leff_kernel_size: int = 3
    activation_fn: Callable = nn.activation.gelu
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        encoder_block = partial(EncoderBlock,
                                num_heads=self.num_heads,
                                expand_ratio=self.expand_ratio,
                                leff_kernel_size=self.leff_kernel_size,
                                activation_fn=self.activation_fn,
                                bn_momentum=self.bn_momentum,
                                bn_epsilon=self.bn_epsilon,
                                dtype=self.dtype)

        x = encoder_block()(inputs, is_training=is_training)
        cls_tokens_lst = [jnp.expand_dims(x[:, 0], axis=1)]
        for _ in range(self.num_layers - 1):
            x = encoder_block()(x, is_training=is_training)
            cls_tokens_lst.append(jnp.expand_dims(x[:, 0], axis=1))

        return jnp.concatenate(cls_tokens_lst, axis=1)


class LCAEncoderBlock(nn.Module):
    num_heads: int
    expand_ratio: float = 4
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = LCSelfAttentionBlock(num_heads=self.num_heads,
                                 dtype=self.dtype)(inputs,
                                                   is_training=is_training)
        x += inputs
        x = nn.LayerNorm(dtype=self.dtype)(x)

        y = FFBlock(expand_ratio=self.expand_ratio,
                    activation_fn=self.activation_fn,
                    dtype=self.dtype)(x, is_training=is_training)
        y = x + y
        output = nn.LayerNorm(dtype=self.dtype)(y)
        return output


class CeiT(nn.Module):
    num_classes: int
    num_layers: int
    num_heads: int
    embed_dim: int
    patch_shape: Tuple[int] = (4, 4)
    num_ch: int = 32
    conv_kernel_size: int = 7
    conv_stride: int = 2
    pool_window_size: int = 3
    pool_stride: int = 2
    expand_ratio: float = 4
    leff_kernel_size: int = 3
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        assert self.embed_dim % self.num_heads == 0

        x = Image2TokenBlock(patch_shape=self.patch_shape,
                             num_ch=self.num_ch,
                             conv_kernel_size=self.conv_kernel_size,
                             conv_stride=self.conv_stride,
                             pool_window_size=self.pool_window_size,
                             pool_stride=self.pool_stride,
                             embed_dim=self.embed_dim,
                             bn_momentum=self.bn_momentum,
                             bn_epsilon=self.bn_epsilon,
                             dtype=self.dtype)(inputs, is_training=is_training)

        b = x.shape[0]
        cls_shape = (1, 1, self.embed_dim)
        cls_token = self.param('cls', initializers.zeros, cls_shape)
        cls_token = jnp.tile(cls_token, [b, 1, 1])

        x = jnp.concatenate([cls_token, x], axis=1)

        cls_tokens = Encoder(num_layers=self.num_layers,
                             num_heads=self.num_heads,
                             expand_ratio=self.expand_ratio,
                             leff_kernel_size=self.leff_kernel_size,
                             bn_momentum=self.bn_momentum,
                             bn_epsilon=self.bn_epsilon,
                             dtype=self.dtype)(x, is_training=is_training)

        cls_tokens = LCSelfAttentionBlock(num_heads=self.num_heads,
                                          dtype=self.dtype)(
                                              cls_tokens,
                                              is_training=is_training)
        cls = cls_tokens[:, -1]
        output = nn.Dense(features=self.num_classes,
                          use_bias=True,
                          dtype=self.dtype,
                          kernel_init=nn.initializers.zeros)(cls)
        return output
