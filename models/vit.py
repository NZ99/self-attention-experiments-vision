from typing import Tuple, Callable

import haiku as hk
import jax
from jax import numpy as jnp

from models.layers import SelfAttentionBlock, FFBlock, AddAbsPosEmbed, PatchEmbedBlock


class EncoderBlock(hk.Module):

    def __init__(self,
                 num_heads: int,
                 expand_ratio: float = 4,
                 attn_drop_rate: float = 0.,
                 out_drop_rate: float = 0.,
                 activation_fn: Callable = jax.nn.gelu):
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.attn_drop_rate = attn_drop_rate
        self.out_drop_rate = out_drop_rate
        self.activation_fn = activation_fn

    def __call__(self, inputs, is_training: bool):
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(inputs)
        x = SelfAttentionBlock(num_heads=self.num_heads,
                               attn_dropout_rate=self.attn_dropout_rate,
                               out_dropout_rate=self.dropout_rate)(
                                   x, is_training=is_training)
        x = x + inputs

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = FFBlock(expand_ratio=self.expand_ratio,
                    dropout_rate=self.dropout_rate,
                    activation_fn=self.activation_fn,
                    dtype=self.dtype)(y, is_training=is_training)
        output = x + y
        return output


class Encoder(nn.Module):
    num_layers: int
    num_heads: int
    expand_ratio: float = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = AddAbsPosEmbed()(inputs)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)

        for _ in range(self.num_layers):
            x = EncoderBlock(num_heads=self.num_heads,
                             expand_ratio=self.expand_ratio,
                             attn_dropout_rate=self.attn_dropout_rate,
                             dropout_rate=self.dropout_rate,
                             activation_fn=self.activation_fn,
                             dtype=self.dtype)(x, is_training=is_training)

        output = nn.LayerNorm(dtype=self.dtype)(x)
        return output


class ViT(nn.Module):
    num_classes: int
    num_layers: int
    num_heads: int
    embed_dim: int
    patch_shape: Tuple[int]
    expand_ratio: float = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        assert self.embed_dim % self.num_heads == 0

        x = PatchEmbedBlock(patch_shape=self.patch_shape,
                            embed_dim=self.embed_dim,
                            dtype=self.dtype)(inputs)

        b, l, _ = x.shape
        cls_shape = (1, 1, self.embed_dim)
        cls_token = self.param('cls', nn.initializers.zeros, cls_shape)
        cls_token = jnp.tile(cls_token, [b, 1, 1])
        x = jnp.concatenate([cls_token, x], axis=1)

        x = Encoder(num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    expand_ratio=self.expand_ratio,
                    attn_dropout_rate=self.attn_dropout_rate,
                    dropout_rate=self.dropout_rate,
                    activation_fn=self.activation_fn,
                    dtype=self.dtype)(x, is_training=is_training)

        cls_token = x[:, 0]
        output = nn.Dense(features=self.num_classes,
                          dtype=self.dtype,
                          kernel_init=nn.initializers.zeros)(cls_token)
        return output
