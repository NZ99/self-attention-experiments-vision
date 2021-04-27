from typing import Callable, Tuple

import jax
from jax import numpy as jnp
from jax.lax import Precision

from flax.linen import initializers
from flax import linen as nn

from models.layers import FFBlock, PatchEmbedBlock, SelfAttentionBlock, AddAbsPosEmbed


class EncoderBlock(nn.Module):

    num_heads: int
    head_ch: int
    mlp_ch: int
    dropout_rate: float = 0.
    attn_dropout_rate: float = 0.
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        in_ch = inputs.shape[-1]
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = SelfAttentionBlock(num_heads=self.num_heads,
                               head_ch=self.head_ch,
                               dropout_rate=self.attn_dropout_rate,
                               dtype=self.dtype)(x, train=train)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = x + inputs

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = FFBlock(hidden_ch=self.mlp_ch,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype)(y, train=train)

        output = x + y
        return output


class Encoder(nn.Module):

    num_layers: int
    num_heads: int
    head_ch: int
    mlp_ch: int
    dropout_rate: float = 0.
    attn_dropout_rate: float = 0.
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool):
        x = AddAbsPosEmbed()(inputs)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        for _ in range(self.num_layers):
            x = EncoderBlock(num_heads=self.num_heads,
                             head_ch=self.head_ch,
                             mlp_ch=self.mlp_ch,
                             dropout_rate=self.dropout_rate,
                             attn_dropout_rate=self.attn_dropout_rate,
                             dtype=self.dtype)(x)

        output = nn.LayerNorm(dtype=self.dtype)(x)
        return output


class ViT(nn.Module):

    num_classes: int
    patch_shape: Tuple[int, int]
    embed_dim: int
    num_layers: int
    num_heads: int
    head_ch: int
    mlp_ch: int
    dropout_rate: float = 0.
    attn_dropout_rate: float = 0.
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool):
        x = PatchEmbedBlock(patch_shape=self.patch_shape,
                            dtype=self.dtype)(inputs)
        b, l, _ = x.shape
        cls_shape = (1, 1, self.embed_dim)
        cls_token = self.param('cls', initializers.zeros, cls_shape)
        cls_token = jnp.tile(cls_token, [b, 1, 1])
        x = jnp.concatenate([cls_token, x], axis=1)
        x = Encoder(num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    head_ch=self.head_ch,
                    mlp_ch=self.embed_dim,
                    dropout_rate=self.dropout_rate,
                    attn_dropout_rate=self.attn_dropout_rate,
                    dtype=self.dtype)(x)
        x = x[:, 0]
        output = nn.Dense(
            features=self.num_classes,
            use_bias=True,
            dtype=self.dtype,
            kernel_init=initializers.zeros,
        )(x)
        return output
