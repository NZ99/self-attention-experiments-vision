from flax import linen as nn
from jax import numpy as jnp

from models.layers import SelfAttentionBlock, FFBlock, AddAbsPosEmbed


class EncoderBlock(nn.Module):
    num_heads: int
    head_ch: int
    out_ch: int
    mlp_ch: int

    is_lca: bool = False
    dropout_rate: float = 0.
    attn_dropout_rate: float = 0.
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = SelfAttentionBlock(num_heads=self.num_heads,
                               head_ch=self.head_ch,
                               out_ch=self.out_ch,
                               dropout_rate=self.attn_dropout_rate,
                               dtype=self.dtype)(x, train=train)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # LCA: only consider last token in sequence
        if self.is_lca:
            x += jnp.expand_dims(inputs[:, -1, :], axis=1)
        else:
            x += inputs

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
    out_ch: int
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
                             out_ch=self.out_ch,
                             mlp_ch=self.mlp_ch,
                             dropout_rate=self.dropout_rate,
                             attn_dropout_rate=self.attn_dropout_rate,
                             dtype=self.dtype)(x)

        output = nn.LayerNorm(dtype=self.dtype)(x)
        return output