from flax import linen as nn
from jax import numpy as jnp

from models.layers.encoder import EncoderBlock


class LCAEncoder(nn.Module):
    num_heads: int
    head_ch: int
    mlp_ch: int

    num_layers: int = 1
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            # an LCA block follows the Transformer MSA with FFN Block,
            # but only computes representation for the last token
            x = EncoderBlock(
                num_heads=self.num_heads,
                head_ch=self.head_ch,
                out_ch=self.head_ch,
                mlp_ch=self.mlp_ch,
                is_lca=True
            )(x)

        output = nn.LayerNorm(dtype=self.dtype)(x)
        return output
