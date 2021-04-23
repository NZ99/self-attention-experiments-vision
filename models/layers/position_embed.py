from typing import Callable

from flax.linen import initializers
from flax import linen as nn


class AddAbsPosEmbed(nn.Module):

    embed_init: Callable = initializers.normal(stddev=0.02)

    @nn.compact
    def __call__(self, inputs):
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pos_emb = self.param('pos_embed', self.embed_init, pos_emb_shape)
        output = inputs + pos_emb
        return output