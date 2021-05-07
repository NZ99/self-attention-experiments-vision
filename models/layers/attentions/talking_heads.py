from flax import linen as nn
from jax import numpy as jnp


class TalkingHeadsBlock(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, inputs):
        transform = self.param('talking_heads_transform',
                               nn.initializers.orthogonal,
                               (self.num_heads, self.num_heads))
        output = jnp.einsum('h i, ... h q k -> ... i q k', transform, inputs)
        return output