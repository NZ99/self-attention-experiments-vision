import haiku as hk
from jax import numpy as jnp


class TalkingHeadsBlock(hk.Module):

    def __init__(self, num_heads: int):
        self.num_heads = num_heads

    def __call__(self, inputs):
        transform_shape = (self.num_heads, self.num_heads)
        transform = hk.get_parameter('pre_softmax',
                                     hk.initializers.orthogonal(),
                                     transform_shape)
        output = jnp.einsum('h i, b h ... -> b i ...', transform, inputs)
        return output
