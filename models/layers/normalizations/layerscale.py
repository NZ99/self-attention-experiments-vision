import haiku as hk
import jax.numpy as jnp


class LayerScaleBlock(hk.Module):

    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, inputs):
        scale = hk.get_parameter('layerscale',
                                 inputs.shape[-1],
                                 inputs.dtype,
                                 init=hk.initializers.constant(self.eps))
        scale = jnp.broadcast_to(scale, inputs.shape)
        return inputs * scale
