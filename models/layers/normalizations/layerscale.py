import haiku as hk
import jax.numpy as jnp


def full(eps: float, shape, dtype=jnp.float32):
    return jnp.full(shape, eps, dtype)


class LayerScaleBlock(hk.Module):

    def __init__(self, in_ch: int, eps: float):
        self.eps = eps

    def __call__(self, inputs):
        scale = hk.get_parameter('layerscale',
                                 inputs.shape[-1],
                                 inputs.dtype,
                                 init=full(self.eps))
        scale = jnp.broadcast_to(scale, inputs.shape)
        return inputs * scale
