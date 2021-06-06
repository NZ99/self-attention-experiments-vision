import haiku as hk
import jax.numpy as jnp
import jax.random as random


class StochasticDepthBlock(hk.Module):

    def __init__(self, drop_rate: float, scale_by_keep: bool = True):
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def __call__(self, inputs, is_training: bool):
        if not is_training or self.drop_rate == 0.:
            return inputs

        b = inputs.shape[0]
        keep_prob = 1. - self.drop_rate
        shape = [b,] + ([1,] * (inputs.ndim - 1))

        random_tensor = random.uniform(hk.next_rng_key(),
                                       shape,
                                       dtype=inputs.dtype)
        binary_tensor = jnp.floor(keep_prob + random_tensor)

        if self.scale_by_keep:
            x = inputs / keep_prob

        output = x * binary_tensor
        return output
