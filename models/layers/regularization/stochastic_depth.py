import jax.numpy as jnp
import jax.random as random
import flax.linen as nn


class StochDepth(hk.Module):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False, name=None):
        super().__init__(name=name)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def __call__(self, x, is_training) -> jnp.ndarray:
        if not is_training:
            return x
        batch_size = x.shape[0]
        r = jax.random.uniform(hk.next_rng_key(), [batch_size, 1, 1, 1],
                               dtype=x.dtype)
        keep_prob = 1. - self.drop_rate
        binary_tensor = jnp.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


class StochasticDepthBlock(nn.Module):
    drop_rate: float
    scale_by_keep: bool = True

    @nn.compact
    def __call__(self, inputs, is_training: bool):

        if not is_training or self.drop_rate == 0.:
            return inputs

        keep_prob = 1. - self.drop_rate
        rng = self.make_rng('stochastic_depth')

        b = inputs.shape[0]
        random_tensor = random.uniform(rng, [b, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = jnp.floor(keep_prob + random_tensor)

        if self.scale_by_keep:
            x = inputs / keep_prob

        output = x * binary_tensor
        return output
