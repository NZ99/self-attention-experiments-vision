from typing import Tuple, Callable

from jax import numpy as jnp
from jax.lax import Precision
from flax.linen import initializers
from flax import linen as nn
from einops import rearrange

from models.layers import PatchEmbedBlock, FFBlock


class MixerBlock(nn.Module):
    tokens_expand_ratio: float
    channels_expand_ratio: float
    activation_fn = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        x = nn.LayerNorm(dtype=self.dype)(inputs)
        x = rearrange(x, '... l d -> ... d l')
        x = FFBlock(expand_ratio=self.tokens_expand_ratio,
                    activation_fn=self.activation_fn,
                    dype=self.dtype,
                    precision=self.precision,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init)(x)
        x = rearrange(x, '... d l -> ... l d')
        x = x + inputs

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = FFBlock(expand_ratio=self.channels_expand_ratio,
                    activation_fn=self.activation_fn,
                    dype=self.dtype,
                    precision=self.precision,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init)(y)
        output = x + y
        return output


class MLPMixer(nn.Module):
    num_classes: int
    num_layers: int
    embed_dim: int
    patch_shape: Tuple[int, int]
    tokens_expand_ratio: float = 0.5
    channels_expand_ratio: float = 4
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, *unused_args, **unused_kwargs):
        x = PatchEmbedBlock(patch_shape=self.patch_shape,
                            embed_dim=self.embed_dim,
                            dtype=self.dtype,
                            precision=self.precision,
                            kernel_init=self.kernel_init)(inputs)
        x = rearrange(x, 'b h w d -> b (h w) d')

        for _ in range(self.num_layers):
            x = MixerBlock(tokens_expand_ratio=self.tokens_expand_ratio,
                           channels_expand_ratio=self.channels_expand_ratio,
                           activation_fn=self.activation_fn,
                           dtype=self.dtype,
                           precision=self.precision,
                           kernel_init=self.kernel_init,
                           bias_init=self.bias_init)(x)

        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = jnp.mean(x, axis=1)
        output = nn.Dense(features=self.num_classes,
                          dtype=self.dtype,
                          precision=self.precision,
                          kernel_init=nn.initializers.zeros,
                          bias_init=self.bias_init)(x)
        return output