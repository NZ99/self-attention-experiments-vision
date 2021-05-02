from functools import partial
from typing import Any, Callable

from einops import rearrange
from flax import linen as nn
from flax.linen import initializers
from jax import numpy as jnp
from jax import random
from jax.lax import Precision

PRNGKey = Any


class SelfAttentionBlock(nn.Module):
    num_heads: int
    head_ch: int
    out_ch: int
    talking_heads: bool = False
    dropout_rate: float = 0.
    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        dense = partial(nn.DenseGeneral,
                        axis=-1,
                        features=(self.num_heads, self.head_ch),
                        use_bias=self.use_bias,
                        dtype=self.dtype,
                        precision=self.precision,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init)

        queries, keys, values = (dense(name='queries')(inputs),
                                 dense(name='keys')(inputs),
                                 dense(name='values')(inputs))

        queries = queries / jnp.sqrt(self.head_ch)

        attn_weights = jnp.einsum('... q h d, ... k h d -> ... h q k',
                                  queries,
                                  keys,
                                  precision=self.precision)
        if self.talking_heads:
            pre_softmax_transform = self.param('pre_softmax', self.kernel_init,
                                               (self.num_heads, self.num_heads))
            attn_weights = jnp.einsum('... h q k, h i -> ... i q k',
                                      attn_weights,
                                      pre_softmax_transform,
                                      precision=self.precision)

        attn_weights = nn.softmax(attn_weights)

        if self.talking_heads:
            post_softmax_transform = self.param(
                'post_softmax', self.kernel_init,
                (self.num_heads, self.num_heads))
            attn_weights = jnp.einsum('... i q k, i h -> ... h q k',
                                      attn_weights,
                                      post_softmax_transform,
                                      precision=self.precision)

        if is_training and self.dropout_rate > 0.:
            keep_prob = 1.0 - self.dropout_rate
            dropout_rng = self.make_rng('dropout')
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
            multiplier = (keep.astype(attn_weights.dtype) /
                          jnp.asarray(keep_prob, dtype=self.dtype))
            attn_weights = attn_weights * multiplier

        attn_scores = jnp.einsum('... h q k, ... k h d -> ... q h d',
                                 attn_weights,
                                 values,
                                 precision=self.precision)

        if (self.num_heads * self.head_ch) == self.out_ch:
            output = rearrange(attn_scores, '... q h d -> ... q (h d)')
        else:
            output = nn.DenseGeneral(features=self.out_ch,
                                     axis=(-2, -1),
                                     use_bias=self.use_bias,
                                     dtype=self.dtype,
                                     precision=self.precision,
                                     kernel_init=self.kernel_init,
                                     bias_init=self.bias_init)(attn_scores)
        return output
