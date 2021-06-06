from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j=2)[:, None, :],
                   sincos)
    return (x * cos) + (rotate_every_two(x) * sin)


class FixedPositionalEmbedding(hk.Module):

    def __call__(self, inputs, seq_dim=0):
        dim = inputs.shape[-1]
        intervals = jnp.arange(start=0, stop=dim, step=2, dtype=inputs.dtype)
        inv_freq = 1. / (10e4**intervals / dim)
        t = jnp.arange(inputs.shape[seq_dim], dtype=inputs.dtype)

        freqs = jnp.einsum('i , j -> i j', t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return emb


class RotaryPositionalEmbedding(FixedPositionalEmbedding):

    def __call__(self, inputs, seq_dim=0):
        sincos = super(RotaryPositionalEmbedding,
                       self).__call__(inputs, seq_dim)
        emb = apply_rotary_pos_emb(inputs, sincos)
        return emb


class AddAbsPosEmbed(hk.Module):

    def __init__(self,
                 embed_init: Callable = hk.initializers.normal(stddev=0.02)):
        self.embed_init = embed_init

    def __call__(self, inputs):
        assert inputs.ndim == 3
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pos_emb = hk.get_parameter('pos_embed', pos_emb_shape, self.embed_init)
        output = inputs + pos_emb
        return output
