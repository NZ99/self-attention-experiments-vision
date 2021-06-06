from typing import Optional

import haiku as hk
import jax
from jax import numpy as jnp
from einops import rearrange

from models.layers.attentions import TalkingHeadsBlock


class AttentionBlock(hk.Module):

    def __init__(self,
                 in_ch: int,
                 out_ch: Optional[int] = None,
                 num_heads: Optional[int] = None,
                 head_ch: Optional[int] = None,
                 talking_heads: bool = False,
                 attn_drop_rate: float = 0.,
                 out_drop_rate: float = 0.,
                 use_bias: bool = False):

        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch

        if num_heads is None:
            if head_ch is None:
                raise ValueError('Must provide one of num_heads or head_ch')
            self.head_ch = head_ch
            self.num_heads = self.in_ch // head_ch
        else:
            self.head_ch = self.in_ch // num_heads
            self.num_heads = num_heads

        self.talking_heads = talking_heads

        if self.talking_heads:
            self.pre_softmax = TalkingHeadsBlock(num_heads=self.num_heads)
            self.post_softmax = TalkingHeadsBlock(num_heads=self.num_heads)

        self.attn_drop_rate = attn_drop_rate
        self.out_drop_rate = out_drop_rate

        self.to_q = hk.Linear(self.num_heads * self.head_ch, with_bias=use_bias)
        self.to_k = hk.Linear(self.num_heads * self.head_ch, with_bias=use_bias)
        self.to_v = hk.Linear(self.num_heads * self.head_ch, with_bias=use_bias)

        self.to_out = hk.Linear(self.out_ch, with_bias=use_bias)

    def __call__(self, inputs_q, inputs_kv, is_training: bool):
        assert inputs_q.ndim == inputs_kv.ndim == 3

        q = self.to_q(inputs_q)
        k = self.to_k(inputs_kv)
        v = self.to_v(inputs_kv)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        q = q / jnp.sqrt(self.head_ch)

        weights = jnp.einsum('b h l d, b h k d -> b h l k', q, k)

        if self.talking_heads:
            weights = self.pre_softmax(weights)

        weights = jax.nn.softmax(weights)

        if self.talking_heads:
            weights = self.post_softmax(weights)

        weights = hk.dropout(hk.next_rng_key(), self.attn_drop_rate, weights)

        scores = jnp.einsum('b h l k, b h k d -> b h l d', weights, v)

        scores = rearrange(scores, 'b h l d -> b l (h d)')

        output = self.to_out(scores)
        output = hk.dropout(hk.next_rng_key(), self.out_drop_rate, output)
        return output


class SelfAttentionBlock(AttentionBlock):

    def __call__(self, inputs, is_training: bool):
        return super().__call__(inputs, inputs, is_training=is_training)
