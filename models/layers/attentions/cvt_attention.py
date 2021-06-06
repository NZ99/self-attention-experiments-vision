from typing import Optional, Tuple

import haiku as hk
import jax
from jax import numpy as jnp
from einops import rearrange

from models.layers.attentions import TalkingHeadsBlock


class ConvProjectionBlock(hk.Module):

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 use_bias: bool = True):

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = use_bias

        self.depthwise_conv = hk.DepthwiseConv2D(channel_multiplier=1,
                                                 kernel_shape=self.kernel_size,
                                                 stride=self.stride,
                                                 with_bias=False)
        self.bn = hk.BatchNorm(create_scale=True,
                               create_offset=True,
                               decay_rate=0.999)
        self.pointwise_conv = hk.Conv2D(self.out_ch,
                                        kernel_shape=1,
                                        use_bias=self.use_bias)

    def __call__(self, inputs, is_training: bool, test_local_stats: bool):
        x = self.depthwise_conv(inputs)
        x = self.bn(x, is_training, test_local_stats)
        x = self.pointwise_conv(x)
        return x


class CvTAttentionBlock(hk.Module):

    def __init__(self,
                 in_ch: int,
                 out_ch: Optional[int] = None,
                 num_heads: Optional[int] = None,
                 head_ch: Optional[int] = None,
                 kernel_size: int = 3,
                 strides: Tuple[int, int, int] = (1, 2, 2),
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

        self.kernel_size = kernel_size
        q_stride, k_stride, v_stride = strides

        self.talking_heads = talking_heads

        if self.talking_heads:
            self.pre_softmax = TalkingHeadsBlock(num_heads=self.num_heads)
            self.post_softmax = TalkingHeadsBlock(num_heads=self.num_heads)

        self.attn_drop_rate = attn_drop_rate
        self.out_drop_rate = out_drop_rate

        self.to_q = ConvProjectionBlock(in_ch=self.in_ch,
                                        out_ch=self.num_heads * self.head_ch,
                                        kernel_size=self.kernel_size,
                                        stride=q_stride,
                                        use_bias=use_bias)
        self.to_k = ConvProjectionBlock(in_ch=self.in_ch,
                                        out_ch=self.num_heads * self.head_ch,
                                        kernel_size=self.kernel_size,
                                        stride=k_stride,
                                        use_bias=use_bias)
        self.to_v = ConvProjectionBlock(in_ch=self.in_ch,
                                        out_ch=self.num_heads * self.head_ch,
                                        kernel_size=self.kernel_size,
                                        stride=v_stride,
                                        use_bias=use_bias)

        self.to_out = hk.Linear(self.out_ch, with_bias=use_bias)

    def __call__(self, inputs_q, inputs_kv, is_training: bool,
                 test_local_stats: bool):
        assert inputs_q.ndim == inputs_kv.ndim == 4

        q = self.to_q(inputs_q, is_training, test_local_stats)
        k = self.to_k(inputs_kv, is_training, test_local_stats)
        v = self.to_v(inputs_kv, is_training, test_local_stats)

        q = rearrange(q, 'b H W (h d) -> b h (H W) d', h=self.num_heads)
        k = rearrange(k, 'b H W (h d) -> b h (H W) d', h=self.num_heads)
        v = rearrange(v, 'b H W (h d) -> b h (H W) d', h=self.num_heads)

        q = q / jnp.sqrt(self.head_ch)

        weights = jnp.einsum('b h l d, b h k d -> b h l k', q, k)

        if self.talking_heads:
            weights = self.pre_softmax(weights)

        weights = jax.nn.softmax(weights)

        if self.talking_heads:
            weights = self.post_softmax(weights)

        weights = hk.dropout(hk.next_rng_key(), self.attn_drop_rate, weights)

        scores = jnp.einsum('b h l k, b h k d  -> b h l d', weights, v)

        scores = rearrange(scores, 'b h l d -> b l (h d)')

        output = self.to_out(scores)
        output = hk.dropout(hk.next_rng_key(), self.out_drop_rate, output)
        return output


class SelfAttentionBlock(CvTAttentionBlock):

    def __call__(self, inputs, is_training: bool, test_local_stats: bool):
        return super().__call__(inputs, inputs, is_training, test_local_stats)
