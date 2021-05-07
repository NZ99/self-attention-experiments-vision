from functools import partial
from typing import Callable, Optional, Tuple

from flax import linen as nn
from flax.linen import initializers
from jax import numpy as jnp
from jax.lax import Precision

from einops import rearrange


class ConvProjectionBlock(nn.Module):
    out_ch: int
    kernel_size: int = 3
    strides: int = 1
    use_bias: bool = True
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        in_ch = inputs.shape[-1]

        conv = partial(nn.Conv,
                       dtype=self.dtype,
                       precision=self.precision,
                       kernel_init=self.kernel_init)

        x = conv(features=in_ch,
                 kernel_size=(self.kernel_size, self.kernel_size),
                 strides=(self.strides, self.strides),
                 padding='SAME',
                 feature_group_count=in_ch,
                 use_bias=False)(inputs)
        x = nn.BatchNorm(use_running_average=not is_training,
                         momentum=self.bn_momentum,
                         epsilon=self.bn_epsilon,
                         dtype=self.dtype)(x)
        output = conv(features=self.out_ch,
                      kernel_size=(1, 1),
                      use_bias=self.use_bias,
                      bias_init=self.bias_init)(x)
        return output


def zero_pad_and_reshape(inputs):
    assert inputs.ndim == 3
    _, l, in_ch = inputs.shape
    spatial_ch = int(jnp.ceil(jnp.sqrt(l)))
    inputs = jnp.pad(inputs, ((0, 0), (0, spatial_ch**2 - l), (0, 0)))
    inputs = rearrange(inputs, 'b (H W) c -> b H W c', W=spatial_ch)
    return inputs


class CvTAttentionBlock(nn.Module):
    num_heads: int
    head_ch: Optional[int] = None
    out_ch: Optional[int] = None
    talking_heads: bool = False
    attn_dropout_rate: float = 0.
    out_dropout_rate: float = 0.
    kernel_size: int = 3
    strides: Tuple[int] = (1, 2, 2)
    use_bias: bool = False
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, is_training: bool):
        assert len(self.strides) == 3
        q_strides, k_strides, v_strides = self.strides

        in_ch = inputs_q.shape[-1]
        assert in_ch % self.num_heads == 0
        head_ch = self.head_ch or int(in_ch / self.num_heads)
        out_ch = self.out_ch or in_ch

        inputs_q = zero_pad_and_reshape(inputs_q)
        inputs_kv = zero_pad_and_reshape(inputs_kv)

        conv_proj = partial(ConvProjectionBlock,
                            out_ch=self.num_heads * head_ch,
                            kernel_size=self.kernel_size,
                            use_bias=self.use_bias,
                            bn_momentum=self.bn_momentum,
                            bn_epsilon=self.bn_epsilon,
                            dtype=self.dtype,
                            precision=self.precision,
                            kernel_init=self.kernel_init,
                            bias_init=self.bias_init)

        query = conv_proj(strides=q_strides)(inputs_q, is_training=is_training)
        key = conv_proj(strides=k_strides)(inputs_kv, is_training=is_training)
        value = conv_proj(strides=v_strides)(inputs_kv, is_training=is_training)

        query = rearrange(query, 'b H W (h d) -> b (H W) h d', h=self.num_heads)
        key = rearrange(key, 'b H W (h d) -> b (H W) h d', h=self.num_heads)
        value = rearrange(value, 'b H W (h d) -> b (H W) h d', h=self.num_heads)

        query = query / jnp.sqrt(head_ch)

        attn_weights = jnp.einsum('... q h d, ... k h d -> ... h q k',
                                  query,
                                  key,
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

        attn_weights = nn.Dropout(rate=self.attn_dropout_rate)(
            attn_weights, deterministic=not is_training)

        attn_scores = jnp.einsum('... h q k, ... k h d -> ... q h d',
                                 attn_weights,
                                 value,
                                 precision=self.precision)

        output = nn.DenseGeneral(features=out_ch,
                                 axis=(-2, -1),
                                 use_bias=self.use_bias,
                                 dtype=self.dtype,
                                 precision=self.precision,
                                 kernel_init=self.kernel_init,
                                 bias_init=self.bias_init)(attn_scores)

        output = nn.Dropout(rate=self.out_drop_rate)(
            output, deterministic=not is_training)

        return output


class CvTSelfAttentionBlock(CvTAttentionBlock):

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        return super().__call__(inputs, inputs, is_training)
