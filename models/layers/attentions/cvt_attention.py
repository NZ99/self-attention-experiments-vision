from typing import Callable, Sequence

from functools import partial

from jax import numpy as jnp
from jax.nn import initializers
from jax.lax import Precision

from flax import linen as nn

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


class CvTSelfAttentionBlock(nn.Module):
    num_heads: int
    head_ch: int
    out_ch: int
    kernel_size: int = 3
    strides: Sequence[int] = [1, 2, 2]
    use_bias: bool = False
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        assert len(self.strides) == 3
        assert inputs.ndim == 3
        q_strides, k_strides, v_strides = self.strides
        b, l, c = inputs.shape
        out_ch = self.out_ch if self.out_ch is not None else c
        spatial_ch = int(jnp.ceil(jnp.sqrt(l)))
        inputs = jnp.pad(inputs, ((0, 0), (0, spatial_ch**2 - l), (0, 0)))
        inputs = rearrange(inputs, 'b (H W) c -> b H W c', W=spatial_ch)

        conv_proj = partial(ConvProjectionBlock,
                            out_ch=self.num_heads * self.head_ch,
                            kernel_size=self.kernel_size,
                            use_bias=self.use_bias,
                            bn_momentum=self.bn_momentum,
                            bn_epsilon=self.bn_epsilon,
                            dtype=self.dtype,
                            precision=self.precision,
                            kernel_init=self.kernel_init,
                            bias_init=self.bias_init)

        query = conv_proj(strides=q_strides)(inputs, is_training=is_training)
        key = conv_proj(strides=k_strides)(inputs, is_training=is_training)
        value = conv_proj(strides=v_strides)(inputs, is_training=is_training)

        query = rearrange(query, 'b H W (h d) -> b (H W) h d', h=self.num_heads)
        key = rearrange(key, 'b H W (h d) -> b (H W) h d', h=self.num_heads)
        value = rearrange(value, 'b H W (h d) -> b (H W) h d', h=self.num_heads)

        query = query / jnp.sqrt(self.head_ch)

        attn_weights = jnp.einsum('... q h d, ... k h d -> ... h q k',
                                  query,
                                  key,
                                  precision=self.precision)

        attn_weights = nn.softmax(attn_weights)

        attn_scores = jnp.einsum('... h q k, ... k h d -> ... q h d',
                                 attn_weights,
                                 value,
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
