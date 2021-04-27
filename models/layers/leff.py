import math
from typing import Callable

from einops import rearrange
from flax import linen as nn
from flax.linen import initializers
from jax import numpy as jnp
from jax.lax import Precision

from models.layers import AddAbsPosEmbed, SelfAttentionBlock


class DWConvBlock(nn.Module):
    num_features: int
    kernel_size: int
    activation_fn: Callable = nn.activation.gelu
    padding: str = 'SAME'
    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform
    bias_init: Callable = initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, x_s,):
        # depthwise conv
        h = nn.Conv(
            features=self.num_features,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_bias=self.use_bias,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x_s)
        # h.shape = [b, sqrt(n), sqrt(n), c]
        h = nn.BatchNorm()(h)
        x_d = self.activation_fn(h)

        return x_d


class LeFFBlock(nn.Module):
    dw_conv_kernel_size: int
    expand_ratio: int

    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, x, train: bool = False):
        assert not (self.expand_ratio is None)

        # x.shape = [b, n, c]
        in_ch = x.shape[-1]
        n = x.shape[1]
        hidden_ch = max(1, self.expand_ratio * in_ch)
        h = nn.DenseGeneral(
            axis=-1,
            features=hidden_ch,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)

        # h.shape = [b, n, c*e]
        h = nn.BatchNorm(use_running_average=True)(h)
        x_l1 = self.activation_fn(h)

        # spatial restore
        x_s = rearrange(x_l1, 'b (h w) c -> b h w c', h=int(math.sqrt(n)), w=int(math.sqrt(n)))
        # x_s.shape = [b, sqrt(n), sqrt(n), c]

        # depthwise conv
        h = nn.Conv(
            features=hidden_ch,
            kernel_size=(self.dw_conv_kernel_size,)*2,
            dtype=self.dtype,
            precision=self.precision,
            feature_group_count=hidden_ch,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x_s)

        # h.shape = [b, sqrt(n), sqrt(n), c]
        h = nn.BatchNorm(use_running_average=True)(h)
        x_d = self.activation_fn(h)

        # flatten
        x_f = rearrange(x_d, 'b h w c -> b (h w) c', h=int(math.sqrt(n)), w=int(math.sqrt(n)))
        # x_f.shape = [b, n, c*e]

        h = nn.Dense(
            features=in_ch,
            kernel_init=self.kernel_init,
        )(x_f)
        # h.shape = [b, n, c]

        h = nn.BatchNorm(use_running_average=True)(h)
        x_l2 = self.activation_fn(h)

        return x_l2


class LeFFEncoderBlock(nn.Module):
    num_heads: int
    head_ch: int
    attn_dropout_rate: int

    dw_conv_kernel_size: int
    expand_ratio: int

    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        # x = LN(x + MSA(x))
        # y = LN(x + LeFF(x))

        h = SelfAttentionBlock(
            num_heads=self.num_heads,
            head_ch=self.head_ch,
            out_ch=self.head_ch,
            dropout_rate=self.attn_dropout_rate,
            dtype=self.dtype
        )(inputs, train=train)

        res = inputs + h

        a = nn.LayerNorm(dtype=self.dtype)(res)

        h = LeFFBlock(
            dw_conv_kernel_size=self.dw_conv_kernel_size,
            expand_ratio=self.expand_ratio,
        )(a[:, 1:])

        res = a[:, 1:] + h

        leff = nn.LayerNorm(dtype=self.dtype)(res)

        cls = a[:, 0]
        return cls, jnp.concatenate([jnp.expand_dims(cls, axis=1), leff], axis=1)


class LeFFEncoder(nn.Module):
    num_layers: int
    expand_ratio: int
    dw_conv_kernel_size: int

    num_heads: int
    head_ch: int

    dropout_rate: float = 0.
    attn_dropout_rate: float = 0.
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        x = AddAbsPosEmbed()(inputs)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        cls_tokens = []
        for _ in range(self.num_layers):
            cls, x = LeFFEncoderBlock(
                num_heads=self.num_heads,
                head_ch=self.head_ch,
                attn_dropout_rate=self.attn_dropout_rate,
                dw_conv_kernel_size=self.dw_conv_kernel_size,
                expand_ratio=self.expand_ratio,
                dtype=self.dtype
            )(x)
            cls_tokens.append(cls)

        output = nn.LayerNorm(dtype=self.dtype)(x)
        return output, rearrange(jnp.stack(cls_tokens), 'l b d -> b l d')
