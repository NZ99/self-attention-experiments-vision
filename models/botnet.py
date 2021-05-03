from functools import partial
from typing import Any, Tuple, Callable, Sequence

from jax import numpy as jnp
from jax.nn import initializers
from jax.lax import Precision

from flax import linen as nn
from flax import struct

from einops import rearrange

from models.layers import SqueezeExciteBlock

ModuleDef = Any


class BottleneckResNetBlock(nn.Module):
    """ResNet bottleneck block.
    Attributes:
        filters: number of filters to use in the first and second convolutions
        conv: a convolution module
        norm: a normalization module such as nn.BatchNorm
        strides: strides to use in the second convolution, either (1, 1) or (2, 2)
    """
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    strides: Tuple[int, int]
    se_ratio: float = 0.0625
    projection_factor: int = 4
    activation_fn: Callable = nn.activation.swish
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        """Passes the input through a resnet bottleneck block.
        Arguments:
            inputs:     [batch_size, height, width, dim]
        Returns:
            output:     [batch_size, height, width, dim * config.projection_factor]
        """
        residual = inputs

        y = self.conv(self.filters, kernel_size=(1, 1))(inputs)
        y = self.norm()(y)
        y = self.activation_fn(y)
        y = self.conv(self.filters, kernel_size=(3, 3), strides=self.strides)(y)
        y = self.norm()(y)
        y = self.activation_fn(y)
        y = self.conv(self.filters * self.projection_factor,
                      kernel_size=(1, 1))(y)
        y = self.norm(scale_init=initializers.zeros)(y)

        if self.se_ratio is not None:
            y = SqueezeExciteBlock(se_ratio=self.se_ratio,
                                   activation_fn=self.activation_fn,
                                   dtype=self.dtype)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * self.projection_factor,
                                 kernel_size=(1, 1),
                                 strides=self.strides)(residual)
            residual = self.norm()(residual)
            residual = self.activation_fn(residual)

        y = self.activation_fn(residual + y)
        return y


class RelativeLogits(nn.Module):
    """Relative logits module.
    Attributes:
        config: BoTNet configuration
    """
    head_ch: int

    @staticmethod
    def _to_absolute_logits(rel_logits):
        """Converts relative logits into absolute logits.
        Arguments:
            rel_logits: [batch_size, heads, length, 2 * length - 1]
        Returns:
            output:     [batch_size, heads, length, length]
        """
        b, h, l, _ = rel_logits.shape
        col_pad = jnp.zeros((b, h, l, 1))
        x = jnp.concatenate((rel_logits, col_pad), axis=3)
        x = rearrange(x, 'b h l v -> b h (l v)')
        flat_pad = jnp.zeros((b, h, l - 1))
        x = jnp.concatenate((x, flat_pad), axis=2)
        x = rearrange(x, 'b h (l v) -> b h l v', l=l + 1)
        out = x[:, :, :l, (l - 1):]
        return out

    @staticmethod
    def _relative_logits_1d(query, rel_pos_emb):
        """Computes relative logits along one dimension.
        Arguments:
            query:          [batch_size, heads, height, width, dim]
            rel_pos_emb:    [2 * width - 1, dim]
        Returns:
            output:         [batch_size, heads, height, height, width, width]
        """
        H = query.shape[2]
        x = jnp.einsum('b h H W d, m d -> b h H W m', query, rel_pos_emb)
        x = rearrange(x, 'b h H W m -> b (h H) W m', H=H)
        x = RelativeLogits._to_absolute_logits(x)
        x = rearrange(x, 'b (h H) W V -> b h H W V', H=H)
        x = jnp.expand_dims(x, axis=3)
        x = jnp.tile(x, [1, 1, 1, H, 1, 1])
        return x

    @nn.compact
    def __call__(self, query):
        """Computes relative position embedding logits.
        Arguments:
            query:      [batch_size, heads, height, width, dim]
        Returns:
            output:     [batch_size, heads, height, width, height, width]
        """

        _, _, H, W, _ = query.shape

        rel_pos_emb_w_shape = (2 * W - 1, self.head_ch)
        rel_pos_emb_w = self.param(
            'rel_pos_emb_w', initializers.normal(stddev=self.head_ch**-0.5),
            rel_pos_emb_w_shape)

        rel_pos_emb_h_shape = (2 * H - 1, self.head_ch)
        rel_pos_emb_h = self.param(
            'rel_pos_emb_h', initializers.normal(stddev=self.head_ch**-0.5),
            rel_pos_emb_h_shape)

        rel_logits_w = self._relative_logits_1d(query, rel_pos_emb_w)
        rel_logits_w = rearrange(rel_logits_w, 'b h H I W V -> b h H W I V')

        rel_logits_h = self._relative_logits_1d(
            rearrange(query, 'b h H W d -> b h W H d'), rel_pos_emb_h)
        rel_logits_h = rearrange(rel_logits_h, 'b h W V H I -> b h H W I V')
        out = rel_logits_h + rel_logits_w
        return out


class BoTMHSA(nn.Module):
    """Multi-head self-attention module as described in figure 4.
    Attributes:
        config: BoTNet configuration
    """
    num_heads: int
    head_ch: int

    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.he_uniform()
    bias_init: Callable = initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs):
        """Applies multi-head self-attention on the input data.
        Arguments:
            inputs_q:   [batch_size, height, width, dim]
        Returns:
            output:     [batch_size, height, width, dim]
        """
        conv = partial(nn.Conv,
                       features=self.num_heads * self.head_ch,
                       kernel_size=(1, 1),
                       use_bias=False,
                       precision=self.precision,
                       kernel_init=self.kernel_init)

        query, key, value = (conv(dtype=self.dtype, name="query")(inputs),
                             conv(dtype=self.dtype, name="key")(inputs),
                             conv(dtype=self.dtype, name="value")(inputs))
        query, key, value = (rearrange(query,
                                       'b H W (h d) -> b h H W d',
                                       h=self.num_heads),
                             rearrange(key,
                                       'b H W (h d) -> b h H W d',
                                       h=self.num_heads),
                             rearrange(value,
                                       'b H W (h d) -> b h H W d',
                                       h=self.num_heads))

        query = query / jnp.sqrt(self.head_dim).astype(self.dtype)

        attn_weights = jnp.einsum('b h H W d, b h P Q d -> b h H W P Q',
                                  query,
                                  key,
                                  precision=self.precision)
        attn_weights = attn_weights + RelativeLogits(
            head_ch=self.head_ch)(query)
        attn_weights = nn.softmax(attn_weights).astype(self.dtype)
        attn_out = jnp.einsum('b h H W P Q, b h H W d -> b H W h d',
                              attn_weights,
                              value,
                              precision=self.precision)
        attn_out = rearrange(attn_out, 'b H W h d -> b H W (h d)')
        return attn_out


class BoTBlock(nn.Module):
    """Bottleneck Transformer block module.
    Attributes:
        config: BoTNet configuration
        filters: number of filters to use in the first convolution
        conv: a convolution module
        norm: a normalization module such as nn.BatchNorm
        strides: strides to use in the second convolution, either (1, 1) or (2, 2)
    """
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    strides: Tuple[int, int]
    projection_factor: int = 4
    activation_fn: ModuleDef = nn.swish

    @nn.compact
    def __call__(self, inputs):
        """Passes the input through a bottleneck transformer block.
        Arguments:
            inputs:     [batch_size, height, width, dim]
        Returns:
            output:     [batch_size, height, width, dim * config.projection_factor]
        """
        residual = inputs
        cfg = self.config

        y = self.conv(self.filters, kernel_size=(1, 1))(inputs)
        y = self.norm()(y)
        y = self.activation_fn(y)
        y = BoTMHSA(config=cfg)(y)
        if self.strides == (2, 2):
            y = nn.avg_pool(y,
                            window_shape=(2, 2),
                            strides=self.strides,
                            padding='SAME')
        y = self.norm()(y)
        y = self.activation_fn(y)
        y = self.conv(self.filters * self.projection_factor,
                      kernel_size=(1, 1))(y)
        y = self.norm(scale_init=initializers.zeros)(y)

        if self.strides == (2, 2) or residual.shape != y.shape:
            residual = self.conv(self.filters * self.projection_factor,
                                 kernel_size=(1, 1),
                                 strides=self.strides)(residual)
            residual = self.norm()(residual)
            residual = self.activation_fn(residual)

        y = self.activation_fn(residual + y)
        return y


class BoTNet(nn.Module):
    """Bottleneck Transformer Network module.
    Attributes:
        config: BoTNet configuration
    """
    num_classes: int
    stage_sizes: Sequence[int]
    stride_one: bool = True
    se_ratio: float = 0.0625
    activation_fn: ModuleDef = nn.swish
    num_heads: int = 4
    head_dim: int = 128
    initial_filters: int = 64
    projection_factor: int = 4
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    precision: Any = Precision.DEFAULT
    kernel_init: Callable = initializers.he_uniform()
    bias_init: Callable = initializers.normal(stddev=1e-6)
    posemb_init: Callable = initializers.normal(stddev=head_dim**-0.5)

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        """Passes the input through the network.
        Arguments:
            inputs:     [batch_size, height, width, channels]
            train:      bool
        Returns:
            output:     [batch_size, config.num_classes]
        """
        conv = partial(nn.Conv,
                       use_bias=False,
                       dtype=self.dtype,
                       precision=self.precision,
                       kernel_init=self.kernel_init)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=self.bn_momentum,
                       epsilon=self.bn_epsilon,
                       dtype=self.dtype)

        y = conv(self.initial_filters,
                 kernel_size=(7, 7),
                 strides=(2, 2),
                 padding=[(3, 3), (3, 3)])(inputs)
        y = norm()(y)
        y = self.activation_fn(y)
        y = nn.max_pool(y, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(self.stage_sizes[:-1]):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                y = BottleneckResNetBlock(
                    filters=self.initial_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    se_ratio=self.se_ratio,
                    projection_factor=self.projection_factor,
                    activation_fn=self.activation_fn,
                    dtype=self.dtype,
                )(y)
        for j in range(self.stage_sizes[-1]):
            strides = (2, 2) if j == 0 and self.stride_one is False else (1, 1)
            y = BoTBlock(filters=self.initial_filters * 2**(i + 1),
                         strides=strides,
                         conv=conv,
                         norm=norm,
                         projection_factor=self.projection_factor,
                         activation_fn=self.activation_fn)(y)
        y = jnp.mean(y, axis=(1, 2))
        y = nn.Dense(self.num_classes,
                     dtype=self.dtype,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(y)
        y = jnp.asarray(y, dtype=self.dtype)
        return y
