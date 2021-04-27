from functools import partial
from typing import Any, Tuple

from jax import numpy as jnp
from jax.nn import initializers

from flax import linen as nn

from einops import rearrange

from models.botnet_config import BoTNetConfig

ModuleDef = Any


class SELayer(nn.Module):
    """Squeeze and Excite module.
    Attributes:
        config: BoTNet configuration
    """
    config: BoTNetConfig

    @nn.compact
    def __call__(self, inputs):
        """Passes the input through a squeeze and excite block.
        Arguments:
            inputs:     [batch_size, height, width, dim]
        Returns:
            output:     [batch_size, height, width, dim]
        """
        cfg = self.config
        out_dim = inputs.shape[-1]
        se_features = max(1, int(out_dim * cfg.se_ratio))

        dense = partial(nn.Dense,
                        dtype=cfg.dtype,
                        precision=cfg.precision,
                        kernel_init=cfg.kernel_init,
                        bias_init=cfg.bias_init)

        y = jnp.mean(inputs, axis=(1, 2), dtype=cfg.dtype, keepdims=True)
        y = dense(features=se_features)(y)
        y = cfg.activation_fn(y)
        y = dense(features=out_dim)(y)
        y = nn.sigmoid(y) * inputs
        return y


class BottleneckResNetBlock(nn.Module):
    """ResNet bottleneck block.
    Attributes:
        config: BoTNet configuration
        filters: number of filters to use in the first and second convolutions
        conv: a convolution module
        norm: a normalization module such as nn.BatchNorm
        strides: strides to use in the second convolution, either (1, 1) or (2, 2)
    """
    config: BoTNetConfig
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    strides: Tuple[int, int]

    @nn.compact
    def __call__(self, inputs):
        """Passes the input through a resnet bottleneck block.
        Arguments:
            inputs:     [batch_size, height, width, dim]
        Returns:
            output:     [batch_size, height, width, dim * config.projection_factor]
        """
        residual = inputs
        cfg = self.config

        y = self.conv(self.filters, kernel_size=(1, 1))(inputs)
        y = self.norm()(y)
        y = cfg.activation_fn(y)
        y = self.conv(self.filters, kernel_size=(3, 3), strides=self.strides)(y)
        y = self.norm()(y)
        y = cfg.activation_fn(y)
        y = self.conv(self.filters * cfg.projection_factor,
                      kernel_size=(1, 1))(y)
        y = self.norm(scale_init=initializers.zeros)(y)

        if cfg.se_ratio is not None:
            y = SELayer(config=cfg)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * cfg.projection_factor,
                                 kernel_size=(1, 1),
                                 strides=self.strides)(residual)
            residual = self.norm()(residual)
            residual = cfg.activation_fn(residual)

        y = cfg.activation_fn(residual + y)
        return y


class RelativeLogits(nn.Module):
    """Relative logits module.
    Attributes:
        config: BoTNet configuration
    """
    config: BoTNetConfig

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

        cfg = self.config
        _, _, H, W, _ = query.shape

        rel_pos_emb_w_shape = (2 * W - 1, cfg.head_dim)
        rel_pos_emb_w = self.param('rel_pos_emb_w', cfg.posemb_init,
                                   rel_pos_emb_w_shape)

        rel_pos_emb_h_shape = (2 * H - 1, cfg.head_dim)
        rel_pos_emb_h = self.param('rel_pos_emb_h', cfg.posemb_init,
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
    config: BoTNetConfig

    @nn.compact
    def __call__(self, inputs_q):
        """Applies multi-head self-attention on the input data.
        Arguments:
            inputs_q:   [batch_size, height, width, dim]
        Returns:
            output:     [batch_size, height, width, dim]
        """
        cfg = self.config
        conv = partial(nn.Conv,
                       features=cfg.num_heads * cfg.head_dim,
                       kernel_size=(1, 1),
                       use_bias=False,
                       precision=cfg.precision,
                       kernel_init=cfg.kernel_init)

        query, key, value = (conv(dtype=cfg.dtype, name="query")(inputs_q),
                             conv(dtype=cfg.dtype, name="key")(inputs_q),
                             conv(dtype=cfg.dtype, name="value")(inputs_q))
        query, key, value = (rearrange(query,
                                       'b H W (h d) -> b h H W d',
                                       h=cfg.num_heads),
                             rearrange(key,
                                       'b H W (h d) -> b h H W d',
                                       h=cfg.num_heads),
                             rearrange(value,
                                       'b H W (h d) -> b h H W d',
                                       h=cfg.num_heads))

        query = query / jnp.sqrt(cfg.head_dim).astype(cfg.dtype)

        attn_weights = jnp.einsum('b h H W d, b h P Q d -> b h H W P Q',
                                  query,
                                  key,
                                  precision=cfg.precision)
        attn_weights = attn_weights + RelativeLogits(config=cfg)(query)
        attn_weights = nn.softmax(attn_weights).astype(cfg.dtype)
        attn_out = jnp.einsum('b h H W P Q, b h H W d -> b H W h d',
                              attn_weights,
                              value,
                              precision=cfg.precision)
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
    config: BoTNetConfig
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    strides: Tuple[int, int]

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
        y = cfg.activation_fn(y)
        y = BoTMHSA(config=cfg)(y)
        if self.strides == (2, 2):
            y = nn.avg_pool(y,
                            window_shape=(2, 2),
                            strides=self.strides,
                            padding='SAME')
        y = self.norm()(y)
        y = cfg.activation_fn(y)
        y = self.conv(self.filters * cfg.projection_factor,
                      kernel_size=(1, 1))(y)
        y = self.norm(scale_init=initializers.zeros)(y)

        if self.strides == (2, 2) or residual.shape != y.shape:
            residual = self.conv(self.filters * cfg.projection_factor,
                                 kernel_size=(1, 1),
                                 strides=self.strides)(residual)
            residual = self.norm()(residual)
            residual = cfg.activation_fn(residual)

        y = cfg.activation_fn(residual + y)
        return y


class BoTNet(nn.Module):
    """Bottleneck Transformer Network module.
    Attributes:
        config: BoTNet configuration
    """
    config: BoTNetConfig

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        """Passes the input through the network.
        Arguments:
            inputs:     [batch_size, height, width, channels]
            train:      bool
        Returns:
            output:     [batch_size, config.num_classes]
        """
        cfg = self.config
        conv = partial(nn.Conv,
                       use_bias=False,
                       dtype=cfg.dtype,
                       precision=cfg.precision,
                       kernel_init=cfg.kernel_init)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=cfg.bn_momentum,
                       epsilon=cfg.bn_epsilon,
                       dtype=cfg.dtype)

        y = conv(cfg.initial_filters,
                 kernel_size=(7, 7),
                 strides=(2, 2),
                 padding=[(3, 3), (3, 3)])(inputs)
        y = norm()(y)
        y = cfg.activation_fn(y)
        y = nn.max_pool(y, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(cfg.stage_sizes[:-1]):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                y = BottleneckResNetBlock(filters=cfg.initial_filters * 2**i,
                                          strides=strides,
                                          config=cfg,
                                          conv=conv,
                                          norm=norm)(y)
        for j in range(cfg.stage_sizes[-1]):
            strides = (2, 2) if j == 0 and cfg.stride_one is False else (1, 1)
            y = BoTBlock(filters=cfg.initial_filters * 2**(i + 1),
                         strides=strides,
                         config=cfg,
                         conv=conv,
                         norm=norm)(y)
        y = jnp.mean(y, axis=(1, 2))
        y = nn.Dense(cfg.num_classes,
                     dtype=cfg.dtype,
                     kernel_init=cfg.kernel_init,
                     bias_init=cfg.bias_init)(y)
        y = jnp.asarray(y, dtype=cfg.dtype)
        return y
