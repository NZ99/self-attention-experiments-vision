from typing import Callable, Tuple

import jax.numpy as jnp
from jax.nn import initializers
from jax.lax import Precision
import flax.linen as nn

from models.layers import PatchEmbedBlock, AddAbsPosEmbed, SelfAttentionBlock, LayerScaleBlock, StochasticDepthBlock, FFBlock, ClassSelfAttentionBlock


class EncoderBlock(nn.Module):
    num_heads: int
    expand_ratio: int = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    stoch_depth_rate: float = 0.
    layerscale_eps = float = 0.
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = self.SelfAttentionBlock(num_heads=self.num_heads,
                                    attn_drop_rate=self.attn_dropout_rate,
                                    out_drop_rate=self.dropout_rate,
                                    dtype=self.dtype,
                                    precision=self.precision,
                                    kernel_init=self.kernel_init,
                                    bias_init=self.bias_init)(
                                        x, is_training=is_training)
        x = LayerScaleBlock(eps=self.layerscale_eps,
                            dtype=self.dtype)(x, is_training=is_training)
        x = StochasticDepthBlock(drop_rate=self.stoch_depth_rate)(
            x, is_training=is_training)
        x = x + inputs

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = FFBlock(expand_ratio=self.expand_ratio,
                    dropout_rate=self.dropout_rate,
                    activation_fn=self.activation_fn,
                    dtype=self.dtype,
                    precision=self.precision,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init)(y, train=is_training)
        y = LayerScaleBlock(eps=self.layerscale_eps,
                            dtype=self.dtype)(y, is_training=is_training)
        y = StochasticDepthBlock(drop_rate=self.stoch_depth_rate)(
            y, is_training=is_training)

        output = x + y
        return output


class Encoder(nn.Module):
    num_layers: int
    num_heads: int
    expand_ratio: int = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    stoch_depth_rate: float = 0.
    layerscale_eps = float = 0.
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = AddAbsPosEmbed()(inputs)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)

        for _ in range(self.num_layers):
            x = EncoderBlock(num_heads=self.num_heads,
                             expand_ratio=self.expand_ratio,
                             attn_dropout_rate=self.attn_dropout_rate,
                             dropout_rate=self.dropout_rate,
                             stoch_depth_rate=self.stoch_depth_rate,
                             layerscale_eps=self.layerscale_eps,
                             attn_class=SelfAttentionBlock,
                             activation_fn=self.activation_fn,
                             dtype=self.dtype,
                             precision=self.precision,
                             kernel_init=self.kernel_init,
                             bias_init=self.bias_init)(x,
                                                       is_training=is_training)

        output = x
        return output


class CAEncoderBlock(nn.Module):
    num_heads: int
    expand_ratio: int = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    stoch_depth_rate: float = 0.
    layerscale_eps = float = 0.
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, cls_token, is_training: bool):
        x = jnp.concatenate([cls_token, inputs], axis=1)
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = self.SelfAttentionBlock(num_heads=self.num_heads,
                                    attn_drop_rate=self.attn_dropout_rate,
                                    out_drop_rate=self.dropout_rate,
                                    dtype=self.dtype,
                                    precision=self.precision,
                                    kernel_init=self.kernel_init,
                                    bias_init=self.bias_init)(
                                        x, is_training=is_training)
        x = LayerScaleBlock(eps=self.layerscale_eps,
                            dtype=self.dtype)(x, is_training=is_training)
        x = StochasticDepthBlock(drop_rate=self.stoch_depth_rate)(
            x, is_training=is_training)
        cls_token = cls_token + x

        y = nn.LayerNorm(dtype=self.dtype)(cls_token)
        y = FFBlock(expand_ratio=self.expand_ratio,
                    dropout_rate=self.dropout_rate,
                    activation_fn=self.activation_fn,
                    dtype=self.dtype,
                    precision=self.precision,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init)(y, train=is_training)
        y = LayerScaleBlock(eps=self.layerscale_eps,
                            dtype=self.dtype)(y, is_training=is_training)
        y = StochasticDepthBlock(drop_rate=self.stoch_depth_rate)(
            y, is_training=is_training)

        output = cls_token + y
        return output


class CaiT(nn.Module):
    num_classes: int
    num_layers: int
    num_layers_token_only: int
    num_heads: int
    embed_dim: int
    patch_shape: Tuple[int, int]
    expand_ratio: int = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    stoch_depth_rate: float = 0.
    layerscale_eps = float = 0.
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = PatchEmbedBlock(
            patch_shape=self.patch_shape,
            embed_dim=self.embed_dim,
            dtype=self.dtype,
            precision=self.precision,
        )(inputs)

        x = Encoder(num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    expand_ratio=self.expand_ratio,
                    attn_dropout_rate=self.attn_dropout_rate,
                    dropout_rate=self.dropout_rate,
                    stoch_depth_rate=self.stoch_depth_rate,
                    layerscale_eps=self.layerscale_eps,
                    activation_fn=self.activation_fn,
                    dtype=self.dtype,
                    precision=self.precision,
                    kernel_init=self.kernel_init)(x, is_training=is_training)

        b = x.shape[0]
        cls_shape = (1, 1, self.embed_dim)
        cls_token = self.param('cls', initializers.zeros, cls_shape)
        cls_token = jnp.tile(cls_token, [b, 1, 1])

        for _ in range(self.num_layers_token_only):
            cls_token = CAEncoderBlock(num_heads=self.num_heads,
                                       expand_ratio=self.expand_ratio,
                                       attn_dropout_rate=self.attn_dropout_rate,
                                       dropout_rate=self.dropout_rate,
                                       stoch_depth_rate=self.stoch_depth_rate,
                                       layerscale_eps=self.layerscale_eps,
                                       attn_class=ClassSelfAttentionBlock,
                                       activation_fn=self.activation_fn,
                                       dtype=self.dtype,
                                       precision=self.precision,
                                       kernel_init=self.kernel_init)(
                                           x,
                                           cls_token,
                                           is_training=is_training)

        x = jnp.concatenate([cls_token, x], axis=1)
        x = nn.LayerNorm(dtype=self.dtype)(x)

        cls_token = x[:, 0]
        output = nn.Dense(
            features=self.num_classes,
            use_bias=True,
            dtype=self.dtype,
            kernel_init=initializers.zeros,
            bias_init=self.bias_init,
        )(cls_token)
        return output
