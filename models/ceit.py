from functools import partial
from typing import Tuple, Callable, Optional

from flax import linen as nn
from flax.linen import initializers
from jax import numpy as jnp
from jax.lax import Precision

from models.layers.stems import Image2TokenBlock
from models.layers.attentions import SelfAttentionBlock
from models.layers.feedforwards import LeFFBlock, FFBlock


class EncoderBlock(nn.Module):
    is_lca: bool
    num_heads: int
    head_ch: int
    expand_ratio: int = 4
    leff_kernel_size: Optional[int] = 3
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, is_training: bool):

        x = SelfAttentionBlock(num_heads=self.num_heads,
                               is_lca=self.is_lca,
                               head_ch=self.head_ch,
                               out_ch=self.head_ch * self.num_heads,
                               dropout_rate=0.,
                               dtype=self.dtype,
                               precision=self.precision,
                               kernel_init=self.kernel_init,
                               bias_init=self.bias_init)(
                                   inputs, is_training=is_training)
        x += inputs
        x = nn.LayerNorm(dtype=self.dtype)(x)

        if self.is_lca is False:
            y = LeFFBlock(expand_ratio=self.expand_ratio,
                          kernel_size=self.leff_kernel_size,
                          bn_momentum=self.bn_momentum,
                          bn_epsilon=self.bn_epsilon,
                          dtype=self.dtype,
                          precision=self.precision,
                          kernel_init=self.kernel_init,
                          bias_init=self.bias_init)(x, is_training=is_training)
        else:
            y = FFBlock(expand_ratio=self.expand_ratio,
                        activation_fn=self.activation_fn,
                        dtype=self.dtype,
                        precision=self.precision,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init)(x, is_training=is_training)

        y = x + y
        output = nn.LayerNorm(dtype=self.dtype)(y)
        return output


class Encoder(nn.Module):
    num_layers: int
    num_heads: int
    head_ch: int
    expand_ratio: int = 4
    leff_kernel_size: int = 3
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, is_training: bool):

        encoder_block = partial(EncoderBlock,
                                is_lca=False,
                                num_heads=self.num_heads,
                                head_ch=self.head_ch,
                                expand_ratio=self.expand_ratio,
                                leff_kernel_size=self.leff_kernel_size,
                                bn_momentum=self.bn_momentum,
                                bn_epsilon=self.bn_epsilon,
                                dtype=self.dtype,
                                kernel_init=self.kernel_init,
                                bias_init=self.bias_init)

        x = encoder_block()(inputs, is_training=is_training)
        cls_tokens_lst = [jnp.expand_dims(x[:, 0], axis=1)]
        for _ in range(self.num_layers - 1):
            x = encoder_block()(x, is_training=is_training)
            cls_tokens_lst.append(jnp.expand_dims(x[:, 0], axis=1))

        return jnp.concatenate(cls_tokens_lst, axis=1)


class CeiT(nn.Module):
    num_classes: int
    num_layers: int = 12
    patch_shape: Tuple[int, int] = (4, 4)
    num_heads: int = 3
    num_ch: int = 32
    conv_kernel_size: int = 7
    conv_stride: int = 2
    pool_window_size: int = 3  #
    pool_stride: int = 2
    embed_dim: int = 192
    expand_ratio: int = 4
    leff_kernel_size: int = 3
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32
    precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.normal(stddev=1e-6)
    dropout_rate: float = 0
    attn_dropout_rate: float = 0

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        assert self.embed_dim % self.num_heads == 0

        x = Image2TokenBlock(patch_shape=self.patch_shape,
                             num_ch=self.num_ch,
                             conv_kernel_size=self.conv_kernel_size,
                             conv_stride=self.conv_stride,
                             pool_window_size=self.pool_window_size,
                             pool_stride=self.pool_stride,
                             embed_dim=self.embed_dim,
                             bn_momentum=self.bn_momentum,
                             bn_epsilon=self.bn_epsilon,
                             dtype=self.dtype,
                             precision=self.precision,
                             kernel_init=self.kernel_init,
                             bias_init=self.bias_init)(inputs,
                                                       is_training=is_training)

        b = x.shape[0]
        cls_shape = (1, 1, self.embed_dim)
        cls_token = self.param('cls', initializers.zeros, cls_shape)
        cls_token = jnp.tile(cls_token, [b, 1, 1])

        x = jnp.concatenate([cls_token, x], axis=1)

        head_ch = int(self.embed_dim / self.num_heads)
        cls_tokens = Encoder(num_layers=self.num_layers,
                             expand_ratio=self.expand_ratio,
                             leff_kernel_size=self.leff_kernel_size,
                             num_heads=self.num_heads,
                             head_ch=head_ch,
                             dtype=self.dtype,
                             precision=self.precision,
                             kernel_init=self.kernel_init,
                             bias_init=self.bias_init)(x,
                                                       is_training=is_training)

        cls_tokens = EncoderBlock(is_lca=True,
                                  num_heads=self.num_heads,
                                  head_ch=self.head_ch,
                                  expand_ratio=self.expand_ratio,
                                  bn_momentum=self.bn_momentum,
                                  bn_epsilon=self.bn_epsilon,
                                  dtype=self.dtype,
                                  kernel_init=self.kernel_init,
                                  bias_init=self.bias_init)(
                                      cls_tokens, is_training=is_training)
        cls = cls_tokens[:, -1]
        output = nn.Dense(features=self.num_classes,
                          use_bias=True,
                          dtype=self.dtype,
                          kernel_init=self.kernel_init,
                          bias_init=self.bias_init)(cls)
        return output
