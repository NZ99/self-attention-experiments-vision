from typing import Callable, Optional

import haiku as hk
import jax
from jax import numpy as jnp
from einops import rearrange


class LeFFBlock(hk.Module):

    def __init__(self,
                 in_ch: int,
                 out_ch: Optional[int] = None,
                 expand_ratio: float = None,
                 hidden_ch: int = None,
                 kernel_size: int = 5,
                 activation_fn: Callable = jax.nn.gelu):

        self.out_ch = out_ch if out_ch else in_ch
        if expand_ratio is None:
            if hidden_ch is None:
                raise ValueError(
                    'Must provide one of expand_ratio or hidden_ch')
            self.hidden_ch = hidden_ch
        else:
            self.hidden_ch = max(1, int(expand_ratio * in_ch))
        self.kernel_size = kernel_size,
        self.activation_fn = activation_fn
        self.to_hidden = hk.Linear(self.hidden_ch, with_bias=True)
        self.to_out = hk.Linear(self.out_ch, with_bias=True)
        self.bn_1 = hk.BatchNorm(create_scale=True,
                                 create_offset=True,
                                 decay_rate=0.999)
        self.bn_2 = hk.BatchNorm(create_scale=True,
                                 create_offset=True,
                                 decay_rate=0.999)
        self.bn_3 = hk.BatchNorm(create_scale=True,
                                 create_offset=True,
                                 decay_rate=0.999)
        self.conv = hk.Conv2d(self.hidden_ch, kernel_shape=self.kernel_size)

    def __call__(self, inputs, is_training: bool, test_local_stats: bool):
        cls, tokens = inputs[:, 0], inputs[:, 1:]
        l = tokens.shape[1]

        x = self.to_hidden(tokens)
        x = self.bn_1(x, is_training, test_local_stats)
        x = self.activation_fn(x)

        spatial_ch = int(jnp.sqrt(l))
        x = rearrange(x, 'b (h w) c -> b h w c', h=spatial_ch, w=spatial_ch)

        x = self.conv(x)
        x = self.bn_2(x)
        x = self.activation_fn(x)

        x = rearrange(x, 'b h w c -> b (h w) c')

        x = self.to_out(x)
        x = self.bn_3(x)
        x = self.activation_fn(x)

        cls_token = jnp.expand_dims(cls, axis=1)
        output = jnp.concatenate([cls_token, x], axis=1)
        return output
