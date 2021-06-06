from functools import partial
from typing import Callable, Optional

import haiku as hk
import jax
from jax import numpy as jnp


class SqueezeExciteBlock(hk.Module):

    def __init__(self,
                 in_ch: int,
                 out_ch: Optional[int] = None,
                 se_ratio: Optional[float] = None,
                 hidden_ch: Optional[int] = None,
                 activation_fn: Callable = jax.nn.gelu):
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        if se_ratio is None:
            if hidden_ch is None:
                raise ValueError('Must provide one of se_ratio or hidden_ch')
            self.hidden_ch = hidden_ch
        else:
            self.hidden_ch = max(1, int(in_ch * se_ratio))
        self.activation_fn = activation_fn
        self.to_hidden = hk.Linear(self.hidden_ch, with_bias=True)
        self.to_out = hk.Linear(self.out_ch, with_bias=True)

    def __call__(self, inputs):
        x = jnp.mean(inputs, axis=(1, 2), keepdims=True)(inputs)
        x = self.to_hidden(x)
        x = self.activation_fn(x)
        x = self.to_out(x)
        output = jax.nn.sigmoid(x) * inputs
        return output
