from functools import partial
from typing import Callable, Optional

import jax
import haiku as hk


class FFBlock(hk.Module):

    def __init__(self,
                 in_ch: int,
                 out_ch: Optional[int] = None,
                 expand_ratio: Optional[float] = None,
                 hidden_ch: Optional[int] = None,
                 dropout_rate: float = 0.,
                 activation_fn: Callable = jax.nn.gelu):

        if expand_ratio is None:
            if hidden_ch is None:
                raise ValueError(
                    'Must provide one of expand_ratio or hidden_ch')
            self.hidden_ch = hidden_ch
        else:
            self.hidden_ch = max(1, int(expand_ratio * in_ch))
        self.out_ch = out_ch if out_ch else in_ch
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.to_hidden = hk.Linear(self.hidden_ch, with_bias=True)
        self.to_out = hk.Linear(self.out_ch, with_bias=True)

    def __call__(self, inputs, is_training: bool):

        x = self.to_hidden(inputs)
        x = self.activation_fn(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)
        x = self.to_out(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rat, x)
        return x
