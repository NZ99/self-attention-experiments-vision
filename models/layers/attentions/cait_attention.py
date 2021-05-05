from typing import Callable, Optional

from flax import linen as nn
from flax.linen import initializers
from jax import numpy as jnp
from jax.lax import Precision

from models.layers.attentions.attention import AttentionBlock


class ClassSelfAttentionBlock(nn.Module):
    num_heads: int
    head_ch: Optional[int] = None
    out_ch: Optional[int] = None
    talking_heads: bool = False
    attn_dropout_rate: float = 0.
    out_dropout_rate: float = 0.
    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32
    precision: Precision = Precision.DEFAULT
    kernel_init: Callable = initializers.kaiming_uniform()
    bias_init: Callable = initializers.zeros

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        inputs_q = jnp.enpand_dims(inputs[:, 0, :], axis=1)
        return AttentionBlock(num_heads=self.num_heads,
                              head_ch=self.head_ch,
                              out_ch=self.out_ch,
                              talking_heads=self.talking_heads,
                              attn_drop_rate=self.attn_drop_rate,
                              out_drop_rate=self.out_drop_rate,
                              use_bias=self.use_bias,
                              dtype=self.dtype,
                              precision=self.precision,
                              kernel_init=self.kernel_init,
                              bias_init=self.bias_init)(inputs_q,
                                                        inputs,
                                                        is_training=is_training)
