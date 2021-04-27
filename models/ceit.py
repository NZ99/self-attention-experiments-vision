from typing import Tuple

from flax import linen as nn
from flax.linen import initializers
from flax.struct import field
from jax import numpy as jnp
from jax import random

from models.layers import CeiTImage2TokenPatchEmbedBlock, LCAEncoder, LeFFEncoder
from models.ceit_config import CeiTConfig


class CeiT(nn.Module):

    config: CeiTConfig

    num_classes: int
    patch_shape: Tuple[int, int] = field(default_factory=lambda: [4, 4], pytree_node=False)
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool = False):
        x = CeiTImage2TokenPatchEmbedBlock(
            patch_shape=self.patch_shape,
            num_ch=self.config.num_ch,
            conv_kernel_size=self.config.conv_kernel_size,
            conv_stride=self.config.conv_stride,
            pool_window_size=self.config.pool_window_size,
            pool_stride=self.config.pool_stride,
            embed_dim=self.config.embed_dim,
            dtype=self.dtype
        )(inputs)

        b, *(_) = x.shape
        cls_shape = (1, 1, self.config.embed_dim)
        cls_token = self.param('cls', initializers.zeros, cls_shape)
        cls_token = jnp.tile(cls_token, [b, 1, 1])

        x = jnp.concatenate([cls_token, x], axis=1)

        x, layer_cls_tokens = LeFFEncoder(
            num_layers=self.config.num_layers,
            expand_ratio=self.config.expand_ratio,
            dw_conv_kernel_size=self.config.dw_conv_kernel_size,
            num_heads=self.config.num_heads,
            head_ch=self.config.embed_dim,
            dropout_rate=self.config.dropout_rate,
            attn_dropout_rate=self.config.attn_dropout_rate,
            dtype=self.dtype
        )(x, train=train)

        cls = LCAEncoder(
            num_heads=self.config.num_heads,
            head_ch=self.config.embed_dim,
            mlp_ch=self.config.embed_dim,
            num_layers=self.config.lca_num_layers
        )(layer_cls_tokens)

        cls = cls[:, 0]
        cls_out = nn.Dense(
            features=self.num_classes,
            use_bias=True,
            dtype=self.dtype,
            kernel_init=initializers.zeros,
        )(cls)
        return cls_out


if __name__ == '__main__':
    init_input = jnp.ones(shape=(1, 224, 224, 3))
    batch = jnp.ones(shape=(10, 224, 224, 3))

    model = CeiT(num_classes=1000, config=CeiTConfig())

    variables = model.init(random.PRNGKey(seed=0), init_input)
    state, params = variables.pop('params')

    from clu.parameter_overview import count_parameters
    print(f'Num parameters: {count_parameters(params)}')

    out = model.apply(variables, batch)
    print(f'Output shape: {out.shape}')

