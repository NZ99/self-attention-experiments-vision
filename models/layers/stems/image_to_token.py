from typing import Tuple, Optional

import haiku as hk
from einops import rearrange


class Image2TokenBlock(hk.Module):

    def __init__(self,
                 in_ch: int,
                 patch_shape: Tuple[int, int],
                 embed_dim: int,
                 expand_ratio: Optional[int],
                 hidden_ch: Optional[int],
                 conv_kernel_size: int = 7,
                 conv_stride: int = 2,
                 pool_window_size: int = 3,
                 pool_stride: int = 2,
                 with_bias: bool = False):

        self.ph, self.pw = patch_shape
        self.embed_dim = embed_dim

        if expand_ratio is None:
            if hidden_ch is None:
                raise ValueError(
                    'Must provide one of expand_ratio or hidden_ch')
            self.hidden_ch = hidden_ch
        else:
            self.hidden_ch = max(1, int(expand_ratio * in_ch))

        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.pool_window_size = pool_window_size
        self.pool_stride = pool_stride
        self.with_bias = with_bias

        self.to_hidden = hk.Conv2D(self.hidden_ch,
                                   kernel_shape=self.conv_kernel_size,
                                   stride=self.conv_stride,
                                   padding=[(self.ph,) * 2, (self.pw,) * 2],
                                   with_bias=self.with_bias)
        self.bn = hk.BatchNorm(create_scale=True,
                               create_offset=True,
                               decay_rate=0.999)
        self.to_out = hk.Linear(self.embed_dim, with_bias=self.with_bias)

    def __call__(self, inputs, is_training: bool, test_local_stats: bool):
        x = self.to_hidden(inputs)
        x = self.bn(x, is_training, test_local_stats)
        x = hk.max_pool(
            x,
            window_shape=(self.pool_window_size,) * 2,
            strides=(self.pool_stride,) * 2,
        )
        x = rearrange(
            x,
            'b (h ph) (w pw) c -> b (h w) (ph pw c)',
            ph=self.ph,
            pw=self.pw,
        )
        x = self.to_out(x)
        return x
