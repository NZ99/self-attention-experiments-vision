from typing import Tuple

import haiku as hk
from einops import rearrange


class PatchEmbedBlock(hk.Module):

    def __init__(self,
                 patch_shape: Tuple[int, int],
                 embed_dim: int,
                 use_bias: bool = False):
        self.ph, self.pw = patch_shape
        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.to_embed = hk.Linear(embed_dim, use_bias=self.use_bias)

    def __call__(self, inputs):
        x = rearrange(inputs,
                      'b (h ph) (w pw) c -> b (h w) (ph pw c)',
                      ph=self.ph,
                      pw=self.pw)
        x = self.to_embed(x)
        return x
