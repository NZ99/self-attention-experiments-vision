from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax.numpy as jnp
import jax.random as random

from models import CeiT


class CeiTTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ('CeiT-T', 224, 1000, 3, 192),
        ('CeiT-S', 224, 1000, 6, 384),
        ('CeiT-B', 224, 1000, 12, 768)
    )
    def test_logits_shape(self, img_resolution, num_classes, num_heads, embed_dim):
        model = CeiT(num_classes=num_classes, num_heads=num_heads, embed_dim=embed_dim)
        rng = dict(params=random.PRNGKey(0))
        x = jnp.ones((2, img_resolution, img_resolution, 3))
        logits, _ = model.init_with_output(rng, x)
        chex.assert_shape(logits, (2, num_classes))

if __name__ == '__main__':
  absltest.main()