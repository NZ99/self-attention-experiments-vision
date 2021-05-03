from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax.numpy as jnp
import jax.random as random

from models import CvT


class CeiTTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ('CvT-13', 224, 1000, [1, 2, 10], [1, 3, 6], [64, 192, 384]),
        ('CvT-21', 224, 1000, [1, 4, 16], [1, 3, 6], [64, 192, 384]),
        ('CvT-W24', 384, 1000, [2, 2, 20], [3, 12, 16], [192, 768, 1024])
    )
    def test_logits_shape(self, img_resolution, num_classes, size, num_heads, embed_dim):
        model = CvT(num_classes=num_classes, size=size, num_heads=num_heads, embed_dim=embed_dim)
        rng = dict(params=random.PRNGKey(0))
        x = jnp.ones((2, img_resolution, img_resolution, 3))
        logits, _ = model.init_with_output(rng, x)
        chex.assert_shape(logits, (2, num_classes))

if __name__ == '__main__':
  absltest.main()