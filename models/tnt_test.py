from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax.numpy as jnp
import jax.random as random

from models import TNT


class TNTTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ('TNT-S', 224, 1000, 12, 4, 6, 24, 384),
        ('TNT-B', 224, 1000, 12, 4, 10, 40, 640)
    )
    def test_logits_shape(self,
                          img_resolution,
                          num_classes,
                          num_layers,
                          inner_num_heads,
                          outer_num_heads,
                          inner_embed_dim,
                          outer_embed_dim):

        model = TNT(num_classes=num_classes,
                    num_layers=num_layers,
                    inner_num_heads=inner_num_heads,
                    outer_num_heads=outer_num_heads,
                    inner_embed_dim=inner_embed_dim,
                    outer_embed_dim=outer_embed_dim
        )

        rng = dict(params=random.PRNGKey(0))
        x = jnp.ones((2, img_resolution, img_resolution, 3))
        logits, _ = model.init_with_output(rng, x, is_training=True)
        chex.assert_shape(logits, (2, num_classes))


if __name__ == '__main__':
    absltest.main()
