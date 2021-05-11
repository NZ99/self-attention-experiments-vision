from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax.numpy as jnp
import jax.random as random

from models.cait import CaiT


class CaiTTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ('CaiT-XXS-24', 224, 1000, 24, 2, 4, 192, (16, 16), 0.05, 1e-5),
        ('CaiT-XXS-36', 224, 1000, 36, 2, 4, 192, (16, 16), 0.1, 1e-6),
        ('CaiT-XS-24', 224, 1000, 24, 2, 6, 288, (16, 16), 0.05, 1e-5),
        ('CaiT-XS-36', 224, 1000, 36, 2, 6, 288, (16, 16), 0.1, 1e-6),
        ('CaiT-S-24', 224, 1000, 24, 2, 8, 384, (16, 16), 0.1, 1e-5),
        ('CaiT-S-36', 224, 1000, 36, 2, 8, 384, (16, 16), 0.2, 1e-6),
        ('CaiT-S-48', 224, 1000, 48, 2, 8, 384, (16, 16), 0.3, 1e-6),
        ('CaiT-M-24', 224, 1000, 24, 2, 16, 768, (16, 16), 0.2, 1e-5),
        ('CaiT-M-36', 224, 1000, 36, 2, 16, 768, (16, 16), 0.3, 1e-6),
        ('CaiT-M-36', 224, 1000, 36, 2, 16, 768, (16, 16), 0.4, 1e-6))
    def test_logits_shape(self, img_resolution, num_classes, num_layers,
                          num_layers_token_only, num_heads, embed_dim,
                          patch_shape, stoch_depth_rate, layerscale_eps):

        model = CaiT(num_classes=num_classes,
                     num_layers=num_layers,
                     num_layers_token_only=num_layers_token_only,
                     num_heads=num_heads,
                     embed_dim=embed_dim,
                     patch_shape=patch_shape,
                     stoch_depth_rate=stoch_depth_rate,
                     layerscale_eps=layerscale_eps)

        rng = dict(params=random.PRNGKey(0))
        x = jnp.ones((2, img_resolution, img_resolution, 3))
        logits, _ = model.init_with_output(rng, x, is_training=True)
        chex.assert_shape(logits, (2, num_classes))


if __name__ == '__main__':
    absltest.main()
