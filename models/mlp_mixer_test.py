from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax.numpy as jnp
import jax.random as random

from models import MLPMixer


class MLPMixerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        ('MLPMixer-S/32', 224, 1000, 8, 512, (32, 32)),
        ('MLPMixer-S/16', 224, 1000, 8, 512, (16, 16)),
        ('MLPMixer-B/32', 224, 1000, 12, 768, (32, 32)),
        ('MLPMixer-B/16', 224, 1000, 12, 768, (16, 16)),
        ('MLPMixer-L/32', 224, 1000, 24, 1024, (32, 32)),
        ('MLPMixer-L/16', 224, 1000, 24, 1024, (16, 16))
    )
    def test_logits_shape(self,
                          img_resolution,
                          num_classes,
                          num_layers,
                          embed_dim,
                          patch_shape):

        model = MLPMixer(num_classes=num_classes,
                         num_layers=num_layers,
                         embed_dim=embed_dim,
                         patch_shape=patch_shape)

        rng = dict(params=random.PRNGKey(0))
        x = jnp.ones((2, img_resolution, img_resolution, 3))
        logits, _ = model.init_with_output(rng, x, is_training=True)
        chex.assert_shape(logits, (2, num_classes))


if __name__ == '__main__':
    absltest.main()