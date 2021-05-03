from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax.numpy as jnp
import jax.random as random

from models import BoTNet


class BoTNetTest(parameterized.TestCase):

    @parameterized.named_parameters(('BoTNet-T3', 224, 1000, [3, 4, 6, 6]),
                                    ('BoTNet-T4', 224, 1000, [3, 4, 23, 6]),
                                    ('BoTNet-T5', 256, 1000, [3, 4, 23, 12]),
                                    ('BoTNet-T6', 320, 1000, [3, 4, 6, 12]),
                                    ('BoTNet-T7', 384, 1000, [3, 4, 23, 12]))
    def test_logits_shape(self, img_resolution, num_classes, stage_sizes):
        model = BoTNet(num_classes=num_classes, stage_sizes=stage_sizes)
        rng = dict(params=random.PRNGKey(0))
        x = jnp.ones((2, img_resolution, img_resolution, 3))
        logits, _ = model.init_with_output(rng, x)
        chex.assert_shape(logits, (2, num_classes))


if __name__ == '__main__':
    absltest.main()

if __name__ == '__main__':
    absltest.main()
