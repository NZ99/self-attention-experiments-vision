# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import jax
from jax import numpy as jnp



def topk_correct(logits, labels, mask=None, prefix='', topk=(1, 5)):
    """Calculate top-k error for multiple k values."""
    metrics = {}
    argsorted_logits = jnp.argsort(logits)
    for k in topk:
        pred_labels = argsorted_logits[..., -k:]
        # Get the number of examples where the label is in the top-k predictions
        correct = any_in(pred_labels, labels).any(axis=-1).astype(jnp.float32)
        if mask is not None:
            correct *= mask
        metrics[f'{prefix}top_{k}_acc'] = correct
    return metrics


@jax.vmap
def any_in(prediction, target):
    """For each row in a and b, checks if any element of a is in b."""
    return jnp.isin(prediction, target)
