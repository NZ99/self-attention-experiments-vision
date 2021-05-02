import jax
from jax import numpy as jnp

from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils

import numpy as np

from einops import rearrange

from flax import traverse_util

import optax

from clu.parameter_overview import count_parameters

import sys

from absl import flags
from absl import logging

import models
import input_pipeline
import utils

from ml_collections import config_dict


def get_config():
    config = base_config.get_base_config()

    config.random_seed = 0
    images_per_epoch = 1281167
    train_batch_size = 2048
    num_epochs = 300
    steps_per_epoch = images_per_epoch / train_batch_size
    config.training_steps = ((images_per_epoch * num_epochs) //
                             train_batch_size)
    config.experiment_kwargs = config_dict.ConfigDict(
        dict(config=dict(
            lr=1e-3,
            num_epochs=num_epochs,
            image_size=224,
            num_classes=1000,
            which_dataset='imagenet',
            loss='softmax_cross_entropy',
            transpose=True,
            dtype=jnp.bfloat16,
            lr_schedule=dict(name='cosine_decay_schedule',
                             kwargs=dict(init_value=1e-3,
                                         decay_steps=config.training_steps)),
            optimizer_weights=dict(
                name='adamw', kwargs=dict(b1=0.9, b2=0.999, weight_decay=0.05)),
            optimizer_biases=dict(name='adam', kwargs=dict(b1=0.9, b2=0.999)),
            model=dict(name='BoTNet',
                       config_kwargs=dict(stage_sizes=[3, 4, 6, 6],
                                          dtype=jnp.bfloat16)),
            augment_name='cutmix_mixup_randaugment_405')))
