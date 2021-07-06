# coding=utf-8
# Copyright 2020 The Nested-Transformer Authors.
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
# See the License for the specific Nested-Transformer governing permissions and
# limitations under the License.
# ==============================================================================
"""Deterministic input pipeline for ImageNet."""
# Modified by Niccolò Zanichelli

import functools
from typing import Callable, Dict, Tuple, Union

from absl import logging
from clu import deterministic_data
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from data.custom_datasets import Imagenet21kPWinter
from data.preprocess import preprocess, augment_utils
from data.constants import IMAGENET_1K_DEFAULT_MEAN, IMAGENET_1K_DEFAULT_STD
from data.constants import IMAGENET_21K_DEFAULT_MEAN, IMAGENET_21K_DEFAULT_STD

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Features = Dict[str, Tensor]


def preprocess_with_per_batch_rng(ds: tf.data.Dataset,
                                  preprocess_fn: Callable[[Features], Features],
                                  *, rng: jnp.ndarray) -> tf.data.Dataset:
    """Maps batched `ds` using the preprocess_fn and a deterministic RNG per batch.

    This preprocess_fn usually contains data preprcess needs a batch of data, like
    Mixup.

    Args:
      ds: Dataset containing Python dictionary with the features. The 'rng'
        feature should not exist.
      preprocess_fn: Preprocessing function that takes a Python dictionary of
        tensors and returns a Python dictionary of tensors. The function should be
        convertible into a TF graph.
      rng: Base RNG to use. Per example RNGs will be derived from this by folding
        in the example index.

    Returns:
      The dataset mapped by the `preprocess_fn`.
    """
    rng = list(jax.random.split(rng, 1)).pop()

    def _fn(example_index: int, features: Features) -> Features:
        example_index = tf.cast(example_index, tf.int32)
        features["rng"] = tf.random.experimental.stateless_fold_in(
            tf.cast(rng, tf.int64), example_index)
        processed = preprocess_fn(features)
        if isinstance(processed, dict) and "rng" in processed:
            del processed["rng"]
        return processed

    return ds.enumerate().map(_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_dataset_fns(
    dataset_name: str,
    data_dir: str = 'gs://neo-datasets/vision_datasets/',
    image_size: int = 224,
    use_randaugment: bool = True,
    randaugment_use_cutout: bool = False,
    randaugment_size: int = 224,
    randaugment_num_layers: int = 2,
    randaugment_num_levels: int = 10,
    randaugment_magnitude: int = 9,
    randaugment_magnitude_std: float = 0.5,
    randaugment_prob: float = 0.5,
    use_random_erasure: bool = True,
    random_erasuse_prob: float = 0.25,
    use_mix: bool = True,
    mix_type: str = 'mixup',
    mix_smoothing: float = 0.1,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 0.5,
    mix_prob_to_apply: float = 1.0,
    use_color_jitter: bool = True,
    color_jitter_size: int = 224,
    color_jitter_strength: float = 0.3,
    color_jitter_use_crop: bool = False,
) -> Tuple[tfds.core.DatasetBuilder, tfds.core.ReadInstruction, Callable[
    [Features], Features], Callable[[Features], Features], str, Union[Callable[
        [Features], Features], None]]:
    """Gets dataset specific functions."""

    use_custom_process = (use_randaugment or use_random_erasure or
                          use_color_jitter)

    label_key = "label"
    image_key = "image"
    if dataset_name == "imagenet_1k":
        train_dataset_builder = tfds.builder("imagenet2012:5.1.0",
                                             data_dir=data_dir)
        train_num_examples = train_dataset_builder.info.splits[
            "train"].num_examples
        train_split = deterministic_data.get_read_instruction_for_host(
            "train", train_num_examples)

        eval_dataset_builder = tfds.builder("imagenet_v2:3.0.0",
                                            data_dir=data_dir)
        eval_split_name = "test"
        eval_num_examples = eval_dataset_builder.info.splits[
            eval_split_name].num_examples
        eval_split = deterministic_data.get_read_instruction_for_host(
            eval_split_name, eval_num_examples)

        # If there is resource error during preparation, checkout
        # https://github.com/tensorflow/datasets/issues/1441#issuecomment-581660890
        # dataset_builder.download_and_prepare()

        # Create augmentaton fn.
        if use_custom_process:
            # When using custom augmentation, we use mean/std normalization.
            logging.info("Configuring augmentation")
            mean, std = IMAGENET_1K_DEFAULT_MEAN, IMAGENET_1K_DEFAULT_STD
            mean = tf.constant(mean, dtype=tf.float32, shape=[1, 1, 3])
            std = tf.constant(std, dtype=tf.float32, shape=[1, 1, 3])
            basic_preprocess_fn = functools.partial(preprocess.train_preprocess,
                                                    input_size=image_size)

            preprocess_fn = preprocess.get_augment_preprocess(
                use_randaugment=use_randaugment,
                randaugment_use_cutout=randaugment_use_cutout,
                randaugment_size=randaugment_size,
                randaugment_num_layers=randaugment_num_layers,
                randaugment_num_levels=randaugment_num_levels,
                randaugment_magnitude=randaugment_magnitude,
                randaugment_magnitude_std=randaugment_magnitude_std,
                randaugment_prob=randaugment_prob,
                use_random_erasure=use_random_erasure,
                random_erasuse_prob=random_erasuse_prob,
                use_color_jitter=use_color_jitter,
                color_jitter_size=color_jitter_size,
                color_jitter_strength=color_jitter_strength,
                color_jitter_use_crop=color_jitter_use_crop,
                mean=mean,
                std=std,
                basic_process=basic_preprocess_fn)
            eval_preprocess_fn = functools.partial(preprocess.eval_preprocess,
                                                   mean=mean,
                                                   std=std,
                                                   input_size=image_size)
        else:
            # Standard imagenet preprocess with 0-1 normalization
            preprocess_fn = functools.partial(preprocess.train_preprocess,
                                              input_size=image_size)
            eval_preprocess_fn = functools.partial(preprocess.eval_preprocess,
                                                   input_size=image_size)

    elif dataset_name == 'imagenet_21k':
        train_dataset_builder = tfds.builder("imagenet21k_p_winter:1.0.0",
                                             data_dir=data_dir)
        train_num_examples = train_dataset_builder.info.splits[
            "train"].num_examples
        train_split = deterministic_data.get_read_instruction_for_host(
            "train", train_num_examples)

        eval_dataset_builder = tfds.builder("imagenet21k_p_winter:1.0.0",
                                            data_dir=data_dir)
        eval_num_examples = eval_dataset_builder.splits[
            "validation"].num_examples
        eval_split_name = "validation"

        # Create augmentaton fn.
        if use_custom_process:
            # When using custom augmentation, we use mean/std normalization.
            mean, std = IMAGENET_21K_DEFAULT_MEAN, IMAGENET_21K_DEFAULT_STD
            mean = tf.constant(mean, dtype=tf.float32, shape=[1, 1, 3])
            std = tf.constant(std, dtype=tf.float32, shape=[1, 1, 3])
            basic_preprocess_fn = functools.partial(preprocess.train_preprocess,
                                                    input_size=image_size)

            train_preprocess_fn = preprocess.get_augment_preprocess(
                use_randaugment=use_randaugment,
                randaugment_use_cutout=randaugment_use_cutout,
                randaugment_size=randaugment_size,
                randaugment_num_layers=randaugment_num_layers,
                randaugment_num_levels=randaugment_num_levels,
                randaugment_magnitude=randaugment_magnitude,
                randaugment_magnitude_std=randaugment_magnitude_std,
                randaugment_prob=randaugment_prob,
                use_random_erasure=use_random_erasure,
                random_erasuse_prob=random_erasuse_prob,
                use_color_jitter=use_color_jitter,
                color_jitter_size=color_jitter_size,
                color_jitter_strength=color_jitter_strength,
                color_jitter_use_crop=color_jitter_use_crop,
                mean=mean,
                std=std,
                basic_process=basic_preprocess_fn)
            eval_preprocess_fn = functools.partial(preprocess.eval_preprocess,
                                                   mean=mean,
                                                   std=std,
                                                   input_size=image_size)
        else:
            # When not using use_custom_process, we use 0-1 normalization.
            train_preprocess_fn = functools.partial(preprocess.train_preprocess,
                                                    input_size=image_size)
            eval_preprocess_fn = functools.partial(preprocess.eval_preprocess,
                                                   input_size=image_size)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if use_mix:
        logging.info("Configuring mix augmentation")
        # When batch augmentation is enabled.

        if mix_type == 'mixup':
            batch_preprocess_fn = augment_utils.create_mix_augment(
                num_classes=train_dataset_builder.info.features[label_key].
                num_classes,
                smoothing=mix_smoothing,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=0.,
                prob_to_apply=mix_prob_to_apply)
        elif mix_type == 'cutmix':
            batch_preprocess_fn = augment_utils.create_mix_augment(
                num_classes=train_dataset_builder.info.features[label_key].
                num_classes,
                smoothing=mix_smoothing,
                mixup_alpha=0.,
                cutmix_alpha=cutmix_alpha,
                prob_to_apply=mix_prob_to_apply)
        else:
            raise ValueError(f"Mix type {mix_type} not supported.")

    else:
        batch_preprocess_fn = None

    return (train_dataset_builder, train_split, eval_dataset_builder,
            eval_split_name, eval_num_examples, train_preprocess_fn,
            eval_preprocess_fn, batch_preprocess_fn)


def create_datasets(
    dataset_name: str,
    data_rng: jax.random.PRNGKey,
    data_dir: str = 'gs://neo-datasets/vision_datasets/',
    num_epochs: int = 300,
    per_device_batch_size: int = 256,
    eval_pad_last_batch: bool = True,
    shuffle_buffer_size: int = 1000,
    image_size: int = 224,
    use_randaugment: bool = True,
    randaugment_use_cutout: bool = False,
    randaugment_size: int = 224,
    randaugment_num_layers: int = 2,
    randaugment_num_levels: int = 10,
    randaugment_magnitude: int = 9,
    randaugment_magnitude_std: float = 0.5,
    randaugment_prob: float = 0.5,
    use_random_erasure: bool = True,
    random_erasuse_prob: float = 0.25,
    use_mix: bool = True,
    mix_type: str = 'mixup',
    mix_smoothing: float = 0.1,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 0.5,
    mix_prob_to_apply: float = 1.0,
    use_color_jitter: bool = True,
    color_jitter_size: int = 224,
    color_jitter_strength: float = 0.3,
    color_jitter_use_crop: bool = False,
) -> Tuple[tfds.core.DatasetInfo, tf.data.Dataset, tf.data.Dataset]:
    """Create datasets for training and evaluation.

    Args:
      # TO DO
      data_rng: PRNGKey for seeding operations in the training dataset.

    Returns:
      A tuple with the dataset info, the training dataset and the evaluation
      dataset.
    """
    (train_dataset_builder, train_split, eval_dataset_builder, eval_split_name,
     eval_num_examples, train_preprocess_fn, eval_preprocess_fn,
     eval_num_examples, batch_preprocess_fn) = get_dataset_fns(
         dataset_name=dataset_name,
         data_dir=data_dir,
         image_size=image_size,
         use_randaugment=use_randaugment,
         randaugment_use_cutout=randaugment_use_cutout,
         randaugment_size=randaugment_size,
         randaugment_num_layers=randaugment_num_layers,
         randaugment_num_levels=randaugment_num_levels,
         randaugment_magnitude=randaugment_magnitude,
         randaugment_magnitude_std=randaugment_magnitude_std,
         randaugment_prob=randaugment_prob,
         use_random_erasure=use_random_erasure,
         random_erasuse_prob=random_erasuse_prob,
         use_mix=use_mix,
         mix_type=mix_type,
         mix_smoothing=mix_smoothing,
         mixup_alpha=mixup_alpha,
         cutmix_alpha=cutmix_alpha,
         mix_prob_to_apply=mix_prob_to_apply,
         use_color_jitter=use_color_jitter,
         color_jitter_size=color_jitter_size,
         color_jitter_strength=color_jitter_strength,
         color_jitter_use_crop=color_jitter_use_crop)

    data_rng1, data_rng2 = jax.random.split(data_rng, 2)
    skip_batching = batch_preprocess_fn is not None
    batch_dims = [jax.local_device_count(), per_device_batch_size]
    train_ds = deterministic_data.create_dataset(
        train_dataset_builder,
        split=train_split,
        rng=data_rng1,
        preprocess_fn=train_preprocess_fn,
        cache=False,
        decoders={"image": tfds.decode.SkipDecoding()},
        shuffle_buffer_size=shuffle_buffer_size,
        batch_dims=batch_dims if not skip_batching else None,
        num_epochs=num_epochs,
        shuffle=True,
    )

    if batch_preprocess_fn:
        # Perform batch augmentation on each device and them batch devices.
        train_ds = train_ds.batch(batch_dims[-1], drop_remainder=True)
        train_ds = preprocess_with_per_batch_rng(train_ds,
                                                 batch_preprocess_fn,
                                                 rng=data_rng2)
        for batch_size in reversed(batch_dims[:-1]):
            train_ds = train_ds.batch(batch_size, drop_remainder=True)
        train_ds = train_ds.prefetch(4)

    options = tf.data.Options()
    options.experimental_external_state_policy = (
        tf.data.experimental.ExternalStatePolicy.WARN)
    options.experimental_threading.private_threadpool_size = 48
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.autotune = True
    options.experimental_optimization.hoist_random_uniform = True
    train_ds = train_ds.with_options(options)

    eval_split = deterministic_data.get_read_instruction_for_host(
        eval_split_name, eval_num_examples, drop_remainder=False)

    eval_num_batches = None
    if eval_pad_last_batch:
        eval_batch_size = jax.local_device_count() * per_device_batch_size
        eval_num_batches = int(
            np.ceil(eval_num_examples / eval_batch_size / jax.process_count()))

    eval_ds = deterministic_data.create_dataset(
        eval_dataset_builder,
        split=eval_split,
        preprocess_fn=eval_preprocess_fn,
        # Only cache dataset in distributed setup to avoid consuming a lot of
        # memory in Colab and unit tests.
        cache=(jax.process_count() > 1),
        batch_dims=[jax.local_device_count(), per_device_batch_size],
        num_epochs=1,
        shuffle=False,
        pad_up_to_batches=eval_num_batches,
    )

    return train_dataset_builder.info, eval_dataset_builder.info, train_ds, eval_ds
