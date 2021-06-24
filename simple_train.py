import wandb

import jax
from jax import numpy as jnp

from flax import jax_utils
from flax.training.train_state import TrainState
from flax.training import checkpoints

from einops import rearrange

import optax

import utils
import input_pipeline
from models import create_model


def one_hot(x):
    return jax.nn.one_hot(x, num_classes=1000)


def create_train_state(model):
    rng = jax.random.PRNGKey(42)
    tx = optax.chain(optax.clip_by_global_norm(1),
                     optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
                     optax.scale(-0.003))
    params = model.init(rng, jax.numpy.ones((1, 224, 224, 3)), is_training=True)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return train_state


def build_train_input():
    num_devices = jax.device_count()
    bs_per_device, ragged = divmod(256, num_devices)
    if ragged:
        raise ValueError(
            f'Batch size {256} must be divisible by num devices {num_devices}')
    return input_pipeline.load(
        input_pipeline.Split.TRAIN,
        data_dir='gs://neo-datasets/vision_datasets/',
        is_training=True,
        batch_dims=[jax.local_device_count(), bs_per_device],
        transpose=True,
        image_size=(224, 224),
        augment_name='cutmix_mixup_0.4_randaugment_415',
        augment_before_mix=True,
        name='imagenet',
        fake_data=False)


def build_eval_input():
    bs_per_device = (256 // jax.local_device_count())
    split = input_pipeline.Split.TEST
    eval_preproc = 'crop_resize'
    return input_pipeline.load(
        split,
        data_dir='gs://neo-datasets/vision_models/',
        is_training=False,
        batch_dims=[jax.local_device_count(), bs_per_device],
        transpose=True,
        image_size=(224, 224),
        name='imagenet',
        eval_preproc=eval_preproc,
        fake_data=False)


def train_step(train_state, batch):

    def loss_fn(params):
        images = rearrange(batch['images'], 'B W C N -> B H W C')
        images = images.astype(jnp.bfloat16)
        logits = train_state.apply_fn(params, images, is_training=True)
        y = one_hot(batch['labels'])
        logits = logits.astype(jnp.float32)
        loss = optax.softmax_cross_entropy(logits, y)
        scaled_loss = loss / jax.device_count()
        return scaled_loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(train_state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss, logits = aux
    top_k_acc = utils.topk_correct(logits, batch['labels'], prefix='train_')
    top_k_acc = jax.tree_map(jnp.mean, top_k_acc)
    new_train_state = train_state.apply_gradients(grads=grads)

    if jax.process_index() == 0:
        wandb.log(
            {
                'train/loss': float(loss),
                'train/top-1-acc': top_k_acc['train_top_1_acc']
            }, train_state.step)

    return new_train_state


def eval_step(train_state, batch):
    images = rearrange(batch['images'], 'H W C N -> N H W C')
    variables = {'params': train_state.params}
    logits = train_state.apply_fn(variables, images, is_training=False)
    logits = logits.astype(jnp.float32)
    y = one_hot(batch['labels'])
    loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
    loss = loss / jax.device_count()
    return jax.lax.psum(loss, axis_name='batch')


def save_checkpoint(train_state, dir):
    if jax.process_index() == 0:
        train_state = jax.device_get(jax.tree_map(lambda x: x[0], train_state))
        step = int(train_state.step)
        checkpoints.save_checkpoint(dir, train_state, step, keep=3)


def main():
    if jax.process_index() == 0:
        wandb.init(project='self-attention-experiments',
                   config={
                       'dataset': 'ImageNet1K',
                       'model': 'ViT',
                   })

    checkpoint_dir = '/home/connor/checkpoints'
    images_per_epoch = 1281167
    steps_per_epoch = images_per_epoch // 256
    steps_per_eval = steps_per_epoch * 5
    steps_per_checkpoint = steps_per_epoch * 10
    steps_total = steps_per_epoch * 300

    train_input = build_train_input()
    eval_input = build_eval_input()

    model = create_model('vit_l_patch32', dtype=jnp.bfloat16)

    train_state = create_train_state(model)
    train_state = jax_utils.replicate(train_state)

    train_fn = jax.pmap(train_step, axis_name='batch')
    eval_fn = jax.pmap(eval_step, axis_name='batch')

    for step, batch in zip(range(steps_total), train_input):
        train_state = train_fn(train_state, batch)

        if (step % steps_per_checkpoint == 0 and step) or step == steps_total:
            save_checkpoint(train_state, checkpoint_dir)

        if step % steps_per_eval == 0:
            num_samples = 0.
            sum_loss = None
            for batch in eval_input:
                num_samples += jnp.prod(batch['labels'].shape[:2])
                loss = eval_fn(train_state, batch)
                if sum_loss is None:
                    sum_loss = loss
                else:
                    sum_loss = jax.tree_multimap(jnp.add, sum_loss, loss)

            mean_loss = jax.tree_map(lambda x: x / num_samples, sum_loss)
            if jax.process_index() == 0:
                wandb.log({'eval/loss': float(mean_loss)}, train_state.step)

        if (step % steps_total) == 0 and step:
            exit()


if __name__ == '__main__':
    main()
