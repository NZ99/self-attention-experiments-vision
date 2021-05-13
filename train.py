import click
import flax.jax_utils
import wandb
from click_option_group import optgroup
import jax
from jax import numpy as jnp
from flax import traverse_util
from flax.training.train_state import TrainState
from flax.training import checkpoints
from einops import rearrange
import optax

import utils
import input_pipeline
from models import create_model


def one_hot(x):
    return jax.nn.one_hot(x, num_classes=1000)


def create_train_state(rng, model, img_size, lr_schedule_fn, weight_decay,
                       max_norm):

    tx = optax.chain(optax.clip_by_global_norm(max_norm), optax.scale_by_adam(),
                     optax.additive_weight_decay(weight_decay),
                     optax.scale_by_schedule(lr_schedule_fn))

    params = model.init(rng,
                        jax.numpy.ones((1, img_size, img_size, 3)),
                        is_training=False)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    return train_state


def build_train_input(data_dir, batch_size, img_size, augmentation):
    num_devices = jax.device_count()
    bs_per_device, ragged = divmod(batch_size, num_devices)
    if ragged:
        raise ValueError(
            f'Batch size {batch_size} must be divisible by num devices {num_devices}'
        )
    return input_pipeline.load(
        input_pipeline.Split.TRAIN_AND_VALID,
        data_dir=data_dir,
        is_training=True,
        batch_dims=[jax.local_device_count(), bs_per_device],
        transpose=True,
        image_size=(img_size, img_size),
        augment_name=augmentation,
        augment_before_mix=True,
        name='imagenet',
        fake_data=False)


def build_eval_input(data_dir, batch_size, img_size):
    bs_per_device = (batch_size // jax.local_device_count())
    split = input_pipeline.Split.TEST
    eval_preproc = 'crop_resize'
    return input_pipeline.load(
        split,
        data_dir=data_dir,
        is_training=False,
        batch_dims=[jax.local_device_count(), bs_per_device],
        transpose=True,
        image_size=(img_size, img_size),
        name='imagenet',
        eval_preproc=eval_preproc,
        fake_data=False)


def train_step(train_state, batch, label_smoothing):

    def loss_fn(params):
        images = rearrange(batch['images'], 'H W C N -> N H W C')
        images = images.astype(jnp.bfloat16)
        logits = train_state.apply_fn(params, images, is_training=True)
        y = one_hot(batch['labels'])
        if 'mix_labels' in batch:
            y1 = one_hot(batch['mix_labels'])
            y = batch['ratio'][:,
                               None] * y + (1. - batch['ratio'][:, None]) * y1
        y = optax.smooth_labels(y, label_smoothing)
        logits = logits.astype(jnp.float32)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
        scaled_loss = loss / jax.device_count()
        return scaled_loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True, axis_name='batch')
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


@click.command()
@optgroup('Dataset configuration')
@optgroup.option('--data_dir',
                 type=str,
                 required=True,
                 help='path to dataset directory')
@optgroup('Training configuration')
@optgroup.option('-s',
                 '--img_size',
                 type=int,
                 default=224,
                 help='image size to use (default: 224x224)')
@optgroup.option('-e',
                 '--num_epochs',
                 type=int,
                 default=300,
                 help='epoch to complete during training (default: 300)')
@optgroup.option('-b',
                 '--batch_size',
                 type=int,
                 default=32,
                 help='batch size to use (default: 32)')
@optgroup.option('--label_smoothing',
                 type=float,
                 default=0.1,
                 help='label smoothing alpha to use (default: 0.1)')
@optgroup('Data augmentation configuration')
@optgroup.option('--augmentation',
                 type=str,
                 default='cutmix_mixup_randaugment_405',
                 help='augmentation strategy to use')
@optgroup('Model configuration')
@optgroup.option('-m',
                 '--model_name',
                 type=str,
                 required=True,
                 help='model to use')
@optgroup('Optimizer and schedule configuration')
@optgroup.option('-l',
                 '--lr',
                 type=float,
                 default=5e-4,
                 help='learning rate to use (default: 0.01)')
@optgroup.option('--weight_decay',
                 type=float,
                 default=0.0001,
                 help='weight decay to use (default: 0.0001)')
@optgroup.option('--clip_grad',
                 type=float,
                 default=None,
                 help='gradient clip value to use (default: None)')
@optgroup('Miscellaneous')
@optgroup.option('-c',
                 '--checkpoint_dir',
                 type=str,
                 required=True,
                 help='path to checkpoint directory')
@optgroup.option('--seed',
                 type=int,
                 default=42,
                 help='path to checkpoint directory')
def main(data_dir, img_size, num_epochs, batch_size, label_smoothing,
         augmentation, model_name, lr, weight_decay, clip_grad, checkpoint_dir,
         seed):

    if jax.process_index() == 0:
        wandb.init(project='self-attention-experiments',
                   config={
                       'dataset': 'ImageNet1K',
                       'model': model_name,
                   })
        config = wandb.config

    rng = jax.random.PRNGKey(seed)
    images_per_epoch = 1281167
    steps_per_epoch = images_per_epoch // batch_size
    steps_per_eval = steps_per_epoch * 5
    steps_per_checkpoint = steps_per_epoch * 10
    steps_total = steps_per_epoch * num_epochs

    train_input = build_train_input(data_dir, batch_size, img_size,
                                    augmentation)
    eval_input = build_eval_input(data_dir, batch_size, img_size)

    base_lr = lr * (batch_size / 512)
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.,
        peak_value=base_lr,
        warmup_steps=5 * steps_per_epoch,
        decay_steps=30 * steps_per_epoch,
        end_value=1e-5)

    model = create_model(model_name=model_name,
                         num_classes=1000,
                         dtype=jnp.bfloat16)

    train_state = create_train_state(rng, model, img_size, schedule_fn,
                                     weight_decay, clip_grad)
    train_state = flax.jax_utils.replicate(train_state)

    train_fn = jax.pmap(train_step, axis_name='batch')
    eval_fn = jax.pmap(eval_step, axis_name='batch')

    for step, batch in zip(range(steps_total), train_input):
        train_state = train_fn(train_state, batch, label_smoothing)

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
