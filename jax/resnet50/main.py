# Copyright 2021 The Flax Authors.
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
#
# Modified by Mikhail Isaev at Nvidia in 2021

"""Main file for running the ImageNet example.

This file is intentionally kept short. The majority for logic is in libraries
than can be easily tested and imported in Colab.
"""

from absl import app
from absl import flags
from absl import logging

from clu import platform
import train
import jax
from ml_collections import config_flags
import tensorflow as tf


import os
if 'XLA_FLAGS' in os.environ and \
        "xla_force_host_platform_device_count" in os.environ['XLA_FLAGS']:
  jax.config.update('jax_platform_name', 'cpu')

from jax.lib import xla_bridge
print("Jax platform:", xla_bridge.get_backend().platform)
print (jax.devices())

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


  import functools
  import optax
  from flax import jax_utils, optim
  from jax import random
  import tensorflow_datasets as tfds
  import models

  batch_size = 512
  half_precision = True
  print (jax.devices())

  def hlo_from_wrapped_trainer():
    rng = random.PRNGKey(0)
    image_size = 224

    if batch_size % jax.device_count() > 0:
      raise ValueError('Batch size must be divisible by the number of devices')
    local_batch_size = batch_size // jax.process_count()

    platform = jax.local_devices()[0].platform

    if half_precision:
      if platform == 'tpu':
        input_dtype = tf.bfloat16
      else:
        input_dtype = tf.float16
    else:
      input_dtype = tf.float32

    dataset_builder = tfds.builder('imagenette')
    train_iter = train.create_input_iter(
        dataset_builder, local_batch_size, image_size, input_dtype, train=True,
        cache=False)
    eval_iter = train.create_input_iter(
        dataset_builder, local_batch_size, image_size, input_dtype, train=False,
        cache=False)

    steps_per_epoch = (
        dataset_builder.info.splits['train'].num_examples // batch_size
    )
    num_steps = 1
    steps_per_eval = 1
    steps_per_checkpoint = steps_per_epoch * 10
    base_learning_rate = 0.4 * batch_size / 256.
    momentum = 0.9

    model_cls = getattr(models, "ResNet50")
    model = train.create_model(
        model_cls=model_cls, half_precision=half_precision)

    def create_learning_rate_fn_colab(
          base_learning_rate: float,
          steps_per_epoch: int):
      """Create learning rate schedule."""
      warmup_fn = optax.linear_schedule(
          init_value=0., end_value=base_learning_rate,
          transition_steps=1 * steps_per_epoch)
      cosine_epochs = 1
      cosine_fn = optax.cosine_decay_schedule(
          init_value=base_learning_rate,
          decay_steps=cosine_epochs * steps_per_epoch)
      schedule_fn = optax.join_schedules(
          schedules=[warmup_fn, cosine_fn],
          boundaries=[1 * steps_per_epoch])
      return schedule_fn
    learning_rate_fn = create_learning_rate_fn_colab(
        base_learning_rate, steps_per_epoch)

    def create_train_state_colab(rng, model, image_size, learning_rate_fn):
      """Create initial training state."""
      dynamic_scale = None
      platform = jax.local_devices()[0].platform
      if half_precision and platform == 'gpu':
          dynamic_scale = optim.DynamicScale()
      else:
          dynamic_scale = None

      params, batch_stats = train.initialized(rng, image_size, model)
      tx = optax.sgd(
          learning_rate=learning_rate_fn,
          momentum=momentum,
          nesterov=True,
      )
      state = train.TrainState.create(
          apply_fn=model.apply,
          params=params,
          tx=tx,
          batch_stats=batch_stats,
          dynamic_scale=dynamic_scale)
      return state
    state = create_train_state_colab(rng, model, image_size, learning_rate_fn)

    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        functools.partial(train.train_step, learning_rate_fn=learning_rate_fn),
        axis_name='batch')

    train_metrics = []
    
    def train_epoch(state):
      for step, batch in zip(range(step_offset, num_steps), train_iter):
          state, metrics = p_train_step(state, batch)
    return jax.xla_computation(train_epoch)(state)
  hlo_computation = hlo_from_wrapped_trainer()
  print ('Saving HLO files')
  if not os.path.isdir('/content/hlo_files'):
    os.makedirs('/content/hlo_files')
  with open("hlo_files/hlo_trace_resnet50.txt", "w") as text_file:
    text_file.write(hlo_computation.as_hlo_text())
  with open("hlo_files/hlo_trace_resnet50.pb", "wb") as proto_file:
    proto_file.write(hlo_computation.as_serialized_hlo_module_proto())


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
