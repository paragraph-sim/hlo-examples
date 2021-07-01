# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Mikhail Isaev at Nvidia

"""An MNIST example with single-program multiple-data (SPMD) data parallelism.

The aim here is to illustrate how to use JAX's `pmap` to express and execute
SPMD programs for data parallelism along a batch dimension, while also
minimizing dependencies by avoiding the use of higher-level layers and
optimizers libraries.
"""


from functools import partial
import time

import numpy as np
import numpy.random as npr

import jax
from jax import jit, grad, pmap
from jax.scipy.special import logsumexp
from jax.lib import xla_bridge
from jax.tree_util import tree_map
from jax import lax
import jax.numpy as jnp

import array
import gzip
import os
from os import path
import struct
import urllib.request

import numpy as np


_DATA = "/tmp/jax_example_data/"

if "xla_force_host_platform_device_count" in os.environ['XLA_FLAGS']:
  jax.config.update('jax_platform_name', 'cpu')

from jax.lib import xla_bridge
print("Jax platform:", xla_bridge.get_backend().platform)
print (jax.devices())


def _download(url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(_DATA):
    os.makedirs(_DATA)
  out_file = path.join(_DATA, filename)
  if not path.isfile(out_file):
    urllib.request.urlretrieve(url, out_file)
    print("downloaded {} to {}".format(url, _DATA))

def _partial_flatten(x):
  """Flatten all but the first dimension of an ndarray."""
  return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)

def mnist_raw():
  """Download and parse the raw MNIST dataset."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()),
                      dtype=np.uint8).reshape(num_data, rows, cols)

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(base_url + filename, filename)

  train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

  return train_images, train_labels, test_images, test_labels

def mnist(permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = _partial_flatten(train_images) / np.float32(255.)
  test_images = _partial_flatten(test_images) / np.float32(255.)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def predict(params, inputs):
  activations = inputs
  for w, b in params[:-1]:
    outputs = jnp.dot(activations, w) + b
    activations = jnp.tanh(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(activations, final_w) + final_b
  return logits - logsumexp(logits, axis=1, keepdims=True)

def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

@jit
def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)

if __name__ == "__main__":
  layer_sizes = [784, 1024, 1024, 10]
  param_scale = 0.1
  step_size = 0.001
  num_epochs = 2
  batch_size = len(jax.local_devices()) * 32

  train_images, train_labels, test_images, test_labels = mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  # For this manual SPMD example, we get the number of devices (e.g. GPUs or
  # TPU cores) that we're using, and use it to reshape data minibatches.
  num_devices = xla_bridge.device_count()
  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        images, labels = train_images[batch_idx], train_labels[batch_idx]
        # For this SPMD example, we reshape the data batch dimension into two
        # batch dimensions, one of which is mapped over parallel devices.
        batch_size_per_device, ragged = divmod(images.shape[0], num_devices)
        if ragged:
          msg = "batch size must be divisible by device count, got {} and {}."
          raise ValueError(msg.format(batch_size, num_devices))
        shape_prefix = (num_devices, batch_size_per_device)
        images = images.reshape(shape_prefix + images.shape[1:])
        labels = labels.reshape(shape_prefix + labels.shape[1:])
        yield images, labels
  batches = data_stream()

  @partial(pmap, axis_name='batch')
  def spmd_update(params, batch):
    grads = grad(loss)(params, batch)
    # We compute the total gradients, summing across the device-mapped axis,
    # using the `lax.psum` SPMD primitive, which does a fast all-reduce-sum.
    grads = [(lax.psum(dw, 'batch'), lax.psum(db, 'batch')) for dw, db in grads]
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

  # We replicate the parameters so that the constituent arrays have a leading
  # dimension of size equal to the number of devices we're pmapping over.
  init_params = init_random_params(param_scale, layer_sizes)
  replicate_array = lambda x: np.broadcast_to(x, (num_devices,) + x.shape)
  replicated_params = tree_map(replicate_array, init_params)

  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      replicated_params = spmd_update(replicated_params, next(batches))
    epoch_time = time.time() - start_time

    # We evaluate using the jitted `accuracy` function (not using pmap) by
    # grabbing just one of the replicated parameter values.
    params = tree_map(lambda x: x[0], replicated_params)
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))

if __name__ == "__main__":
  layer_sizes = [784, 1024, 1024, 10]
  param_scale = 0.1
  step_size = 0.001
  num_epochs = 2
  batch_size = 128

  train_images, train_labels, test_images, test_labels = mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  # For this manual SPMD example, we get the number of devices (e.g. GPUs or
  # TPU cores) that we're using, and use it to reshape data minibatches.
  num_devices = xla_bridge.device_count()
  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        images, labels = train_images[batch_idx], train_labels[batch_idx]
        # For this SPMD example, we reshape the data batch dimension into two
        # batch dimensions, one of which is mapped over parallel devices.
        batch_size_per_device, ragged = divmod(images.shape[0], num_devices)
        if ragged:
          msg = "batch size must be divisible by device count, got {} and {}."
          raise ValueError(msg.format(batch_size, num_devices))
        shape_prefix = (num_devices, batch_size_per_device)
        images = images.reshape(shape_prefix + images.shape[1:])
        labels = labels.reshape(shape_prefix + labels.shape[1:])
        yield images, labels
  batches = data_stream()

  @partial(pmap, axis_name='batch')
  def spmd_update(params, batch):
    grads = grad(loss)(params, batch)
    # We compute the total gradients, summing across the device-mapped axis,
    # using the `lax.psum` SPMD primitive, which does a fast all-reduce-sum.
    grads = [(lax.psum(dw, 'batch'), lax.psum(db, 'batch')) for dw, db in grads]
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

  # We replicate the parameters so that the constituent arrays have a leading
  # dimension of size equal to the number of devices we're pmapping over.
  init_params = init_random_params(param_scale, layer_sizes)
  replicate_array = lambda x: np.broadcast_to(x, (num_devices,) + x.shape)
  replicated_params = tree_map(replicate_array, init_params)

  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      replicated_params = spmd_update(replicated_params, next(batches))
    epoch_time = time.time() - start_time

    # We evaluate using the jitted `accuracy` function (not using pmap) by
    # grabbing just one of the replicated parameter values.
    params = tree_map(lambda x: x[0], replicated_params)
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))

# HLO extraction
print (jax.devices())

def wrapped_training(replicated_params):
  for epoch in range(num_epochs):
    for _ in range(num_batches):
      replicated_params = spmd_update(replicated_params, next(batches))
    params = tree_map(lambda x: x[0], replicated_params)
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))

hlo_computation = jax.xla_computation(wrapped_training)(replicated_params)
print ('Saving HLO files')
if not os.path.isdir('hlo_files'):
  os.makedirs('hlo_files')
with open("hlo_files/hlo_trace_wave_equation.txt", "w") as text_file:
  text_file.write(hlo_computation.as_hlo_text())
with open("hlo_files/hlo_trace_wave_equation.pb", "wb") as proto_file:
  proto_file.write(hlo_computation.as_serialized_hlo_module_proto())
