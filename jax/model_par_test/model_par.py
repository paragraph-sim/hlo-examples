import os
import jax

os.environ['XLA_FLAGS'] =f'--xla_dump_to=./hlo_files --xla_dump_hlo_as_text=true'
print ("XLA_FLAGS =", os.environ['XLA_FLAGS'])

from jax.lib import xla_bridge
print("Jax platform:", xla_bridge.get_backend().platform)
print (jax.devices())

from jax import random
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.nn import one_hot, relu
from jax.scipy.special import logsumexp

num_devices = jax.device_count()
key = jax.random.PRNGKey(0)

w1 = random.normal(key, (10240, 1024))
w2 = random.normal(key, (1024, 128))
inputs = random.normal(key, (512, 10240))
labels = random.randint(key, (512,), 0, 128, dtype=jnp.int32)

def predict(w1, w2, inputs):
  hiddens = relu(jnp.dot(inputs, w1))
  logits = jnp.dot(hiddens, w2)
  return logits - logsumexp(logits, axis=1, keepdims=True)

def loss(w1, w2, inputs, labels):
  predictions = predict(w1, w2, inputs)
  targets = one_hot(labels, predictions.shape[-1])
  losses = jnp.sum(targets * predictions, axis=1)
  return -jnp.mean(losses, axis=0)

def named_predict(w1, w2, image):
  hidden = relu(lax.pdot(image, w1, 'inputs'))
  logits = lax.pdot(hidden, w2, 'hidden')
  def my_logsumexp(x):
    c = x.max()
    return c + jnp.log(lax.psum(jnp.exp(x - c), 'classes'))
  return logits - my_logsumexp(logits)

def named_loss(w1, w2, inputs, labels):
  predictions = named_predict(w1, w2, inputs)
  num_classes = lax.psum(1, 'classes')
  targets = one_hot(labels, num_classes, axis='classes')
  losses = lax.psum(targets * predictions, 'classes')
  return -lax.pmean(losses, 'batch')

from jax.experimental.maps import xmap, mesh
from jax import grad 

in_axes = [['inputs', 'hidden', ...],
           ['hidden', 'classes', ...],
           ['batch', 'inputs', ...],
           ['batch', ...]]
grad_out_axes =  (['inputs', 'hidden', ...],
                  ['hidden', 'classes', ...])

print(loss(w1, w2, inputs, labels))

def g(w1, w2, inputs, labels):
    grad_fun = xmap(jax.grad(named_loss, (0, 1), reduce_axes=('classes', 'batch')),
                    in_axes=[['inputs', 'hidden', ...],
                             ['hidden', 'classes', ...],
                             ['batch', 'inputs', ...],
                             ['batch', ...]],
                    out_axes=(['inputs', 'hidden', ...],
                              ['hidden', 'classes', ...]),
                     axis_resources={'batch': 'x',
                                     'hidden': 'y',
                                     'inputs': 'z',
                                     'classes': 'z'})

    devices = np.array(jax.local_devices()).reshape((2, 2, 2))
    with mesh(devices, ('x', 'y', 'z')):
        grads = grad_fun(w1, w2, inputs, labels)
    return grads
print (g(w1, w2, inputs, labels))


