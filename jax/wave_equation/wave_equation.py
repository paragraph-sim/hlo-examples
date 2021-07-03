from functools import partial
import jax
from jax import jit, pmap
from jax import lax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.filters
import proglog
from moviepy.editor import ImageSequenceClip

if 'XLA_FLAGS' in os.environ and \
        "xla_force_host_platform_device_count" in os.environ['XLA_FLAGS']:
  jax.config.update('jax_platform_name', 'cpu')

from jax.lib import xla_bridge
print("Jax platform:", xla_bridge.get_backend().platform)
print (jax.devices())


device_count = jax.device_count()

# Spatial partitioning via halo exchange

def send_right(x, axis_name):
  # Note: if some devices are omitted from the permutation, lax.ppermute
  # provides zeros instead. This gives us an easy way to apply Dirichlet
  # boundary conditions.
  left_perm = [(i, (i + 1) % device_count) for i in range(device_count - 1)]
  return lax.ppermute(x, perm=left_perm, axis_name=axis_name)

def send_left(x, axis_name):
  left_perm = [((i + 1) % device_count, i) for i in range(device_count - 1)]
  return lax.ppermute(x, perm=left_perm, axis_name=axis_name)

def axis_slice(ndim, index, axis):
  slices = [slice(None)] * ndim
  slices[axis] = index
  return tuple(slices)

def slice_along_axis(array, index, axis):
  return array[axis_slice(array.ndim, index, axis)]

def tree_vectorize(func):
  def wrapper(x, *args, **kwargs):
    return tree_util.tree_map(lambda x: func(x, *args, **kwargs), x)
  return wrapper

@tree_vectorize
def halo_exchange_padding(array, padding=1, axis=0, axis_name='x'):
  if not padding > 0:
    raise ValueError(f'invalid padding: {padding}')
  array = jnp.array(array)
  if array.ndim == 0:
    return array
  left = slice_along_axis(array, slice(None, padding), axis)
  right = slice_along_axis(array, slice(-padding, None), axis)
  right, left = send_left(left, axis_name), send_right(right, axis_name)
  return jnp.concatenate([left, array, right], axis)

@tree_vectorize
def halo_exchange_inplace(array, padding=1, axis=0, axis_name='x'):
  left = slice_along_axis(array, slice(padding, 2*padding), axis)
  right = slice_along_axis(array, slice(-2*padding, -padding), axis)
  right, left = send_left(left, axis_name), send_right(right, axis_name)
  array = jax.ops.index_update(
      array, axis_slice(array.ndim, slice(None, padding), axis), left)
  array = jax.ops.index_update(
      array, axis_slice(array.ndim, slice(-padding, None), axis), right)
  return array

# Reshaping inputs/outputs for pmap

def split_with_reshape(array, num_splits, *, split_axis=0, tile_id_axis=None):
  if tile_id_axis is None:
    tile_id_axis = split_axis
  tile_size, remainder = divmod(array.shape[split_axis], num_splits)
  if remainder:
    raise ValueError('num_splits must equally divide the dimension size')
  new_shape = list(array.shape)
  new_shape[split_axis] = tile_size
  new_shape.insert(split_axis, num_splits)
  return jnp.moveaxis(jnp.reshape(array, new_shape), split_axis, tile_id_axis)

def stack_with_reshape(array, *, split_axis=0, tile_id_axis=None):
  if tile_id_axis is None:
    tile_id_axis = split_axis
  array = jnp.moveaxis(array, tile_id_axis, split_axis)
  new_shape = array.shape[:split_axis] + (-1,) + array.shape[split_axis+2:]
  return jnp.reshape(array, new_shape)

def shard(func):
  def wrapper(state):
    sharded_state = tree_util.tree_map(
        lambda x: split_with_reshape(x, device_count), state)
    sharded_result = func(sharded_state)
    result = tree_util.tree_map(stack_with_reshape, sharded_result)
    return result
  return wrapper

# Physics

def shift(array, offset, axis):
  index = slice(offset, None) if offset >= 0 else slice(None, offset)
  sliced = slice_along_axis(array, index, axis)
  padding = [(0, 0)] * array.ndim
  padding[axis] = (-min(offset, 0), max(offset, 0))
  return jnp.pad(sliced, padding, mode='constant', constant_values=0)

def laplacian(array, step=1):
  left = shift(array, +1, axis=0)
  right = shift(array, -1, axis=0)
  up = shift(array, +1, axis=1)
  down = shift(array, -1, axis=1)
  convolved = (left + right + up + down - 4 * array)
  if step != 1:
    convolved *= (1 / step ** 2)
  return convolved

def scalar_wave_equation(u, c=1, dx=1):
  return c ** 2 * laplacian(u, dx)

@jax.jit
def leapfrog_step(state, dt=0.5, c=1):
  # https://en.wikipedia.org/wiki/Leapfrog_integration
  u, u_t = state
  u_tt = scalar_wave_equation(u, c)
  u_t = u_t + u_tt * dt
  u = u + u_t * dt
  return (u, u_t)

# Time stepping

@jax.jit
def multi_step(state, count, dt=1/jnp.sqrt(2), c=1):
  return lax.fori_loop(0, count, lambda i, s: leapfrog_step(s, dt, c), state)

def multi_step_pmap(state, count, dt=1/jnp.sqrt(2), c=1, exchange_interval=1,
                    save_interval=1):

  def exchange_and_multi_step(state_padded):
    c_padded = halo_exchange_padding(c, exchange_interval)
    evolved = multi_step(state_padded, exchange_interval, dt, c_padded)
    return halo_exchange_inplace(evolved, exchange_interval)

  @shard
  @partial(jax.pmap, axis_name='x')
  def simulate_until_output(state):
    stop = save_interval // exchange_interval
    state_padded = halo_exchange_padding(state, exchange_interval)
    advanced = lax.fori_loop(
        0, stop, lambda i, s: exchange_and_multi_step(s), state_padded)
    xi = exchange_interval
    return tree_util.tree_map(lambda array: array[xi:-xi, ...], advanced)

  results = [state]
  for _ in range(count // save_interval):
    state = simulate_until_output(state)
    tree_util.tree_map(lambda x: x.copy_to_host_async(), state)
    results.append(state)
  results = jax.device_get(results)
  return tree_util.tree_multimap(lambda *xs: np.stack([np.array(x) for x in xs]), *results)

multi_step_jit = jax.jit(multi_step)


#Initial Conditions
x = jnp.linspace(0, 8, num=8*1024, endpoint=False)
y = jnp.linspace(0, 1, num=1*1024, endpoint=False)
x_mesh, y_mesh = jnp.meshgrid(x, y, indexing='ij')

# NOTE: smooth initial conditions are important, so we aren't exciting
# arbitrarily high frequencies (that cannot be resolved)
u = skimage.filters.gaussian(
    ((x_mesh - 1/3) ** 2 + (y_mesh - 1/4) ** 2) < 0.1 ** 2,
    sigma=1)

# u = jnp.exp(-((x_mesh - 1/3) ** 2 + (y_mesh - 1/4) ** 2) / 0.1 ** 2)

# u = skimage.filters.gaussian(
#     (x_mesh > 1/3) & (x_mesh < 1/2) & (y_mesh > 1/3) & (y_mesh < 1/2),
#     sigma=5)

v = jnp.zeros_like(u)
c = 1  # could also use a 2D array matching the mesh shape

# single chip
u_final, _ = multi_step_jit((u, v), count=2**13, c=c, dt=0.5)

# 8x chips, 4x more steps in roughly half the time!
u_final, _ = multi_step_pmap(
    (u, v), count=2**15, c=c, dt=0.5, exchange_interval=4, save_interval=2**15)


# Extracting HLOs
count=1
save_interval=2**15
def multi_step_pmap_traced(
    state, dt=1/jnp.sqrt(2), c=1, exchange_interval=1):

  def exchange_and_multi_step(state_padded):
    c_padded = halo_exchange_padding(c, exchange_interval)
    evolved = multi_step(state_padded, exchange_interval, dt, c_padded)
    return halo_exchange_inplace(evolved, exchange_interval)

  @shard
  @partial(jax.pmap, axis_name='x')
  def simulate_until_output(state):
    stop = save_interval // exchange_interval
    state_padded = halo_exchange_padding(state, exchange_interval)
    advanced = lax.fori_loop(
        0, stop, lambda i, s: exchange_and_multi_step(s), state_padded)
    xi = exchange_interval
    return tree_util.tree_map(lambda array: array[xi:-xi, ...], advanced)

  results = [state]
  for _ in range(count):
    state = simulate_until_output(state)
    def pull_back(state):
        state
    tree_util.tree_map(pull_back, state)
    results.append(state)
  results = jax.device_get(results)
  return tree_util.tree_multimap(lambda *xs: jnp.stack([jnp.array(x) for x in xs]), *results)

def wrapped_solver():
    multi_step_pmap_traced(
        (u, v), c=c, dt=0.5, 
        exchange_interval=4)

hlo_computation = jax.xla_computation(wrapped_solver)()
if not os.path.isdir('hlo_files'):
  os.makedirs('hlo_files')
print ('Saving HLO files')
with open("hlo_files/hlo_trace_wave_equation.txt", "w") as text_file:
    text_file.write(hlo_computation.as_hlo_text())
with open("hlo_files/hlo_trace_wave_equation.pb", "wb") as proto_file:
    proto_file.write(hlo_computation.as_serialized_hlo_module_proto())
