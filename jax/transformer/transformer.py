import jax
import trax
from trax.data import inputs
from trax import layers as tl
from trax.supervised import training

import numpy as np

from termcolor import colored
import random


import os
if 'XLA_FLAGS' in os.environ and \
        "xla_force_host_platform_device_count" in os.environ['XLA_FLAGS']:
  jax.config.update('jax_platform_name', 'cpu')

from jax.lib import xla_bridge
print("Jax platform:", xla_bridge.get_backend().platform)
print (jax.devices())

# global variables that state the filename and directory of the vocabulary file
VOCAB_FILE = 'ende_32k.subword'
VOCAB_DIR = 'gs://trax-ml/vocabs/'
data_path = 'data'

EOS = 1

# generator helper function to append EOS to each sentence
def append_eos(stream):
    for (inputs, targets) in stream:
        inputs_with_eos = list(inputs) + [EOS]
        targets_with_eos = list(targets) + [EOS]
        yield np.array(inputs_with_eos), np.array(targets_with_eos)

train_batches_stream = trax.data.Serial(
    trax.data.TFDS('para_crawl/ende',
                   data_dir=data_path,
                   keys=('en', 'de'),
                   eval_holdout_size=0.01, # 1% for eval
                   train=True), # replace TFDS with lambda _: train_stream_fn() if you want to run with your own data
    trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR),
    lambda _: append_eos(_),
    trax.data.Shuffle(),
    trax.data.FilterByLength(max_length=512, length_keys=[0, 1]),
    trax.data.BucketByLength(boundaries =  [  8,  16,  32,  64, 128, 256],
                             batch_sizes = [128, 128, 128, 128, 128, 128, 128],
                             length_keys=[0, 1]),
    trax.data.AddLossWeights(id_to_mask=0)
  )

eval_batches_stream = trax.data.Serial(
    trax.data.TFDS('para_crawl/ende',
                   data_dir=data_path,
                   keys=('en', 'de'),
                   eval_holdout_size=0.01, # 1% for eval
                   train=False),
    trax.data.Tokenize(vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR),
    trax.data.Shuffle(),
    trax.data.FilterByLength(max_length=1024, length_keys=[0, 1]),
    trax.data.BucketByLength(boundaries =  [  8,  16,  32,  64, 128, 256],
                             batch_sizes = [128, 128, 128, 128, 128, 128, 128],
                             length_keys=[0, 1]),
    trax.data.AddLossWeights(id_to_mask=0)
  )

# Exploring the data
train_batch_stream = train_batches_stream()
eval_batch_stream = eval_batches_stream()
input_batch, target_batch, mask_batch = next(train_batch_stream)
# let's see the data type of a batch
print("input_batch data type: ", type(input_batch))
print("target_batch data type: ", type(target_batch))
# let's see the shape of this particular batch (batch length, sentence length)
print("input_batch shape: ", input_batch.shape)
print("target_batch shape: ", target_batch.shape)

# pick a random index less than the batch size.
index = random.randrange(len(input_batch))
# use the index to grab an entry from the input and target batch
print(colored('ENGLISH SENTENCE: \n', 'red'), trax.data.detokenize(input_batch[index], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR), '\n')
print(colored('THE TOKENIZED VERSION OF THE ENGLISH SENTENCE: \n ', 'red'), input_batch[index], '\n')
print(colored('THE GERMAN TRANSLATION: \n', 'red'), trax.data.detokenize(target_batch[index], vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR), '\n')
print(colored('THE TOKENIZED VERSION OF THE GERMAN TRANSLATION: \n', 'red'), target_batch[index], '\n')

# Create a Transformer model.
model = trax.models.Transformer(
    input_vocab_size=33600,
    d_model=512, d_ff=2048, dropout = 0.1,
    n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
    max_len=2048, mode='train')

# Pre-trained Transformer model config in gs://trax-ml/models/translation/ende_wmt32k.gin
# Initialize Transformer using pre-trained weights.
model.init_from_file('gs://trax-ml/models/translation/ende_wmt32k.pkl.gz',
                     weights_only=True)

train_task = training.TrainTask(
    # use the train batch stream as labeled data
    labeled_data= train_batch_stream,
    # use the cross entropy loss with LogSoftmax
    loss_layer= tl.CrossEntropyLossWithLogSoftmax(),
    # use the Adafactor optimizer with learning rate of 0.001
    optimizer= trax.optimizers.Adafactor(learning_rate=0.001, epsilon1=1e-30),
    # have 500 warmup steps
    lr_schedule= trax.lr.multifactor(constant=1.0, warmup_steps=500),
    # have a checkpoint every 100 steps
    n_steps_per_checkpoint= 100,
    # saving a checkpoint every 1000 steps on the output_dir
    n_steps_per_permanent_checkpoint = 1000
)

eval_task = training.EvalTask(
    # use the eval batch stream as labeled data
    labeled_data=eval_batch_stream,
    # use the cross entropy loss with LogSoftmax and accuracy as metrics
    metrics=[tl.CrossEntropyLossWithLogSoftmax(), tl.WeightedCategoryAccuracy()],
    # you could specify the number of eval batch by n_eval_batches = 64 or any other number
    # but it not specified here as we want to evaluate the whole eval data
    # n_eval_batches = 64
)

# define the output directory
output_dir = 'Transformer_DE_pretrained_336'

# # remove old model if it exists. restarts training.
# !rm -rf output_dir

# define the training loop
training_loop = training.Loop(model,
                              train_task,
                              eval_tasks=[eval_task],
                              output_dir=output_dir)

training_loop.run(10)

# Extracting HLOs
import functools
from trax import fastmath
def wrapped_run():
    """Runs this training loop for 1 steps.
    Optionally runs evals and saves checkpoints at specified points.
    Args:
      n_steps: Stop training after completing n steps.
    """
    with training_loop._open_summary_writers() as (
        train_summary_writers, eval_summary_writers):
      loss_acc, step_acc = 0.0, 0
      for i in range(3):
        prev_task_index = training_loop._which_task(training_loop._step)
        training_loop._step += 1
        task_index = training_loop._which_task(training_loop._step)
        task_changed = task_index != prev_task_index

        if task_changed:
          loss_acc, step_acc = 0.0, 0

        loss, optimizer_metrics = training_loop._run_one_step(task_index, task_changed)

        # optimizer_metrics and loss are replicated on self.n_devices, a few
        # metrics are replicated (ex: gradients_l2, weights_l2) - i.e. they are
        # the same across devices, whereas some (ex: loss) aren't because they
        # are different on different devices (due to different data).
        # Taking the average does the correct thing in both the cases.
        #
        # NOTE: Only the weights and gradients are synced across the hosts. This
        # implies the loss here is averaged from this hosts' devices and not
        # across all hosts.
        optimizer_metrics, loss = fastmath.nested_map(
            functools.partial(tl.mean_or_pmean, training_loop._n_devices),
            (optimizer_metrics, loss))

        loss_acc += loss
        step_acc += 1

        if training_loop._eval_at(training_loop.step):
          training_loop.run_evals(eval_summary_writers)
          loss_acc, step_acc = 0.0, 0

    # Store the final values back into their respective objects, for testing
    # or other inspection/use.
    #
    # We keep the standard model weights/state unreplicated and
    # tl.Accelerate(model) will carry the replicated weights/state.
    # TODO(afrozm): Try to use tl.Accelerate(model) everywhere in the Loop.
    training_loop._eval_model.weights = training_loop._model.weights

hlo_computation = jax.xla_computation(wrapped_run)()
print ('Saving HLO files')
if not os.path.isdir('hlo_files'):
  os.makedirs('hlo_files')
with open("hlo_files/hlo_trace_transformer.txt", "w") as text_file:
  text_file.write(hlo_computation.as_hlo_text())
with open("hlo_files/hlo_trace_transformer.pb", "wb") as proto_file:
  proto_file.write(hlo_computation.as_serialized_hlo_module_proto())
