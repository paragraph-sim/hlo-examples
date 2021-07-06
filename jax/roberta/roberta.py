import jax
import optax
import flax
import jax.numpy as jnp

from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard

import numpy as np

from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from transformers import AutoTokenizer, AutoConfig, FlaxAutoModelForMaskedLM

import os
from pathlib import Path

language = "is"
model_config = "roberta-base"
model_dir = model_config + f"-pretrained-{language}"
Path(model_dir).mkdir(parents=True, exist_ok=True)
config = AutoConfig.from_pretrained(model_config)
config.save_pretrained(f"{model_dir}")

if 'XLA_FLAGS' in os.environ and \
        "xla_force_host_platform_device_count" in os.environ['XLA_FLAGS']:
  jax.config.update('jax_platform_name', 'cpu')

from jax.lib import xla_bridge
print("Jax platform:", xla_bridge.get_backend().platform)
print (jax.devices())


raw_dataset = load_dataset("oscar", f"unshuffled_deduplicated_{language}")
tokenizer = ByteLevelBPETokenizer()
def batch_iterator(batch_size=1000):
    for i in range(0, len(raw_dataset), batch_size):
        yield raw_dataset["train"][i: i + batch_size]["text"]

tokenizer.train_from_iterator(batch_iterator(), vocab_size=config.vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save(f"{model_dir}/tokenizer.json")

max_seq_length = 128
raw_dataset["train"] = load_dataset("oscar", f"unshuffled_deduplicated_{language}", split="train[5%:]")
raw_dataset["validation"] = load_dataset("oscar", f"unshuffled_deduplicated_{language}", split="train[:5%]")

# these cells should be commented out to run on full dataset
raw_dataset["train"] = raw_dataset["train"].select(range(10000))
raw_dataset["validation"] = raw_dataset["validation"].select(range(1000))

tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}")

def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=raw_dataset["train"].column_names)

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)


per_device_batch_size = 64
num_epochs = 2      # We reduced number of steps to just a single epoch
training_seed = 0
learning_rate = 5e-5

total_batch_size = per_device_batch_size * jax.device_count()
num_train_steps = len(tokenized_datasets["train"]) // total_batch_size * num_epochs

print ("creating the model")
model = FlaxAutoModelForMaskedLM.from_config(config, seed=training_seed, dtype=jnp.dtype("float32"))

linear_decay_lr_schedule_fn = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps)
adamw = optax.adamw(learning_rate=linear_decay_lr_schedule_fn, b1=0.9, b2=0.98, eps=1e-8, weight_decay=0.01)
state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)

@flax.struct.dataclass
class FlaxDataCollatorForMaskedLanguageModeling:
    mlm_probability: float = 0.15

    def __call__(self, examples, tokenizer):
        batch = tokenizer.pad(examples, return_tensors="np")

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask, tokenizer
        )

        return batch

    def mask_tokens(self, inputs, special_tokens_mask, tokenizer):
        labels = inputs.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        indices_random &= masked_indices & ~indices_replaced
        random_words = np.random.randint(tokenizer.vocab_size, size=labels.shape, dtype="i4")
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

data_collator = FlaxDataCollatorForMaskedLanguageModeling(mlm_probability=0.15)

def generate_batch_splits(num_samples, batch_size, rng=None):
    samples_idx = jax.numpy.arange(num_samples)

    # if random seed is provided, then shuffle the dataset
    if input_rng is not None:
        samples_idx = jax.random.permutation(input_rng, samples_idx)

    samples_to_remove = num_samples % batch_size

    # throw away incomplete batch
    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    
    batch_idx = np.split(samples_idx, num_samples // batch_size)
    return batch_idx

def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        labels = batch.pop("labels")

        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

        # compute loss, ignore padded input tokens
        label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

        # take average
        loss = loss.sum() / label_mask.sum()

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
        {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
    )

    return new_state, metrics, new_dropout_rng

parallel_train_step = jax.pmap(train_step, "batch")

def eval_step(params, batch):
    labels = batch.pop("labels")

    logits = model(**batch, params=params, train=False)[0]

    label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
    loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

    # compute accuracy
    accuracy = jax.numpy.equal(jax.numpy.argmax(logits, axis=-1), labels) * label_mask

    # summarize metrics
    metrics = {"loss": loss.sum(), "accuracy": accuracy.sum(), "normalizer": label_mask.sum()}
    metrics = jax.lax.psum(metrics, axis_name="batch")

    return metrics

parallel_eval_step = jax.pmap(eval_step, "batch")

print ("Replicating the state across devices")
state = flax.jax_utils.replicate(state)

def process_eval_metrics(metrics):
    metrics = get_metrics(metrics)
    metrics = jax.tree_map(jax.numpy.sum, metrics)
    normalizer = metrics.pop("normalizer")
    metrics = jax.tree_map(lambda x: x / normalizer, metrics)
    return metrics

rng = jax.random.PRNGKey(training_seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())

for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)

    # -- Train --
    train_batch_idx = generate_batch_splits(len(tokenized_datasets["train"]), total_batch_size, rng=input_rng)

    for batch_idx in train_batch_idx:
        model_inputs = data_collator(tokenized_datasets["train"][batch_idx], tokenizer=tokenizer)

        # Model forward
        model_inputs = shard(model_inputs.data)
        state, train_metric, dropout_rngs = parallel_train_step(state, model_inputs, dropout_rngs)


    # -- Eval --
    eval_batch_idx = generate_batch_splits(len(tokenized_datasets["validation"]), total_batch_size)
    eval_metrics = []

    for batch_idx in eval_batch_idx:
        model_inputs = data_collator(tokenized_datasets["validation"][batch_idx], tokenizer=tokenizer)

        # Model forward
        model_inputs = shard(model_inputs.data)
        eval_metric = parallel_eval_step(state.params, model_inputs)
        eval_metrics.append(eval_metric)

    eval_metrics_dict = process_eval_metrics(eval_metrics)


# Extracting HLOs
def hlo_from_training_function(state):
    rng = jax.random.PRNGKey(0)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    rng, input_rng = jax.random.split(rng)

    # -- Train --
    train_batch_idx = generate_batch_splits(len(tokenized_datasets["train"]), total_batch_size, rng=input_rng)

    for batch_idx in train_batch_idx:
        model_inputs = data_collator(tokenized_datasets["train"][batch_idx], tokenizer=tokenizer)

        def wrapped_train_step(model_inputs, state, dropout_rngs):
            # Model forward
            model_inputs = shard(model_inputs)
            #for i in range(len(train_batch_idx)):
            for i in range(2):
                state, train_metric, dropout_rngs = parallel_train_step(state, model_inputs, dropout_rngs)
        return jax.xla_computation(wrapped_train_step)(model_inputs.data, state, dropout_rngs)


hlo_computation = hlo_from_training_function(state)
print ('Saving HLO files')
if not os.path.isdir('hlo_files'):
  os.makedirs('hlo_files')
with open("hlo_files/hlo_trace_roberta.txt", "w") as text_file:
  text_file.write(hlo_computation.as_hlo_text())
with open("hlo_files/hlo_trace_roberta.pb", "wb") as proto_file:
  proto_file.write(hlo_computation.as_serialized_hlo_module_proto())
