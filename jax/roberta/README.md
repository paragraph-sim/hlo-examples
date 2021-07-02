# Extracting HLO from JAX MNIST example

<a href="https://colab.research.google.com/drive/1gImVNzilEhLKhIEJwCZeZXhtfrH-rKI1" ><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook walks you through extracting HLO representation from a JAX application. This HLO can be fed to HLO bridge of ParaGraph and simulated in a hardware simulator supported by it. The notebook is originally based on [Masked Language Model Pretraining on TPU with 🤗 Transformers  & JAX](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/masked_language_modeling_flax.ipynb) available on the [Hugginface Transformers repo](https://github.com/huggingface/transformers/tree/master/examples/flax/language-modeling). The file was origianlly made available by Google under [Apache License 2.0](https://github.com/huggingface/transformers/blob/master/LICENSE), and modified by Mikhail Isaev at Nvidia. Only a few minor changes are required to extract the HLOs, and they are related to eliminating CPU code (anything that cannot be compiled to accelerators like TPUs or GPUs) from the compiled function.

In order to run the example, you need to have installed [JAX](https://github.com/google/jax#installation) and other dependencies from `reuirements.txt`. Simply run it as 
```bash
python roberta.py
```
It will store the traced HLOs under `hlo_files` directory.

If you want to store HLO graph corresponding to the running program, you need to setup `XLA_FLAGS` (HLO files before and after optimizations will be dumped together with low level code and object files):
```bash
export XLA_FLAGS="--xla_dump_to=./hlo_files --xla_dump_hlo_as_text=true "
python roberta.py
```

If you want to store HLO graph corresponding to the program running on a fake system with N processors (i.e. 32), you need to add flag `--xla_force_host_platform_device_count=32` (only HLO files after optimizations will be dumped together with low level code and object files):
```bash
export XLA_FLAGS="--xla_dump_to=./hlo_files --xla_dump_hlo_as_text=true --xla_force_host_platform_device_count=32 "
python roberta.py
```

