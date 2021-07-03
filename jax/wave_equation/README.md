# Extracting HLO from JAX Wave Equation example

<a href="https://colab.research.google.com/drive/158HjHaxfYZ0gkO0PgyHFtduGB_Z6pcNG" ><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook walks you through extracting HLO representation from a JAX application. This HLO can be fed to HLO bridge of ParaGraph and simulated in a hardware simulator supported by it. The notebook is originally based on [JAX TPU Wave Equation Colab](https://colab.research.google.com/github/google/jax/blob/master/cloud_tpu_colabs/Wave_Equation.ipynb) available on the [JAX repo](https://github.com/google/jax/blob/85796cc7e3430536a68df72f7d60b0504787e235/cloud_tpu_colabs/Wave_Equation.ipynb) available on the [JAX repo](https://github.com/google/jax/tree/85796cc7e3430536a68df72f7d60b0504787e235/cloud_tpu_colabs). The file was origianlly made available by Google under [Apache License 2.0](https://github.com/google/jax/blob/85796cc7e3430536a68df72f7d60b0504787e235/LICENSE), and modified by Mikhail Isaev at Nvidia. Only a few minor changes are required to extract the HLOs, and they are related to eliminating CPU code (anything that cannot be compiled to accelerators like TPUs or GPUs) from the compiled function.

In order to run the example, you need to have [JAX installed](https://github.com/google/jax#installation). Simply run it as 
```bash
python wave_equation.py
```
It will store the traced HLOs under `hlo_files` directory.

If you want to store HLO graph corresponding to the running program, you need to setup `XLA_FLAGS` (HLO files before and after optimizations will be dumped together with low level code and object files):
```bash
export XLA_FLAGS="--xla_dump_to=./hlo_files --xla_dump_hlo_as_text=true "
python wave_equation.py
```

If you want to store HLO graph corresponding to the program running on a fake system with N processors (i.e. 32), you need to add flag `--xla_force_host_platform_device_count=32` (only HLO files after optimizations will be dumped together with low level code and object files):
```bash
export XLA_FLAGS="--xla_dump_to=./hlo_files --xla_dump_hlo_as_text=true --xla_force_host_platform_device_count=32 "
python wave_equation.py
```

