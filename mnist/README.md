# Extracting HLO from JAX MNIST example

<a href="https://colab.research.google.com/drive/1nZjleMTiCst5W14sm6vbcZQaZlPWXkSN" ><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook walks you through extracting HLO representation from a JAX application. This HLO can be fed to HLO bridge of ParaGraph and simulated in a hardware simulator supported by it. The code is originally based on [JAX SPMD MNIST example](https://github.com/google/jax/blob/main/examples/spmd_mnist_classifier_fromscratch.py) available on the [JAX repo](https://github.com/google/jax/blob/main/examples). Only a few minor changes are required to extract the HLOs, and they are related to eliminating CPU code (anything that cannot be compiled to accelerators like TPUs or GPUs) from the compiled function. We will mark such cells with <font color='red'>**[ParaGraph edits]**</font> badge.
