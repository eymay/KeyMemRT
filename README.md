# KeyMemRT

KeyMemRT is a compiler and runtime framework that reduces the memory footprint of FHE applications. It introduces infrastructure and optimizations to manage rotation keys in FHE.

This repo includes the runtime `include/KeyMemRT.hpp` and the flows with the benchmarks used for KeyMemRT. Please check [KeyMemRT-Compiler](https://github.com/eymay/KeyMemRT-Compiler) for the compiler part.

More information is available in the KeyMemRT paper [here](https://arxiv.org/abs/2601.18445).

## Dependencies

1. Install the prerequisites:
```shell
sudo apt update && sudo apt install  \
    build-essential cmake clang ninja-build \
    git python3 python3-pip python3-venv \
    wget curl libssl-dev libomp-dev lld mold zlib1g-dev libcereal-dev
```
2. Install OpenFHE

For OpenFHE installation, please take a look at the [documentation](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/linux.html). We use a custom fork of it for experimental features, however they are not a core part of KeyMemRT.

```shell
git clone https://github.com/eymay/openfhe-development && \
cd openfhe-development && \
mkdir build && cd build && \
cmake -B . -S .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ && \
make -j $(nproc) && \
sudo make install
```

3. Install Orion and Orion-HEIR-Translator

To make use of the Orion frontend, we need to install [Orion](https://github.com/baahl-nyu/orion) FHE compiler and [Orion-HEIR-Translator](https://github.com/eymay/orion_heir_translator) to make it work with MLIR.

```shell
git clone https://github.com/eymay/orion_heir_translator.git && \
python3 -m venv .venv && . .venv/bin/activate && \
cd orion_heir_translator && pip install .[orion-fhe] 
```
4. Install KeyMemRT-Compiler

Please follow the README of KeyMemRT-Compiler [here](https://github.com/eymay/KeyMemRT-Compiler) to build. With `Bazelisk` and the dependencies installed:

```shell
git clone https://github.com/eymay/KeyMemRT-Compiler.git && cd KeyMemRT-Compiler/ && \
bazel build @heir//tools:keymemrt-opt && \
bazel build @heir//tools:keymemrt-translate
```
This repo expects `KEYMEMRT_COMPILER` env variable to point to the path of `KeyMemRT-Compiler`. To do that:
```shell
export KEYMEMRT_COMPILER=<path-to-keymemrt-compiler>
```
This can be added to `~/.bashrc` to make the env variable persistent across terminal sessions.

## Running the Benchmarks
The following ML models are supported: `mlp lola lenet alexnet vgg yolo resnet1 resnet8 resnet10 resnet18 resnet20 resnet32 resnet34 resnet44 resnet50 resnet56`. To build and run all variants of any model:

```shell
just run-servers mlp
```

The memory and time results can be found in `build/results` and output of the runs are in `build/logs`.
