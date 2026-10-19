# Installation

## Pre-requisites

Install [Miniforge](https://github.com/conda-forge/miniforge) or another Conda distribution.

## Install DAS

Create an isolated environment. Conda provides Python, Qt, FFmpeg, `numcodecs`, and `uv`; `uv` installs DAS and its remaining Python dependencies. Installing Qt and FFmpeg together through Conda keeps their native libraries compatible.

Users who need the TensorFlow backend should follow the [TensorFlow installation instructions](install_tf.md) for the final TensorFlow-backed release.

On Windows and Linux:

```shell
conda create -n das -c conda-forge --strict-channel-priority python=3.14 pyside6 ffmpeg "numcodecs<0.16" uv -y
```

On macOS, select the OpenBLAS build that avoids loading a second OpenMP runtime alongside PyTorch:

```shell
conda create -n das -c conda-forge --strict-channel-priority python=3.14 pyside6 ffmpeg "numcodecs<0.16" "libopenblas=*=*pthreads*" uv -y
```

Then activate the environment and install DAS:

```shell
conda activate das
uv pip install das --torch-backend=auto
```

`--torch-backend=auto` selects a suitable PyTorch build for the available hardware. To request a specific build instead, replace `auto` with one of these backends:

```shell
uv pip install das --torch-backend=cpu       # CPU only
uv pip install das --torch-backend=cu130     # NVIDIA CUDA 13.0
uv pip install das --torch-backend=rocm7.2   # AMD ROCm 7.2 on Linux
uv pip install das --torch-backend=xpu       # Intel GPU
```

Available accelerator builds depend on the operating system and hardware. On macOS, the standard PyTorch build supports Apple Metal acceleration.

## Verify the installation
```shell
das version
das gui
```

## Next steps
If all is working, you can now use _DAS_ to annotate song. To get started, you will first need to train a network on your own data. For that you need annotated audio - either create new annotations [using the GUI](/tutorials_gui/tutorials_gui) or convert existing annotations [using python scripts](/tutorials/tutorials).
