# Installation

## Pre-requisites

Install [Miniforge](https://github.com/conda-forge/miniforge) or another Conda distribution.

## Install DAS

Create and activate an isolated environment. Conda provides Python, FFmpeg, and `uv`; `uv` installs DAS and its Python dependencies.

Users who need the TensorFlow backend can install the final TensorFlow-backed release with `uv pip install das==0.32.13`.

```shell
conda create -n das -c conda-forge python=3.14 ffmpeg uv -y
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
