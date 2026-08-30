---
orphan: true
---

# TensorFlow installation

Version 0.32.13 is the final TensorFlow-backed release of _DAS_. Use its OS-specific Conda environment file to install compatible versions of Python, FFmpeg, TensorFlow, and _DAS_.

## Pre-requisites

Install the [Anaconda distribution](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). If Conda is already installed, use version 23.10.0 or later. Update an older installation with:

```shell
conda update conda -n base
```

On Linux, install `libsndfile` to load audio formats other than WAV:

```shell
sudo apt-get install libsndfile1
```

## Install _DAS_

### Windows

```shell
conda env create -n das -y -f https://raw.githubusercontent.com/janclemenslab/das/v0.32.13/env/das_win.yaml
```

### Mac (M1 and later)

```shell
conda env create -n das -y -f https://raw.githubusercontent.com/janclemenslab/das/v0.32.13/env/das_mac.yaml
```

### Linux

```shell
conda env create -n das -y -f https://raw.githubusercontent.com/janclemenslab/das/v0.32.13/env/das_linux.yaml
```

## Open the graphical user interface

```shell
conda activate das
das gui
```

## Next steps

You can now annotate song and train a network on your own data. Start by creating annotations [using the GUI](/tutorials_gui/tutorials_gui) or convert existing annotations [using Python scripts](/tutorials/tutorials).
