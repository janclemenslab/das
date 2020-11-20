# Install

## Pre-requisites


__Anaconda__: _DeepSS_ is installed using an anaconda environment. For that, you first have to [install anaconda](https://docs.anaconda.com/anaconda/install/) (or use [miniconda](https://docs.conda.io/en/latest/miniconda.html)).

__CUDA libraries for using the GPU__: While _DeepSS_ works well for annotating song using CPUs, GPUs will greatly improve annotation speed and are in particular recommended for training a _DeepSS_ network. The network is implement in the deep-learning framework Tensorflow. To make sure that Tensorflow can utilize the GPU, make sure you have the required CUDA libraries installed: [https://www.tensorflow.org/install/gpu]().

__Libsoundfile on linux__: The graphical user interface (GUI) reads audio data using [soundfile](http://pysoundfile.readthedocs.io/), which relies on `libsndfile`. `libsndfile` will be automatically installed on Windows and macOS but needs to be installed separately with: `sudo apt-get install libsndfile1`. Alternatively, _DeepSS_ can be installed and used w/o the GUI (see below).

## Install _DeepSS_ with or without the GUI
We then create an anaconda environment called `deepss` that contains all the required packages:
```shell
conda env create -f https://raw.githubusercontent.com/janclemenslab/deepsongsegmenter/master/env/deepss_gui.yml -n dss
```

This will also install the graphical user interface (GUI) `xb`. The information about the required packages is in [deepss_gui.yml](https://raw.githubusercontent.com/janclemenslab/deepsongsegmenter/master/env/deepss_gui.yml). If you do not need the `xb` GUI (for instance on a server), you can also install a plain version:

```shell
conda env create -f https://raw.githubusercontent.com/janclemenslab/deepsongsegmenter/master/env/deepss_plain.yml -n dss
```


## Test the installation
To test the installation, activate the conda environment and run these three commands in the terminal:
```shell
conda activate dss
dss-train --help
dss-predict --help
xb
```
The first two should display the command line arguments for `dss-train` and `dss-predict`. The last should start the graphical user interface.

## Next steps
If all is working, you probably want to adapt _DeepSS_ to annotate song in your own recordings. For that, you first need to manually [annotate song using the GUI](annotate). If you already have annotations, you are already able to [make a dataset for training](make_dataset) _DeepSS_.

<!--
# Manual
## Install python
Install conda (see [here](https://docs.conda.io/en/latest/miniconda.html)). DeepSS has been tested with python 3.7 and 3.8. Then create a fresh python environment and activate it:
```shell
conda create -n dss python=3.7 -y
conda activate dss
```
All of the following steps should be performed with this environment active.
```shell
conda env create -f deepss_gui_osx.yml -n dss
```



## Install _DeepSS_
This is sufficient if you want to use the _dss_ module via the command line or python:
```shell
conda install zarr tensorflow # or `tensorflow-gpu`
pip install git+https://github.com/janclemenslab/deepsongsegmenter
```
Tensorflow is the deep learning framework used to implement the DeepSS network. Tensorflow is not installed automatically to avoid interference with existing installations and to provide more control over versions. We recommend installation via conda install, but pip install should also work and typically installs a newer version. The manual install of zarr is required in some windows systems since the install via pip can be wonky. DeepSS has been tested with tensorflow versions 2.1, 2.2, and 2.3.

## Install the GUI
If you want to use the GUI to annotate song manually and with DeepSS and to train DeepSS networks:
```shell
conda install pyside2
conda install pyqtgraph=0.11.0rc0 python-sounddevice hdbscan umap-learn -c conda-forge -y
pip install xarray-behave[gui]
```


## Optional
Install tools for unsupervised analyses of song:
```shell
conda install hdbscan umap-learn -c conda-forge -y
pip install deepss-unsupervised
```


## Test the installation
To test the installation, run these three commands in the terminal:
```shell
dss-train --help
dss-predict --help
xb
```
The first two should display the command line arguments for `dss-train` and `dss-predict`. The last should start the graphical user interface.


```
conda create -n dss python=3.7 -y
conda activate dss
conda install zarr tensorflow -y # or `tensorflow-gpu`
conda install python-sounddevice hdbscan umap-learn -c conda-forge -y
conda install hdbscan umap-learn -c conda-forge -y
pip install git+https://github.com/janclemenslab/deepsongsegmenter
pip install xarray-behave[gui]
pip install deepss-unsupervised
dss-train --help
dss-predict --help
xb
``` -->

