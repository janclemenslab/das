# Installation

## Pre-requisites

_Anaconda python_: Install the [anaconda python distribution](https://docs.anaconda.com/anaconda/install/) (or [miniconda](https://docs.conda.io/en/latest/miniconda.html)). If you have conda already installed, make sure you have at least conda v23.10.0. If not, update from an older version with `conda update conda -n base`.

_Libsoundfile (Linux only)_: If you are on Linux and want to load audio from a wide range of audio formats (other than `wav`), then you need to install `libsndfile`. The GUI uses the [soundfile](http://pysoundfile.readthedocs.io/) python package, which relies on `libsndfile`. `libsndfile` will be automatically installed on Windows and macOS. On Linux, the library needs to be installed manually with: `sudo apt-get install libsndfile1`. Again, this is only required if you work with more exotic audio files.

## Install _DAS_
<!-- Create an anaconda environment called `das` that contains all the required packages.

On windows:
```shell
conda create python=3.10 das=0.32.5 -c conda-forge -c ncb -c nvidia -n das -y
```

On Linux or MacOS (arm only):
```shell
conda create python=3.11 das=0.32.5 -c conda-forge -c ncb -c nvidia -c apple -n das -y
``` -->

### Windows
```shell
conda env create -n das -y -f https://raw.githubusercontent.com/janclemenslab/das/refs/heads/master/env/das_win.yaml
```

### Mac (M1 and later)
```shell
conda env create -n das -y -f https://raw.githubusercontent.com/janclemenslab/das/refs/heads/master/env/das_mac.yaml
```

### Linux
```shell
conda env create -n das -y -f https://raw.githubusercontent.com/janclemenslab/das/refs/heads/master/env/das_linux.yaml
```




## Open the graphical user interface
```shell
conda activate das  # activate the conda environment
das gui  # start the DAS GUI
```

## Next steps
If all is working, you can now use _DAS_ to annotate song. To get started, you will first need to train a network on your own data. For that you need annotated audio - either create new annotations [using the GUI](/tutorials_gui/tutorials_gui) or convert existing annotations [using python scripts](/tutorials/tutorials).
