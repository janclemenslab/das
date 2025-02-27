<!-- [![Test install](https://github.com/janclemenslab/das/actions/workflows/main.yml/badge.svg)](https://github.com/janclemenslab/das/actions/workflows/main.yml) -->

# Deep Audio Segmenter
_DAS_ is a method for automatically annotating song from raw audio recordings based on a deep neural network. _DAS_ can be used with a graphical user interface, from the terminal, or from within python scripts.

If you have questions, feedback, or find bugs please raise an [issue](https://github.com/janclemenslab/das/issues).

Please cite _DAS_ as:

Elsa Steinfath, Adrian Palacios, Julian Rottschäfer, Deniz Yuezak, Jan Clemens (2021).
_Fast and accurate annotation of acoustic signals with deep neural networks._
[eLife](https://doi.org/10.7554/eLife.68837)

## Installation
### Pre-requisites


_Anaconda_: _DAS_ is installed using an anaconda environment. For that, first install the [anaconda python distribution](https://docs.anaconda.com/anaconda/install/) (or [miniconda](https://docs.conda.io/en/latest/miniconda.html)). If you have conda already installed, make sure you have at least conda v23.10.0. If not, update from an older version with `conda update conda -n base`.

_Libsoundfile on linux_: The graphical user interface (GUI) reads audio data using [soundfile](http://pysoundfile.readthedocs.io/), which relies on `libsndfile`. `libsndfile` will be automatically installed on Windows and macOS. On Linux, the library needs to be installed manually with: `sudo apt-get install libsndfile1`. Note that _DAS_ will work w/o `libsndfile` but will not be able to load exotic audio formats.

### Install _DAS_
Create an anaconda environment called `das` that contains all the required packages.

On windows:
```shell
conda create python=3.10 das=0.32.5 -c conda-forge -c ncb -c nvidia -n das -y
```

On Linux or MacOS (arm only):
```shell
conda create python=3.11 das=0.32.5 -c conda-forge -c ncb -c nvidia -c apple -n das -y
```

## Usage
To start the graphical user interface:
```shell
conda activate das
das gui
```

The documentation at [https://janclemenslab.org/das/](https://janclemenslab.org/das/) provides information on the usage of _DAS_:

- A [quick start tutorial](https://janclemenslab.org/das/quickstart.html) walks through all steps from manually annotating song, over training a network, to generating new annotations.
- How to use the [graphical user interface](https://janclemenslab.org/das/tutorials_gui/tutorials_gui.html).
- How to use _DAS_ [from the terminal or from python scripts](https://janclemenslab.org/das/tutorials/tutorials.html).



## Acknowledgements
The following packages were modified and integrated into das:

- Keras implementation of TCN models modified from [keras-tcn](https://github.com/philipperemy/keras-tcn) (in `das.tcn`)
- Trainable STFT layer implementation modified from [kapre](https://github.com/keunwoochoi/kapre) (in `das.kapre`)

See the sub-module directories for the original READMEs.
