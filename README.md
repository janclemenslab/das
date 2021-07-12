[![Test install](https://github.com/janclemenslab/das/actions/workflows/main.yml/badge.svg)](https://github.com/janclemenslab/das/actions/workflows/main.yml)

# Deep Audio Segmenter
_DAS_ is a method for automatically annotating song from raw audio recordings based on a deep neural network. _DAS_ can be used with a graphical user interface, from the terminal, or from within python scripts.

If you have questions, feedback, or find bugs please raise an [issue](https://github.com/janclemenslab/das/issues).

Please cite _DAS_ as:

Elsa Steinfath, Adrian Palacios, Julian Rottschäfer, Deniz Yuezak, Jan Clemens (2021).   
_Fast and accurate annotation of acoustic signals with deep neural networks._   
bioRxiv, [https://doi.org/10.1101/2021.03.26.436927]()

## Installation
### Pre-requisites


__Anaconda__: _DAS_ is installed using an anaconda environment. For that, first install the [anaconda python distribution](https://docs.anaconda.com/anaconda/install/) (or [miniconda](https://docs.conda.io/en/latest/miniconda.html)).

If you have conda already installed, make sure you have conda v4.8.4+. If not, update from an older version with `conda update conda`.

<!-- ```shell
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
``` -->

__CUDA libraries for using the GPU__: While _DAS_ works well for annotating song using the CPU, a GPU will greatly improve annotation speed and is recommended for training a _DAS_ network. The network is implemented in the deep-learning framework Tensorflow. To make sure that Tensorflow can use your GPU, the required CUDA libraries need to be installed. See the [tensorflow docs](https://www.tensorflow.org/install/gpu) for details.

__Libsoundfile on linux__: The graphical user interface (GUI) reads audio data using [soundfile](http://pysoundfile.readthedocs.io/), which relies on `libsndfile`. `libsndfile` will be automatically installed on Windows and macOS. On Linux, the library needs to be installed manually with: `sudo apt-get install libsndfile1`. Note that _DAS_ will work w/o `libsndfile` but will not be able to load exotic audio formats.

### Install _DAS_ with or without the GUI
Create an anaconda environment called `das` that contains all the required packages, including the GUI:
```shell
conda env create -f https://raw.githubusercontent.com/janclemenslab/das/master/env/das_gui.yml -n das
```

If you do not need the graphical user interface, for instance, when training _DAS_ on a server, install the plain version:
```shell
conda env create -f https://raw.githubusercontent.com/janclemenslab/das/master/env/das_plain.yml -n das
```

## Usage
To start the graphical user interface:
```shell
conda activate das
das gui
```

The documentation at [https://janclemenslab.org/das/](https://janclemenslab.org/das/) provides information on the usage of _DAS_:

- A [quick start tutorial](https://janclemenslab.org/das/quick_start.html) walks through all steps from manually annotating song, over training a network, to generating new annotations.
- How to use the [graphical user interface](https://janclemenslab.org/das/tutorials_gui/tutorials_gui.html).
- How to use _DAS_ [from the terminal or from python scripts](https://janclemenslab.org/das/tutorials/tutorials.html).



## Acknowledgements
The following packages were modified and integrated into das:

- Keras implementation of TCN models modified from [keras-tcn](https://github.com/philipperemy/keras-tcn) (in `das.tcn`)
- Trainable STFT layer implementation modified from [kapre](https://github.com/keunwoochoi/kapre) (in `das.kapre`)

See the sub-module directories for the original READMEs.
