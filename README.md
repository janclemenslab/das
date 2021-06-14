[![Test install](https://github.com/janclemenslab/deepss/actions/workflows/main.yml/badge.svg)](https://github.com/janclemenslab/deepss/actions/workflows/main.yml)

# DeepSS
_DeepSS_ is a method for automatically annotating song from raw audio recordings based on a deep neural network. _DeepSS_ can be used with a graphical user interface, from the terminal or from with in python scripts.

If you have questions, feedback, or find bugs please raise in [issue](https://github.com/janclemenslab/deepss/issues).

Please cite DeepSS as:

Elsa Steinfath, Adrian Palacios, Julian Rottschäfer, Deniz Yuezak, Jan Clemens (2021). _Fast and accurate annotation of acoustic signals with deep neural networks_, bioRxiv, [https://doi.org/10.1101/2021.03.26.436927]()

## Installation
### Pre-requisites


__Anaconda__: _DeepSS_ is installed using an anaconda environment. For that, first install the [anaconda python distribution](https://docs.anaconda.com/anaconda/install/) (or [miniconda](https://docs.conda.io/en/latest/miniconda.html)).

If you have conda already installed, make sure you have conda v4.8.4+. If not, update from an older version with `conda update conda`.

<!-- ```shell
curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
``` -->

__CUDA libraries for using the GPU__: While _DeepSS_ works well for annotating song using CPUs, GPUs will greatly improve annotation speed and are in particular recommended for training a _DeepSS_ network. The network is implement in the deep-learning framework Tensorflow. To make sure that Tensorflow can utilize the GPU, the required CUDA libraries need to be installed. See the [tensorflow docs](https://www.tensorflow.org/install/gpu) for details.

__Libsoundfile on linux__: The graphical user interface (GUI) reads audio data using [soundfile](http://pysoundfile.readthedocs.io/), which relies on `libsndfile`. `libsndfile` will be automatically installed on Windows and macOS. On Linux, the library needs to be installed manually with: `sudo apt-get install libsndfile1`. Note that _DeepSS_ will work w/o `libsndfile` but will only be able to load more unusual audio file formats.

### Install _DeepSS_ with or without the GUI
Create an anaconda environment called `deepss` that contains all the required packages, including the GUI:
```shell
conda env create -f https://raw.githubusercontent.com/janclemenslab/deepss/master/env/deepss_gui.yml -n dss
```

If you do not need require the graphical user interface `dss-gui` (for instance, for training _DeepSS_ on a server), install the plain version:
```shell
conda env create -f https://raw.githubusercontent.com/janclemenslab/deepss/master/env/deepss_plain.yml -n dss
```

## Usage
To start the graphical user interface:
```shell
conda activate dss
dss gui
```

The documentation at [https://janclemenslab.org/deepss/] provides for information on the usage of _DeepSS_:

- A [quick start tutorial](https://janclemenslab.org/deepss/tutorials_gui/quick_start.html) that walks through all steps from manually annotating song, over training a network, to generating new annotations.
- How to use [graphical user interface](https://janclemenslab.org/deepss/tutorials_gui).
- How to use _DeepSS_ from [the terminal or python scripts](https://janclemenslab.org/deepss/tutorials/tutorials.html).



## Acknowledgements
The following packages were modified and integrated into dss:

- Keras implementation of TCN models modified from [keras-tcn](https://github.com/philipperemy/keras-tcn) (in `dss.tcn`)
- Trainable STFT layer implementation modified from [kapre](https://github.com/keunwoochoi/kapre) (in `dss.kapre`)

See the sub-module directories for the original READMEs.
