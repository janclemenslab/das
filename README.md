
![](https://github.com/janclemenslab/deepss/workflows/Python%20Package%20using%20Conda/badge.svg)

# DeepSS
DeepSS is supervised. See [dss-unsupervised](https://github.com/janclemenslab/dss-unsupervised) for unsupervised tools to augment DeepSS.

## Installation

```shell
conda create -n dss python=3.7
conda activate dss
conda install zarr
conda install tensorflow  # add -gpu to ensure GPU support
pip install deepss
```
DeepSS has been tested with python 3.7 and 3.8 and tensorflow versions 2.1, 2.2, and 2.3.

Tensorflow is *not* installed automatically to avoid interference with existing installations and to provide more control over versions. We recommend installation via `conda install`, but `pip install` should also work and typically installs a newer version. The manual install of zarr is required in some windows systems since the install via pip can be wonky.


## Tutorials
There are four tutorial notebooks that illustrate all steps required for going from annotated data via training and evaluating a network to segmenting new recordings:

- [Prepare training data](tutorials/1_prepare_data.ipynb)
- [Train the network](tutorials/2_training.ipynb)
- Evaluate the network and fine tune inference parameters for predicting [events like Drosophila song pulses](tutorials/3a_evaluate_events.ipynb) or [segments like Drosophila sine song or bird song syllables](tutorials/3b_evaluate_segments.ipynb)
- [Inference on new data](tutorials/4_inference.ipynb)
- [Realtime inference](tutorials/5_realtime.ipynb)

For the tutorials to work, you first need to download some data and example models (266MB) from [here](https://www.dropbox.com/sh/wnj3389k8ei8i1c/AACy7apWxW87IS_fBjI8-7WDa?dl=0) and put the four folders in the same folder as the tutorials notebooks. The tutorial notebooks also have extra dependencies:
`conda install jupyterlab ipython tqdm ipywidgets -y`


## Acknowledgements
The following packages were modified and integrated into dss:

- Keras implementation of TCN models modified from [keras-tcn](https://github.com/philipperemy/keras-tcn) (in `dss.tcn`)
- Trainable STFT layer implementation modified from [kapre](https://github.com/keunwoochoi/kapre) (in `dss.kapre`)
- FCN model code modified from [LEAP](https://github.com/talmo/leap)

See the sub-module directories for the original READMEs.
