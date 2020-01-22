# DSS segments song[^1]

## Installation

```shell
conda create -n dss "python>=3.7,<3.8"
conda activate dss
pip install -e git+https://github.com/janclemenslab/deepsongsegmenter#egg=deepsongsegmenter[cpu]
```
Replace `[cpu]` at the end of the last command with `[gpu]` if you want to use the GPU. By default (without any `[...]` in the last command) tensorflow will no be installed - do this to avoid conflict with already installed cpu or gpu versions of tensorflow.

Local install: `pip install -e ".[cpu]"` or `".[gpu]"`.

To use the tutorial notebooks, you need to install a couple of extra dependencies:
```shell
conda install jupyterlab ipython tqdm ipywidgets -y
```

Troubleshooting:
- Pip install straight into a vanilla env failed on my mac due to zarr - I had to first manually install zarr using `conda install zarr`.
- [tensorflow-auto-detect](https://pypi.org/project/tensorflow-auto-detect/) could be used to automatically select the right tensorflow variant (CPU/GPU) but does as of Nov 2019 not support tf2.0.
- You can also manually install all dependencies (replace `"tensorflow>=2.0"` with `"tensorflow-gpu>=2.0"` if you want to use the GPU):
```shell
conda install "numpy>=1.8.0" scikit-learn scipy scikit-image "tensorflow>=2.0" pandas h5py yaml pywavelets matplotlib seaborn zarr
conda install librosa peakutils -c conda-forge
pip install flammkuchen defopt matplotlib-scalebar
```

## Tutorials
For the tutorials to work, you first need to download some data and example models (266MB) from [here](https://www.dropbox.com/sh/wnj3389k8ei8i1c/AACy7apWxW87IS_fBjI8-7WDa?dl=0) and put the four folders in the `tutorials` folder. Note also that you need to install some extra dependencies (see _Installation_ above)

There are four tutorial notebooks that illustrate all steps required for going from annotated data via training and evaluating a network to segmenting new recordings:
- [Prepare training data](tutorials/1_prepare_data.ipynb)
- [Train the network](tutorials/2_training.ipynb)
- Evaluate the network and fine tune inference parameters for predicting [events like Drosophila song pulses](tutorials/3a_evaluate_events.ipynb) or [segments like Drosophila sine song or bird song syllables](tutorials/3b_evaluate_segments.ipynb)
- [Inference on new data](tutorials/4_inference.ipynb)

## Acknowledgements
The following packages were modified and integrated into dss.
- Keras implementation of TCN models modified from [keras-tcn](https://github.com/philipperemy/keras-tcn) (in `dss.tcn`)
- Trainable STFT layer implementation modified from [kapre](https://github.com/keunwoochoi/kapre) (in `dss.kapre`)
- FCN model code modified from [LEAP](https://github.com/talmo/leap)

See the sub-module directories for the original READMEs.


[^1]: As in ["LEAP estimates animal pose"](https://github.com/talmo/leap)
