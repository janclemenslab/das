# DSS segments song

## Installation
Install all dependencies
```shell
conda create -n dss "python>=3.7,<3.8"
conda activate dss
conda install "tensorflow>=2.0"
pip install git+https://github.com/janclemenslab/deepsongsegmenter
```
(replace "tensorflow>=2.0" with "tensorflow-gpu>=2.0" if you want to use the GPU)

Troubleshooting:
- Pip install straight into a vanilla env failed on my mac due to zarr - I had to first manually install zarr using `conda install zarr`.
- [tensorflow-auto-detect](https://pypi.org/project/tensorflow-auto-detect/) could be used to automatically select the right tensorflow variant (CPU/GPU) but does as of Nov 2019 not work support tf2.0.
- You can also manually install all dependencies (replace "tensorflow>=2.0" with "tensorflow-gpu>=2.0" if you want to use the GPU):
```shell
conda install "numpy>=1.8.0" scikit-learn scipy scikit-image "tensorflow>=2.0" pandas h5py yaml pywavelets librosa matplotlib seaborn tqdm jupyterlab
conda install peakutils -c conda-forge
pip install flammkuchen defopt matplotlib-scalebar
```
- the notebooks have a couple extra deps:
```shell
conda install jupyterlab ipython tqdm ipywidgets -y
pip install
```

## Docs
- [How to train](doc/training.md)
- [How to predict](doc/predict.md)
- [Data schema](doc/data.md)

## Acknowledgements
The following packages were modified and integrated into dss.
- Keras implementation of TCNs: [keras-tcn](https://github.com/philipperemy/keras-tcn)
- Trainable STFT layer: [kapre](https://github.com/keunwoochoi/kapre)

See the module directories for the original READMEs.
