# DSS segments song

## Installation
Install all dependencies
```shell
conda create -n dss python=3.7
conda activate dss
conda install numpy scikit-learn scipy scikit-image tensorflow=2.0 pandas h5py yaml pywavelets matplotlib seaborn tqdm jupyterlab
conda install peakutils -c conda-forge
pip install flammkuchen defopt matplotlib-scalebar
pip install git+https://github.com/postpop/kapre.git
```
Important: If you want to use the GPU replace `tensorflow=2.0` with `tensorflow-gpu=2.0` in the above command

Finally, install _DSS_ itself:
```shell
pip install git+https://github.com/janclemenslab/deepsongsegmenter@tf2
```
or if you want to develop, clone the repo
```shell
git clone https://github.com/janclemenslab/deepsongsegmenter.git
cd deepsongsegmenter
pip install -e .
```

Use the [tf2](https://github.com/janclemenslab/deepsongsegmenter/tree/tf2) branch and the corresponding [tf2 branch of keras-tcn](https://github.com/postpop/keras-tcn/tree/tf2). Not thoroughly tested but runs and seems to reproduce the predictions of models trained with the old tf1.x.

## Docs
- [How to train](doc/training.md)
- [How to predict](doc/predict.md)
- [Data schema](doc/data.md)
