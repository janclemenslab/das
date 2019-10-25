# Deep song segmenter (DSS)

## Installation
Install all dependencies
```shell
conda create -n dss python=3.7
source activate dss
conda install keras numpy scikit-learn scipy scikit-image tensorflow-gpu pandas h5py yaml pywavelets matplotlib seaborn
conda install peakutils deepdish -c conda-forge
pip install defopt
pip install matplotlib-scalebar
pip install git+https://github.com/postpop/keras-tcn
```

Finally, install _DSS_ itself:
```shell
pip install git+https://github.com/janclemenslab/deepsongsegmenter
```
or if you want to develop, clone the repo
```shell
git clone https://github.com/janclemenslab/deepsongsegmenter.git
cd deepsongsegmenter
pip install -e .
```

Use the [tf2](https://github.com/janclemenslab/deepsongsegmenter/tree/tf2) branch and the corresponding [tf2 branch of keras-tcn](https://github.com/postpop/keras-tcn/tree/tf2). Not thoroughly tested but runs and seems to reproduce the predictions of models trains with the old tf1.x.