# Apple ARM (M1/M1 pro/M1 max)
Due to dependency conflicts, the GUI and the deep learning cannot be installed at the same time. But you can install each component into different conda environments:

## GUI
```shell
CONDA_SUBDIR=osx-64 conda create -y -n das_gui python=3.9   # create a new environment called rosetta with intel packages.
conda activate das_gui
python -c "import platform;print(platform.machine())"
conda env config vars set CONDA_SUBDIR=osx-64  #
conda install xarray-behave -y
xb gui
```

Note that the GUI is started with `xb gui`, *not* `das gui`.


## Training

```shell
conda env create -f env/das-nogui-osxarm-env.yml -n das_dev
conda activate das_dev
conda install numba -c conda-forge
conda install llvmlite -c conda-forge
conda install libsndfile -c conda-forge
pip install librosa
pip install das
das version
```
