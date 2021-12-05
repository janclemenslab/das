
install micromamba (or miniconda)

```shell
micromamba install -y -n base -c conda-forge python=3.8 mamba boa anaconda-client conda-build conda-verify
micromamba clean --all --yes

conda config --set anaconda_upload yes
anaconda login

conda mambabuild das-nogui -c conda-forge -c ncb --python 3.9 --user ncb
conda mambabuild das-nogui -c conda-forge -c ncb --python 3.8 --user ncb
conda mambabuild das-nogui -c conda-forge -c ncb --python 3.7 --user ncb
conda mambabuild das -c conda-forge -c ncb --python 3.9 --user ncb
conda mambabuild das -c conda-forge -c ncb --python 3.8 --user ncb
conda mambabuild das -c conda-forge -c ncb --python 3.7 --user ncb
```




```shell
mamba create -y -n das_build -c conda-forge -c ncb -c anaconda python=3.8 mamba boa anaconda-client conda-build conda-verify
conda activate das_build
conda config --set anaconda_upload yes
anaconda login

conda mambabuild das-nogui -c conda-forge -c ncb -c anaconda --python 3.9 --user ncb & conda mambabuild das-nogui -c conda-forge -c ncb -c anaconda --python 3.8 --user ncb & conda mambabuild das-nogui -c conda-forge -c ncb -c anaconda --python 3.7 --user ncb & conda mambabuild das -c conda-forge -c ncb -c anaconda --python 3.9 --user ncb & conda mambabuild das -c conda-forge -c ncb -c anaconda --python 3.8 --user ncb & conda mambabuild das -c conda-forge -c ncb -c anaconda --python 3.7 --user ncb
```