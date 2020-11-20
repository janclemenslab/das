# Configure and train a network

## Train DeepSS using the GUI
Select the `*.npy` with the dataset, which song type to train the network for (will train network for all song types by default), and network and training parameters (see [DeepSS]() for details). Training can be started locally in a separate process or a script can be generated that can be execute to train elsewhere, for instance on a cluster.

edit script to fix paths, add cluster config commands or activate conda env.

Will run and put results in the save dir - four files:
- model (arch + params in one file but sth fails to load across versions)
- params
- arch (weights only - for robustness - load by making arch from params and load weights into it
- results

## Train notebook

## Train cli



## Network configuration
Explain important parameters here.



## Test the network
Load `_results.h5` into [test_deepss.ipynb]() notebook.