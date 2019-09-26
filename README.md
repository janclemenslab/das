# DNNs for song segmentation


## Installation

## TODO:
[ ] clean up and git-ize
[x] use data from multiple recordings for training
[-] memoize/speed up cwt
[s] select best architecture
[x] test different losses (MSE, categorical cross-entropy) - makes these params for reproducibility
[x] predict should use `power_only` param from `params.yaml`
[s] model_dict should be part of `models.py`
[x] steps_per_epoch=4*batch_size seemed to work best - maybe increase further (to 8*batch_size?)

# loading models
Due to python 3.5 on the cluster, straight out `keras.models.load_model` does not work for networks with custom/lambda layers that were trained on the cluster. However, saved weights can be loaded into an existing architecture — this is what `dd.utils.load_model_from_params` does.


# Outdated: note on package versions (outdated - update!!)
- tensorflow w/ gpu on the cluster only works with python 3.5 for now due to constraints on CUDA/CUDNN library versions I think.
- requires py3.5.2+ for all dependencies to work
- only works reproducibly with tensorflow 1.11.0/1.13.0 and keras 2.2.4 (install with pip - conda install will try to upgrade to py3.6) - NO - works with py37 and tf1.13.1
- see `environment.yml`
