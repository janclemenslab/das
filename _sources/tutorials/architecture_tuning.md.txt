# Architecture tuning
Often, the standard parameters are sufficient to get a performing model - see the [Advice on architecture tuning](/tutorials/struct_params).

However, sometimes "you want more"...

- Given that training is often fast, you can __manually__ run an exhaustive search by fitting models with different architectures.
- We have also added experimental support for __automatic__ architecture tuning via [Keras Tuner](https://keras.io/keras_tuner) for when the search space is too big for an exhaustive search.
It is particularly useful to track the tuning on [Weights & Biases](http://wandb.ai) - see the examples below and [experiment tracking](experiment_tracking).


## Manual tuning
If the search space is small and training fast, it is often best to do an exhaustive search by fitting models with all parameter combinations. It is advisable to run repeated fits with the same combination to be robust to random effects.

```python
import das.train
from itertools import product
from pprint import pprint

nb_convs = [2, 3, 4]
nb_filters = [32, 64, 96]
learning_rates = [0.0001, 0.00001]
repeats = list(range(4))

parameter_combinations = product(nb_convs, nb_filters, learning_rates, repeats)

results = []

for nb_conv, nb_filters, learning_rate, repeat in  parameter_combinations:
    model, params, fit_hist = das.train.train(data_dir='tutorial_dataset.npy', save_dir='res', nb_conv=nb_conv, nb_filters=nb_filters, learning_rate=learning_rate, nb_epoch=20, WANDB_ARGS)
    results.append({'nb_conv': nb_conv,
                    'nb_filters':nb_filters,
                    'learning_rate': learning_rate,
                    'training_loss': min(fit_hist.history['loss']),
                    'validation_loss': min(fit_hist.history['val_loss']),})
    pprint(results[-1])

# plot results
```
During the training, the results can be tracked on [https://wandb.ai]().

A model with A TCN blocks, B filters, and a learning rate of C yields the lowest validation loss.


## Automatic architecture tuning
The interface is similar to das.train:
`das.train_tune.train(data_dir='tutorial_data.npy', save_dir='res', kernel_size=3, tune_config='tune.yml')`. There is also a CLI that you can access via `das tune` (see [CLI documentation](/technical/cli)).

Crucially, it accepts a yaml file with the parameter names and a set of values you want the optimizer to try. For instance, if you want the optimizer to try models with

- 2, 3, or 3 TCN blocks,
- 32, 64, or 96 filters, and
- learning rates of 0.0001 or 0.00001

then the `tune.yml` file should look like this:
```yaml
nb_conv: [2, 3, 4]
nb_filters: [32, 64, 96]
learning_rate: [0.0001, 0.00001]
```

The tuner will then run a bunch of fits to find an optimal parameter combination in the search space defined in the yaml file - see keras tuner for how this works.

```python
import das.train_tune

model, params, tuner = das.train_tune.train(
    data_dir="tutorial_dataset.npy",
    save_dir="res",
    kernel_size=16,
    tune_config="tuning.yml",
    WANDB_ARGS,
)
```
Will take N minutes, during which results will can be inspected at [Link to wandb site for this project]().

