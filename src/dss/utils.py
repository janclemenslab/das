"""General utilities"""
import tensorflow.keras as keras
import logging
import time
import numpy as np
import yaml
import h5py
from . import kapre
from . import tcn
from . import models


class LossHistory(keras.callbacks.Callback):
    """[summary]

    Args:
        keras ([type]): [description]
    """

    def __init__(self):
        self.min_loss = np.inf

    def on_train_begin(self, logs):
        logging.info('training started')

    def on_epoch_begin(self, epoch, logs):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs):
        duration = time.time() - self.t0
        current_loss = logs.get('val_loss')
        current_train_loss = logs.get('loss')
        self.min_loss = np.min((current_loss, self.min_loss))
        logging.info('ep{0:>5d} ({1:1.4f} seconds): train_loss={4:1.5f} val_loss= {2:1.5f}/{3:1.5f} (current/min)'.format(epoch, duration, current_loss, self.min_loss, current_train_loss))

    def on_training_end(self, logs):
        logging.info('trained ended')


def save_model(model, file_trunk, weights_ext='_weights.h5', architecture_ext='_arch.yaml'):
    """Save model weights and architecture to separate files.

    Args:
        model ([type]): [description]
        file_trunk ([type]): [description]
        weights_ext (str, optional): [description]. Defaults to '_weights.h5'.
        architecture_ext (str, optional): [description]. Defaults to '_arch.yaml'.
    """
    save_model_architecture(model, file_trunk, architecture_ext)
    model.save_weights(file_trunk + weights_ext)


def save_model_architecture(model, file_trunk, architecture_ext='_arch.yaml'):
    """Save model architecture as yaml to separate files.

    Args:
        model ([type]): [description]
        file_trunk ([type]): [description]
        architecture_ext (str, optional): [description]. Defaults to '_arch.yaml'.
    """
    with open(file_trunk + architecture_ext, 'w') as f:
        f.write(model.to_yaml())


def load_model(file_trunk, model_dict, weights_ext='_weights.h5', from_epoch=False,
               params_ext='_params.yaml', compile=True):
    """Load model.

    First tries to load the full model directly using keras.models.load_model - this will likely fail for models with custom layers.
    Second, try to init model from parameters and then add weights...

    Args:
        file_trunk ([type]): [description]
        model_dict ([type]): [description]
        weights_ext (str, optional): [description]. Defaults to '_weights.h5'.
        from_epoch ([type], optional): [description]. Defaults to None.
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
        compile (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    if from_epoch:
        file_trunk_params = file_trunk[:-4]  # remove epoch number from params file name
        weights_ext = file_trunk[-4:] + '_weights.h5'  # add epoch number to weight file to load epoch specific weights
        model_ext = '_weights.h5'
    else:
        file_trunk_params = file_trunk  # remove epoch number from params file name
        weights_ext = '_model.h5'  # add epoch number to weight file to load epoch specific weights
        model_ext = '_model.h5'

    try:
        model = keras.models.load_model(file_trunk + model_ext,
                                        custom_objects={'Spectrogram': kapre.time_frequency.Spectrogram,
                                                        'TCN': tcn.tcn_new.TCN})
    except (SystemError, ValueError):
        logging.debug('Failed to load model using keras, likely because it contains custom layers. Will try to init model architecture from code and load weights into it.', exc_info=False)
        logging.debug('', exc_info=True)
        model = load_model_from_params(file_trunk_params, model_dict, weights_ext, compile=compile)
    return model


def load_model_from_params(file_trunk, model_dict, weights_ext='_weights.h5', params_ext='_params.yaml', compile=True):
    """Load model weights and architecture from separate files.

    Args:
        file_trunk ([type]): [description]
        models_dict ([type]): [description]
        weights_ext (str, optional): [description]. Defaults to '_weights.h5'.
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
        compile (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    params = load_params(file_trunk, params_ext)
    model = model_dict[params['model_name']](**params)  # get the model - calls the function that generates a model with parameters
    model.load_weights(file_trunk + weights_ext)
    if compile:
        # Compile with random standard optimizer and loss so we can use the model for prediction
        # Just re-compile the model if you want a particular optimizer and loss.
        model.compile(optimizer=keras.optimizers.Adam(amsgrad=True),
                      loss="mean_squared_error")
    return model


def save_params(params, file_trunk, params_ext='_params.yaml'):
    """Save model/training parameters to yaml.

    Args:
        params ([type]): [description]
        file_trunk ([type]): [description]
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.
    """
    with open(file_trunk + params_ext, 'w') as f:
        yaml.dump(params, f)


def load_params(file_trunk, params_ext='_params.yaml'):
    """Load model/training parameters from yaml

    Args:
        file_trunk ([type]): [description]
        params_ext (str, optional): [description]. Defaults to '_params.yaml'.

        Returns:
        [type]: [description]
    """
    with open(file_trunk + params_ext, 'r') as f:
        try:
            params = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            params = yaml.load(f)
    return params


def load_model_and_params(model_save_name):
    # load parameters and model
    params = load_params(model_save_name)
    model = load_model(model_save_name, models.model_dict, from_epoch=False)
    return model, params


def load_from(filename, datasets):
    """Load datasets from h5 file.

    Args:
        filename ([type]): [description]
        datasets ([type]): [description]

        Returns:
        [type]: [description]
    """
    data = dict()
    with h5py.File(filename, 'r') as f:
        data = {dataset: f[dataset][:] for dataset in datasets}
    return data


class Timer:

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = None
        self.end = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        if self.verbose:
            print(self)

    def __str__(self):
        if self.start is None:
            s = "Timer not started yet."
        elif self.end is None:
            s = "Timer still running."
        elif self.elapsed is not None:
            s = f'Time elapsed {self.elapsed:1.2f} seconds.'
        else:
            s = "Timer in unexpected state."
        return s
