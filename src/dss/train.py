"""Code for training and evaluating networks."""
import time
import logging
import deepdish as dd
import numpy as np
# import dss.utils as ut
import sklearn.metrics
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
# import keras
import defopt
import pandas as pd
import os
from glob import glob

from . import data, models, utils, predict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# HACK https://github.com/uchicago-cs/deepdish/issues/36
dd.io.hdf5io._pandas = False


# this function should not be here...
def predict_evaluate(model: keras.models.Model, test_gen, return_x: bool = True, verbose: int = 2):
    """[summary]

    Args:
        model ([type]): [description]
        test_gen ([type]): [description]
        no_x (bool): Whether unrolling should also return x. May results in very large matrices, Defaults to True.
        verbose (int): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    try:
        y_pred = model.predict_generator(test_gen.for_prediction(), verbose=verbose, workers=4, use_multiprocessing=True)
    except ValueError:
        logging.warning('Predict generator requires (x,y) or (x,y,weights) pairs - retrying with the full generator.')
        y_pred = model.predict_generator(test_gen, verbose=verbose, workers=4, use_multiprocessing=True)
    x_test, y_test = test_gen.unroll(return_x, merge_batches=True)  # make this AudioSequence.method

    # reshape from [batches, nb_hist, ...] to [time, ...]
    if return_x:
        x_test = data.unpack_batches(x_test, test_gen.data_padding)
    y_test = data.unpack_batches(y_test, test_gen.data_padding)
    y_pred = data.unpack_batches(y_pred, test_gen.data_padding)

    # trim to ensure equal length
    min_len = min(len(y_test), len(y_pred))
    if return_x:
        x_test = x_test[:min_len]
    y_test = y_test[:min_len]
    y_pred = y_pred[:min_len]

    return x_test, y_test, y_pred


# this function should not be here...
def evaluate(labels_test, labels_pred, class_names):
    """[summary]

    Args:
        labels_test ([type]): [description]
        labels_pred ([type]): [description]
        class_names ([type]): [description]

    Returns:
        [type]: [description]
    """
    nb_classes = len(np.unique(labels_test))
    conf_mat = pd.DataFrame(data=sklearn.metrics.confusion_matrix(labels_test, labels_pred),
                            columns=['true ' + p for p in class_names[:nb_classes]],
                            index=['pred ' + p for p in class_names[:nb_classes]])

    report = sklearn.metrics.classification_report(labels_test, labels_pred, target_names=class_names[:nb_classes])
    return conf_mat, report


def train(*, model_name: str = 'tcn_seq', nb_filters: int = 16, kernel_size: int = 3,
          nb_conv: int = 3, dwnsmp: int = 1, nb_hist: int = 1024, batch_norm: bool = True,
          data_dir: str = '../dat.song', save_dir: str = 'res.pulse', mode: int = 1,
          loss: str = 'categorical_crossentropy', nb_stacks: int = 2, with_y_hist: bool = True,
          keep_intermediates: bool = False, fraction_data: float = 1.0, ignore_boundaries: bool = False):
    """[summary]

    Args:
        model_name (str): [description]. Defaults to 'tcn_seq'.
        nb_filters (int): [description]. Defaults to 16.
        kernel_size (int): [description]. Defaults to 3.
        nb_conv (int): [description]. Defaults to 3.
        dwnsmp (int): [description]. Defaults to 1.
        nb_hist (int): [description]. Defaults to 1024.
        batch_norm (bool): [description]. Defaults to True.
        data_dir (str): [description]. Defaults to '../dat.song'.
        save_dir (str): [description]. Defaults to 'res.pulse'.
        mode (int): [description]. Defaults to 1.
        loss (str): [description]. Defaults to 'categorical_crossentropy'.
        nb_stacks (int): [description]. Defaults to 2.
        with_y_hist (bool): [description]. Defaults to True.
        keep_intermediates (bool): [description]. Defaults to False.
        fraction_data (float): [description]. Defaults to 1.0.
        ignore_boundaries (bool): [description]. Defaults to False.
    """

    # THIS IS NOT GREAT:
    batch_size = 32
    sample_weight_mode = None
    if with_y_hist:  # regression
        cut_trailing_dim = True
        return_sequences = True
        stride = nb_hist
        y_offset = 0
        sample_weight_mode = 'temporal'
        if ignore_boundaries:
            stride = stride - kernel_size
            data_padding = int(np.ceil(kernel_size / 2))
    else:  # classification
        cut_trailing_dim = True
        return_sequences = False
        stride = 10
        y_offset = int(round(nb_hist / 2))

    params = locals()
    logging.info('loading data')
    x_train, y_train, x_val, y_val, x_test, y_test, *_ = data.load_data(data_dir)
    print(y_train.shape)
    # HACK TO MAKE THIS WORK WITH pre-processed (multi-frequency), single-channel data
    compute_power = False
    if 'tcn' in model_name and 'coefs' in data_dir:
        print('tcn coefs')
        compute_power = True

    if fraction_data < 1.0:  # train on a subset
        logging.info('Using {} of data for validation and validation.'.format(fraction_data))
        # min_nb_samples makes sure the generator contains at least one full batch
        x_train, y_train = data.sub_range((x_train, y_train), fraction_data, min_nb_samples=nb_hist * (batch_size + 2), seed=1)
        x_val, y_val = data.sub_range((x_val, y_val), fraction_data, min_nb_samples=nb_hist * (batch_size + 2), seed=1)

    class_names = ("nix", "pulse", "sine")
    if mode == 1:  # [NIX, PULSE]
        class_names = ("nix", "pulse")
    elif mode == 2:  # [NIX, SINE]
        class_names = ("nix", "sine")
    elif mode == 3:  # [NIX, SINE+PULSE]
        class_names = ("nix", "sine+pulse")

    params.update({'nb_freq': x_train.shape[1], 'nb_channels': x_train.shape[-1],
                   'nb_classes': len(class_names), 'class_names': class_names,
                   'compute_power': compute_power})
    print(params)

    logging.info('preparing data')
    data_gen = data.AudioSequence(x_train, y_train, shuffle=True, **params)
    val_gen = data.AudioSequence(x_val, y_val, shuffle=False, **params)
    params['nb_classes'] = data_gen.nb_classes
    params['classs_names'] = class_names[:params['nb_classes']]
    print(data_gen)
    print(val_gen)
    logging.info('building network')
    print(params)
    model = models.model_dict[model_name](**params)
    # save_name = 'res.all/20190801_070823'
    # model = utils.load_model(save_name, models.model_dict, from_epoch=False)
    logging.info(model.summary())
    dd.io._pandas = False
    save_name = '{0}/{1}'.format(save_dir, time.strftime('%Y%m%d_%H%M%S'))
    utils.save_params(params, save_name)
    utils.save_model_architecture(model, file_trunk=save_name, architecture_ext='_arch.yaml')

    if keep_intermediates:
        checkpoint_save_name = save_name + "_{epoch:03d}_weights.h5"
    else:
        checkpoint_save_name = save_name + "_model.h5"  # this will overwrite intermediates from previous epochs

    # TRAIN NETWORK
    logging.info('start training')
    # parallel_model = models.ModelMGPU(model, gpus=2)
    # fit_hist = parallel_model.fit_generator(
    fit_hist = model.fit_generator(
        data_gen,
        epochs=400,
        steps_per_epoch=min(len(data_gen) * 100, 1000),
        verbose=2,
        validation_data=val_gen,
        callbacks=[ModelCheckpoint(checkpoint_save_name, save_best_only=True, save_weights_only=False, monitor='val_loss', verbose=1),
                   EarlyStopping(monitor='val_loss', patience=20),
                   ],
    )

    # TEST
    logging.info('re-loading last best model')
    if keep_intermediates:  # load last checkpoint and save
        best_weight_file = sorted(glob("{0}_*_weights.h5".format(save_name)))[-1]
        model.load_weights(best_weight_file)
        model.save("{0}_model.h5".format(save_name))
    else:  # otherwise just load the best model
        model.load_weights(checkpoint_save_name)

    logging.info('predicting')
    test_gen = data.AudioSequence(x_test, y_test, shuffle=False, **params)
    x_test, y_test, y_pred = predict_evaluate(model, test_gen)
    labels_test = predict.labels_from_probabilities(y_test)
    labels_pred = predict.labels_from_probabilities(y_pred)
    logging.info('evaluating')
    conf_mat, report = evaluate(labels_test, labels_pred, params['class_names'])
    logging.info(conf_mat)
    logging.info(report)

    save_filename = "{0}_results.h5".format(save_name)
    logging.info('saving to ' + save_filename + '.')
    d = {'fit_hist': fit_hist.history,
         'confusion_matrix': conf_mat,
         'classification_report': report,
         'x_test': x_test,
         'y_test': y_test,
         'y_pred': y_pred,
         'labels_test': labels_test,
         'labels_pred': labels_pred,
         }

    dd.io.save(save_filename, d)


if __name__ == '__main__':
    baselogger = logging.getLogger()
    baselogger.setLevel(logging.INFO)
    defopt.run(train)
