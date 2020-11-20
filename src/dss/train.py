"""Code for training networks."""
import time
import logging
import flammkuchen as fl
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import defopt
import os
from glob import glob
from typing import List
from . import data, models, utils, predict, io, evaluate  #, timeseries

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def train(*, data_dir: str, model_name: str = 'tcn', nb_filters: int = 16, kernel_size: int = 3,
          nb_conv: int = 3, nb_hist: int = 1024, batch_norm: bool = True,
          save_dir: str = './', verbose: int = 2,
          nb_stacks: int = 2, with_y_hist: bool = True, nb_epoch: int = 400,
          fraction_data: float = None, seed: int = None, ignore_boundaries: bool = False,
          x_suffix: str = '', y_suffix: str = '', nb_pre_conv: int = 0,
          learning_rate: float = None, reduce_lr: bool = False, reduce_lr_patience: int = 5, batch_level_subsampling: bool = False,
          tensorboard: bool = False, use_separable: List[bool] = False,
          pre_kernel_size: int = 3, pre_nb_filters: int = 16, pre_nb_conv: int = 2,
          save_prefix: str = None, log_messages: bool = False):
    """[summary]

    Args:
        model_name (str): [description]. Defaults to 'tcn_seq'.
        nb_filters (int): [description]. Defaults to 16.
        kernel_size (int): [description]. Defaults to 3.
        nb_conv (int): [description]. Defaults to 3.
        nb_hist (int): [description]. Defaults to 1024.
        batch_norm (bool): [description]. Defaults to True.
        data_dir (str): [description]. Defaults to '../dat.song'.
        save_dir (str): [description]. Defaults to current directory ('./').
        verbose (int): Verbosity of training output (0 - no output(?), 1 - progress bar, 2 - one line per epoch). Defaults to 2.
        nb_stacks (int): [description]. Defaults to 2.
        with_y_hist (bool): [description]. Defaults to True.
        nb_epoch (int): Defaults to 400.
        fraction_data (float): [description]. Defaults to 1.0.
        seed (int): Seed for selecting random subsets of the data. Defaults to None (no seed).
        ignore_boundaries (bool): [description]. Defaults to False.
        x_suffix (str): ... Defaults to '' (will use 'x')
        y_suffix (str): ... Defaults to '' (will use 'y')
        nb_pre_conv (int): adds a frontend of N conv blocks (conv-relu-batchnorm-maxpool2) to the TCN - useful for reducing the sampling rate for USV. Defaults to 0 (no frontend).
        learning_rate (float): learning rate of the model. Defaults to None (values set in the model def)
        reduce_lr (bool): reduce learning rate on plateau
        batch_level_subsampling (bool): if true fraction data will select random subset of shuffled batches, otherwise will select a continuous chunk of the recording
        tensorboard (bool): whether to write tensorboard logs to save_dir Defaults to False.
        use_separable: use separable convs in TCN. Defaults to False.
        pre_nb_filters (int): [description]. Defaults to 16.
        pre_kernel_size (int): [description]. Defaults to 3.
        pre_nb_conv (int): [description]. Defaults to 3.
        save_prefix (str): prepend to save file name. Defaults to ''.
        log_messages (bool): sets logging level to INFO. Defaults to False (will follow existing settings).
        """

    if log_messages:
        logging.basicConfig(level=logging.INFO)

    # FIXME THIS IS NOT GREAT:
    batch_size = 32
    sample_weight_mode = None
    data_padding = 0
    if with_y_hist:  # regression
        return_sequences = True
        stride = nb_hist
        y_offset = 0
        sample_weight_mode = 'temporal'
        if ignore_boundaries:
            data_padding = int(np.ceil(kernel_size * nb_conv))  # this does not completely avoid boundary effects but should minimize them sufficiently
            stride = stride - 2 * data_padding
    else:  # classification
        return_sequences = False
        stride = 1  # should take every sample, since sampling rates of both x and y are now the same
        y_offset = int(round(nb_hist / 2))

    output_stride = 1  # since we upsample output to original sampling rate. w/o upsampling: `output_stride = int(2**nb_pre_conv)` since each pre-conv layer does 2x max pooling

    if save_prefix is None:
        save_prefix = ''

    params = locals()

    if stride <=0:
        raise ValueError('Stride <=0 - needs to be >0. Possible solutions: reduce kernel_size, increase nb_hist parameters, uncheck ignore_boundaries')

    # remove learning rate param if not set so the value from the model def is used
    if params['learning_rate'] is None:
        del params['learning_rate']

    if '_multi' in model_name:
        params['unpack_channels'] = True

    logging.info('loading data')
    d = io.load(data_dir, x_suffix=x_suffix, y_suffix=y_suffix)
    params.update(d.attrs)  # add metadata from data.attrs to params for saving

    if fraction_data is not None:
        if fraction_data > 1.0:  # seconds
            logging.info(f"{fraction_data} seconds corresponds to {fraction_data / (d['train']['x'].shape[0] / d.attrs['samplerate_x_Hz']):1.4f} of the training data.")
            fraction_data = np.min((fraction_data / (d['train']['x'].shape[0] / d.attrs['samplerate_x_Hz']), 1.0))
        elif fraction_data < 1.0:
            logging.info(f"Using {fraction_data:1.4f} of data for training and validation.")

    if fraction_data is not None and not batch_level_subsampling:  # train on a subset
        min_nb_samples = nb_hist * (batch_size + 2)  # ensure the generator contains at least one full batch
        first_sample_train, last_sample_train = data.sub_range(d['train']['x'].shape[0], fraction_data, min_nb_samples, seed=seed)
        first_sample_val, last_sample_val = data.sub_range(d['val']['x'].shape[0], fraction_data, min_nb_samples, seed=seed)
    else:
        first_sample_train, last_sample_train = 0, None
        first_sample_val, last_sample_val = 0, None

    # TODO clarify nb_channels, nb_freq semantics - always [nb_samples,..., nb_channels] -  nb_freq is ill-defined for 2D data
    params.update({'nb_freq': d['train']['x'].shape[1], 'nb_channels': d['train']['x'].shape[-1], 'nb_classes': len(params['class_names']),
                   'first_sample_train': first_sample_train, 'last_sample_train': last_sample_train,
                   'first_sample_val': first_sample_val, 'last_sample_val': last_sample_val,
                   })
    logging.info('Parameters:')
    logging.info(params)

    logging.info('preparing data')
    if fraction_data is not None and batch_level_subsampling:  # train on a subset
        np.random.seed(seed)
        shuffle_subset = fraction_data
    else:
        shuffle_subset = None

    data_gen = data.AudioSequence(d['train']['x'], d['train']['y'],
                                  shuffle=True, shuffle_subset=shuffle_subset,
                                  first_sample=first_sample_train, last_sample=last_sample_train, nb_repeats=1,
                                  **params)
    val_gen = data.AudioSequence(d['val']['x'], d['val']['y'],
                                 shuffle=False, shuffle_subset=shuffle_subset,
                                 first_sample=first_sample_val, last_sample=last_sample_val,
                                 **params)
    # data_gen = timeseries.timeseries_dataset_from_array(d['train']['x'], d['train']['y'],
    #                               sequence_length=params['nb_hist'], sequence_stride=stride,
    #                               shuffle=True, batch_size=batch_size,
    #                               start_index=first_sample_train, end_index=last_sample_train)
    # val_gen = timeseries.timeseries_dataset_from_array(d['val']['x'], d['val']['y'],
    #                              sequence_length=params['nb_hist'], sequence_stride=stride,
    #                              shuffle=False, batch_size=batch_size,
    #                              start_index=first_sample_val, end_index=last_sample_val)

    logging.info('Training data:')
    logging.info(data_gen)
    logging.info('Validation data:')
    logging.info(val_gen)

    logging.info('building network')
    model = models.model_dict[model_name](**params)

    logging.info(model.summary())
    os.makedirs(os.path.abspath(save_dir), exist_ok=True)
    save_name = '{0}/{1}_{2}'.format(save_dir, save_prefix, time.strftime('%Y%m%d_%H%M%S'))
    utils.save_params(params, save_name)
    utils.save_model_architecture(model, file_trunk=save_name, architecture_ext='_arch.yaml')

    checkpoint_save_name = save_name + "_model.h5"  # this will overwrite intermediates from previous epochs

    callbacks = [ModelCheckpoint(checkpoint_save_name, save_best_only=True, save_weights_only=False, monitor='val_loss', verbose=1),
                 EarlyStopping(monitor='val_loss', patience=20),]
    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(patience=reduce_lr_patience, verbose=1))

    if tensorboard:
        callbacks.append(TensorBoard(log_dir=save_dir))

    # TRAIN NETWORK
    logging.info('start training')
    fit_hist = model.fit(
        data_gen,
        epochs=nb_epoch,
        steps_per_epoch=min(len(data_gen), 1000),
        verbose=verbose,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # TEST
    if len(d['test']['x']) < nb_hist:
        logging.info('No test data - skipping final evaluation step.')
        return
    else:
        logging.info('re-loading last best model')
        model.load_weights(checkpoint_save_name)

        logging.info('predicting')
        x_test, y_test, y_pred = evaluate.evaluate_probabilities(x=d['test']['x'], y=d['test']['y'],
                                                                model=model, params=params)

        labels_test = predict.labels_from_probabilities(y_test)
        labels_pred = predict.labels_from_probabilities(y_pred)

        logging.info('evaluating')
        conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred, params['class_names'])
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
            'params': params,
            }

        fl.save(save_filename, d)

def main():
    logging.basicConfig(level=logging.INFO)
    defopt.run(train)

if __name__ == '__main__':
    main()
