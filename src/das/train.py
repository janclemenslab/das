"""Code for training networks."""
import time
import logging
import flammkuchen as fl
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import defopt
import os
from glob import glob
from typing import List, Optional
from . import data, models, utils, predict, io, evaluate  #, timeseries

try:
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()
except Exception as e:
    logging.exception(e)


def train(*, data_dir: str, y_suffix: str = '',
          save_dir: str = './', save_prefix: Optional[str] = None,
          model_name: str = 'tcn', nb_filters: int = 16, kernel_size: int = 16,
          nb_conv: int = 3, use_separable: List[bool] = False, nb_hist: int = 1024,
          ignore_boundaries: bool = True, batch_norm: bool = True,
          nb_pre_conv: int = 0, pre_nb_dft: int = 64,
          pre_kernel_size: int = 3, pre_nb_filters: int = 16, pre_nb_conv: int = 2,
          verbose: int = 2, batch_size: int = 32,
          nb_epoch: int = 400,
          learning_rate: Optional[float] = None, reduce_lr: bool = False, reduce_lr_patience: int = 5,
          fraction_data: Optional[float] = None, seed: Optional[int] = None, batch_level_subsampling: bool = False,
          tensorboard: bool = False, log_messages: bool = False,
          nb_stacks: int = 2, with_y_hist: bool = True, x_suffix: str = '',
          _qt_progress: bool = False):
    """Train a DeepSS network.

    Args:
        data_dir (str): Path to the directory or file with the dataset for training.
                        Accepts npy-dirs (recommended), h5 files or zarr files.
                        See documentation for how the dataset should be organized.
        y_suffix (str): Select training target by suffix.
                        Song-type specific targets can be created with a training dataset,
                        Defaults to '' (will use the standard target 'y')
        save_dir (str): Directory to save training outputs.
                        The path of output files will constructed from the SAVE_DIR, an optional prefix, and the time stamp of the start of training.
                        Defaults to current directory ('./').
        save_prefix (Optional[str]): Prepend to timestamp.
                           Name of files created will be SAVE_DIR/SAVE_PREFIX + "_" + TIMESTAMP
                           or SAVE_DIR/ TIMESTAMP if SAVE_PREFIX is empty.
                           Defaults to '' (empty).
        model_name (str): Network architecture to use.
                          Use "tcn" (TCN) or "tcn_stft" (TCN with STFT frontend).
                          See das.models for a description of all models.
                          Defaults to 'tcn'.
        nb_filters (int): Number of filters per layer.
                          Defaults to 16.
        kernel_size (int): Duration of the filters (=kernels) in samples.
                           Defaults to 16.
        nb_conv (int): Number of TCN blocks in the network.
                       Defaults to 3.
        use_separable (List[bool]): Specify which TCN blocks should use separable convolutions.
                                    Provide as a space-separated sequence of "False" or "True.
                                    For instance: "True False False" will set the first block in a
                                    three-block (as given by nb_conv) network to use separable convolutions.
                                    Defaults to False (no block uses separable convolution).
        nb_hist (int): Number of samples processed at once by the network (a.k.a chunk size).
                       Defaults to 1024.
        ignore_boundaries (bool): Minimize edge effects by discarding predictions at the edges of chunks.
                                  Defaults to True.
        batch_norm (bool): Batch normalize.
                           Defaults to True.
        nb_pre_conv (int): Downsampling rate. Adds downsampling frontend if not 0.
                           TCN_TCN: adds a frontend of N conv blocks (conv-relu-batchnorm-maxpool2) to the TCN.
                           TCN_STFT: adds a trainable STFT frontend.
                           Defaults to 0 (no frontend).
        pre_nb_dft (int): Number of filters (roughly corresponding to filters) in the STFT frontend.
                          Defaults to 64.
        pre_nb_filters (int): Number of filters per layer in the pre-processing TCN.
                              Defaults to 16.
        pre_kernel_size (int): Duration of filters (=kernels) in samples in the pre-processing TCN.
                               Defaults to 3.
        verbose (int): Verbosity of training output (0 - no output(?), 1 - progress bar, 2 - one line per epoch).
                       Defaults to 2.
        batch_size (int): Batch size
                          Defaults to 32.
        nb_epoch (int): Maximal number of training epochs.
                        Training will stop early if validation loss did not decrease in the last 20 epochs.
                        Defaults to 400.
        learning_rate (Optional[float]): Learning rate of the model. Defaults should work in most cases.
                               Values typically range between 0.1 and 0.00001.
                               If None, uses per model defaults: "tcn" 0.0001, "tcn_stft" 0.0005).
                               Defaults to None.
        reduce_lr (bool): Reduce learning rate on plateau.
                          Defaults to False.
        reduce_lr_patience (int): Number of epochs w/o a reduction in validation loss after which to trigger a reduction in learning rate.
                                  Defaults to 5.
        fraction_data (Optional[float]): Fraction of training and validation to use for training.
                               Defaults to 1.0.
        seed (Optional[int]): Random seed to reproducible select fractions of the data.
                    Defaults to None (no seed).
        batch_level_subsampling (bool): Select fraction of data for training from random subset of shuffled batches.
                                        If False, select a continuous chunk of the recording.
                                        Defaults to False.
        tensorboard (bool): Write tensorboard logs to save_dir.
                            Defaults to False.
        log_messages (bool): Sets logging level to INFO.
                             Defaults to False (will follow existing settings).
        nb_stacks (int): Unused if model name is "tcn" or "tcn_stft". Defaults to 2.
        with_y_hist (bool): Unused if model name is "tcn" or "tcn_stft". Defaults to True.
        x_suffix (str): Select specific training data based on suffix (e.g. x_suffix).
                        Defaults to '' (will use the standard data 'x')
        """
        # _qt_progress: tuple of (multiprocessing.Queue, threading.Event)
        #        The queue is used to transmit progress updates to the GUI,
        #        the event is set in the GUI to stop training.
    if log_messages:
        logging.basicConfig(level=logging.INFO)

    # FIXME THIS IS NOT GREAT:
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

    if len(save_prefix):
        save_prefix = save_prefix + '_'
    params = locals()
    del params['_qt_progress']

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
    save_name = '{0}/{1}{2}'.format(save_dir, save_prefix, time.strftime('%Y%m%d_%H%M%S'))
    logging.info(f'Will save to {save_name}.')

    utils.save_params(params, save_name)
    checkpoint_save_name = save_name + "_model.h5"  # this will overwrite intermediates from previous epochs

    callbacks = [ModelCheckpoint(checkpoint_save_name, save_best_only=True, save_weights_only=False, monitor='val_loss', verbose=1),
                 EarlyStopping(monitor='val_loss', patience=20),]

    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(patience=reduce_lr_patience, verbose=1))

    if _qt_progress:
        callbacks.append(utils.QtProgressCallback(nb_epoch, _qt_progress))

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
