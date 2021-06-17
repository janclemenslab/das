
"""Code for training networks."""
import time
import logging
import flammkuchen as fl
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import tensorflow.keras as keras
import tensorflow.keras.layers as kl

import defopt
from glob import glob

from . import data, models, utils, predict, io, evaluate



def train(load_name, *, data_dir: str = '../dat.song', save_dir: str = './',
          verbose: int = 2, nb_epoch: int = 400, fraction_data=None, seed: int = None,
          freeze: bool = False, reshape_output: bool = False, learning_rate: float = 0.0001,
          reduce_lr: bool = False):
    """Transfer learning - load existing network and train with new data

    Args:
        load_name (str): old model to load.
        model_name (str): [description]. Defaults to 'tcn_seq'.
        data_dir (str): [description]. Defaults to '../dat.song'.
        save_dir (str): [description]. Defaults to current directory ('./').
        verbose (int): Verbosity of training output (0 - no output(?), 1 - progress bar, 2 - one line per epoch). Defaults to 2.
        nb_epoch (int): Defaults to 400.
        fraction_data (float): [description]. Defaults to 1.0.
        seed (int): Random seed for selecting subsets of the data. Defaults to None (no seed).
        freeze (bool): freeze TCN layers of the pre-trained network
        reshape_output (bool): reshape output layers of the pre-trained network to match new data
        learning_rate (float): lr
        reduce_lr (bool): reduce learning rate
    """

    params_given = locals()

    logging.info(f'loading old params and network from {load_name}.')
    model, params = utils.load_model_and_params(load_name)
    params.update(params_given)  # override loaded params with given params
    params['data_dir'] = data_dir
    logging.info(f"loading data from {params['data_dir']}")

    d = io.load(params['data_dir'], x_suffix=params['x_suffix'], y_suffix=params['y_suffix'])
    params.update(d.attrs)  # add metadata from data.attrs to params for saving

    if fraction_data is not None:  # train on a subset
        if fraction_data > 1.0:  # seconds
            logging.info(f"{fraction_data} seconds corresponds to {fraction_data / d.attrs['samplerate_x_Hz']} of the training data.")
            fraction_data = fraction_data / d.attrs['samplerate_x_Hz']
        logging.info(f"Using {fraction_data} of data for training and validation.")
        min_nb_samples = params['nb_hist'] * (params['batch_size'] + 2)  # ensure the generator contains at least one full batch
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
    # logging.info('Parameters:')
    # logging.info(params)


    logging.info('preparing data')
    data_gen = data.AudioSequence(d['train']['x'], d['train']['y'], shuffle=True,
                                  first_sample=first_sample_train, last_sample=last_sample_train, nb_repeats=100,
                                  **params)
    val_gen = data.AudioSequence(d['val']['x'], d['val']['y'], shuffle=False,
                                 first_sample=first_sample_val, last_sample=last_sample_val,
                                 **params)
    logging.info('Training data:')
    logging.info(data_gen)
    logging.info('Validation data:')
    logging.info(val_gen)


    nb_classes = d['train']['y'].shape[1]
    if freeze or nb_classes != model.output_shape[-1]:

        sample_weight_mode = params['sample_weight_mode']
        nb_pre_conv = params['nb_pre_conv']
        upsample = True
        loss = 'categorical_crossentropy'

        new_model = keras.Model(model.inputs, model.layers[-4].output)
        # freeze layers
        if freeze:
            for layer in new_model.layers:
                if 'conv1d' not in layer.name:
                    layer.trainable = False

        x = new_model.output
        x = kl.Dense(nb_classes, name='dense_new')(x)
        x = kl.Activation('softmax', name='activation_new')(x)
        if nb_pre_conv > 0 and upsample:
            x = kl.UpSampling1D(size=2**nb_pre_conv, name='upsampling_new')(x)
        output_layer = x
        model = keras.models.Model(new_model.inputs, output_layer, name='TCN_new')
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, amsgrad=True, clipnorm=1.),
                      loss=loss, sample_weight_mode=sample_weight_mode)


    logging.info(model.summary())
    save_name = '{0}/{1}'.format(save_dir, time.strftime('%Y%m%d_%H%M%S'))
    utils.save_params(params, save_name)
    checkpoint_save_name = save_name + "_model.h5"  # this will overwrite intermediates from previous epochs

    callbacks = [ModelCheckpoint(checkpoint_save_name, save_best_only=True, save_weights_only=False, monitor='val_loss', verbose=1),
                 EarlyStopping(monitor='val_loss', patience=20),]
    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(patience=5, verbose=1))

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
    logging.info('re-loading last best model')
    # model.load_weights(checkpoint_save_name)
    model = utils.load_model(save_name, models.model_dict, from_epoch=False)

    logging.info('predicting')
    # x_test, y_test, y_pred = predict.predict_with_y(x=d['test']['x'], y=d['test']['y'], model=model, params=params)
    # evaluate.evaluate_probabilities(x, y, model, params, verbose=None)
    x_test, y_test, y_pred = evaluate.evaluate_probabilities(x=d['test']['x'], y=d['test']['y'],
                                                             model=model, params=params)

    labels_test = predict.labels_from_probabilities(y_test)
    labels_pred = predict.labels_from_probabilities(y_pred)

    logging.info('evaluating')
    conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred, np.array(params['class_names'])[np.unique(labels_test)])
    # conf_mat, report = evaluate.evaluate_segments(labels_test, labels_pred, params['class_names'])
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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(train)
