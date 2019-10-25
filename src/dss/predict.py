"""Code for training and evaluating networks."""
import time
import logging
import numpy as np
import defopt
import pandas as pd
import os
from glob import glob
import h5py
import deepdish as dd

from . import data, models, train, utils, pulse_utils, predict


def probabilities(model, test_gen, verbose=1):
    y_pred = model.predict_generator(test_gen.for_prediction(), verbose=verbose)    
    # reshape from [batches, nb_hist, ...] to [time, ...]
    y_pred = data.unpack_batches(y_pred, test_gen.data_padding)    
    return y_pred


def labels_from_probabilities(probability):
    """[summary]
    
    Args:
        probability ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    labels = np.argmax(probability, axis=1)
    return labels


def run(data_name: str, model_save_name: str):
    # load model
    params = utils.load_params(model_save_name)
    model = utils.load_model(model_save_name, models.model_dict, from_epoch=False)

    # load h5 data
    with h5py.File(data_name, 'r') as f:
        data = f['samples'][:]
        sampling_rate = 10000  # read this from the file!
    
    # merge recording channels
    song_merged = utils.merge_channels(data, sampling_rate)

    # make generator
    pred_gen = data.AudioSequence(x=data, y=None, batch_size=32, shuffle=False, **params)
    # predict
    song_probabilities = probabilities(model, pred_gen, verbose=1)
    song_labels = labels_from_probabilities(song_probabilities)

    # detect pulses
    tol = 100
    try: 
        pulse_pred_index = params['class_names'].index('pulse')
    except ValueError:
        print('THIS MODEL DOES NOT PREDICT PULSES.')
        pulse_pred_index = None

    if pulse_pred_index is not None:
        pulsetimes_pred, pulsetimes_pred_confidence = pulse_utils.detect_pulses(song_probabilities[:,pulse_pred_index], thres=0.7, min_dist=tol)
        # d, nn_pred_pulse, nn_true_pulse, nn_dist = dss.pulse_utils.eval_pulse_times(pulsetimes_true, pulsetimes_pred, tol)

        # extract pulse shapes
        win_hw = 100
        pulseshapes_pred = pulse_utils.get_pulseshapes(pulsetimes_pred + win_hw, song_merged, win_hw)
        pulsenorm_pred = np.linalg.norm(np.abs(pulseshapes_pred[50:-50,:]), axis=0)
        pulseshapes_pred = np.apply_along_axis(pulse_utils.normalize_pulse, axis=-1, arr=pulseshapes_pred.T).T
    # assemble everything into dict/xarray


def eval(save_name: str):
    logging.info('Evaluating model {}.'.format(save_name))
    params = utils.load_params(save_name)
    print(params)

    # load data
    _, _, _, _, x_test0, y_test0, song_test, pulse_times_test = data.load_data(params['data_dir'])

    try:
        datasets = utils.load_from(save_name + '_results.h5', ['x_test', 'y_test', 'y_pred'])
        x_test, y_test, y_pred = [datasets[key] for key in ['x_test', 'y_test', 'y_pred']]  # unpack dict items to vars
        logging.info('   loading test data.')
    except OSError:
        model = utils.load_model(save_name, models.model_dict, from_epoch=False)
        logging.info('   loading model and predicting test data.')
        test_gen = data.AudioSequence(x_test0, y_test0, shuffle=False, **params)
        x_test, y_test, y_pred = train.predict_evaluate(model, test_gen)

    logging.info('   evaluating')
    labels_test = predict.labels_from_probabilities(y_test)
    labels_pred = predict.labels_from_probabilities(y_pred)
    conf_mat, report = train.evaluate(labels_test, labels_pred, params['class_names'][:2])

    logging.info(conf_mat)
    logging.info(report)

    # logging.info('   evaluating pulses.')
    # try:
    #     pulse_pred_index = params['class_names'].index('pulse')
    # except ValueError:
    #     print('THIS MODEL DOES NOT PREDICT PULSES.')
    #     pulse_pred_index = None
    #     d = None

    # if pulse_pred_index is not None:
    #     tol = 100
    #     pulsetimes_true = np.unique(np.sort(pulse_times_test.copy()))
    #     pulsetimes_true = pulsetimes_true[pulsetimes_true<song_test.shape[0]]

    #     pulsetimes_pred, pulsetimes_pred_confidence = pulse_utils.detect_pulses(song_probabilities[:,pulse_pred_index], thres=0.7, min_dist=tol)
    #     d, nn_pred_pulse, nn_true_pulse, nn_dist = dss.pulse_utils.eval_pulse_times(pulsetimes_true, pulsetimes_pred, tol)

    #     # # extract pulse shapes
    #     # win_hw = 100
    #     # pulseshapes_pred = pulse_utils.get_pulseshapes(pulsetimes_pred + win_hw, song_merged, win_hw)
    #     # pulsenorm_pred = np.linalg.norm(np.abs(pulseshapes_pred[50:-50,:]), axis=0)
    #     # pulseshapes_pred = np.apply_along_axis(pulse_utils.normalize_pulse, axis=-1, arr=pulseshapes_pred.T).T

    logging.info('   saving results.')
    d = {'fit_hist': [],
         'confusion_matrix': conf_mat,
         'classification_report': report,
         # 'x_test': x_test,
         'y_test': y_test,
         'y_pred': y_pred,
         'labels_test': labels_test,
         'labels_pred': labels_pred,
         }

    dd.io.save("{0}_results.h5".format(save_name), d)


if __name__ == "__main__":
    defopt.run(run)