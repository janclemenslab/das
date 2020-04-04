import sklearn.metrics
import defopt
import logging
from . import predict, data, utils, models
import os
import numpy as np
import pandas as pd
from .event_utils import evaluate_eventtimes


# to segment_utils
def evaluate_segments(labels_test, labels_pred, class_names, confmat_as_pandas: bool = False,
                      report_as_dict: bool = False, labels=None):
    """


    Args:
        labels_test (List): [nb_samples,]
        labels_pred (List): [nb_samples,]
        class_names ([type]): [description]
        confmat_as_pandas (bool, optional): [description]. Defaults to False.
        report_as_dict (bool, optional): [description]. Defaults to False.
        labels ([type], optional): [description]. Defaults to None.

    Returns:
        conf_mat
        report
    """
    conf_mat = sklearn.metrics.confusion_matrix(labels_test, labels_pred)
    if confmat_as_pandas:
        conf_mat = pd.DataFrame(data=conf_mat,
                                columns=['true ' + p for p in class_names],
                                index=['pred ' + p for p in class_names])
    if labels is None:
        labels = np.arange(len(class_names))
    report = sklearn.metrics.classification_report(labels_test, labels_pred, labels=labels,
                                                   target_names=class_names, output_dict=report_as_dict, digits=3)
    return conf_mat, report


def evaluate_segment_timing(segment_labels_true, segment_labels_pred, samplerate, event_tol):
    """[summary]

    Args:
        segment_labels_true ([type]): [description]
        segment_labels_pred ([type]): [description]
        samplerate ([type]): Hz [description]
        event_tol ([type]): seconds [description]

    Returns:
        [type]: [description]
    """
    segment_onsets_true, segment_offsets_true = segment_timing(segment_labels_true, samplerate)
    segment_onsets_pred, segment_offsets_pred = segment_timing(segment_labels_pred, samplerate)

    # ensure evaluate_eventtimes returns nearest_predicted_onsets (nearest true event for each predicted event),
    # if not, rename var
    segment_onsets_report, _, _, nearest_predicted_onsets = evaluate_eventtimes(segment_onsets_true, segment_onsets_pred, samplerate, event_tol)
    segment_offsets_report, _, _, nearest_predicted_offsets = evaluate_eventtimes(segment_offsets_true, segment_offsets_pred, samplerate, event_tol)
    return segment_onsets_report, segment_offsets_report, nearest_predicted_onsets, nearest_predicted_offsets


def segment_timing(labels, samplerate):
    """Get onset and offset time (in seconds) for each segment."""
    segment_onset_times = np.where(np.diff(labels) == 1)[0].astype(np.float) / samplerate  # explicit cast required?
    segment_offset_times = np.where(np.diff(labels) == -1)[0].astype(np.float) / samplerate
    return segment_onset_times, segment_offset_times


# DEPRECATED
def evaluate_probabilities(x, y, model, params, verbose=None):
    y_pred = predict.predict_probabililties(x, model, params, verbose)

    eval_gen = data.AudioSequence(x, y, shuffle=False, **params)
    x, y = data.get_data_from_gen(eval_gen)

    return x, y, y_pred


def evaluate(x, y, model_savename):
    # model, params = utils.load_model_and_params(model_save_name)
    # x, y, y_pred = predict(x, y, model, params)
    segments, events, probabilities = predict.predict(x, model_savename)

    eval_gen = data.AudioSequence(x, y, shuffle=False, **params)
    x, y = data.get_data_from_gen(eval_gen)
    # eval_segments(segments)
    # eval_events(events)


# move to cli module
def run(data_name: str, model_save_name: str, *,
        data_key: str = 'samples', segment_labels_key: str = 'probabilities', event_times_key: str = 'eventtimes',
        save_name: str = None, event_class_name: str = 'event', event_thres: float = 0.75, event_tol: float = 100):
    """[summary]

    Args:
        data_name (str): [description]
        model_save_name (str): [description]
        data_key (str): [description]. Defaults to 'samples'.
        segment_labels_key (str): [description]. Defaults to 'probabilities'.
        event_times_key (str): [description]. Defaults to 'eventtimes'.
        save_name (str): [description]. Defaults to None.
        event_class_name (str): [description]. Defaults to 'event'.
        event_thres (float): [description]. Defaults to 0.75.
        event_tol (float): [description]. Defaults to 100.

    Raises:
        ValueError: [description]
    """
    # load model
    params = utils.load_params(model_save_name)
    model = utils.load_model(model_save_name, models.model_dict, from_epoch=False)

    # load data
    x, samplerate = predict._load_data(data_name, data_key)
    if samplerate is None:
        logging.warning(f"unknown samplerate - assuming data has same samplerate as the training data the model was trained with ({params['samplerate_x_Hz']}Hz).")
        samplerate = params['samplerate_x_Hz']

    try:
        segment_probabilities_true = predict._load_data(data_name, segment_labels_key)
        # segment_labels_true = predict.labels_from_probabilities(segment_probabilities_true)

        # get only the relevant event type
        test_gen = data.AudioSequence(x, segment_probabilities_true, shuffle=False, **params)
        xx, yy = test_gen.unroll()
        segment_probabilities_true = data.unpack_batches(yy, test_gen.data_padding)
        segment_labels_true = predict.labels_from_probabilities(segment_probabilities_true)
        x_test = data.unpack_batches(xx, test_gen.data_padding)
    except KeyError:
        segment_labels_true = None
        logging.info(f'Could not load labels from key "{segment_labels_key}" from file "{data_name}".')

    try:
        event_times_true = predict._load_data(data_name, event_times_key)
    except KeyError:
        event_times_true = None
        logging.info(f'Could not load labels from key "{event_times_key}" from file "{data_name}".')

    try:
        event_index = params['class_names'].index(event_class_name)
    except ValueError:
        logging.info(f'model does not predict events of the type "{event_class_name}".')
        event_index = None

    segment_probabilities, segment_labels_pred, event_indices_pred, event_confidence = predict.predict(data, model, params, event_index)

    # evaluate
    if event_times_true is not None and event_times_pred is not None:
        event_times_pred = event_indices_pred / samplerate
        event_times_report, _, _, nearest_predicted_events = evaluate_eventtimes(event_times_true, event_times_pred, event_tol)
    else:
        event_times_report, nearest_predicted_events = None, None

    if segment_labels_true is not None:
        min_len = min(segment_labels_true.shape[0], segment_labels_pred.shape[0])
        segment_labels_true = segment_labels_true[:min_len]
        segment_labels_pred = segment_labels_pred[:min_len]

        segment_confmat, segment_report = evaluate_segments(segment_labels_true, segment_labels_pred, params['class_names'])

        segment_onsets_true, segment_offsets_true = segment_timing(segment_labels_true, samplerate)
        segment_onsets_pred, segment_offsets_pred = segment_timing(segment_labels_pred, samplerate)

        # ensure evaluate_eventtimes returns nearest_predicted_onsets (nearest true event for each predicted event),
        # if not, rename var
        segment_onsets_report, _, _, nearest_predicted_onsets = evaluate_eventtimes(segment_onsets_true, segment_onsets_pred, event_tol)
        segment_offsets_report, _, _, nearest_predicted_offsets = evaluate_eventtimes(segment_offsets_true, segment_offsets_pred, event_tol)
    else:
        segment_confmat, segment_report = None, None
        segment_onsets_true, segment_offsets_true, segment_onsets_pred, segment_offsets_pred = None, None, None, None
        segment_onsets_report, segment_offsets_report = None, None
        nearest_predicted_onsets, nearest_predicted_offsets = None, None

    # save
    if save_name is None:
        save_name = os.path.splitext(data_name)[0] + '_eval.h5'

    logging.info(f'   saving results to "{save_name}".')
    d = {'x_test': x_test,
         'y_test': segment_probabilities_true,
         'y_pred': segment_probabilities,
         'labels_true': segment_labels_true,
         'labels_pred': segment_labels_pred,
         # events
         'event_times_pred': event_times_pred,
         'event_times_true': event_times_true,
         'event_confidence': event_confidence,
         'event_times_report': event_times_report,
         'nearest_predicted_events': nearest_predicted_events,
         # segment overlap
         'segment_confmat': segment_confmat,
         'segment_report': segment_report,
         # segment timing
         'segment_onsets_true': segment_onsets_true,
         'segment_offsets_true': segment_offsets_true,
         'segment_onsets_pred': segment_onsets_pred,
         'segment_offsets_pred': segment_offsets_pred,
         'segment_onsets_report': segment_onsets_report,
         'segment_offsets_report': segment_offsets_report,
         'nearest_predicted_onsets': nearest_predicted_onsets,
         'nearest_predicted_offsets': nearest_predicted_offsets,
         }
    fl.save(save_name, d)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    defopt.run(run)
