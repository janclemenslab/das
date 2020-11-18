"""Code for training and evaluating networks."""
import logging
import os
import defopt
import scipy
import flammkuchen
import numpy as np
from . import utils, data, models, event_utils, segment_utils
from typing import List, Union

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def predict_probabililties(x, model, params, verbose=None):
    """[summary]

    Args:
        x ([samples, ...]): [description]
        model (tf.keras.Model): [description]
        params ([type]): [description]
        verbose (int, optional): Verbose level for predict_generator (see tf.keras docs). Defaults to None.

    Returns:
        y_pred - output of network for each sample
    """
    pred_gen = data.AudioSequence(x=x, y=None, shuffle=False, **params)  # prep data
    y_pred = model.predict(pred_gen, verbose=verbose)  # run the network
    y_pred = data.unpack_batches(y_pred, pred_gen.data_padding)  # reshape from [batches, nb_hist, ...] to [time, ...]
    return y_pred


def labels_from_probabilities(probabilities, threshold=None):
    """Convert class-wise probabilities into labels.

    Args:
        probabilities ([type]): [samples, classes] or [samples, ]
        threshold (float, Optional): Argmax over all classes (Default, 2D - corresponds to 1/nb_classes or 0.5 if 1D).
                                     If float, each class probability is compared to the threshold.
                                     First class to cross threshold wins.
                                     If no class crosses threshold label will default to the first class.
    Returns:
        labels [samples,] - index of "winning" dimension for each sample
    """
    if probabilities.ndim == 1:
        if threshold is None:
            threshold = 0.5
        labels = (probabilities > threshold).astype(np.intp)
    elif probabilities.ndim == 2:
        if threshold is None:
            labels = np.argmax(probabilities, axis=1)
        else:
            thresholded_probabilities = probabilities.copy()
            thresholded_probabilities[thresholded_probabilities < threshold] = 0
            labels = np.argmax(thresholded_probabilities > threshold, axis=1)
    else:
        raise ValueError(f'Probabilities have to many dimensions ({probabilities.ndim}). Can only be 1D or 2D.')

    return labels


def predict_segments(class_probabilities, samplerate=1, segment_dims=None, segment_names=None,
                     segment_thres=0.5, segment_minlen=None, segment_fillgap=None):
    """[summary]

    Args:
        class_probabilities ([type]): [description]
        samplerate [float, optional): Hz
        segement_dims ([type], optional): [description]. Defaults to None.
        segment_names ([type], optional): [description]. Defaults to None.
        segment_thres (float, optional): [description]. Defaults to 0.5.
        segment_minlen ([type], optional): seconds [description]. Defaults to None.
        segment_fillgap ([type], optional): seconds [description]. Defaults to None.

    Returns:
        dict['segmentnames']['denselabels-samples'/'onsets'/'offsets'/'probabilities']

    """
    if segment_dims is None:
        nb_classes = class_probabilities.shape[1]
        segment_dims = range(nb_classes)

    if segment_names is None:
        segment_names = segment_dims

    # cleanup
    segments = dict()
    if len(segment_dims):
        for segment_dim, segment_name in zip(segment_dims, segment_names):
            segments[segment_name] = dict()
            segments[segment_name]['index'] = segment_dim
            prob = class_probabilities[:, segment_dim]
            segments[segment_name]['probabilities'] = prob
            labels = labels_from_probabilities(prob, segment_thres)
            if segment_fillgap is not None:
                labels = segment_utils.fill_gaps(labels, segment_fillgap * samplerate)
            if segment_minlen is not None:
                labels = segment_utils.remove_short(labels, segment_minlen * samplerate)
            segments[segment_name]['samples'] = labels

            segments[segment_name]['onsets_seconds'] = np.where(np.diff(np.insert(labels, 0, values=[0], axis=0)) == 1)[0].astype(np.float) / samplerate
            segments[segment_name]['offsets_seconds'] = np.where(np.diff(np.append(labels, values=[0], axis=0)) == -1)[0].astype(np.float) / samplerate
            segments[segment_name]['durations_seconds'] = segments[segment_name]['offsets_seconds'] - segments[segment_name]['onsets_seconds']
    return segments


def predict_events(class_probabilities, samplerate: float = 1.0,
                   event_dims: int = None, event_names: List[str] = None,
                   event_thres: float = 0.5, events_offset: float = 0, event_dist: float=100,
                   event_dist_min: float = 0, event_dist_max: float = None):
    """[summary]

    Args:
        class_probabilities ([type]): [samples, classes][description]
        samplerate (float, optional): Hz
        event_dims ([type], optional): [description]. Defaults to range(nb_classes).
        event_names ([type], optional): [description]. Defaults to event_dims.
        event_thres (float, optional): [description]. Defaults to 0.5.
        events_offset (float, optional): . Defaults to 0 seconds.
        event_dist (float, optional): minimal distance between events for detection (in seconds). Defaults to 100 seconds.
        event_dist_min (float, optional): minimal distance to nearest event for post detection interval filter (in seconds). Defaults to 0 seconds.
        event_dist_max (float, optional): maximal distance to nearest event for post detection interval filter (in seconds). Defaults to None (no upper limit).

    Raises:
        ValueError: [description]

    Returns:
        dict['eventnames']['seconds'/'probabilities'/'index']
    """
    if event_dims is None:
        nb_classes = class_probabilities.shape[1]
        event_dims = range(nb_classes)

    if event_names is None:
        event_names = event_dims

    if event_dist_max is None:
        event_dist_max = class_probabilities.shape[0] + 1

    events = dict()
    if len(event_dims):
        for event_dim, event_name in zip(event_dims, event_names):
            events[event_name] = dict()
            events[event_name]['index'] = event_dim

            events[event_name]['seconds'], events[event_name]['probabilities'] = event_utils.detect_events(
                                                                          class_probabilities[:, event_dim],
                                                                          thres=event_thres, min_dist=event_dist * samplerate)
            events[event_name]['seconds'] = events[event_name]['seconds'].astype(np.float) / samplerate
            events[event_name]['seconds'] += events_offset

            good_event_indices = event_utils.event_interval_filter(events[event_name]['seconds'],
                                                                   event_dist_min, event_dist_max)
            events[event_name]['seconds'] = events[event_name]['seconds'][good_event_indices]
            events[event_name]['probabilities'] = events[event_name]['probabilities'][good_event_indices]

    return events


def predict(x: np.array, model_save_name: str = None, verbose: int = None, batch_size: int = None,
            model: models.keras.models.Model = None, params: dict = None,
            event_thres: float = 0.5, event_dist: float = 0.01,
            event_dist_min: float = 0, event_dist_max: float = None,
            segment_thres: float = 0.5, segment_minlen: float = None,
            segment_fillgap: float = None):
    """[summary]

    Usage:
    Calling predict with the path to the model will load the model and the
    associated params and run inference:
    `dss.predict.predict(x=data, model_save_name='tata')`

    To re-use the same model with multiple recordings, load the modal and params
    once and pass them to `predict`
    ```my_model, my_params = dss.utils.load_model_and_params(model_save_name)
    for data in data_list:
        dss.predict.predict(x=data, model=my_model, params=my_params)
    ```

    Args:
        x (np.array): Audio data [samples, channels]
        model_save_name (str): path with the trunk name of the model. Defaults to None.
        model (keras.model.Models): Defaults to None.
        params (dict): Defaults to None.

        verbose (int): display progress bar during prediction. Defaults to 1.
        batch_size (int): number of chunks processed at once . Defaults to None (the default used during training).
                         Larger batches lead to faster inference. Limited by memory size, in particular for GPUs which typically have 8GB.
                         Large batch sizes lead to loss of samples since only complete batches are used.

        event_thres (float): Confidence threshold for detecting peaks. Range 0..1. Defaults to 0.5.
        event_dist (float): Minimal distance between adjacent events during thresholding.
                            Prevents detecting duplicate events when the confidence trace is a little noisy.
                            Defaults to 0.01.
        event_dist_min (float): MINimal inter-event interval for the event filter run during post processing.
                                Defaults to 0.
        event_dist_max (float): MAXimal inter-event interval for the event filter run during post processing.
                                Defaults to None (no upper limit).

        segment_thres (float): Confidence threshold for detecting segments. Range 0..1. Defaults to 0.5.
        segment_minlen (float): Minimal duration of a segment used for filtering out spurious detections. Defaults to None.
        segment_fillgap (float): Gap between adjacent segments to be filled. Useful for correcting brief lapses. Defaults to None.


    Raises:
        ValueError: [description]

    Returns:
        events: [description]
        segments: [description]
        class_probabilities: [description]
    """

    if model_save_name is not None:
        model, params = utils.load_model_and_params(model_save_name)
    else:
        assert isinstance(model, models.keras.models.Model)
        assert isinstance(params, dict)

    samplerate = params['samplerate_x_Hz']

    # if model.input_shape[2:] != x.shape[1:]:
    #     raise ValueError(f'Input x has wrong shape - expected [samples, {model.input_shape[2:]}], got [samples, {x.shape[1:]}]')

    if batch_size is not None:
        params['batch_size'] = batch_size

    events_offset = params['data_padding'] / samplerate

    class_probabilities = predict_probabililties(x, model, params, verbose)

    segment_dims = np.where([val == 'segment' for val in params['class_types']])[0]
    segment_names = [params['class_names'][segment_dim] for segment_dim in segment_dims]
    segments = predict_segments(class_probabilities, samplerate,
                                segment_dims, segment_names,
                                segment_thres, segment_minlen, segment_fillgap)
    segments['samplerate_Hz'] = samplerate


    event_dims = np.where([val == 'event' for val in params['class_types']])[0]
    event_names = [params['class_names'][event_dim] for event_dim in event_dims]
    events = predict_events(class_probabilities, samplerate,
                            event_dims, event_names,
                            event_thres, events_offset, event_dist,
                            event_dist_min, event_dist_max)


    events['samplerate_Hz'] = samplerate

    return events, segments, class_probabilities


def cli_predict(recording_filename: str, model_save_name: str, *, save_filename: str = None,
         verbose: int = 1, batch_size: int = None,
         event_thres: float = 0.5, event_dist: float = 0.01,
         event_dist_min: float = 0, event_dist_max: float = None,
         segment_thres: float = 0.5, segment_minlen: float = None,
         segment_fillgap: float = None):
    """[summary]

    Saves hdf5 file with keys: events, segments,class_probabilities

    Args:
        recording_filename (str): path to the WAV file with the audio data.
        model_save_name (str): path with the trunk name of the model.
        save_filename (str): path to save annotations to. [Optional] - will strip extension from recording_filename and add '_dss.h5'.

        verbose (int): display progress bar during prediction. Defaults to 1.
        batch_size (int): number of chunks processed at once . Defaults to None (the default used during training).
                         Larger batches lead to faster inference. Limited by memory size, in particular for GPUs which typically have 8GB.

        event_thres (float): Confidence threshold for detecting peaks. Range 0..1. Defaults to 0.5.
        event_dist (float): Minimal distance between adjacent events during thresholding.
                            Prevents detecting duplicate events when the confidence trace is a little noisy.
                            Defaults to 0.01.
        event_dist_min (float): MINimal inter-event interval for the event filter run during post processing.
                                Defaults to 0.
        event_dist_max (float): MAXimal inter-event interval for the event filter run during post processing.
                                Defaults to None (no upper limit).

        segment_thres (float): Confidence threshold for detecting segments. Range 0..1. Defaults to 0.5.
        segment_minlen (float): Minimal duration of a segment used for filtering out spurious detections. Defaults to None.
        segment_fillgap (float): Gap between adjacent segments to be filled. Useful for correcting brief lapses. Defaults to None.
    """

    logging.info(f"   Loading data from {recording_filename}.")
    _, x = scipy.io.wavfile.read(recording_filename)
    x = np.atleast_2d(x).T

    logging.info(f"   Annotating using model at {model_save_name}.")
    events, segments, class_probabilities = predict(x, model_save_name, verbose, batch_size,
                                                    None, None,
                                                    event_thres, event_dist, event_dist_min, event_dist_max,
                                                    segment_thres, segment_minlen, segment_fillgap)

    d = {'events': events,
        'segments': segments,
        'class_probabilities': class_probabilities}

    if save_filename is None:
        save_filename = os.path.splitext(recording_filename)[0] + '_dss.h5'

    logging.info(f"   Saving results to {save_filename}.")
    flammkuchen.save(save_filename, d)
    logging.info(f"Done.")


def main():
    logging.basicConfig(level=logging.INFO)
    defopt.run(cli_predict)


if __name__ == '__main__':
    main()
