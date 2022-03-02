"""Utilities for handling events."""
import numpy as np
import peakutils
from typing import Iterable


def find_nearest(array, values):
    """Find nearest occurrence of each item of values in array.

    Args:
        array: find nearest in this list
        values: queries

    Returns:
        val: nearest val in array to each item in values
        idx: index of nearest val in array to each item in values
        dist: distance to nearest val in array for each item in values
        NOTE: Returns nan-arrays of the same size as values if `array` is empty.
    """
    if len(values) and len(array):  # only do this if both inputs are non-empty lists
        values = np.atleast_1d(values)
        abs_dist = np.abs(np.int64(np.subtract.outer(array, values)))
        idx = abs_dist.argmin(0)
        dist = abs_dist.min(0)
        val = array[idx]
    else:
        idx = np.full_like(values, fill_value=np.nan)
        dist = np.full_like(values, fill_value=np.nan)
        val = np.full_like(values, fill_value=np.nan)
    return val, idx, dist


def detect_events(event_probability, thres=0.70, min_dist=100):
    """[summary]

    Args:
        event_probability ([type]): [description]
        thres (float, optional): [description]. Defaults to 0.70.
        min_dist (int, optional): [description]. Defaults to 100 samples.

    Returns:
        event_indices: index of each detected event
        event_confidence: event_probability at the event_index
    """
    event_indices = peakutils.indexes(event_probability, thres=thres, min_dist=min_dist)

    if len(event_indices):  # guard against empty event_indices
        event_confidence = event_probability[event_indices]
    else:
        event_confidence = []

    return event_indices, event_confidence


def match_events(eventindices_true, eventindices_pred, tol=100):
    """Find events eventindices_pred that match those (within tol) in eventindices_true.

    Args:
        eventindices_true: list of reference event indices
        eventindices_pred: list of detected event indices
        tol: n samples within which events are deemed identical
    Returns:
        nearest_event: masked array copy of eventindices_pred, mask=True indicates entries in pred not closest within tol in true
        nearest_dist: dist of each eventindices_pred to the nearest true_event
    """

    nearest_dist = np.zeros_like(eventindices_pred)
    nearest_event = np.zeros_like(eventindices_pred)
    # find nearest true event for each predicted event
    _, nearest_event, nearest_dist = find_nearest(eventindices_true, eventindices_pred)
    nearest_event = nearest_event.astype(np.float)

    # flag those that have no nearby event
    nearest_event = np.ma.masked_array(nearest_event, mask=nearest_dist > tol)
    if len(eventindices_true) == 0:
        nearest_event.mask = True  # all detections are false positives
    else:
        # flag doublettes - keep only nearest
        for idx in np.unique(nearest_event[nearest_event >= 0]):
            hits = np.where(nearest_event == idx)[0]
            if len(hits) > 1:
                nearest = np.argmin(nearest_dist[nearest_event == idx])  # find closest hit
                hits = np.delete(hits, nearest)
                nearest_event.mask[hits] = True

    return nearest_event.astype(np.uintp), nearest_dist


def event_interval_filter(events: Iterable, event_dist_min: float = 0, event_dist_max: float = np.inf) -> Iterable:
    """[summary]

    Args:
        events (Iterable): Iterable (list, np.array) of event times in seconds.
        event_dist_min (float, optional): [description]. Defaults to 0 second.
        event_dist_max (float, optional): [description]. Defaults to np.inf seconds.

    Returns:
        good_event_indices (Iterable): indices of events to keep `events[good_event_indices]`.
    """

    ipi_pre = np.diff(events, prepend=np.inf)
    ipi_post = np.diff(events, append=np.inf)

    ipi_too_long = np.logical_and(ipi_pre>event_dist_max, ipi_post>event_dist_max)
    ipi_too_short = np.logical_or(ipi_pre<event_dist_min, ipi_post<event_dist_min)
    ipi_too_short[0] = False  # otherwise first event will always be removed

    good_event_indices = ~np.logical_or(ipi_too_long, ipi_too_short)

    return good_event_indices


def evaluate_eventtimes(eventtimes_true, eventtimes_pred, samplerate, tol=0.01):
    """[summary]

    Args:
        eventtimes_true ([type]): in seconds
        eventtimes_pred ([type]): in seconds
        samplerate (float): in Hz
        tol (int, optional): in seconds [description]. Defaults to 0.01 seconds.

    Returns:
        [type]: [description]
    """

    # match_events works with indices - so need to convert eventtimes to indices and distance back to times
    eventindices_true = eventtimes_true * samplerate
    eventindices_pred = eventtimes_pred * samplerate

    nearest_pred_event, nearest_dist_indices = match_events(eventindices_true, eventindices_pred, tol * samplerate)
    nearest_true_event, _ = match_events(eventindices_pred, eventindices_true, tol * samplerate)
    nearest_dist = nearest_dist_indices / samplerate  # convert back to seconds

    d = dict()
    d['FP'] = np.sum(nearest_pred_event.mask)  # pred events that have no nearby true event (or there is another pred event nearer to the true event)
    d['TP'] = len(nearest_pred_event[np.isfinite(nearest_dist)].compressed())
    d['FN'] = max(0, np.sum(nearest_true_event.mask))

    if d['FP'] == 0:  # if there are no false positives (even for no detections), then precision is 1.0
        d['precision'] = 1.0
    elif (d['TP'] + d['FP']) == 0:
        d['precision'] = 0.0
    else:
        d['precision'] = d['TP'] / (d['TP'] + d['FP'])

    if d['FN'] == 0:  # if there are no false negatives (even for no detections), the recall is 1.0 (?!)
        d['recall'] = 1.0
    elif d['TP'] + d['FN'] == 0:
        d['recall'] = 0
    else:
        d['recall'] = d['TP'] / (d['TP'] + d['FN'])

    if (d['precision'] + d['recall']) == 0:
        d['f1_score'] = 0
    else:
        d['f1_score'] = 2 * (d['precision'] * d['recall']) / (d['precision'] + d['recall'])

    return d, nearest_pred_event, nearest_true_event, nearest_dist
