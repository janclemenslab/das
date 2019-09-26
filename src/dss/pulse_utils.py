"""Utilities for handling pulses."""
import numpy as np
import peakutils
import scipy.signal as ss


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
    if len(values) and len(array):  # only do this if boh inputs are non-empty lists
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


def normalize_pulse(pulse, smooth_win=15, flip_win=10):
    """Normalize pulses.

    1. scales to unit-norm,
    2. aligns to energy maximum,
    3. flips so that pre-peak mean is positive

    Args:
        pulse: should be [T,]
        smooth_win: n samples of rect window used to smooth squared pulse for peak detection
        flip_win: number of samples pre-peak used for determining sign of pulse for flipping.
    Returns:
        normalized pulse
    """
    # scale
    pulse /= np.linalg.norm(pulse)
    pulse_len = len(pulse)
    pulse_len_half = int(pulse_len / 2)
    # center
    gwin = ss.windows.boxcar(int(smooth_win))
    pulse_env = np.convolve(pulse**2, gwin, mode='valid')
    offset = np.argmax(pulse_env) + int(np.ceil(smooth_win / 2)) + 1
    pulse = np.pad(pulse, (len(pulse) - offset, offset), mode='constant', constant_values=0)
    # flip
    if np.sum(pulse[pulse_len - flip_win:pulse_len]) < 0:
        pulse *= -1
    return pulse[pulse_len_half:-pulse_len_half]


def center_of_mass(x, y, thres=0.5):
    y /= np.max(y)
    y -= thres
    y[y < 0] = 0
    y /= np.sum(y)
    com = np.dot(x, y)
    return com


def pulse_freq(pulse, fftlen=1000, sampling_rate=10000, mean_subtract=True):
    """Calculate pulse frequency as center of mass of the freq spectrum.

    Args:
        pulse - [T,]
        fftlen - sets freq resolution of the spectrum
        sampling_rate of the pulse
        mean_subtract - removes f0 component
    Returns:
        pulse frequency
        frequency values and amplitude of the pulse spectrum
    """
    if mean_subtract:
        pulse -= np.mean(pulse)
    F = np.fft.rfftfreq(fftlen, 1 / sampling_rate)
    A = np.abs(np.fft.rfft(pulse, fftlen))
    idx = np.argmax(F > 1000)
    center_freq = center_of_mass(F[:idx], A[:idx])
    return center_freq, F[:idx], A[:idx]


def match_pulses(true_pulses, pred_pulses, tol=100):
    """Find pulses pred_pulses that match those (within tol) in true_pulses.

    Args:
        true_pulses: list of reference pulse indices
        pred_pulses: list of detected pulse indices
        tol: n samples within which pulses are deemed identical
    Returns:
        nn_pulse: masked array copy of pred_pulses, mask=True indicates entries in pred not closest within tol in true
        nn_dist: dist of each pred_pulses to the nearest true_pulse
    """

    nn_dist = np.zeros_like(pred_pulses)
    nn_pulse = np.zeros_like(pred_pulses)
    # find nearest true pulse for each predicted pulse
    _, nn_pulse, nn_dist = find_nearest(true_pulses, pred_pulses)
    nn_pulse = nn_pulse.astype(np.float)

    # flag those that have no nearby pulse
    nn_pulse = np.ma.masked_array(nn_pulse, mask=nn_dist > tol)

    # flag doublettes - keep only nearest
    for idx in np.unique(nn_pulse[nn_pulse >= 0]):
        hits = np.where(nn_pulse == idx)[0]
        if len(hits) > 1:
            nearest = np.argmin(nn_dist[nn_pulse == idx])  # find closest hit
            hits = np.delete(hits, nearest)
            nn_pulse.mask[hits] = True

    return nn_pulse.astype(np.uintp), nn_dist


def eval_pulse_times(pulsetimes_true, pulsetimes_pred, tol=100):
    """[summary]

    Args:
        pulsetimes_true ([type]): [description]
        pulsetimes_pred ([type]): [description]
        tol (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    nn_pred_pulse, nn_dist = match_pulses(pulsetimes_true, pulsetimes_pred, tol)
    nn_true_pulse, _ = match_pulses(pulsetimes_pred, pulsetimes_true, tol)
    
    d = dict()
    d['FP'] = np.sum(nn_pred_pulse.mask)  # pred pulses that have no nearby true pulse (or there is another pred pulses nearer to the true pulse)
    d['TP'] = len(nn_pred_pulse.compressed())
    d['FN'] = len(nn_true_pulse) - d['TP']
    d['precision'] = d['TP'] / (d['TP'] + d['FP'])
    d['recall'] = d['TP'] / (d['TP'] + d['FN'])
    d['f1_score'] = 2 * (d['precision'] * d['recall']) / (d['precision'] + d['recall'])
    return d, nn_pred_pulse, nn_true_pulse, nn_dist


def get_pulseshapes(pulsecenters, song, win_hw):
    pulseshapes = np.zeros((2 * win_hw, len(pulsecenters)))
    for cnt, p in enumerate(pulsecenters):
        t0 = int(p - 2 * win_hw)
        t1 = int(p + 0 * win_hw)
        if t0 >= 0 and t1 < song.shape[0]:
            pulseshapes[:, cnt] = song[t0:t1, 0].copy()
    return pulseshapes


def detect_pulses(pulse_probability, thres=0.70, min_dist=100):
    """[summary]
    
    Args:
        pulse_probability ([type]): [description]
        thres (float, optional): [description]. Defaults to 0.70.
        min_dist (int, optional): [description]. Defaults to 100.
    
    Returns:
        [type]: [description]
    """
    pulse_indices = peakutils.indexes(pulse_probability, thres=thres, min_dist=min_dist)
    if len(pulse_indices):  # guard against empty pulse_indices
        pulse_confidence = pulse_probability[pulse_indices]
    else:
        pulse_confidence = []
    return pulse_indices, pulse_confidence
