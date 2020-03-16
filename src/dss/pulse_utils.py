"""Utilities for handling pulses."""
import numpy as np
import peakutils
import scipy.signal as ss


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


def get_pulseshapes(pulsecenters, song, win_hw):
    """[summary]

    In case of multi-channel recordings, will return the shape for the loudest channel.

    Args:
        pulsecenters ([type]): [description]
        song ([type]): samples x channels
        win_hw ([type]): [description]

    Returns:
        [type]: [description]
    """
    pulseshapes = np.zeros((2 * win_hw, len(pulsecenters)))
    nb_channels = song.shape[1]
    for cnt, p in enumerate(pulsecenters):
        t0 = int(p - 2 * win_hw)
        t1 = int(p + 0 * win_hw)
        if t0 > 0 and t1 < song.shape[0]:
            if nb_channels > 1:
                tmp = song[t0:t1, :]
                loudest_channel = np.argmax(np.max(tmp, axis=0))
                pulseshapes[:, cnt] = tmp[:, loudest_channel].copy()
            else:
                pulseshapes[:, cnt] = song[t0:t1, 0].copy()
    return pulseshapes
