"""Utilities for handling pulses."""
import numpy as np
import scipy.signal as ss
from typing import List, Tuple


def normalize_pulse(pulse: np.ndarray, smooth_win: int = 15, flip_win: int = 10) -> np.ndarray:
    """Normalize pulses.

    1. scales to unit-norm,
    2. aligns to energy maximum,
    3. flips so that pre-peak mean is positive

    Args:
        pulse (np.ndarray): should be [T,]
        smooth_win (int, optional): n samples of rect window used to smooth squared pulse for peak detection
        flip_win (int, optional): number of samples pre-peak used for determining sign of pulse for flipping.

    Returns:
        np.ndarray: normalized pulse [T,]
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


def center_of_mass(x: np.ndarray, y: np.ndarray, thres: float = 0.5) -> float:
    """Calculate center of mass of y.

    Args:
        x (np.ndarray):
        y (np.ndarray):
        thres (float, optional): Threshold. Defaults to 0.5.

    Returns:
        float: Center of mass.
    """
    y /= np.max(y)
    y -= thres
    y[y < 0] = 0
    y /= np.sum(y)
    com = np.dot(x, y)
    return com


def pulse_freq(pulse: np.ndarray,
               fftlen: int = 1000,
               sampling_rate: int = 10000,
               mean_subtract: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculate pulse frequency as center of mass of the pulse's amplitude spectrum.

    Args:
        pulse (np.ndarray): Waveform (shape [T,]).
        fftlen (int, optional): Sets freq resolution of the spectrum. Defaults to 1_000.
        sampling_rate (float, optional): Sample rate of the pulse, in Hz. Defaults to 10_000.
        mean_subtract (bool, optional): If true, removes f0 component by mean subtraction. Defaults to True.
    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Center frequency,
                                              frequency and
                                              amplitude values of the pulse spectrum (cut off at 1000 Hz).
    """
    if mean_subtract:
        pulse -= np.mean(pulse)
    F = np.fft.rfftfreq(fftlen, 1 / sampling_rate)
    A = np.abs(np.fft.rfft(pulse, fftlen))
    idx = int(np.argmax(F > 1_000))
    center_freq = center_of_mass(F[:idx], A[:idx])
    return center_freq, F[:idx], A[:idx]


def get_pulseshapes(pulsecenters: List[int], song: np.ndarray, win_hw: int) -> np.ndarray:
    """Extract waveforms around `pulsecenters` from `song`.

    In case of multi-channel recordings, will return the waveform on the channel
    with the maximum absolute value within +/-`win_hw` around the each pulsecenter.

    Args:
        pulsecenters (List[int]): Location of each pulse center in `song`, in samples
        song (np.ndarray): Audio data ([samples, channels]).
        win_hw (int): Half-width of the waveform cut out around each pulse center, in samples.

    Returns:
        np.ndarray: Extracted waveforms [2 * win_hw, nb_centers]
    """
    pulseshapes = np.zeros((2 * win_hw, len(pulsecenters)))
    nb_channels = song.shape[1]
    for cnt, p in enumerate(pulsecenters):
        t0 = int(p - win_hw)
        t1 = int(p + win_hw)
        if t0 > 0 and t1 < song.shape[0]:
            if nb_channels > 1:
                tmp = song[t0:t1, :]
                loudest_channel = np.argmax(np.max(np.abs(tmp), axis=0))
                pulseshapes[:, cnt] = tmp[:, loudest_channel].copy()
            else:
                pulseshapes[:, cnt] = song[t0:t1, 0].copy()
    return pulseshapes
