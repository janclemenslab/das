import UMAP
import hdbscan
import numpy as np
from PIL import Image
from typing import Optional, List


def cut_syllables(spec: np.ndarray, onsets: List[int], offsets: List[int],
                  thres_noisefloor: bool = True, thres_bias: float = 0):
    """[summary]

    Args:
        spec (np.ndarray): [F x T]
        onsets (List[int]): Indices of syllable starts. Should match offsets.
        offsets (List[int]): Indices of syllable ends. Should match onsets.
        thres_noisefloor (bool, optional):
                Divide spec at each frequency by noise floor.
                Noise floor for each frequency is given by the median over time.
                Defaults to True.
        thres_bias (float, optional): Bias to subtract before thresholding at 0. Defaults to 0.

    Returns:
        [type]: [description]
    """

    if thres_noisefloor:
        noise_floor = np.median(spec, axis=1)[:, np.newaxis]
    else:
        noise_floor = 1.0

    specs = []
    for onset, offset in zip(onsets, offsets):
        syll = np.log2(spec[:, onset:offset] / noise_floor[:, np.newaxis])
        syll -= thres_bias
        syll = np.clop(syll, a_min=0, a_max=np.inf)
        specs.append(syll)
    return specs


def log_resize_spec(spec: np.ndarray, scaling_factor=10) -> np.ndarray:
    """Log resize time axis. SCALING_FACTOR determines nonlinearity of scaling."""
    #from https://github.com/timsainb/avgn_paper
    resize_shape = [int(np.log(spec.shape[1]) * scaling_factor), spec.shape[0]]
    resize_spec = np.array(Image.fromarray(spec).resize(resize_shape, Image.ANTIALIAS))
    return resize_spec


def pad_spec(spec: np.ndarray, pad_length: float) -> np.ndarray:
    """Pads a spectrogram to PAD_LENGTH."""
    #from https://github.com/timsainb/avgn_paper
    excess_needed = pad_length - spec.shape[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")
    return np.pad(spec, [(0, 0), (pad_left, pad_right)], "constant", constant_values=0)


def center_spec(spec: np.ndarray, halfwidth: Optional[int] = None) -> np.ndarray:
    """Center spectrogram around peak frequency.

    Args:
        spec (np.ndarray): [F x T]
        halfwidth (Optional[int], optional):
                Range of freqs to around best freq to keep.
                Will pad overhang with edge values.
                Defaults to number of freqs / 2.

    Returns:
        np.ndarray: [description]
    """
    freqs = spec.shape[0]

    if halfwidth is None:
        halfwidth = np.floor(freqs/2)

    peak_freq = np.clip(np.argmax(np.mean(spec, axis=1)), halfwidth, freqs - halfwidth)

    return spec[peak_freq-halfwidth:peak_freq+halfwidth,:]
