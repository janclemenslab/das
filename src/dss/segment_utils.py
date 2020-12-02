"""Segment (syllable) utilities."""
import numpy as np
import scipy.stats
from typing import List, Tuple


def fill_gaps(sine_pred: np.array, gap_dur: int = 100) -> np.array:
    onsets = np.where(np.diff(sine_pred.astype(np.int))==1)[0]
    offsets = np.where(np.diff(sine_pred.astype(np.int))==-1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets<offsets[-1]]
        offsets = offsets[offsets>onsets[0]]
        durations = offsets - onsets
        for idx, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if idx>0 and offsets[idx-1]>onsets[idx]-gap_dur:
                sine_pred[offsets[idx-1]:onsets[idx]+1] = 1
    return sine_pred


def remove_short(sine_pred: np.array, min_len: int = 100) -> np.array:
    # remove too short sine songs
    onsets = np.where(np.diff(sine_pred.astype(np.int)) == 1)[0]
    offsets = np.where(np.diff(sine_pred.astype(np.int)) == -1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets < offsets[-1]]
        offsets = offsets[offsets > onsets[0]]
        durations = offsets - onsets
        for cnt, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if duration < min_len:
                sine_pred[onset:offset + 1] = 0
    return sine_pred


def label_syllables_by_majority(labels: np.array,
                                onsets_seconds: List[float], offsets_seconds: List[float],
                                samplerate: float) -> Tuple[np.array, np.array]:
    syllables = []
    labels_clean = np.zeros_like(labels, dtype=np.int)

    for onset_seconds, offset_seconds in zip(onsets_seconds, offsets_seconds):
        onset_sample = int(onset_seconds * samplerate)
        offset_sample = int(offset_seconds * samplerate)

        syllables.append(int(scipy.stats.mode(labels[onset_sample:offset_sample])[0]))
        labels_clean[onset_sample:offset_sample] = syllables[-1]

    syllables = np.array(syllables)

    return syllables, labels_clean


def levenshtein(seq1, seq2):
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len(seq2) - 1]


def syllable_error_rate(true, pred):
    """Levenshtein/edit distance normalized by length of true sequence

    Args:
        true ([str]): ground truth labels for a series of songbird syllables
        pred ([str]): predicted labels for a series of songbird syllables

    Raises:
        TypeError: if either input is not a str

    Returns:
        float: Levenshtein distance / len(true)

    """

    if type(true) != str or type(pred) != str:
        raise TypeError('Both `true` and `pred` must be of type `str')

    return levenshtein(pred, true) / len(true)
