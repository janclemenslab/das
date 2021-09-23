"""Segment (syllable) utilities."""
import numpy as np
import scipy.stats
from typing import List, Tuple, Any


def fill_gaps(labels: np.array, gap_dur: int = 100) -> np.array:
    """Fill short gaps in a sequence of labelled samples.

    `---111111-1111---` -> `---11111111111---`

    Args:
        labels (np.array): Sequence of labelled samples.
        gap_dur (int, optional): Minimal gap duration, in samples. Defaults to 100.

    Returns:
        np.array: Labelled samples with short gaps filled.
    """
    onsets = np.where(np.diff(labels.astype(np.int))==1)[0]
    offsets = np.where(np.diff(labels.astype(np.int))==-1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets<offsets[-1]]
        offsets = offsets[offsets>onsets[0]]
        durations = offsets - onsets
        for idx, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if idx>0 and offsets[idx-1]>onsets[idx]-gap_dur:
                labels[offsets[idx-1]:onsets[idx]+1] = 1
    return labels


def remove_short(labels: np.array, min_len: int = 100) -> np.array:
    """Remove short syllables from sequence of labelled samples.

    `---1111-1---1--` -> `---1111--------`

    Args:
        labels (np.array): Sequence of labelled samples.
        min_len (int, optional): Minimal segment (syllable) duration, in samples. Defaults to 100.

    Returns:
        np.array: Labelled samples with short syllables removed.
    """
    onsets = np.where(np.diff(labels.astype(np.int)) == 1)[0]
    offsets = np.where(np.diff(labels.astype(np.int)) == -1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets < offsets[-1]]
        offsets = offsets[offsets > onsets[0]]
        durations = offsets - onsets
        for cnt, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if duration < min_len:
                labels[onset:offset + 1] = 0
    return labels


def label_syllables_by_majority(labels: np.array,
                                onsets_seconds: List[float], offsets_seconds: List[float],
                                samplerate: float) -> Tuple[np.array, np.array]:
    """Label syllables by a majority vote.

    Args:
        labels (np.array): Sequence of dirty, per sample, labels.
        onsets_seconds (List[float]): Onset of each syllable in `labels`, in seconds.
        offsets_seconds (List[float]): Offset of each syllable in `labels`, in seconds.
        samplerate (float): Samplerate of `labels`, in Hz.

    Returns:
        Tuple[np.array, np.array]: Sequence of syllables, clean sequence of per sample labels.
    """
    syllables = []
    labels_clean = np.zeros_like(labels, dtype=np.int)

    for onset_seconds, offset_seconds in zip(onsets_seconds, offsets_seconds):
        onset_sample = int(onset_seconds * samplerate)
        offset_sample = int(offset_seconds * samplerate)

        syllables.append(int(scipy.stats.mode(labels[onset_sample:offset_sample])[0]))
        labels_clean[onset_sample:offset_sample] = syllables[-1]

    syllables = np.array(syllables)

    return syllables, labels_clean


def levenshtein(seq1: str, seq2: str) -> float:
    """Compute the Levenshtein edit distance between two strings.

    Corresponds to the minimal number of insertions, deletions, and subsitutions
    required to transform `seq1` into `seq2`.

    Args:
        seq1 (str)
        seq2 (str)

    Returns:
        float: The Levenshtein distance between seq1 and seq1.
    """
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


def syllable_error_rate(true, pred) -> float:
    """Compute the Levenshtein edit distance normalized by length of `true`.

    Args:
        true (str): Ground truth labels for a sequence of syllables.
        pred (str): Predicted labels for a sequence of syllables.

    Raises:
        TypeError: if either input is not a `str`

    Returns:
        float: Levenshtein distance normalized by length of `true`.
    """

    if type(true) != str or type(pred) != str:
        raise TypeError("Both `true` and `pred` must be of type `str`")

    return levenshtein(pred, true) / len(true)
