"""Segment (syllable) utilities."""

import numpy as np
from typing import Tuple


def fill_gaps(labels: np.ndarray, gap_dur: int = 100) -> np.ndarray:
    """Fill short gaps in a sequence of labelled samples.

    `---111111-1111---` -> `---11111111111---`

    Args:
        labels (np.ndarray): Sequence of labelled samples.
        gap_dur (int, optional): Minimal gap duration, in samples. Defaults to 100.

    Returns:
        np.ndarray: Labelled samples with short gaps filled.
    """
    onsets = np.where(np.diff(labels.astype(int)) == 1)[0]
    offsets = np.where(np.diff(labels.astype(int)) == -1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets < offsets[-1]]
    if len(onsets) and len(offsets):
        offsets = offsets[offsets > onsets[0]]
    if len(onsets) and len(offsets) and len(onsets) == len(offsets):
        for idx in range(len(onsets)):
            if idx > 0 and offsets[idx - 1] > onsets[idx] - gap_dur:
                labels[offsets[idx - 1] : onsets[idx] + 1] = 1
    return labels


def remove_short(labels: np.ndarray, min_len: int = 100) -> np.ndarray:
    """Remove short syllables from sequence of labelled samples.

    `---1111-1---1--` -> `---1111--------`

    Args:
        labels (np.ndarray): Sequence of labelled samples.
        min_len (int, optional): Minimal segment (syllable) duration, in samples. Defaults to 100.

    Returns:
        np.ndarray: Labelled samples with short syllables removed.
    """
    onsets = np.where(np.diff(labels.astype(int)) == 1)[0]
    offsets = np.where(np.diff(labels.astype(int)) == -1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets < offsets[-1]]
    if len(onsets) and len(offsets):
        offsets = offsets[offsets > onsets[0]]
    if len(onsets) and len(offsets) and len(onsets) == len(offsets):
        durations = offsets - onsets
        for onset, offset, duration in zip(onsets, offsets, durations):
            if duration < min_len:
                labels[onset : offset + 1] = 0
    return labels


def label_syllables_by_majority(
    labels: np.ndarray, onsets_seconds: np.ndarray, offsets_seconds: np.ndarray, samplerate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Label syllables by a majority vote.

    Args:
        labels (np.ndarray): Sequence of dirty, per sample, labels.
        onsets_seconds (List[float]): Onset of each syllable in `labels`, in seconds.
        offsets_seconds (List[float]): Offset of each syllable in `labels`, in seconds.
        samplerate (float): Samplerate of `labels`, in Hz.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Sequence of syllables, clean sequence of per-sample labels.
    """
    syllables = []
    labels_clean = np.zeros_like(labels, dtype=int)

    onsets_sample = (onsets_seconds * samplerate).astype(int)
    offsets_sample = (offsets_seconds * samplerate).astype(int)

    for onset_sample, offset_sample in zip(onsets_sample, offsets_sample):
        # max sure the segment is at least one sample long
        os = max(onset_sample + 1, offset_sample)
        values, counts = np.unique(labels[onset_sample:os], return_counts=True)
        if len(values):
            majority_label = values[counts.argmax()]
            syllables.append(int(majority_label))
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


def syllable_error_rate(true: str, pred: str) -> float:
    """Compute the Levenshtein edit distance normalized by length of `true`.

    Args:
        true (str): Ground truth labels for a sequence of syllables. For instance, 'ABCDAAE'.
        pred (str): Predicted labels for a sequence of syllables.

    Raises:
        TypeError: if either input is not a `str`

    Returns:
        float: Levenshtein distance normalized by length of `true`.
    """

    if not isinstance(true, str) or not isinstance(pred, str):
        raise TypeError("Both `true` and `pred` must be of type `str`")

    return levenshtein(pred, true) / len(true)
