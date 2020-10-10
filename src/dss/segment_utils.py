"""Segment (syllable) utilities."""
import numpy as np


def fill_gaps(sine_pred, gap_dur=100):
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


def remove_short(sine_pred, min_len=100):
    # remove too short sine songs
    onsets = np.where(np.diff(sine_pred.astype(np.int))==1)[0]
    offsets = np.where(np.diff(sine_pred.astype(np.int))==-1)[0]
    if len(onsets) and len(offsets):
        onsets = onsets[onsets<offsets[-1]]
        offsets = offsets[offsets>onsets[0]]
        durations = offsets - onsets
        for cnt, (onset, offset, duration) in enumerate(zip(onsets, offsets, durations)):
            if duration<min_len:
                sine_pred[onset:offset+1] = 0
    return sine_pred


# def levenshtein(source, target):
#     """Levenshstein (edit) distance

#        Corresponds to the number of edits (deletions, insertions, or substitutions)
#        required to convert source string into target string.

#     Args:
#         source (str): in this context, predicted labels for songbird syllables
#         target (str): in this context, ground truth labels for songbird syllables

#     Returns:
#         [type]: Levenshtein distance

#     code copied from from https://github.com/NickleDave/vak/blob/master/src/vak/metrics.py
#     see from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
#     """

#     if len(source) < len(target):
#         return levenshtein(target, source)

#     # So now we have len(source) >= len(target).
#     if len(target) == 0:
#         return len(source)

#     # We call tuple() to force strings to be used as sequences
#     # ('c', 'a', 't', 's') - numpy uses them as values by default.
#     source = np.array(tuple(source))
#     target = np.array(tuple(target))

#     # We use a dynamic programming algorithm, but with the
#     # added optimization that we only need the last two rows
#     # of the matrix.
#     previous_row = np.arange(target.size + 1)
#     for s in source:
#         # Insertion (target grows longer than source):
#         current_row = previous_row + 1

#         # Substitution or matching:
#         # Target and source items are aligned, and either
#         # are different (cost of 1), or are the same (cost of 0).
#         current_row[1:] = np.minimum(
#                 current_row[1:],
#                 np.add(previous_row[:-1], target != s))

#         # Deletion (target grows shorter than source):
#         current_row[1:] = np.minimum(
#                 current_row[1:],
#                 current_row[0:-1] + 1)

#         previous_row = current_row

#     return previous_row[-1]


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
