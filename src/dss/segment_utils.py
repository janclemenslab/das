"""Segment (syllable) utilities."""
import numpy as np


def levenshtein(source, target):
    """Levenshstein (edit) distance

       Corresponds to the number of edits (deletions, insertions, or substitutions)
       required to convert source string into target string.

    Args:
        source (str): in this context, predicted labels for songbird syllables
        target (str): in this context, ground truth labels for songbird syllables

    Returns:
        [type]: Levenshtein distance

    code copied from from https://github.com/NickleDave/vak/blob/master/src/vak/metrics.py
    see from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """

    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


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
