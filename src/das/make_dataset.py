import numpy as np
import zarr
from typing import List, Iterable, Mapping, Optional
import pandas as pd
import scipy.signal
import logging
import sklearn.model_selection


def init_store(nb_channels: int,
               nb_classes: int,
               samplerate: Optional[float] = None,
               make_single_class_datasets: bool = False,
               class_names: List[str] = None,
               class_types: List[str] = None,
               store_type=zarr.TempStore,
               store_name: str = 'store.zarr',
               chunk_len: int = 1_000_000):
    """[summary]

    Args:
        nb_channels (int): [description]
        nb_classes (int): [description]  <- should infer from class_names!
        samplerate (float, optional): [description]. Defaults to None.
        make_single_class_datasets (bool, optional): make y_suffix and attrs['class_names/types_suffix']. Defaults to None.
        class_names (List[str], optional): [description]. Defaults to None.
        class_types (List[str], optional): 'event' or 'segment'. Defaults to None.
        store_type ([type], optional): [description]. Defaults to zarr.TemporaryStore.
        store_name (str, optional): [description]. Defaults to 'store.zarr'.
        chunk_len (int, optional): [description]. Defaults to 1_000_000.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    if class_names is not None and nb_classes is not None and len(
            class_names) != nb_classes:
        raise ValueError(
            f'Number of classes ({nb_classes}) needs to match len(class_names) ({len(class_names)}).'
        )
    if class_types is not None and nb_classes is not None and len(
            class_names) != nb_classes:
        raise ValueError(
            f'Number of classes ({nb_classes}) needs to match len(class_names) ({len(class_types)}).'
        )

    # initialize the store
    store = store_type(store_name)
    root = zarr.group(store=store, overwrite=True)  # need to def the root
    for target in ['train', 'val', 'test']:
        root.empty(name=f'{target}/x',
                   shape=(0, nb_channels),
                   chunks=(chunk_len, nb_channels),
                   dtype=np.float16)
        root.empty(name=f'{target}/y',
                   shape=(0, nb_classes),
                   chunks=(chunk_len, nb_classes),
                   dtype=np.float16)
        # root.empty(name=f'{target}/eventtimes', shape=(0, nb_classes), chunks=(1_000,), dtype=np.float)
        if make_single_class_datasets:
            for class_name in class_names[1:]:
                root.empty(name=f'{target}/y_{class_name}',
                           shape=(0, 2),
                           chunks=(chunk_len, nb_classes),
                           dtype=np.float16)

    # init metadata - since attrs cannot be appended to, we init a dict here, populate it with information below and finaly assign it to root.attrs
    root.attrs['samplerate_x_Hz'] = samplerate
    root.attrs['samplerate_y_Hz'] = samplerate

    root.attrs['class_names'] = class_names
    root.attrs['class_types'] = class_types

    if make_single_class_datasets:
        for class_name, class_type in zip(class_names[1:], class_types[1:]):
            root.attrs[f'class_names_{class_name}'] = [
                class_names[0], class_name
            ]
            root.attrs[f'class_types_{class_name}'] = [
                class_types[0], class_type
            ]

    for target in ['train', 'val', 'test']:
        root.attrs[f'filename_startsample_{target}'] = []
        root.attrs[f'filename_endsample_{target}'] = []
        root.attrs[f'filename_{target}'] = []
    return root


def events_to_probabilities(eventsamples: List[int], desired_len: Optional[int] = None, extent: int = 61):
    """Converts list of events to one-hot-encoded probability vectors.

    Args:
        eventsamples (List[int]): List of event "times" in samples.
        desired_len (float, optional): Length of the probability vector.
                                       Events exceeding `desired_len` will be ignored.
                                       Defaults to `max(eventsamples) + extent`.
        extent (int, optional): Temporal extent of an event in the probability vector.
                                Each event will be represented as a box with a duration `exent` samples centered on the event.
                                Defaults to 61 samples (+/-30 samples).
    Returns:
        probabilities: np.array with shape [desired_len, 2]
                       where `probabilities[:, 0]` corresponds to the probability of no event
                       and `probabilities[:, 0]` corresponds to the probability of an event.
    """
    if desired_len is None:
        desired_len = max(eventsamples) + extent
    else:
        eventsamples = eventsamples[
            eventsamples < desired_len -
            extent]  # delete all eventsamples exceeding desired_len
    probabilities = np.zeros((desired_len, 2))
    probabilities[eventsamples, 1] = 1
    probabilities[:, 1] = np.convolve(probabilities[:, 1],
                                      np.ones((extent, )),
                                      mode='same')
    probabilities[:, 0] = 1 - probabilities[:, 1]
    return probabilities


def infer_class_info(df: pd.DataFrame):
    """[summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    class_names, first_indices = np.unique(df['name'], return_index=True)
    class_names = list(class_names)
    class_names.insert(0, 'noise')

    # infer class type - event if start and end are the same
    class_types = ['segment']
    for first_index in first_indices:
        if df.loc[first_index]['start_seconds'] == df.loc[first_index][
                'stop_seconds']:
            class_types.append('event')
        else:
            class_types.append('segment')
    return class_names, class_types


def make_annotation_matrix(df: pd.DataFrame,
                           nb_samples: int,
                           samplerate: float,
                           class_names: Optional[List[str]] = None) -> np.ndarray:
    """One-hot encode a list of song timings to a binary matrix.

    Args:
        df (pd.DataFrame): DataFrame with the following columns:
                            - name: class name of the syllable/song event
                            - start_seconds: start of the song event in the audio recording in seconds.
                            - stop_seconds: stop of the song event in the audio recording in seconds.
        nb_samples ([type]): Length of the annotation matrix in samples.
        samplerate (float): Sample rate for the annotation matrix in Hz.
        class_names (List[str], optional): List of class names.
                            If provided, the annotation matrix will be built only for the events in class_names.
                            Otherwise, the matrix will be build for all class names in the df.
                            Order in class_names determines order in class_matrix

    Returns:
        nd.array: Binary matrix [nb_samples, nb_classes]
                  with 1 indicating the presence of a class at a specific sample.
    """
    if class_names is None:
        class_names, _ = infer_class_info(df)
    class_matrix = np.zeros((nb_samples, len(class_names)))
    for _, row in df.iterrows():
        if not row['name'] in class_names:
            continue
        if np.all(np.isnan(row['start_seconds'])):
            continue
        class_index = class_names.index(row['name'])
        start_index = int(row['start_seconds'] * samplerate)
        stop_index = int(row['stop_seconds'] * samplerate + 1)
        if start_index < stop_index:
            class_matrix[start_index:stop_index, class_index] = 1
        else:
            logging.warning(
                f'{start_index} should be greater than {stop_index} for row {row}'
            )
    return class_matrix


def match_perm(perm: List[int], fractions: List[float],
               block_stats: List[float]):
    """Quantify how well the current permutation of blocks produces
    splits that with class distributions represenative of full data set.

    Args:
        perm (List[int]): Permutated list of block indices.
        fractions (List[float]): Fractions to split the blocks into.
        block_stats (List[float]): Class stats for each block.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if len(fractions) != 3:
        raise ValueError('')
    blocks = dict()
    blocks['train'], val_test = sklearn.model_selection.train_test_split(
        perm, test_size=fractions[0] * (fractions[1] + fractions[2]))
    blocks['val'], blocks['test'] = sklearn.model_selection.train_test_split(
        val_test, test_size=fractions[1] / (fractions[1] + fractions[2]))
    all_stats = np.mean(
        block_stats, axis=0
    )  # better to get this from the intact data array - will be wrong if blocks have uneven sizes
    non_zero_stats = np.where(all_stats > 0)[0]
    stats = dict()
    match = dict()
    for key, val in blocks.items():
        stats[key] = np.mean(block_stats[val], axis=0)
        match[key] = np.log2((stats[key][non_zero_stats] + 0.0000001) /
                             all_stats[non_zero_stats])
    total_match = np.sum([np.sum(np.abs(m)) for m in match.values()])
    return total_match, stats, match, blocks


def do_block_stratify(y: np.ndarray,
                      fractions: List[float],
                      block_size: int,
                      gap: int = 0,
                      nb_perms: int = 100):
    """Find permutation such that class probabilities y in individual splits
       match the class probabilities in the full data array.

    Args:
        y (np.ndarray): [description]
        fractions (List[float]): [description]
        block_size (int): [description]
        gap (int, optional): [description]. Defaults to 0.
        nb_perms (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    # calculate per-block stats
    block_stats = []
    split_points = np.array(range(block_size, y.shape[0], block_size))
    split_points = split_points[:-1]
    for split in split_points[1:]:
        block_stats.append(
            np.mean(y[split - block_size + gap:split - gap], axis=0))
    block_stats = np.array(block_stats)
    nb_splits = len(block_stats)

    while len(fractions) < 3:
        fractions.append(0)
    fractions = np.array(fractions) / np.sum(fractions)

    perms = []
    total_mismatches = []
    total_stats = []
    split_blocks = []
    for _ in range(nb_perms):
        perm = np.random.permutation(nb_splits)
        perms.append(perm)
        total_mismatch, stats, match, blocks = match_perm(
            perm, fractions, block_stats)
        total_mismatches.append(total_mismatch)
        total_stats.append(stats)
        split_blocks.append(blocks)

    best_perm = np.argmin(total_mismatches)

    # split_block to list of names
    split_names = np.array(['train'] * nb_splits)
    for key, val in split_blocks[best_perm].items():
        split_names[val] = key

    # convert split_points from samples to relative position in array
    split_points = split_points / len(y)

    return split_points, split_names


def generate_data_splits(arrays: Mapping[str, np.ndarray],
                         splits: List[float],
                         split_names: List[str],
                         shuffle: bool = True,
                         block_stratify: Optional[np.ndarray] = None,
                         block_size: Optional[int] = None,
                         seed: Optional[float] = None):
    """[summary]

    Args:
        arrays (Mapping[str, np.ndarray]): [description], e.g. {'x': [...], 'y': [...]}
        splits (List[float]): [description], e.g. [0.6, 0.2, 0.2]
        split_names (List[str]): [description], e.g. ['train', 'val', 'test']
        shuffle (bool, optional): Shuffle the splits, (does not shuffle samples). Defaults to True.
        block_stratify (np.ndarray, optional): Label for each sample in arrays used for stratification. Defaults to None (no stratification).
        block_size(int, optional): Size of blocks (in samples) used for stratified sampling. Defaults to None (100_000 samples).
        seed (float, optional): Defaults to None (do not seed random number generator).

    Returns:
        [type]: [description]
    """
    if seed is not None:
        np.random.seed(seed)

    splits = np.array(splits)
    names = np.array(split_names)

    # if not all([len(arrays[0]) == len(a) for a in arrays]):
    #     ValueError('All arrays should have same length')

    split_points = np.cumsum(splits)
    if shuffle and block_stratify is None:
        order = np.random.permutation(np.arange(len(names)))
        splits = splits[order]
        names = names[order]
        split_points = np.cumsum(splits)
    elif block_stratify is not None:
        if block_stratify.ndim < 2:
            raise ValueError(
                'Data for stratification should be one-hot-encoded.')
        if block_size is None:
            block_size = 100_000  # samples
        split_points, names = do_block_stratify(block_stratify, splits,
                                                block_size)

    split_arrays = dict()
    for key, array in arrays.items():
        nb_samples = array.shape[0]
        nb_dims = array.shape[1]
        split_arrays[key] = {name: np.empty((0, nb_dims)) for name in names}
        train_val_test_split = (split_points * nb_samples).astype(np.int)[:-1]
        x_splits = np.split(array, train_val_test_split)

        # distribute splits across train/val/test sets
        for x_split, name in zip(x_splits, names):
            split_arrays[key][name] = np.concatenate(
                (split_arrays[key][name], x_split))
    return split_arrays


def normalize_probabilities(p: np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        p (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    p_song = np.sum(p[:, 1:], axis=-1)

    p[p_song > 1.0,
      1:] = p[p_song > 1.0, 1:] / p_song[p_song > 1.0, np.newaxis]
    p[:, 0] = 1 - np.sum(p[:, 1:], axis=-1)
    return p


def generate_file_splits(file_list: List,
                         splits: List[float],
                         split_names: List[str],
                         shuffle: bool = True,
                         seed: Optional[float] = None) -> List:
    """[summary]

    Args:
        file_list (List): [description]
        splits (List[float]): [description]
        split_names (List[str]): [description]
        shuffle (bool, optional): [description]. Defaults to True.
        seed (float, optional): Defaults to None (do not seed random number generator)

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        List: [description]
    """

    if seed is not None:
        np.random.seed(seed)

    if len(splits) != len(split_names):
        raise ValueError(
            f'there must be one name per split. but there are {len(split_names)} names and {len(splits)} splits.'
        )

    file_list = np.array(file_list)
    nb_files = len(file_list)
    if shuffle:
        order = np.random.permutation(np.arange(nb_files))
        file_list = file_list[order]

    splits = np.concatenate(
        ([0], np.array(splits)))  # prepend 0 as split anchor

    if np.sum(splits) != 1:
        logging.warn(
            f'probs should sum to 1 - but sum({splits}={np.sum(splits)}. Normalizing to {splits / np.sum(splits)}'
        )
        splits /= np.sum(splits)

    file_counts = splits * nb_files
    if not np.all(file_counts[1:] >= 1):
        raise ValueError(
            f'too few files for probs - with {nb_files} files, probs should not be smaller than 1/{nb_files}={1/nb_files} but smallest prob is {np.min(splits[1:])}'
        )

    indices = dict()
    split_files = dict()
    cum_counts = np.cumsum(file_counts)
    file_counts, cum_counts, len(file_list)

    for start, end, name in zip(cum_counts[:-1], cum_counts[1:], split_names):
        indices[name] = list(range(int(np.ceil(start)), int(np.ceil(end))))
        split_files[name] = list(file_list[indices[name]])

    return split_files


def make_gaps(y: np.ndarray,
              gap_seconds: float,
              samplerate: float,
              start_seconds: Optional[List[float]] = None,
              stop_seconds: Optional[List[float]] = None) -> np.ndarray:
    """[summary]

    0011112222000111100 -> 0011100222000111100 (gap_fullwidth=2)

    Args:
        y (np.ndarray): One-hot encoded labels [T, nb_labels]
        gap_seconds (float): [description]
        samplerate (float): [description]
        start_seconds:
        stop_seconds:

    Returns:
        np.ndarray: [description]
    """
    y0 = y.copy()

    if y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)

    gap_halfwidth = int(np.floor(gap_seconds * samplerate) / 2)

    # widen gaps between adjacent syllables of different types
    a = y.copy().astype(np.float)
    label_change = np.where(np.diff(a, axis=0) != 0)[0]
    # remove on and offsets (0->label or label->0)
    onset = a[label_change] == 0
    offset = a[label_change + 1] == 0
    neither_on_nor_off = np.logical_and(~onset, ~offset)

    if np.sum(neither_on_nor_off):
        label_change = label_change[neither_on_nor_off]

        # introduce gap around label changes for adjacent syllables
        for gap_offset in range(-gap_halfwidth, gap_halfwidth + 1):
            y[label_change + gap_offset] = 0

    # one-hot-encode gapped labels
    y0[:] = 0
    for label in range(y0.shape[1]):
        y0[y == label, label] = 1

    # widen gaps between syllables of same type
    for label in range(1, y0.shape[1]):
        label_change = np.where(np.diff(y0[:, label], axis=0) != 0)[0]
        onset = y0[label_change, label] == 0
        offset = y0[label_change + 1, label] == 0

        # there is no gap before the first syll starts and after the last syll ends so ignore those
        gap_onsets = label_change[onset][1:]
        gap_offsets = label_change[offset][:-1]

        # just to be safe - remove all offsets occurring before the first onset and all onsets occurring before the last offset here
        if len(gap_offsets) > 0 and len(gap_onsets) > 0:
            gap_offsets = gap_offsets[gap_offsets > np.min(gap_onsets)]
        # need to check twice since len(gap_offsets) might change above
        if len(gap_offsets) > 0 and len(gap_onsets) > 0:
            gap_onsets = gap_onsets[gap_onsets < np.max(gap_offsets)]

        gaps = gap_onsets - gap_offsets

        for gap, gap_onset, gap_offset in zip(gaps, gap_onsets, gap_offsets):
            if gap < 2 * gap_halfwidth:
                midpoint = int(gap_offset + gap / 2)
                y0[midpoint - gap_halfwidth:midpoint + gap_halfwidth +
                   1, :] = 0

    # ensure gaps exist even when same-type segments touch
    if start_seconds is not None and stop_seconds is not None:
        start_samples = np.array(start_seconds * samplerate).astype(np.uintp)
        stop_samples = np.array(stop_seconds * samplerate).astype(np.uintp)
        for start_sample, stop_sample in zip(start_samples, stop_samples):
            y0[start_sample:int(start_sample + gap_halfwidth), :] = 0
            y0[int(stop_sample - gap_halfwidth):stop_sample, :] = 0

    return y0


def blur_events(event_trace: np.ndarray, event_std_seconds: float,
                samplerate: float) -> np.ndarray:
    """Blur event trace with a gaussian.

    Args:
        event_trace (np.ndarray): shape (N,)
        event_std_seconds (float): With of the Gaussian in seconds
        samplerate (float): sample rate of event_trace

    Returns:
        np.ndarray: blurred event trace
    """
    event_std_samples = event_std_seconds * samplerate
    win = scipy.signal.gaussian(int(event_std_samples * 8),
                                std=event_std_samples)
    event_trace = scipy.signal.convolve(event_trace.astype(float),
                                        win,
                                        mode='same')
    return event_trace
