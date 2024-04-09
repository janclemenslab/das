import numpy as np
import zarr
from typing import List, Dict, Mapping, Optional
import pandas as pd
import scipy.signal
import logging


logger = logging.getLogger(__name__)


def init_store(
    nb_channels: int,
    nb_classes: int,
    store,  # zarr store
    samplerate: Optional[float] = None,
    make_single_class_datasets: bool = False,
    class_names: List[str] = None,
    class_types: List[str] = None,
    # store_type=zarr.TempStore,
    # store_name: str = "store.zarr",
    chunk_len: int = 1_000_000,
):
    """[summary]

    Args:
        nb_channels (int): [description]
        nb_classes (int): [description]  <- should infer from class_names!
        store: zarr store
        samplerate (float, optional): [description]. Defaults to None.
        make_single_class_datasets (bool, optional): make y_suffix and attrs['class_names/types_suffix']. Defaults to None.
        class_names (List[str], optional): [description]. Defaults to None.
        class_types (List[str], optional): 'event' or 'segment'. Defaults to None.
        chunk_len (int, optional): [description]. Defaults to 1_000_000.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    if class_names is not None and nb_classes is not None and len(class_names) != nb_classes:
        raise ValueError(f"Number of classes ({nb_classes}) needs to match len(class_names) ({len(class_names)}).")
    if class_types is not None and nb_classes is not None and len(class_names) != nb_classes:
        raise ValueError(f"Number of classes ({nb_classes}) needs to match len(class_names) ({len(class_types)}).")

    # initialize the store
    root = zarr.group(store=store, overwrite=True)  # need to def the root
    for target in ["train", "val", "test"]:
        root.empty(name=f"{target}/x", shape=(0, nb_channels), chunks=(chunk_len, nb_channels), dtype=np.float16)
        root.empty(name=f"{target}/y", shape=(0, nb_classes), chunks=(chunk_len, nb_classes), dtype=np.float16)
        # root.empty(name=f'{target}/eventtimes', shape=(0, nb_classes), chunks=(1_000,), dtype=float)
        if make_single_class_datasets:
            for class_name in class_names[1:]:
                root.empty(name=f"{target}/y_{class_name}", shape=(0, 2), chunks=(chunk_len, nb_classes), dtype=np.float16)

    # init metadata - since attrs cannot be appended to, we init a dict here, populate it with information below and finaly assign it to root.attrs
    root.attrs["samplerate_x_Hz"] = samplerate
    root.attrs["samplerate_y_Hz"] = samplerate

    root.attrs["class_names"] = [str(cn) for cn in class_names]
    root.attrs["class_types"] = class_types

    if make_single_class_datasets:
        for class_name, class_type in zip(class_names[1:], class_types[1:]):
            root.attrs[f"class_names_{class_name}"] = [class_names[0], class_name]
            root.attrs[f"class_types_{class_name}"] = [class_types[0], class_type]

    for target in ["train", "val", "test"]:
        root.attrs[f"filename_startsample_{target}"] = []
        root.attrs[f"filename_endsample_{target}"] = []
        root.attrs[f"filename_{target}"] = []
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
        eventsamples = eventsamples[eventsamples < desired_len - extent]  # delete all eventsamples exceeding desired_len
    probabilities = np.zeros((desired_len, 2))
    probabilities[eventsamples, 1] = 1
    probabilities[:, 1] = np.convolve(probabilities[:, 1], np.ones((extent,)), mode="same")
    probabilities[:, 0] = 1 - probabilities[:, 1]
    return probabilities


def infer_class_info(df: pd.DataFrame):
    """[summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """
    class_names, first_indices = np.unique(df["name"], return_index=True)
    class_names = list(class_names)
    class_names.insert(0, "noise")

    # infer class type - event if start and end are the same
    class_types = ["segment"]
    for first_index in first_indices:
        if df.loc[first_index]["start_seconds"] == df.loc[first_index]["stop_seconds"]:
            class_types.append("event")
        else:
            class_types.append("segment")
    return class_names, class_types


def make_annotation_matrix(
    df: pd.DataFrame, nb_samples: int, samplerate: float, class_names: Optional[List[str]] = None
) -> np.ndarray:
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
        if row["name"] not in class_names:
            continue
        if np.all(np.isnan(row["start_seconds"])):
            continue
        class_index = class_names.index(row["name"])
        start_index = int(row["start_seconds"] * samplerate)
        stop_index = int(row["stop_seconds"] * samplerate + 1)
        if start_index < stop_index:
            class_matrix[start_index:stop_index, class_index] = 1
        else:
            logger.warning(f"{start_index} should be greater than {stop_index} for row {row}")
    return class_matrix


def normalize_probabilities(p: np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        p (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    p_song = np.sum(p[:, 1:], axis=-1)

    p[p_song > 1.0, 1:] = p[p_song > 1.0, 1:] / p_song[p_song > 1.0, np.newaxis]
    p[:, 0] = 1 - np.sum(p[:, 1:], axis=-1)
    return p


def make_gaps(
    y: np.ndarray,
    gap_seconds: float,
    samplerate: float,
    start_seconds: Optional[List[float]] = None,
    stop_seconds: Optional[List[float]] = None,
) -> np.ndarray:
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
    a = y.copy().astype(float)
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

        if len(gap_offsets) > 0 and len(gap_onsets) > 0:
            gaps = gap_onsets - gap_offsets
        else:
            gaps = []
            gap_onsets = []
            gap_offsets = []

        for gap, gap_onset, gap_offset in zip(gaps, gap_onsets, gap_offsets):
            if gap < 2 * gap_halfwidth:
                midpoint = int(gap_offset + gap / 2)
                y0[midpoint - gap_halfwidth : midpoint + gap_halfwidth + 1, :] = 0

    # ensure gaps exist even when same-type segments touch
    if start_seconds is not None and stop_seconds is not None:
        start_samples = np.array(start_seconds * samplerate).astype(np.uintp)
        stop_samples = np.array(stop_seconds * samplerate).astype(np.uintp)
        for start_sample, stop_sample in zip(start_samples, stop_samples):
            y0[start_sample : int(start_sample + gap_halfwidth), :] = 0
            y0[int(stop_sample - gap_halfwidth) : stop_sample, :] = 0

    return y0


def blur_events(event_trace: np.ndarray, event_std_seconds: float, samplerate: float) -> np.ndarray:
    """Blur event trace with a gaussian.

    Args:
        event_trace (np.ndarray): shape (N,)
        event_std_seconds (float): With of the Gaussian in seconds
        samplerate (float): sample rate of event_trace

    Returns:
        np.ndarray: blurred event trace
    """
    event_std_samples = event_std_seconds * samplerate
    win = scipy.signal.windows.gaussian(int(event_std_samples * 8), std=event_std_samples)
    event_trace = scipy.signal.convolve(event_trace.astype(float), win, mode="same")
    return event_trace
