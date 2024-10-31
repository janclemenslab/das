"""Code for training and evaluating networks."""

import logging
import os
import shutil
import flammkuchen
import numpy as np
from . import utils, data, models, event_utils, segment_utils, annot
from typing import List, Optional, Dict, Any, Sequence, Iterable, Union
import glob
import tensorflow
import librosa
import zarr
from tqdm.autonotebook import tqdm
import dask.config
import dask.array as da
from dask.diagnostics import ProgressBar

dask.config.set(**{"array.slicing.split_large_chunks": True})


def predict_probabilities(
    x: np.ndarray,
    model: tensorflow.keras.Model,
    params: Dict[str, Any],
    verbose: Optional[int] = 1,
    prepend_data_padding: bool = True,
):
    """[summary]

    Args:
        x ([samples, ...]): [description]
        model (tf.keras.Model): [description]
        params ([type]): [description]
        verbose (int, optional): Verbose level for predict() (see keras docs). Defaults to 1.
        prepend_data_padding (bool, optional): Restores samples that are ignored
                    in the beginning of the first and the end of the last chunk
                    because of "ignore_boundaries". Defaults to True.
    Returns:
        y_pred - output of network for each sample [samples, nb_classes]
    """

    pred_gen = data.AudioSequence(x=x, y=None, shuffle=False, **params)  # prep data
    nb_batches = len(pred_gen)
    verbose = 1
    prepend_data_padding = True

    temp_store = zarr.storage.TempStore()
    class_probabilities = zarr.zeros(
        shape=(0, params["nb_classes"]), chunks=(100_000, params["nb_classes"]), dtype="float", store=temp_store, overwrite=True
    )

    # predict and unpack batch by batch to memmaped np or zarr array
    for batch_number, batch_data in tqdm(enumerate(pred_gen), total=nb_batches, disable=verbose < 1):
        y_pred_batch = model.predict_on_batch(batch_data)  # run the network

        # reshape from [batches, nb_hist, ...] to [time, ...]
        y_pred_unpacked_batch = data.unpack_batches(y_pred_batch, pred_gen.data_padding)

        if prepend_data_padding:
            pad_width = None
            if batch_number == 0:  # to account for loss of samples at the first and last chunks
                pad_width = ((params["data_padding"], 0), (0, 0))
            elif batch_number == nb_batches - 1:
                pad_width = ((0, params["data_padding"]), (0, 0))

            if pad_width is not None:
                y_pred_unpacked_batch = np.pad(y_pred_unpacked_batch, pad_width=pad_width, mode="constant", constant_values=0)

        class_probabilities.append(y_pred_unpacked_batch, axis=0)

    class_probabilities = da.from_zarr(class_probabilities, inline_array=True)
    class_probabilities = class_probabilities.rechunk((1_000_000, params["nb_classes"]))
    class_probabilities.temp_dir = temp_store.dir_path()  #  save path to temp store so we can explicitly delete it
    return class_probabilities


def labels_from_probabilities(
    probabilities, threshold: Optional[float] = None, indices: Optional[Union[Sequence[int], slice]] = None
) -> np.ndarray:
    """Convert class-wise probabilities into labels.

    Args:
        probabilities ([type]): [samples, classes] or [samples, ]
        threshold (float, Optional): Argmax over all classes (Default, 2D - corresponds to 1/nb_classes or 0.5 if 1D).
                                     If float, each class probability is compared to the threshold.
                                     First class to cross threshold wins.
                                     If no class crosses threshold label will default to the first class.
        indices: (List[int], Optional): List of indices into axis 1 for which to compute the labels.
                                     Defaults to None (use all indices).
    Returns:
        labels [samples,] - index of "winning" dimension for each sample
    """
    if indices is None:
        indices = slice(None)  # equivalent to ":"

    if probabilities.ndim == 1:
        if threshold is None:
            threshold = 0.5
        labels = (probabilities[:, indices] > threshold).astype(np.intp)
    elif probabilities.ndim == 2:
        if threshold is None:
            labels = np.argmax(probabilities[:, indices], axis=1)
        else:
            thresholded_probabilities = probabilities[:, indices].copy()
            thresholded_probabilities[thresholded_probabilities < threshold] = 0
            labels = np.argmax(thresholded_probabilities > threshold, axis=1)
    else:
        raise ValueError(f"Probabilities have to many dimensions ({probabilities.ndim}). Can only be 1D or 2D.")

    return labels


def predict_segments(
    class_probabilities: np.ndarray,
    samplerate: float = 1.0,
    segment_dims: Optional[Sequence[int]] = None,
    segment_names: Optional[Sequence[str]] = None,
    segment_ref_onsets: Optional[List[float]] = None,
    segment_ref_offsets: Optional[List[float]] = None,
    segment_thres: float = 0.5,
    segment_minlen: Optional[float] = None,
    segment_fillgap: Optional[float] = None,
    segment_labels_by_majority: bool = True,
) -> Dict:
    """[summary]

    TODO: document different approaches for single-type vs. multi-type segment detection

    Args:
        class_probabilities ([type]): [T, nb_classes] with probabilities for each class and sample
                                      or [T,] with integer entries as class labels
        samplerate (float, optional): Hz. Defaults to 1.0.
        segment_dims (Optional[List[int]], optional): set of indices into class_probabilities corresponding
                                                      to segment-like song types.
                                                      Needs to include the noise dim.
                                                      Required to ignore event-like song types.
                                                      Defaults to None (all classes are considered segment-like).
        segment_names (Optional[List[str]], optional): Names for segment-like classes.
                                                       Defaults to None (use indices of segment-like classes).
        segment_ref_onsets (Optional[List[float]], optional):
                            Syllable onsets (in seconds) to use for estimating labels.
                            Defaults to None (will use onsets est from class_probabilitieslabels as ref).
        segment_ref_offsets (Optional[List[float]], optional): [description].
                            Syllable offsets (in seconds) to use for estimating labels.
                            Defaults to None (will use offsets est from class_probabilitieslabels as ref).
        segment_thres (float, optional): [description]. Defaults to 0.5.
        segment_minlen (Optional[float], optional): seconds. Defaults to None.
        segment_fillgap (Optional[float], optional): seconds. Defaults to None.
        segment_labels_by_majority (bool, optional): Segment labels given by majority of label values within on- and offsets.
                                                     Defaults to True.

    Returns:
        dict['segmentnames']['denselabels-samples'/'onsets'/'offsets'/'probabilities']
    """
    probs_are_labels = class_probabilities.ndim == 1
    if segment_dims is None:
        if not probs_are_labels:  # class_probabilities is [T, nb_classes]
            nb_classes = class_probabilities.shape[1]
        else:  # class_probabilities is [T,] with integer entries as class labels
            nb_classes = int(da.max(class_probabilities).compute())
        segment_dims = list(range(nb_classes))

    if segment_names is None:
        segment_names = [str(sd) for sd in segment_dims]

    segments: Dict[str, Any] = dict()
    if len(segment_dims):
        segments["samplerate_Hz"] = samplerate
        segments["index"] = segment_dims
        segments["names"] = segment_names
        if not probs_are_labels:
            segments["probabilities"] = class_probabilities[:, segment_dims]
            labels = da.map_blocks(
                labels_from_probabilities, class_probabilities, segment_thres, segment_dims, dtype=np.intp, drop_axis=1
            )
        else:
            segments["probabilities"] = None
            labels = class_probabilities

        # turn into song (0), no song (1) sequence to detect onsets (0->1) and offsets (1->0)
        song_binary = (labels > 0).astype(np.int8)
        if segment_fillgap is not None:
            song_binary = da.map_overlap(
                segment_utils.fill_gaps,
                song_binary,
                gap_dur=int(segment_fillgap * samplerate),
                depth=(int(segment_fillgap * samplerate + 1), 0),
                boundary="none",
                trim=True,
                align_arrays=True,
            )
        if segment_minlen is not None:
            song_binary = da.map_overlap(
                segment_utils.remove_short,
                song_binary,
                min_len=int(segment_minlen * samplerate),
                depth=(int(segment_minlen * samplerate + 1), 0),
                boundary="none",
                trim=True,
                align_arrays=True,
            )

        # detect syllable on- and offsets
        # pre- and post-pend 0 so we detect on and offsets at boundaries
        onsets = da.where(da.diff(song_binary, prepend=0) == 1)[0]
        offsets = da.where(da.diff(song_binary, append=0) == -1)[0]
        logging.info("   Detecting syllable on and offsets:")
        with ProgressBar(minimum=5):
            onsets, offsets = da.compute(onsets, offsets)
        segments["onsets_seconds"] = onsets.astype(float) / samplerate
        segments["offsets_seconds"] = offsets.astype(float) / samplerate

        # there is just a single segment type plus noise - in that case we use the gap-filled, short-deleted pred
        sequence: List[int] = []  # default to empty list
        if len(segment_dims) == 2:
            labels = song_binary
            # syllable-type for each syllable as int
            sequence = [str(segment_names[1])] * len(segments["offsets_seconds"])
        # if >1 segment type (plus noise) label sylls by majority vote on un-smoothed labels
        elif len(segment_dims) > 2 and segment_labels_by_majority:
            # if no refs provided, use use on/offsets from smoothed labels
            if segment_ref_onsets is None:
                segment_ref_onsets = segments["onsets_seconds"]
            if segment_ref_offsets is None:
                segment_ref_offsets = segments["offsets_seconds"]

            logging.info("   Labeling by majority:")
            if len(segment_dims) < np.iinfo("uint8").max:
                cast_to = np.uint8
            elif len(segment_dims) < np.iinfo("uint16").max:
                cast_to = np.uint16
            elif len(segment_dims) < np.iinfo("uint32").max:
                cast_to = np.uint32
            else:
                cast_to = None

            if cast_to is not None:
                logging.info(f"   Casting labels to {cast_to}:")
                labels = da.map_blocks(lambda x: x.astype(cast_to), labels, dtype=np.int16)

            with ProgressBar(minimum=5):
                labels = labels.compute()

            # syllable-type for each syllable as int
            sequence, labels = segment_utils.label_syllables_by_majority(
                labels, segment_ref_onsets, segment_ref_offsets, samplerate
            )
        segments["samples"] = labels
        segments["sequence"] = sequence
    return segments


def _detect_events_oom(
    event_probability: np.ndarray, thres: float = 0.70, min_dist: int = 100, index: int = 0, block_info=None, pad=0
) -> np.ndarray:
    """Wrapper around detect_events that returns 2D np.ndarray for use in predict_oom."""
    event_indices, event_confidence = event_utils.detect_events(event_probability, thres, min_dist, index)

    if block_info is not None:
        # add offset from position of chunk in
        chunk_start_index = block_info[None]["array-location"][0][0]
        chunk_end_index = block_info[None]["array-location"][0][1]

        event_indices += chunk_start_index
        # correct for overlap in all but the first chunk
        if chunk_start_index > 0:
            event_indices -= pad

        # delete detections in overlapping samples
        good_events = np.logical_and(event_indices >= chunk_start_index, event_indices < chunk_end_index)
        if len(good_events):
            event_confidence = event_confidence[good_events]
            event_indices = event_indices[good_events]
        else:
            event_confidence = []
            event_indices = []
    return np.stack((event_indices, event_confidence), axis=1)


def predict_events(
    class_probabilities: np.ndarray,
    samplerate: float = 1.0,
    event_dims: Optional[Iterable[int]] = None,
    event_names: Optional[Iterable[str]] = None,
    event_thres: float = 0.5,
    events_offset: float = 0,
    event_dist: float = 100,
    event_dist_min: float = 0,
    event_dist_max: float = np.inf,
) -> Dict[str, Any]:
    """[summary]

    Args:
        class_probabilities (np.ndarray): [samples, classes][description]
        samplerate (float, optional): Hz
        event_dims (List[int], optional): [description]. Defaults to np.arange(1, nb_classes).
        event_names ([type], optional): [description]. Defaults to event_dims.
        event_thres (float, optional): [description]. Defaults to 0.5.
        events_offset (float, optional): . Defaults to 0 seconds.
        event_dist (float, optional): minimal distance between events for detection (in seconds). Defaults to 100 seconds.
        event_dist_min (float, optional): minimal distance to nearest event for post detection interval filter (in seconds).
                                          Defaults to 0 seconds.
        event_dist_max (float, optional): maximal distance to nearest event for post detection interval filter (in seconds).
                                          Defaults to None (no upper limit).

    Raises:
        ValueError: [description]

    Returns:
        Dict[str, Any]
    """
    if event_dims is None:
        nb_classes = class_probabilities.shape[1]
        event_dims = np.arange(1, nb_classes)

    if event_names is None:
        event_names = [str(d) for d in event_dims]

    if event_dist_max is None:
        event_dist_max = np.inf

    events: Dict[str, Any] = dict()
    if len(event_dims):
        events["samplerate_Hz"] = samplerate
        events["index"] = event_dims
        events["names"] = event_names

        events["seconds"] = []
        events["probabilities"] = []
        events["sequence"] = []

        pad = int(event_dist * samplerate + 1)
        for event_dim, event_name in zip(event_dims, event_names):
            event_indices_and_probs = da.map_overlap(
                _detect_events_oom,
                class_probabilities,
                thres=event_thres,
                min_dist=event_dist * samplerate,
                index=event_dim,
                pad=pad,
                depth=(pad, 0),
                boundary="none",
                trim=False,
                dtype=int,
                meta=np.array(()),
            )
            with ProgressBar(minimum=5):
                event_indices_and_probs = da.compute(event_indices_and_probs)
            event_indices, event_probabilities = event_indices_and_probs[0][:, 0], event_indices_and_probs[0][:, 1]

            events_seconds = event_indices.astype(float) / samplerate
            events_seconds += events_offset

            good_event_indices = event_utils.event_interval_filter(events_seconds, event_dist_min, event_dist_max)
            events["seconds"].extend(events_seconds[good_event_indices])
            events["probabilities"].extend(event_probabilities[good_event_indices])
            events["sequence"].extend([event_name for _ in events_seconds[good_event_indices]])

    return events


def predict_song(
    class_probabilities: np.ndarray,
    params: Dict[str, Any],
    event_thres: float = 0.5,
    event_dist: float = 0.01,
    event_dist_min: float = 0,
    event_dist_max: float = np.inf,
    segment_ref_onsets: Optional[List[float]] = None,
    segment_ref_offsets: Optional[List[float]] = None,
    segment_thres: float = 0.5,
    segment_minlen: float = None,
    segment_fillgap: float = None,
):
    samplerate = params["samplerate_x_Hz"]
    events_offset = 0

    segment_dims = np.where([val == "segment" for val in params["class_types"]])[0]
    segment_names = [str(params["class_names"][segment_dim]) for segment_dim in segment_dims]
    segments = predict_segments(
        class_probabilities,
        samplerate,
        segment_dims,
        segment_names,
        segment_ref_onsets,
        segment_ref_offsets,
        segment_thres,
        segment_minlen,
        segment_fillgap,
    )

    event_dims = np.where([val == "event" for val in params["class_types"]])[0]
    event_names = [str(params["class_names"][event_dim]) for event_dim in event_dims]
    events = predict_events(
        class_probabilities,
        samplerate,
        event_dims,
        event_names,
        event_thres,
        events_offset,
        event_dist,
        event_dist_min,
        event_dist_max,
    )
    return events, segments


def predict(
    x: np.ndarray,
    model_save_name: str = None,
    verbose: int = 1,
    batch_size: int = None,
    model: models.keras.models.Model = None,
    params: Dict = None,
    event_thres: float = 0.5,
    event_dist: float = 0.01,
    event_dist_min: float = 0,
    event_dist_max: float = np.inf,
    segment_thres: float = 0.5,
    segment_use_optimized: bool = True,
    segment_minlen: float = None,
    segment_fillgap: float = None,
    pad: bool = True,
    prepend_data_padding: bool = True,
    save_memory: bool = False,
    bandpass_low_freq: float = None,
    bandpass_up_freq: float = None,
    resample: bool = True,
    fs_audio: Optional[float] = None,
):
    """[summary]

    Usage:
    Calling predict with the path to the model will load the model and the
    associated params and run inference:
    `das.predict.predict(x=data, model_save_name='tata')`

    To re-use the same model with multiple recordings, load the modal and params
    once and pass them to `predict`
    ```my_model, my_params = das.utils.load_model_and_params(model_save_name)
    for data in data_list:
        das.predict.predict(x=data, model=my_model, params=my_params)
    ```

    Args:
        x (np.array): Audio data [samples, channels]
        model_save_name (str): path with the trunk name of the model. Defaults to None.
        model (keras.model.Models): Defaults to None.
        params (dict): Defaults to None.

        verbose (int): display progress bar during prediction. Defaults to 1.
        batch_size (int): number of chunks processed at once . Defaults to None (the default used during training).
                         Larger batches lead to faster inference.
                         Limited by memory size, in particular for GPUs which typically have 8GB.
                         Large batch sizes lead to loss of samples since only complete batches are used.
        pad (bool): Append zeros to fill up batch. Otherwise the end can be cut.
                    Defaults to False

        event_thres (float): Confidence threshold for detecting peaks. Range 0..1. Defaults to 0.5.
        event_dist (float): Minimal distance between adjacent events during thresholding.
                            Prevents detecting duplicate events when the confidence trace is a little noisy.
                            Defaults to 0.01.
        event_dist_min (float): MINimal inter-event interval for the event filter run during post processing.
                                Defaults to 0.
        event_dist_max (float): MAXimal inter-event interval for the event filter run during post processing.
                                Defaults to np.inf.

        segment_thres (float): Confidence threshold for detecting segments. Range 0..1. Defaults to 0.5.
        segment_use_optimized (bool): Use minlen and fillgap values from param file if they exist.
                                      If segment_minlen and segment_fillgap are provided,
                                      then they will override the values from the param file.
                                      Defaults to True.
        segment_minlen (float): Minimal duration in seconds of a segment used for filtering out spurious detections.
                                Defaults to None.
        segment_fillgap (float): Gap in seconds between adjacent segments to be filled. Useful for correcting brief lapses.
                                 Defaults to None.

        pad (bool): prepend values (repeat last sample value) to fill the last batch.
                    Otherwise, the end of the data will not be annotated because the last, non-full batch will be skipped.
        prepend_data_padding (bool, optional): Restores samples that are ignored
                    in the beginning of the first and the end of the last chunk
                    because of "ignore_boundaries". Defaults to True.
        save_memory (bool): If true, will return memmaped dask.arrays that reside on disk for chunked computations.
                            Convert to np.arrays via the array's `compute()` function.
                            Defaults to False.
    Raises:
        ValueError: [description]

    Returns:
        events: [description]
        segments: [description]
        class_probabilities (np.array): [T, nb_classes]
        class_names (List[str]): [nb_classes]
    """

    if model_save_name is not None:
        model, params = utils.load_model_and_params(model_save_name)
    else:
        assert isinstance(model, models.keras.models.Model)
        assert isinstance(params, dict)

    fs_model = params["samplerate_x_Hz"]

    if fs_audio is not None:
        if bandpass_low_freq is not None or bandpass_up_freq is not None:
            logging.info(f"   Filtering audio between {bandpass_low_freq}Hz and {bandpass_up_freq}Hz.")
            x = utils.bandpass_filter_song(x, fs_audio, bandpass_low_freq, bandpass_up_freq)

        if resample and fs_audio != fs_model:
            logging.info(f"   Resampling. Audio rate is {fs_audio}Hz but model was trained on data with {fs_model}Hz.")
            x = utils.resample(x, fs_audio, fs_model)

    # use postprocessing values from params and/or args
    if segment_use_optimized and "post_opt" in params and isinstance(params["post_opt"], dict):
        if segment_minlen is None:
            segment_minlen = params["post_opt"]["min_len"]
        if segment_fillgap is None:
            segment_fillgap = params["post_opt"]["gap_dur"]

    if batch_size is not None:
        params["batch_size"] = batch_size

    if pad:
        # figure out length in multiples of batches

        batch_len = params["batch_size"] * params["nb_hist"] + params["nb_hist"]
        x_len_original = len(x)
        pad_len = 0
        if np.remainder(len(x), batch_len) > 0:
            pad_len = batch_len - np.remainder(len(x), batch_len)
        # pad with end val to fill
        x = np.pad(x, ((0, pad_len), (0, 0)), mode="edge")

    class_probabilities = predict_probabilities(x, model, params, verbose, prepend_data_padding)
    temp_dir = class_probabilities.temp_dir
    if pad:
        # set all song probs in padded section to zero to avoid out of bounds detections!
        # assumes that the non-song class is at index 0
        class_probabilities[-pad_len:, 1:] = 0
        class_probabilities[-pad_len:, 0] = 1
        # trim probs to original len of x
        class_probabilities = class_probabilities[:x_len_original, :]

    events, segments = predict_song(
        class_probabilities=class_probabilities,
        params=params,
        event_thres=event_thres,
        event_dist=event_dist,
        event_dist_min=event_dist_min,
        event_dist_max=event_dist_max,
        segment_ref_onsets=None,
        segment_ref_offsets=None,
        segment_thres=segment_thres,
        segment_minlen=segment_minlen,
        segment_fillgap=segment_fillgap,
    )
    if not save_memory:
        segments["probabilities"] = _to_np(segments["probabilities"])
        segments["samples"] = _to_np(segments["samples"])
        class_probabilities = _to_np(class_probabilities)
    else:
        class_probabilities.temp_dir = temp_dir
    return events, segments, class_probabilities, params["class_names"]


def _to_np(array):
    if isinstance(array, dask.array.core.Array):
        array = array.compute()
    return array


def cli_predict(
    path: str,
    model_save_name: str,
    *,
    save_filename: Optional[str] = None,
    save_format: str = "csv",
    verbose: int = 1,
    batch_size: Optional[int] = None,
    event_thres: float = 0.5,
    event_dist: float = 0.01,
    event_dist_min: float = 0,
    event_dist_max: float = np.inf,
    segment_thres: float = 0.5,
    segment_use_optimized: bool = True,
    segment_minlen: Optional[float] = None,
    segment_fillgap: Optional[float] = None,
    bandpass_low_freq: float = None,
    bandpass_up_freq: float = None,
    resample: bool = True,
):
    """Predict song labels for a wav file or a folder of wav files.

    Saves hdf5 files with keys: events, segments, class_probabilities
    OR csv files with columns: label/start_seconds/stop_seconds

    Args:
        path (str): Path to a single WAV file with the audio data or to a folder with WAV files.
        model_save_name (str): Stem of the path for the model (and parameters).
                               File to load will be MODEL_SAVE_NAME + _model.h5.
        save_filename (Optional[str]): Path to save annotations to.
                                       If omitted, will construct save_filename by stripping the extension
                                       from recording_filename and adding '_das.h5' or '_annotations.csv'.
                                       Will be ignored if `path` is a folder.
        save_format (str): 'csv' or 'h5'.
                           csv: tabular text file with label, start and end seconds for each predicted song.
                           h5: same information as in csv plus confidence values for each sample and song type.
                           Defaults to 'csv'.
        verbose (int): Display progress bar during prediction. Defaults to 1.
        batch_size (Optional[int]): Number of chunks processed at once.
                                    Defaults to None (the default used during training).

        event_thres (float): Confidence threshold for detecting events. Range 0..1. Defaults to 0.5.
        event_dist (float): Minimal distance between adjacent events during thresholding.
                            Prevents detecting duplicate events when the confidence trace is a little noisy.
                            Defaults to 0.01.
        event_dist_min (float): MINimal inter-event interval for the event filter run during post processing.
                                Defaults to 0.
        event_dist_max (float): MAXimal inter-event interval for the event filter run during post processing.
                                          Defaults to np.inf.

        segment_thres (float): Confidence threshold for detecting segments. Range 0..1. Defaults to 0.5.
        segment_use_optimized (bool): Use minlen and fillgap values from param file if they exist.
                                      If segment_minlen and segment_fillgap are provided,
                                      then they will override the values from the param file.
                                      Defaults to True.
        segment_minlen (Optional[float]): Minimal duration of a segment used for filtering out spurious detections.
                                          Defaults to None (keep all segments).
        segment_fillgap (Optional[float]): Gap between adjacent segments to be filled. Useful for correcting brief lapses.
                                           Defaults to None (do not fill gaps).

        bandpass_low_freq (float): Lower cutoff frequency in Hz for bandpass filtering audio data. Defaults to 1.0.
        bandpass_up_freq (float): Upper cutoff frequency in Hz for bandpass filtering audio data. Defaults to samplingrate / 2.

        resample (bool): Resample audio data to the rate expected by the model. Defaults to True.

    Raises:
        ValueError on unknown save_format
    """
    if not (save_format == "csv" or save_format == "h5"):
        raise ValueError(f"Unknown save_format '{save_format}'. Should be either 'csv' or 'h5'.")

    if os.path.isdir(path) and save_filename is not None:
        logging.warning(f"{path} is a folder. Will ignore save_filename argument {save_filename}.")

    # if path is folder: glob contents - all files
    if os.path.isdir(path):
        filenames = glob.glob(f"{path}/*.wav")
        filenames = [filename for filename in filenames if not os.path.isdir(filename)]
    elif os.path.isfile(path):
        filenames = [path]

    logging.info(f"Loading model from {model_save_name}.")
    model, params = utils.load_model_and_params(model_save_name)
    fs_model = params["samplerate_x_Hz"]

    for recording_filename in filenames:
        logging.info(f"   Loading data from {recording_filename}.")
        try:
            # else if path is file - predict only on file but make it single-item list
            x, fs_audio = librosa.load(recording_filename, sr=None, mono=False)
            x = x.T  # [channels, time] -> [time, channels]
            if x.ndim == 1:
                x = x[:, np.newaxis]

            if bandpass_low_freq is not None or bandpass_up_freq is not None:
                logging.info(f"   Filtering audio between {bandpass_low_freq}Hz and {bandpass_up_freq}Hz.")
                x = utils.bandpass_filter_song(x, fs_audio, bandpass_low_freq, bandpass_up_freq)

            if resample and fs_audio != fs_model:
                logging.info(f"   Resampling. Audio rate is {fs_audio}Hz but model was trained on data with {fs_model}Hz.")
                x = utils.resample(x, fs_audio, fs_model)

            logging.info(f"   Annotating using model at {model_save_name}.")
            # TODO: load model once, provide as direct arg
            events, segments, class_probabilities, class_names = predict(
                x,
                None,
                verbose,
                batch_size,
                model,
                params,
                event_thres,
                event_dist,
                event_dist_min,
                event_dist_max,
                segment_thres,
                segment_use_optimized,
                segment_minlen,
                segment_fillgap,
                save_memory=True,
            )

            if "event" in params["class_types"]:
                logging.info(f"   found {len(events['seconds'])} instances of events '{list(set(events['sequence']))}'.")
            if "segment" in params["class_types"]:
                logging.info(
                    f"   found {len(segments['onsets_seconds'])} instances of segments '{list(set(segments['sequence']))}'."
                )

            if save_format == "h5":
                # turn events and segments into df!
                d = {
                    "events": events,
                    "segments": segments,
                    "class_probabilities": class_probabilities,
                    "class_names": class_names,
                }
                if save_filename is None:
                    save_filename = os.path.splitext(recording_filename)[0] + "_das.h5"
                logging.info(f"   Saving results to {save_filename}.")
                flammkuchen.save(save_filename, d)
                logging.info("Done.")
            elif save_format == "csv":
                evt = annot.Events.from_predict(events, segments)
                if save_filename is None:
                    save_filename = os.path.splitext(recording_filename)[0] + "_annotations.csv"
                logging.info(f"   Saving results to {save_filename}.")
                evt.to_df().to_csv(save_filename)
                logging.info("Done.")
            # reset
            if hasattr(class_probabilities, "temp_dir"):
                temp_dir = class_probabilities.temp_dir
                del class_probabilities
                shutil.rmtree(temp_dir, ignore_errors=True)

            if os.path.isdir(path):
                save_filename = None
        except Exception:
            logging.exception(f"Error processing file {recording_filename}.")
