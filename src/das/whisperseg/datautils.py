import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from copy import deepcopy
import json

from .audio_utils import WhisperSegFeatureExtractor, get_n_fft_given_sr
from ..data import resolve_training_data_dir as _resolve_training_data_dir
from ..data.audio_dir import (
    _ANNOTATION_BOUNDARY_NAMES,
    _ANNOTATION_END_NAME,
    _ANNOTATION_START_NAME,
    _READABLE_AUDIO_ALLOW_PATTERNS,
    _read_resampled_audio_chunk,
    _resampled_sample_count,
    _normalize_include_labels,
    audio_file_info,
    iter_audio_candidate_paths,
    open_audio_file,
)
from .utils import RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP

TRAIN_DATASET_ALLOW_PATTERNS = [
    *_READABLE_AUDIO_ALLOW_PATTERNS,
    "*.json",
    "*.JSON",
    "*.csv",
    "*.CSV",
]
TRAIN_DATASET_IGNORE_PATTERNS = [
    ".ipynb_checkpoints/*",
    "**/.ipynb_checkpoints/*",
]


def _whisperseg_mono_audio(audio) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError(f"Expected audio with 1 or 2 dimensions, got shape {array.shape}.")
    if array.shape[0] <= 32 < array.shape[1]:
        array = array.T
    return np.mean(array, axis=1).astype(np.float32)


def _filter_label_rows(label, keep):
    keep = np.asarray(keep, dtype=bool)
    row_count = int(len(keep))
    for key, value in list(label.items()):
        if key.startswith("_"):
            continue
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)) and len(value) == row_count:
            label[key] = [value[idx] for idx in np.flatnonzero(keep)]


def _annotation_intervals_from_markers(markers, end_seconds, frequency_scale=1.0):
    if not markers:
        return [(0.0, max(float(end_seconds), 0.0))]

    intervals = []
    current_start = None
    current_end = None
    for name, time_seconds in markers:
        time_seconds = float(time_seconds) / float(frequency_scale)
        if name == _ANNOTATION_START_NAME:
            if current_start is None:
                if current_end is not None:
                    intervals.append((0.0, current_end))
                    current_end = None
                current_start = time_seconds
            elif current_end is not None:
                intervals.append((current_start, current_end))
                current_start = time_seconds
                current_end = None
        elif name == _ANNOTATION_END_NAME:
            current_end = time_seconds if current_end is None else max(current_end, time_seconds)

    if current_start is None:
        if current_end is not None:
            intervals.append((0.0, current_end))
    elif current_end is None:
        intervals.append((current_start, float(end_seconds)))
    else:
        intervals.append((current_start, current_end))

    clipped = []
    for start, stop in intervals:
        start = max(float(start), 0.0)
        stop = min(float(stop), float(end_seconds))
        if stop > start:
            clipped.append((start, stop))
    return clipped


def _interval_overlap_mask(onsets, offsets, intervals):
    if not intervals:
        return np.zeros(len(onsets), dtype=bool)
    keep = np.zeros(len(onsets), dtype=bool)
    for start, stop in intervals:
        keep = np.logical_or(keep, np.logical_and(offsets > start, onsets < stop))
    return keep


def read_label(label_path, default_config={}, ignore_cluster=False, include_clusters=None):
    label_path = os.fspath(label_path)
    include_clusters = _normalize_include_labels(include_clusters)
    suffix = Path(label_path).suffix.lower()
    if suffix == ".json":
        with open(label_path, encoding="utf-8") as file:
            label = json.load(file)
    elif suffix == ".csv":
        label = pd.read_csv(label_path)
        label = {k: v.tolist() for k, v in label.items()}
    else:
        assert False, "Unsupported file format!"
    if "onset" not in label and "start_seconds" in label:
        label["onset"] = label["start_seconds"]
    if "offset" not in label and "stop_seconds" in label:
        label["offset"] = label["stop_seconds"]
    if "cluster" not in label and "name" in label:
        label["cluster"] = label["name"]
    assert "onset" in label and "offset" in label
    if "cluster" not in label:
        label["cluster"] = ["Vocal"] * len(label["onset"])
    label["cluster"] = list(map(str, label["cluster"]))
    marker_mask = np.asarray([cluster in _ANNOTATION_BOUNDARY_NAMES for cluster in label["cluster"]], dtype=bool)
    label["_annotation_markers"] = [
        (str(label["cluster"][idx]), float(label["onset"][idx]))
        for idx in np.flatnonzero(marker_mask)
    ]
    if np.any(marker_mask):
        _filter_label_rows(label, ~marker_mask)
    if include_clusters:
        keep = [cluster in include_clusters for cluster in label["cluster"]]
        _filter_label_rows(label, keep)

    for k in default_config:
        if k not in label:
            label[k] = default_config[k]

    ## always ignore species, since it is not actually used
    label["species"] = "unknown"

    if ignore_cluster:
        label["cluster"] = ["Vocal"] * len(label["cluster"])

    return label


def resolve_training_data_dir(data_dir):
    return _resolve_training_data_dir(data_dir)


def get_audio_and_label_paths(folder, *, audio_dataset=None, data_samplerate_hz=None):
    folder_path = Path(folder).expanduser()
    audio_list = []
    for path in iter_audio_candidate_paths(folder_path):
        try:
            audio_file_info(path, audio_dataset=audio_dataset, data_samplerate_hz=data_samplerate_hz)
        except Exception:
            continue
        audio_list.append(path)
    audio_paths = []
    label_paths = []
    for wav_path in audio_list:
        label_path = next(
            (
                path
                for path in (
                    wav_path.with_suffix(".json"),
                    wav_path.with_suffix(".JSON"),
                    wav_path.with_suffix(".csv"),
                    wav_path.with_suffix(".CSV"),
                    wav_path.parent / f"{wav_path.stem}_annotations.csv",
                    wav_path.parent / f"{wav_path.stem}_annotations.CSV",
                )
                if path.exists()
            ),
            None,
        )
        if label_path is not None:
            audio_paths.append(str(wav_path))
            label_paths.append(str(label_path))

    return audio_paths, label_paths


def determine_default_config(
    audio_paths,
    label_paths,
    total_spec_columns,
    ignore_cluster,
    include_clusters=None,
    min_frequency=None,
    frequency_scale=1.0,
    spec_time_step=None,
    audio_dataset=None,
    data_samplerate_hz=None,
):
    sr_list = []
    for audio_fname in audio_paths:
        sr_list.append(
            audio_file_info(audio_fname, audio_dataset=audio_dataset, data_samplerate_hz=data_samplerate_hz)["samplerate"]
        )
    assert len(sr_list) > 0, "No valid audios were provided."
    assert frequency_scale > 0, "frequency_scale must be positive."
    input_sr = int(np.median(sr_list))
    effective_sr = int(round(input_sr * frequency_scale))
    assert effective_sr > 0, "frequency_scale produces an effective samplerate below 1 Hz."
    n_fft = get_n_fft_given_sr(effective_sr)
    time_delta = n_fft / 2 / effective_sr

    onsets = []
    offsets = []
    for audio_fname, label_path in zip(audio_paths, label_paths):
        label = read_label(label_path, ignore_cluster=ignore_cluster, include_clusters=include_clusters)
        info = audio_file_info(audio_fname, audio_dataset=audio_dataset, data_samplerate_hz=data_samplerate_hz)
        audio_dur = (int(info["frames"]) / float(info["samplerate"])) / frequency_scale
        label_onsets = np.asarray(label["onset"], dtype=float) / frequency_scale
        label_offsets = np.asarray(label["offset"], dtype=float) / frequency_scale
        intervals = _annotation_intervals_from_markers(
            label.get("_annotation_markers", []),
            audio_dur,
            frequency_scale=frequency_scale,
        )
        keep = _interval_overlap_mask(label_onsets, label_offsets, intervals)
        label_onsets = label_onsets[keep]
        label_offsets = label_offsets[keep]

        ## assume the time stamps in the input csv/json already eliminate the half FFT blurring effect, here we need to add the blurring effect back to cope with the sampling rate used when computing the spectrogram
        corrected_onsets = [max(0, t - time_delta) for t in label_onsets]
        corrected_offsets = [min(audio_dur, t + time_delta) for t in label_offsets]

        onsets += corrected_onsets
        offsets += corrected_offsets
    onsets = np.array(onsets)
    offsets = np.array(offsets)
    assert len(onsets) > 0, "No vocal segment is annotated in the label files."
    if spec_time_step is None:
        seg_dur_median = np.median(offsets - onsets)
        scale_factor = 25
        spec_time_step = np.ceil(seg_dur_median * scale_factor / 0.5) * 0.5 / total_spec_columns
    if min_frequency is None:
        min_frequency = 0
    species = "unkown"

    return {
        "species": species,
        "sr": effective_sr,
        "input_sr": input_sr,
        "min_frequency": min_frequency,
        "frequency_scale": frequency_scale,
        "spec_time_step": spec_time_step,
    }


def get_cluster_codebook(label_paths, initial_cluster_codebook, ignore_cluster, include_clusters=None):
    cluster_codebook = deepcopy(initial_cluster_codebook)

    unique_clusters = []
    for label_file in label_paths:
        label = read_label(label_file, ignore_cluster=ignore_cluster, include_clusters=include_clusters)
        unique_clusters += [str(cluster) for cluster in label["cluster"]]

    unique_clusters = sorted(list(set(unique_clusters)))

    for cluster in unique_clusters:
        if cluster not in cluster_codebook:
            cluster_codebook[cluster] = len(cluster_codebook)
    return cluster_codebook


@dataclass
class VocalSegIntervalRecord:
    audio_path: str
    label: dict
    interval_start_sample: int
    interval_stop_sample: int
    input_samplerate: int
    audio_dataset: str | None = None
    data_samplerate_hz: float | None = None

    @property
    def num_samples(self) -> int:
        return int(self.interval_stop_sample) - int(self.interval_start_sample)


@dataclass
class VocalSegClipRecord:
    audio_path: str
    label: dict
    interval_start_sample: int
    interval_stop_sample: int
    input_samplerate: int
    padded_start_sample: int
    audio_num_samples: int
    audio_dataset: str | None = None
    data_samplerate_hz: float | None = None

    @property
    def interval_num_samples(self) -> int:
        return int(self.interval_stop_sample) - int(self.interval_start_sample)


def _label_with_rows(label, keep, *, onset, offset, cluster, cluster_id):
    keep = np.asarray(keep, dtype=bool)
    updated = deepcopy(label)
    updated.update(
        {
            "onset": np.asarray(onset, dtype=float),
            "offset": np.asarray(offset, dtype=float),
            "cluster_id": np.asarray(cluster_id),
            "cluster": [cluster[idx] for idx in np.flatnonzero(keep)],
        }
    )
    return updated


def _audio_num_samples_at_input_rate(info, input_sr: int) -> int:
    source_sr = int(info["samplerate"])
    num_samples = int(info["frames"])
    if source_sr == int(input_sr):
        return num_samples
    return _resampled_sample_count(num_samples, source_sr, int(input_sr))


def build_vocalseg_interval_records(
    audio_path_list,
    label_path_list,
    *,
    cluster_codebook=None,
    default_config={},
    ignore_cluster=False,
    include_clusters=None,
    audio_dataset=None,
    data_samplerate_hz=None,
) -> list[VocalSegIntervalRecord]:
    cluster_codebook = {} if cluster_codebook is None else cluster_codebook
    records = []
    for audio_path, label_path in zip(audio_path_list, label_path_list):
        label = read_label(
            label_path,
            default_config,
            ignore_cluster=ignore_cluster,
            include_clusters=include_clusters,
        )
        frequency_scale = float(label.get("frequency_scale", 1.0))
        input_sr = int(label.get("input_sr", label["sr"]))
        label["sr"] = int(round(input_sr * frequency_scale))
        info = audio_file_info(audio_path, audio_dataset=audio_dataset, data_samplerate_hz=data_samplerate_hz)
        num_audio_samples = _audio_num_samples_at_input_rate(info, input_sr)

        n_fft = get_n_fft_given_sr(label["sr"])
        time_delta = n_fft / 2 / label["sr"]
        audio_dur = num_audio_samples / label["sr"]
        intervals = _annotation_intervals_from_markers(
            label.get("_annotation_markers", []),
            audio_dur,
            frequency_scale=frequency_scale,
        )

        scaled_onsets = np.asarray(label["onset"], dtype=float) / frequency_scale
        scaled_offsets = np.asarray(label["offset"], dtype=float) / frequency_scale
        interval_indices = _interval_overlap_mask(scaled_onsets, scaled_offsets, intervals)
        scaled_onsets = scaled_onsets[interval_indices]
        scaled_offsets = scaled_offsets[interval_indices]
        interval_clusters = [label["cluster"][idx] for idx in np.flatnonzero(interval_indices)]

        onset_arr = np.asarray([max(0, t - time_delta) for t in scaled_onsets], dtype=float)
        offset_arr = np.asarray([min(audio_dur, t + time_delta) for t in scaled_offsets], dtype=float)
        valid_indices = np.logical_and(np.logical_and(onset_arr < audio_dur, offset_arr > 0), onset_arr <= offset_arr)
        onset_arr = onset_arr[valid_indices]
        offset_arr = offset_arr[valid_indices]
        onset_arr[onset_arr < 0] = 0
        offset_arr[offset_arr > audio_dur] = audio_dur

        cluster_values = [interval_clusters[idx] for idx in np.flatnonzero(valid_indices)]
        cluster_id_arr = np.asarray([cluster_codebook[value] for value in cluster_values])

        for interval_start, interval_stop in intervals:
            start_sample = max(int(np.ceil(interval_start * label["sr"] - 1e-9)), 0)
            stop_sample = min(int(np.floor(interval_stop * label["sr"] + 1e-9)), num_audio_samples)
            if stop_sample <= start_sample:
                continue

            keep = np.logical_and(onset_arr < interval_stop, offset_arr > interval_start)
            interval_label = _label_with_rows(
                label,
                keep,
                onset=np.maximum(onset_arr[keep], interval_start) - interval_start,
                offset=np.minimum(offset_arr[keep], interval_stop) - interval_start,
                cluster=cluster_values,
                cluster_id=cluster_id_arr[keep],
            )
            records.append(
                VocalSegIntervalRecord(
                    audio_path=str(audio_path),
                    label=interval_label,
                    interval_start_sample=start_sample,
                    interval_stop_sample=stop_sample,
                    input_samplerate=input_sr,
                    audio_dataset=audio_dataset,
                    data_samplerate_hz=data_samplerate_hz,
                )
            )
    return records


def _split_interval_record(record: VocalSegIntervalRecord, split_ratio):
    split_point = int(record.num_samples * float(split_ratio))
    split_time = split_point / record.label["sr"]
    label = record.label

    keep_part1 = np.asarray(label["onset"]) < split_time
    label_part1 = _label_with_rows(
        label,
        keep_part1,
        onset=np.asarray(label["onset"])[keep_part1],
        offset=np.minimum(np.asarray(label["offset"])[keep_part1], split_time),
        cluster=label["cluster"],
        cluster_id=np.asarray(label["cluster_id"])[keep_part1],
    )
    part1 = VocalSegIntervalRecord(
        audio_path=record.audio_path,
        label=label_part1,
        interval_start_sample=record.interval_start_sample,
        interval_stop_sample=record.interval_start_sample + split_point,
        input_samplerate=record.input_samplerate,
        audio_dataset=record.audio_dataset,
        data_samplerate_hz=record.data_samplerate_hz,
    )

    keep_part2 = np.asarray(label["offset"]) > split_time
    label_part2 = _label_with_rows(
        label,
        keep_part2,
        onset=np.maximum(np.asarray(label["onset"])[keep_part2], split_time) - split_time,
        offset=np.asarray(label["offset"])[keep_part2] - split_time,
        cluster=label["cluster"],
        cluster_id=np.asarray(label["cluster_id"])[keep_part2],
    )
    part2 = VocalSegIntervalRecord(
        audio_path=record.audio_path,
        label=label_part2,
        interval_start_sample=record.interval_start_sample + split_point,
        interval_stop_sample=record.interval_stop_sample,
        input_samplerate=record.input_samplerate,
        audio_dataset=record.audio_dataset,
        data_samplerate_hz=record.data_samplerate_hz,
    )

    if part1.num_samples / label["sr"] < 0.1:
        part1 = None
    if part2.num_samples / label["sr"] < 0.1:
        part2 = None
    return part1, part2


def _append_record(bucket, record):
    if record is not None:
        bucket.append(record)


def _random_split_interval_record(record: VocalSegIntervalRecord, split_ratio):
    mode = np.random.choice([0, 1])
    if mode == 0:
        return _split_interval_record(record, split_ratio)
    part2, part1 = _split_interval_record(record, 1 - split_ratio)
    return part1, part2


def split_vocalseg_interval_records(records, validation_fraction, test_fraction):
    train_records = []
    val_records = []
    test_records = []

    validation_fraction = float(validation_fraction)
    test_fraction = float(test_fraction)
    heldout_fraction = validation_fraction + test_fraction

    for record in records:
        train_record = record
        heldout_record = None

        if heldout_fraction > 0:
            heldout_record, train_record = _random_split_interval_record(record, heldout_fraction)

        _append_record(train_records, train_record)

        if heldout_record is None:
            continue

        if validation_fraction > 0 and test_fraction > 0:
            validation_within_heldout = validation_fraction / heldout_fraction
            val_record, test_record = _random_split_interval_record(heldout_record, validation_within_heldout)
            _append_record(val_records, val_record)
            _append_record(test_records, test_record)
        elif validation_fraction > 0:
            _append_record(val_records, heldout_record)
        else:
            _append_record(test_records, heldout_record)

    return train_records, val_records, test_records


def slice_vocalseg_interval_records(records, total_spec_columns):
    clip_records = []
    for record in records:
        label = record.label
        sr = label["sr"]
        clip_duration = total_spec_columns * label["spec_time_step"]
        num_samples_in_clip = int(np.round(clip_duration * sr))
        if num_samples_in_clip <= 0:
            continue

        padded_num_samples = num_samples_in_clip + record.num_samples
        padded_onset = np.asarray(label["onset"]) + clip_duration
        padded_offset = np.asarray(label["offset"]) + clip_duration
        cluster_id = np.asarray(label["cluster_id"])

        for pos in range(0, padded_num_samples, num_samples_in_clip):
            audio_num_samples = min(2 * num_samples_in_clip, padded_num_samples - pos)
            if audio_num_samples / sr < 0.1:
                continue

            start_time = pos / sr
            end_time = (pos + audio_num_samples) / sr
            keep = np.logical_and(padded_onset < end_time, padded_offset > start_time)
            clip_label = _label_with_rows(
                label,
                keep,
                onset=np.maximum(padded_onset[keep], start_time) - start_time,
                offset=np.minimum(padded_offset[keep], end_time) - start_time,
                cluster=label["cluster"],
                cluster_id=cluster_id[keep],
            )
            clip_records.append(
                VocalSegClipRecord(
                    audio_path=record.audio_path,
                    label=clip_label,
                    interval_start_sample=record.interval_start_sample,
                    interval_stop_sample=record.interval_stop_sample,
                    input_samplerate=record.input_samplerate,
                    padded_start_sample=pos,
                    audio_num_samples=audio_num_samples,
                    audio_dataset=record.audio_dataset,
                    data_samplerate_hz=record.data_samplerate_hz,
                )
            )
    return clip_records


def build_vocalseg_clip_records(
    audio_path_list,
    label_path_list,
    *,
    cluster_codebook=None,
    default_config={},
    ignore_cluster=False,
    include_clusters=None,
    total_spec_columns: int,
    validation_fraction: float = 0.0,
    test_fraction: float = 0.0,
    audio_dataset=None,
    data_samplerate_hz=None,
):
    interval_records = build_vocalseg_interval_records(
        audio_path_list,
        label_path_list,
        cluster_codebook=cluster_codebook,
        default_config=default_config,
        ignore_cluster=ignore_cluster,
        include_clusters=include_clusters,
        audio_dataset=audio_dataset,
        data_samplerate_hz=data_samplerate_hz,
    )
    train_records, val_records, test_records = split_vocalseg_interval_records(
        interval_records,
        validation_fraction,
        test_fraction,
    )
    return (
        slice_vocalseg_interval_records(train_records, total_spec_columns),
        slice_vocalseg_interval_records(val_records, total_spec_columns),
        slice_vocalseg_interval_records(test_records, total_spec_columns),
    )


class VocalSegDataset(Dataset):
    def __init__(self, records, tokenizer, max_length, total_spec_columns, species_codebook):
        self.records = list(records)
        self.feature_extractor_bank = self.get_feature_extractor_bank(
            [record.label for record in self.records],
            total_spec_columns,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.total_spec_columns = total_spec_columns
        self.species_codebook = species_codebook

    def get_feature_extractor_bank(self, label_list, total_spec_columns):
        max_clip_duration = max(
            [
                30,
            ]
            + [int(np.ceil(label["spec_time_step"] * total_spec_columns)) for label in label_list]
        )
        feature_extractor_bank = {}
        for label in label_list:
            key = "%s-%s-%s-%s" % (
                label["sr"],
                label["spec_time_step"],
                label["min_frequency"],
                label.get("max_frequency"),
            )
            if key not in feature_extractor_bank:
                feature_extractor_bank[key] = WhisperSegFeatureExtractor(
                    label["sr"],
                    label["spec_time_step"],
                    label["min_frequency"],
                    label.get("max_frequency"),
                    chunk_length=max_clip_duration,
                )
        return feature_extractor_bank

    def map_time_to_spec_col_index(self, t, spec_time_step):
        return min(int(np.round(t / (spec_time_step * RATIO_DECODING_TIME_STEP_TO_SPEC_TIME_STEP))), self.total_spec_columns)

    def __len__(self):
        return len(self.records)

    def _read_record_audio(self, record: VocalSegClipRecord, num_samples_in_clip: int) -> np.ndarray:
        local_start = int(record.padded_start_sample) - int(num_samples_in_clip)
        local_stop = local_start + int(record.audio_num_samples)
        read_start = max(local_start, 0)
        read_stop = min(local_stop, record.interval_num_samples)

        parts = []
        if local_start < 0:
            parts.append(np.zeros(-local_start, dtype=np.float32))

        if read_stop > read_start:
            audio_file = open_audio_file(
                record.audio_path,
                audio_dataset=record.audio_dataset,
                data_samplerate_hz=record.data_samplerate_hz,
            )
            try:
                chunk = _read_resampled_audio_chunk(
                    audio_file,
                    int(record.interval_start_sample) + int(read_start),
                    int(read_stop - read_start),
                    int(record.input_samplerate),
                )
            finally:
                audio_file.close()
            parts.append(_whisperseg_mono_audio(chunk))

        audio = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)
        if len(audio) > int(record.audio_num_samples):
            audio = audio[: int(record.audio_num_samples)]
        return audio.astype(np.float32)

    def __getitem__(self, idx):
        record = self.records[idx]
        label = record.label

        sr = label["sr"]
        spec_time_step = label["spec_time_step"]
        min_frequency = label["min_frequency"]
        feature_extractor = self.feature_extractor_bank[
            "%s-%s-%s-%s" % (sr, spec_time_step, min_frequency, label.get("max_frequency"))
        ]

        num_samples_in_clip = int(np.round(self.total_spec_columns * spec_time_step * sr))
        audio = self._read_record_audio(record, num_samples_in_clip)

        clip_start = np.random.choice(min(num_samples_in_clip + 1, len(audio) - feature_extractor.n_fft + 1))
        audio_clip = audio[clip_start : clip_start + num_samples_in_clip]

        actual_clip_duration = len(audio_clip) / sr
        start_time = clip_start / sr
        end_time = start_time + actual_clip_duration

        intersected_indices = np.logical_and(label["onset"] < end_time, label["offset"] > start_time)

        onset_in_clip = np.maximum(label["onset"][intersected_indices], start_time) - start_time
        offset_in_clip = np.minimum(label["offset"][intersected_indices], end_time) - start_time
        cluster_id_in_clip = label["cluster_id"][intersected_indices]

        """
        The following code part convert the onset, offset, and cluster_id array into label texts
        onset_timestamp + cluster_id + offset_timestamp: e.g.,
        <|zebra_finch|><|0|>7<|6|><|16|>6<|18|>
        """
        label_text = [self.species_codebook.get(label["species"], "<|unknown|>")]

        for pos in range(len(onset_in_clip)):
            label_text.append(
                "<|%d|>%d<|%d|>"
                % (
                    self.map_time_to_spec_col_index(onset_in_clip[pos], spec_time_step),
                    cluster_id_in_clip[pos],
                    self.map_time_to_spec_col_index(offset_in_clip[pos], spec_time_step),
                )
            )
        label_text = "".join(label_text)

        audio_clip = np.concatenate([audio_clip, np.zeros(num_samples_in_clip - len(audio_clip))], axis=0).astype(np.float32)
        input_features = feature_extractor(audio_clip, sampling_rate=sr, padding="do_not_pad")["input_features"][0]
        input_features = input_features[:, : self.total_spec_columns]

        if input_features.shape[1] > 0:
            min_spec_value = input_features.min()
        else:
            min_spec_value = 0
        input_features = np.concatenate(
            [
                input_features,
                min_spec_value * np.ones((input_features.shape[0], self.total_spec_columns - input_features.shape[1])),
            ],
            axis=1,
        ).astype(np.float32)

        decoder_input_ids = self.tokenizer.encode(label_text, max_length=self.max_length + 1, truncation=True, padding=True)
        labels = decoder_input_ids[1:]
        decoder_input_ids = decoder_input_ids[:-1]
        decoder_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(decoder_input_ids))
        labels += [-100] * (self.max_length - len(labels))

        return {"input_features": input_features, "decoder_input_ids": np.array(decoder_input_ids), "labels": np.array(labels)}
