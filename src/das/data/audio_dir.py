import json
import math
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import lightning as L
import numpy as np
import pandas as pd
import soundfile
import torch
from torch.utils.data import DataLoader, Dataset


def _resolve_chunk_stride(num_time_steps: int, chunk_stride: int | None) -> int:
    if chunk_stride is None:
        return max(1, int(num_time_steps) // 2)
    stride = int(chunk_stride)
    if stride <= 0:
        raise ValueError("chunk_stride must be > 0.")
    return stride


def _compute_chunk_starts(num_samples: int, num_time_steps: int, chunk_stride: int) -> np.ndarray:
    if num_samples <= 0:
        return np.zeros((0,), dtype=np.int64)
    if num_samples <= num_time_steps:
        return np.array([0], dtype=np.int64)

    last_start = int(num_samples - num_time_steps)
    starts = np.arange(0, last_start + 1, chunk_stride, dtype=np.int64)
    if starts[-1] != last_start:
        starts = np.concatenate([starts, np.array([last_start], dtype=np.int64)])
    return starts


def _split_lengths(total: int, ratios: Sequence[float]) -> list[int]:
    lengths = [int(total * ratio) for ratio in ratios]
    remainder = int(total) - sum(lengths)
    for idx in range(remainder):
        lengths[idx % len(lengths)] += 1
    return lengths


def _drop_split_boundary_overlaps_with_lengths(
    split_starts: Sequence[np.ndarray],
    split_lengths: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    trimmed_starts = []
    trimmed_lengths = []
    previous_stop = 0
    for starts, lengths in zip(split_starts, split_lengths, strict=True):
        keep = starts >= previous_stop
        starts = starts[keep]
        lengths = lengths[keep]
        trimmed_starts.append(starts)
        trimmed_lengths.append(lengths)
        if len(starts) > 0:
            previous_stop = int(starts[-1]) + int(lengths[-1])
    return trimmed_starts, trimmed_lengths


def _label_frame_count(num_samples: int, hop_samples: int) -> int:
    if num_samples <= 0:
        return 0
    if hop_samples <= 1:
        return int(num_samples)
    return int(num_samples // hop_samples + 1)


def _resampled_sample_count(num_samples: int, source_samplerate: int, target_samplerate: int) -> int:
    if num_samples <= 0:
        return 0
    return max(1, int(round(float(num_samples) * float(target_samplerate) / float(source_samplerate))))


def _resample_audio_array(
    audio: np.ndarray,
    source_samplerate: int,
    target_samplerate: int,
    *,
    axis: int = 0,
) -> np.ndarray:
    source_samplerate = int(round(source_samplerate))
    target_samplerate = int(round(target_samplerate))
    array = np.asarray(audio, dtype=np.float32)
    if source_samplerate == target_samplerate:
        return array

    from scipy.signal import resample_poly

    common = math.gcd(source_samplerate, target_samplerate)
    resampled = resample_poly(array, target_samplerate // common, source_samplerate // common, axis=axis)
    return np.asarray(resampled, dtype=np.float32)


def _normalized_time_length(chunk: np.ndarray) -> int:
    return int(chunk.shape[-1] if chunk.ndim == 2 else chunk.shape[0])


def _slice_normalized_time(chunk: np.ndarray, start: int, stop: int) -> np.ndarray:
    if chunk.ndim == 2:
        return chunk[:, start:stop]
    return chunk[start:stop]


def _pad_or_trim_normalized_time(chunk: np.ndarray, frames: int) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    current = _normalized_time_length(chunk)
    if current > frames:
        return _slice_normalized_time(chunk, 0, frames)
    if current == frames:
        return chunk
    if chunk.ndim == 2:
        pad = np.zeros((chunk.shape[0], frames - current), dtype=np.float32)
        return np.concatenate([chunk, pad], axis=1)
    return np.concatenate([chunk, np.zeros(frames - current, dtype=np.float32)])


def _read_resampled_audio_chunk(audio_file, start: int, frames: int, target_samplerate: int | None) -> np.ndarray:
    source_samplerate = int(audio_file.samplerate)
    if target_samplerate is None or int(target_samplerate) == source_samplerate:
        return audio_file.read(start, frames)

    target_samplerate = int(target_samplerate)
    native_start = max(int(math.floor(float(start) * source_samplerate / target_samplerate)), 0)
    native_stop = int(math.ceil(float(start + frames) * source_samplerate / target_samplerate))
    native_frames = max(native_stop - native_start, 1)
    chunk = audio_file.read(native_start, native_frames)
    axis = 1 if chunk.ndim == 2 else 0
    chunk = _resample_audio_array(chunk, source_samplerate, target_samplerate, axis=axis)

    resampled_start = int(round(float(native_start) * target_samplerate / source_samplerate))
    offset = int(start - resampled_start)
    if offset < 0:
        if chunk.ndim == 2:
            chunk = np.concatenate([np.zeros((chunk.shape[0], -offset), dtype=np.float32), chunk], axis=1)
        else:
            chunk = np.concatenate([np.zeros(-offset, dtype=np.float32), chunk])
        offset = 0
    chunk = _slice_normalized_time(chunk, offset, offset + frames)
    return _pad_or_trim_normalized_time(chunk, frames)


_AUDIO_KEYS = ("audio", "data", "samples")
_SAMPLERATE_KEYS = ("sample_rate", "sample_rate_hz", "samplerate", "samplerate_hz", "fs", "sr", "rate")
_FILEPATH_COLUMN = "filepath"
_ANNOTATION_START_NAME = "annotation_start"
_ANNOTATION_END_NAME = "annotation_end"
_ANNOTATION_BOUNDARY_NAMES = {_ANNOTATION_START_NAME, _ANNOTATION_END_NAME}
_READABLE_AUDIO_ALLOW_PATTERNS = [
    "*.wav",
    "*.WAV",
    "*.flac",
    "*.FLAC",
    "*.aiff",
    "*.AIFF",
    "*.aif",
    "*.AIF",
    "*.h5",
    "*.H5",
    "*.hdf5",
    "*.HDF5",
    "*.mat",
    "*.MAT",
    "*.zarr",
    "*.ZARR",
    "*.npz",
    "*.NPZ",
    "*.npy",
    "*.NPY",
    "*.mmap",
    "*.MMAP",
    "*.audio.json",
]
_TRAIN_DATASET_ALLOW_PATTERNS = [*_READABLE_AUDIO_ALLOW_PATTERNS, "*.json", "*.JSON", "*.csv", "*.CSV"]
_TRAIN_DATASET_IGNORE_PATTERNS = [".ipynb_checkpoints/*", "**/.ipynb_checkpoints/*"]


def audio_sidecar_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}.audio.json")


def load_audio_sidecar(path: str | Path) -> dict[str, Any]:
    sidecar = audio_sidecar_path(path)
    if not sidecar.exists():
        return {}
    with sidecar.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected audio sidecar '{sidecar}' to contain a JSON object.")
    return payload


def _audio_open_options(
    path: str | Path,
    *,
    audio_dataset: str | None = None,
    data_samplerate_hz: float | None = None,
) -> dict[str, Any]:
    sidecar = load_audio_sidecar(path)
    return {
        "audio_dataset": audio_dataset if audio_dataset is not None else sidecar.get("audio_dataset"),
        "samplerate": data_samplerate_hz if data_samplerate_hz is not None else sidecar.get("samplerate_hz"),
        "dtype": sidecar.get("dtype"),
        "shape": sidecar.get("shape"),
        "scale": sidecar.get("scale"),
    }


def _normalized_key(key: object) -> str:
    return str(key).lower()


def _read_scalar(value) -> float:
    if isinstance(value, h5py.Dataset) or (hasattr(value, "shape") and hasattr(value, "__getitem__") and not np.isscalar(value)):
        value = value[()]
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected a scalar sample rate.")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode()
    return float(item)


def _find_mapping_item(mapping, names: Sequence[str]):
    wanted = {_normalized_key(name) for name in names}
    for key in mapping.keys():
        if _normalized_key(key) in wanted:
            return mapping[key]
    return None


def _mapping_get(mapping, key: str):
    try:
        return mapping[key]
    except Exception:
        return None


def _find_audio_dataset(mapping, audio_dataset: str | None = None):
    if audio_dataset:
        dataset = _mapping_get(mapping, audio_dataset)
        if dataset is None:
            raise ValueError(f"Audio dataset '{audio_dataset}' was not found.")
        if getattr(dataset, "ndim", None) in (1, 2):
            return dataset
        raise ValueError(f"Audio dataset '{audio_dataset}' must be 1D or 2D.")

    if getattr(mapping, "ndim", None) in (1, 2):
        return mapping

    wanted = {_normalized_key(name) for name in _AUDIO_KEYS}
    for key in mapping.keys():
        dataset = mapping[key]
        if _normalized_key(key) in wanted and getattr(dataset, "ndim", None) in (1, 2):
            return dataset
    raise ValueError(f"Expected one of {_AUDIO_KEYS} to be a 1D or 2D audio dataset.")


def _find_samplerate(mapping, audio_dataset, samplerate: float | None = None) -> int:
    if samplerate is not None:
        return int(round(float(samplerate)))
    for attrs in (getattr(mapping, "attrs", {}), getattr(audio_dataset, "attrs", {})):
        value = _find_mapping_item(attrs, _SAMPLERATE_KEYS)
        if value is not None:
            return int(round(_read_scalar(value)))

    value = _find_mapping_item(mapping, _SAMPLERATE_KEYS)
    if value is not None:
        return int(round(_read_scalar(value)))

    raise ValueError(f"Expected a sample rate in attrs or one of {_SAMPLERATE_KEYS}.")


def _infer_h5_time_axis(shape: tuple[int, ...]) -> int:
    if len(shape) == 1:
        return 0
    if shape[0] <= 32 < shape[1]:
        return 1
    return 0


def _normalize_chunk(chunk: np.ndarray, num_time_steps: int) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    if len(chunk) < num_time_steps:
        pad_shape = (num_time_steps - len(chunk),)
        if getattr(chunk, "ndim", 1) > 1:
            pad_shape = pad_shape + chunk.shape[1:]
        pad_chunk = np.zeros(pad_shape, dtype=np.float32)
        chunk = np.concatenate([chunk, pad_chunk], axis=0)

    if getattr(chunk, "ndim", 1) == 2 and chunk.shape[1] == 1:
        chunk = chunk[:, 0]
    elif getattr(chunk, "ndim", 1) == 2:
        chunk = chunk.transpose(1, 0)
    return chunk


class SoundFileAudio:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.audio = soundfile.SoundFile(self.path)
        self.samplerate = int(self.audio.samplerate)
        self.channels = int(self.audio.channels)

    def __len__(self):
        return len(self.audio)

    def read(self, start: int, frames: int) -> np.ndarray:
        self.audio.seek(start)
        return _normalize_chunk(self.audio.read(frames=frames, dtype="float32"), frames)

    def close(self) -> None:
        self.audio.close()


class MappingAudio:
    def __init__(self, path: str | Path, mapping, *, audio_dataset: str | None = None, samplerate: float | None = None):
        self.path = Path(path)
        self.handle = mapping
        try:
            self.dataset = _find_audio_dataset(self.handle, audio_dataset=audio_dataset)
            self.samplerate = _find_samplerate(self.handle, self.dataset, samplerate=samplerate)
            self.time_axis = _infer_h5_time_axis(self.dataset.shape)
            self.channels = 1 if self.dataset.ndim == 1 else int(self.dataset.shape[1 - self.time_axis])
        except Exception:
            self.close()
            raise

    def __len__(self):
        return int(self.dataset.shape[self.time_axis])

    def read(self, start: int, frames: int) -> np.ndarray:
        stop = start + frames
        if self.dataset.ndim == 1:
            chunk = self.dataset[start:stop]
        elif self.time_axis == 0:
            chunk = self.dataset[start:stop, :]
        else:
            chunk = self.dataset[:, start:stop].T
        return _normalize_chunk(chunk, frames)

    def close(self) -> None:
        close = getattr(self.handle, "close", None)
        if close is not None:
            close()


class H5Audio(MappingAudio):
    def __init__(self, path: str | Path, *, audio_dataset: str | None = None, samplerate: float | None = None):
        super().__init__(
            path,
            h5py.File(Path(path), "r"),
            audio_dataset=audio_dataset,
            samplerate=samplerate,
        )


class ZarrAudio(MappingAudio):
    def __init__(self, path: str | Path, *, audio_dataset: str | None = None, samplerate: float | None = None):
        import zarr

        super().__init__(
            path,
            zarr.open(Path(path), mode="r"),
            audio_dataset=audio_dataset,
            samplerate=samplerate,
        )


class NumpyArrayAudio:
    def __init__(self, path: str | Path, array, *, samplerate: float | None = None, scale: float | None = None):
        if samplerate is None:
            raise ValueError(f"Audio file '{path}' requires samplerate_hz in a sidecar or data_samplerate_hz config.")
        self.path = Path(path)
        self.array = array
        if self.array.ndim not in (1, 2):
            raise ValueError(f"Expected audio array to be 1D or 2D, got shape {self.array.shape}.")
        self.samplerate = int(round(float(samplerate)))
        self.time_axis = _infer_h5_time_axis(self.array.shape)
        self.channels = 1 if self.array.ndim == 1 else int(self.array.shape[1 - self.time_axis])
        self.scale = None if scale is None else float(scale)

    def __len__(self):
        return int(self.array.shape[self.time_axis])

    def read(self, start: int, frames: int) -> np.ndarray:
        stop = start + frames
        if self.array.ndim == 1:
            chunk = self.array[start:stop]
        elif self.time_axis == 0:
            chunk = self.array[start:stop, :]
        else:
            chunk = self.array[:, start:stop].T
        chunk = np.asarray(chunk, dtype=np.float32)
        if self.scale is not None:
            chunk = chunk / self.scale
        return _normalize_chunk(chunk, frames)

    def close(self) -> None:
        close = getattr(self.array, "close", None)
        if close is not None:
            close()


class NPZAudio(NumpyArrayAudio):
    def __init__(self, path: str | Path, *, audio_dataset: str | None = None, samplerate: float | None = None, scale: float | None = None):
        self.handle = np.load(path)
        try:
            data_key = audio_dataset
            if data_key is None:
                wanted = {_normalized_key(name) for name in _AUDIO_KEYS}
                data_key = next((key for key in self.handle.files if _normalized_key(key) in wanted), None)
            if data_key is None or data_key not in self.handle.files:
                raise ValueError(f"Expected one of {_AUDIO_KEYS} in '{path}'.")

            if samplerate is None:
                for key in self.handle.files:
                    if _normalized_key(key) in {_normalized_key(name) for name in _SAMPLERATE_KEYS}:
                        samplerate = _read_scalar(self.handle[key])
                        break
            super().__init__(path, self.handle[data_key], samplerate=samplerate, scale=scale)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self.handle.close()


class NPYAudio(NumpyArrayAudio):
    def __init__(self, path: str | Path, *, samplerate: float | None = None, scale: float | None = None):
        super().__init__(path, np.load(path, mmap_mode="r"), samplerate=samplerate, scale=scale)


class MmapAudio(NumpyArrayAudio):
    def __init__(self, path: str | Path, *, dtype, shape, samplerate: float | None = None, scale: float | None = None):
        if dtype is None or shape is None:
            raise ValueError(f"mmap audio file '{path}' requires dtype and shape in a sidecar.")
        shape = tuple(int(value) for value in shape)
        if len(shape) not in (1, 2):
            raise ValueError(f"mmap audio shape must be [samples] or [samples, channels], got {shape}.")
        super().__init__(
            path,
            np.memmap(path, mode="r", dtype=np.dtype(dtype), shape=shape),
            samplerate=samplerate,
            scale=scale,
        )


def open_audio_file(
    path: str | Path,
    *,
    audio_dataset: str | None = None,
    data_samplerate_hz: float | None = None,
):
    path = Path(path)
    options = _audio_open_options(path, audio_dataset=audio_dataset, data_samplerate_hz=data_samplerate_hz)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return NPZAudio(
            path,
            audio_dataset=options["audio_dataset"],
            samplerate=options["samplerate"],
            scale=options["scale"],
        )
    if suffix == ".npy":
        return NPYAudio(path, samplerate=options["samplerate"], scale=options["scale"])
    if suffix == ".mmap":
        return MmapAudio(
            path,
            dtype=options["dtype"],
            shape=options["shape"],
            samplerate=options["samplerate"],
            scale=options["scale"],
        )
    if suffix == ".zarr" and path.is_dir():
        return ZarrAudio(path, audio_dataset=options["audio_dataset"], samplerate=options["samplerate"])

    try:
        return SoundFileAudio(path)
    except Exception as soundfile_exc:
        try:
            if h5py.is_hdf5(path):
                return H5Audio(path, audio_dataset=options["audio_dataset"], samplerate=options["samplerate"])
        except Exception as h5_exc:
            raise h5_exc from soundfile_exc
        raise soundfile_exc


def audio_file_info(path: str | Path, *, audio_dataset: str | None = None, data_samplerate_hz: float | None = None):
    audio = open_audio_file(path, audio_dataset=audio_dataset, data_samplerate_hz=data_samplerate_hz)
    try:
        return {
            "frames": len(audio),
            "samplerate": int(audio.samplerate),
            "channels": int(audio.channels),
        }
    finally:
        audio.close()


def load_audio_array(path: str | Path, *, audio_dataset: str | None = None, data_samplerate_hz: float | None = None) -> tuple[np.ndarray, int]:
    audio = open_audio_file(path, audio_dataset=audio_dataset, data_samplerate_hz=data_samplerate_hz)
    try:
        return audio.read(0, len(audio)), int(audio.samplerate)
    finally:
        audio.close()


def _is_zarr_store_path(path: Path) -> bool:
    return path.suffix.lower() == ".zarr" and path.is_dir()


def iter_audio_candidate_paths(root: str | Path) -> list[Path]:
    root = Path(root).expanduser()
    if root.is_file() or _is_zarr_store_path(root):
        return [root]
    if not root.is_dir():
        return []

    candidates = []
    zarr_roots = set()
    for path in sorted(root.rglob("*")):
        if any(parent in zarr_roots for parent in path.parents):
            continue
        if _is_zarr_store_path(path):
            zarr_roots.add(path)
            candidates.append(path)
        elif path.is_file():
            candidates.append(path)
    return candidates


def resolve_training_data_dir(data_dir: str | Path) -> str:
    data_dir_text = os.fspath(data_dir)
    data_path = Path(data_dir_text).expanduser()
    if data_path.exists():
        return str(data_path)

    from ..config import TRAIN_DATASET_HF_OPTIONS

    is_hf_dataset = data_dir_text in TRAIN_DATASET_HF_OPTIONS or (
        not data_path.is_absolute()
        and data_dir_text.count("/") == 1
        and not data_dir_text.startswith((".", "~"))
    )
    if is_hf_dataset:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=data_dir_text,
            repo_type="dataset",
            allow_patterns=_TRAIN_DATASET_ALLOW_PATTERNS,
            ignore_patterns=_TRAIN_DATASET_IGNORE_PATTERNS,
        )
    return str(data_path)


def _resolve_annotation_audio_path(value, annotation_file: str | Path) -> Path | None:
    if pd.isna(value):
        return None
    filepath = str(value).strip()
    if not filepath:
        return None
    path = Path(filepath).expanduser()
    if not path.is_absolute():
        path = Path(annotation_file).parent / path
    return path.resolve()


def _normalize_annotation_table(annotations: pd.DataFrame) -> pd.DataFrame:
    normalized = annotations.copy()
    if "name" not in normalized.columns and "cluster" in normalized.columns:
        normalized["name"] = normalized["cluster"]
    if "name" not in normalized.columns and {"onset", "offset"}.issubset(normalized.columns):
        normalized["name"] = "Vocal"
    if "start_seconds" not in normalized.columns and "onset" in normalized.columns:
        normalized["start_seconds"] = normalized["onset"]
    if "stop_seconds" not in normalized.columns and "offset" in normalized.columns:
        normalized["stop_seconds"] = normalized["offset"]
    if "name" in normalized.columns:
        normalized["name"] = normalized["name"].fillna("Vocal").map(str)
    return normalized


def _annotation_boundary_mask(annotations: pd.DataFrame) -> pd.Series:
    if "name" not in annotations.columns:
        return pd.Series(False, index=annotations.index)
    return annotations["name"].isin(_ANNOTATION_BOUNDARY_NAMES)


def _drop_annotation_boundary_rows(annotations: pd.DataFrame) -> pd.DataFrame:
    if annotations.empty:
        return annotations.copy()
    return annotations.loc[~_annotation_boundary_mask(annotations)].copy()


def _annotation_boundary_time(row) -> float:
    if hasattr(row, "start_seconds"):
        return float(row.start_seconds)
    return float(row.onset)


def _annotation_intervals_seconds(annotations: pd.DataFrame, end_seconds: float) -> list[tuple[float, float]]:
    markers = []
    if "name" in annotations.columns and ("start_seconds" in annotations.columns or "onset" in annotations.columns):
        for row in annotations[_annotation_boundary_mask(annotations)].itertuples(index=False):
            markers.append((str(row.name), _annotation_boundary_time(row)))

    if not markers:
        return [(0.0, max(float(end_seconds), 0.0))]

    intervals = []
    current_start = None
    current_end = None
    for name, time_seconds in markers:
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

    return intervals


def _annotation_intervals_samples(
    annotations: pd.DataFrame,
    *,
    num_samples: int,
    samplerate: int,
) -> list[tuple[int, int]]:
    duration_seconds = float(num_samples) / float(samplerate)
    intervals = []
    for start_seconds, stop_seconds in _annotation_intervals_seconds(annotations, duration_seconds):
        start = max(int(np.ceil(float(start_seconds) * samplerate - 1e-9)), 0)
        stop = min(int(np.floor(float(stop_seconds) * samplerate + 1e-9)), int(num_samples))
        if stop > start:
            intervals.append((start, stop))
    return intervals


def _chunk_starts_and_lengths_for_intervals(
    intervals: Sequence[tuple[int, int]],
    num_time_steps: int,
    chunk_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    lengths = []
    for interval_start, interval_stop in intervals:
        interval_length = int(interval_stop) - int(interval_start)
        if interval_length <= 0:
            continue
        for local_start in _compute_chunk_starts(interval_length, num_time_steps, chunk_stride):
            start = int(interval_start) + int(local_start)
            length = min(int(num_time_steps), int(interval_stop) - start)
            if length > 0:
                starts.append(start)
                lengths.append(length)
    return np.asarray(starts, dtype=np.int64), np.asarray(lengths, dtype=np.int64)


def read_annotation_file(annotation_file: str | Path) -> pd.DataFrame:
    annotation_file = Path(annotation_file)
    if annotation_file.suffix.lower() == ".json":
        with annotation_file.open(encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected '{annotation_file}' to contain a JSON object.")
        onsets = payload.get("onset", [])
        offsets = payload.get("offset", [])
        clusters = payload.get("cluster", ["Vocal"] * len(onsets))
        annotations = pd.DataFrame({"onset": onsets, "offset": offsets, "cluster": clusters})
    else:
        annotations = pd.read_csv(annotation_file)
    return _normalize_annotation_table(annotations)


def _read_annotation_tables(annotation_files: Sequence[Path]) -> dict[Path, pd.DataFrame]:
    tables = {}
    for annotation_file in annotation_files:
        try:
            tables[Path(annotation_file)] = read_annotation_file(annotation_file)
        except Exception:
            pass
    return tables


def _audio_files_from_filepath_annotations(annotation_tables: dict[Path, pd.DataFrame]) -> list[Path]:
    audio_files = []
    seen = set()
    for annotation_file, annotations in annotation_tables.items():
        if _FILEPATH_COLUMN not in annotations.columns:
            continue
        for value in annotations[_FILEPATH_COLUMN].dropna().unique():
            audio_file = _resolve_annotation_audio_path(value, annotation_file)
            if audio_file is None or audio_file in seen:
                continue
            seen.add(audio_file)
            audio_files.append(audio_file)
    return audio_files


def _filter_annotations_for_audio(annotations: pd.DataFrame, annotation_file: str | Path, audio_file: str | Path) -> pd.DataFrame:
    if _FILEPATH_COLUMN not in annotations.columns:
        return annotations.copy()

    target = Path(audio_file).expanduser().resolve()
    mask = annotations[_FILEPATH_COLUMN].map(lambda value: _resolve_annotation_audio_path(value, annotation_file) == target)
    return annotations[mask].copy()


def _annotation_table(annotation_file: Path, annotation_tables: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    if annotation_file not in annotation_tables:
        annotation_tables[annotation_file] = read_annotation_file(annotation_file)
    return annotation_tables[annotation_file]


def _paired_annotation_files(audio_file: Path) -> list[Path]:
    candidates = [
        audio_file.parent / f"{audio_file.stem}_annotations.csv",
        audio_file.parent / f"{audio_file.stem}_annotations.CSV",
        audio_file.with_suffix(".csv"),
        audio_file.with_suffix(".CSV"),
        audio_file.with_suffix(".json"),
        audio_file.with_suffix(".JSON"),
    ]
    seen = set()
    existing = []
    for candidate in candidates:
        resolved = str(candidate.expanduser().resolve()).lower()
        if resolved in seen or not candidate.exists():
            continue
        seen.add(resolved)
        existing.append(candidate)
    return existing


def _collect_annotations_for_audio(
    audio_file: str | Path,
    annotation_tables: dict[Path, pd.DataFrame],
) -> tuple[list[Path], pd.DataFrame] | None:
    audio_file = Path(audio_file)
    sources = []
    annotation_frames = []
    paired_annotation_files = _paired_annotation_files(audio_file)

    for paired_annotation_file in paired_annotation_files:
        annotations = _annotation_table(paired_annotation_file, annotation_tables)
        filtered = _filter_annotations_for_audio(annotations, paired_annotation_file, audio_file)
        if _FILEPATH_COLUMN not in annotations.columns or not filtered.empty:
            sources.append(paired_annotation_file)
            annotation_frames.append(filtered)

    for annotation_file, annotations in annotation_tables.items():
        if annotation_file in paired_annotation_files or _FILEPATH_COLUMN not in annotations.columns:
            continue
        filtered = _filter_annotations_for_audio(annotations, annotation_file, audio_file)
        if filtered.empty:
            continue
        sources.append(annotation_file)
        annotation_frames.append(filtered)

    if not annotation_frames:
        return None
    return sources, pd.concat(annotation_frames, ignore_index=True)


def _collapse_annotation_class_names(annotations: pd.DataFrame) -> pd.DataFrame:
    collapsed = annotations.copy()
    collapsed["name"] = "song"
    return collapsed


def _normalize_include_labels(include_labels: Optional[Sequence[str]] = None) -> list[str]:
    if include_labels is None:
        return []
    values = include_labels.replace("\n", ",").split(",") if isinstance(include_labels, str) else include_labels
    labels = []
    seen = set()
    for label in values:
        label = str(label).strip()
        if not label or label == "noise" or label in seen:
            continue
        labels.append(label)
        seen.add(label)
    return labels


def _filter_annotations_for_include_labels(annotations: pd.DataFrame, include_labels: Sequence[str]) -> pd.DataFrame:
    if not include_labels:
        return annotations.copy()
    return annotations.loc[annotations["name"].isin(include_labels)].copy()


def _validate_include_labels(include_labels: Sequence[str], available_labels: set[str]) -> None:
    if not include_labels:
        return
    unknown = [label for label in include_labels if label not in available_labels]
    if unknown:
        available = ", ".join(sorted(available_labels)) or "none"
        raise ValueError(
            "include_labels contains labels not found in the training data: "
            f"{', '.join(unknown)}. Available labels: {available}."
        )


def _normalize_class_types(*, class_names: Sequence[str], class_types) -> list[str]:
    if class_types is None:
        return ["segment"] * len(class_names)
    normalized = [str(class_type) for class_type in class_types]
    return normalized if len(normalized) == len(class_names) else ["segment"] * len(class_names)


def _update_event_label_evidence(evidence: dict[str, bool], annotations: pd.DataFrame) -> None:
    if annotations.empty or not {"name", "start_seconds", "stop_seconds"}.issubset(annotations.columns):
        return
    for row in annotations.itertuples(index=False):
        name = str(row.name)
        is_event = float(row.stop_seconds) <= float(row.start_seconds)
        evidence[name] = evidence.get(name, True) and is_event


def _class_types_from_event_evidence(class_names: Sequence[str], evidence: dict[str, bool]) -> list[str]:
    event_labels = {name for name, is_event in evidence.items() if is_event}
    return ["event" if str(name) != "noise" and str(name) in event_labels else "segment" for name in class_names]


def _annotation_duration_seconds(annotations: pd.DataFrame) -> float:
    if "start_seconds" not in annotations.columns or "stop_seconds" not in annotations.columns:
        return 0.0
    durations = annotations["stop_seconds"].astype(float) - annotations["start_seconds"].astype(float)
    return float(durations.clip(lower=0.0).sum())


def _annotation_bounds(row, min_duration_s: float) -> tuple[float, float]:
    start = float(row.start_seconds)
    stop = float(row.stop_seconds)
    min_duration_s = max(float(min_duration_s), 0.0)
    if min_duration_s > 0 and stop - start < min_duration_s:
        center = (start + stop) / 2.0
        half_duration = min_duration_s / 2.0
        start = center - half_duration
        stop = center + half_duration
    return start, stop


def _annotations_clipped_to_intervals(
    annotations: pd.DataFrame,
    intervals_seconds: Sequence[tuple[float, float]],
) -> pd.DataFrame:
    if annotations.empty or not intervals_seconds:
        return annotations.iloc[0:0].copy()

    merged_intervals = []
    for start, stop in sorted((float(start), float(stop)) for start, stop in intervals_seconds):
        if merged_intervals and start <= merged_intervals[-1][1]:
            merged_intervals[-1] = (merged_intervals[-1][0], max(stop, merged_intervals[-1][1]))
        else:
            merged_intervals.append((start, stop))

    rows = []
    for row in annotations.to_dict("records"):
        start = float(row["start_seconds"])
        stop = float(row["stop_seconds"])
        for interval_start, interval_stop in merged_intervals:
            clipped_start = max(start, float(interval_start))
            clipped_stop = min(stop, float(interval_stop))
            if clipped_stop > clipped_start or (stop == start and interval_start <= start <= interval_stop):
                clipped = row.copy()
                clipped["start_seconds"] = clipped_start
                clipped["stop_seconds"] = clipped_stop
                if "onset" in clipped:
                    clipped["onset"] = clipped_start
                if "offset" in clipped:
                    clipped["offset"] = clipped_stop
                rows.append(clipped)
    return pd.DataFrame(rows, columns=annotations.columns)


def _pad_normalized_chunk(chunk: np.ndarray, num_time_steps: int) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim == 1:
        if chunk.shape[0] < num_time_steps:
            chunk = np.concatenate([chunk, np.zeros(num_time_steps - chunk.shape[0], dtype=np.float32)])
    elif chunk.ndim == 2 and chunk.shape[1] < num_time_steps:
        pad = np.zeros((chunk.shape[0], num_time_steps - chunk.shape[1]), dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=1)
    return chunk


class AudioDirGenerator(Dataset):
    def __init__(
        self,
        num_time_steps: int,
        chunk_stride: int | None,
        audio_files: Sequence,
        annotation_files: Optional[Sequence] = None,
        annotations: Optional[Sequence[pd.DataFrame]] = None,
        chunk_starts: Optional[Sequence[Sequence[int]]] = None,
        chunk_input_lengths: Optional[Sequence[Sequence[int]]] = None,
        class_names: Optional[Sequence[str]] = None,
        class_types: Optional[Sequence[str]] = None,
        return_targets: bool = True,
        hop_s: float = 0.002,
        min_annotation_duration_s: float = 4e-3,
        target_samplerate: int | None = None,
        audio_dataset: str | None = None,
        data_samplerate_hz: float | None = None,
    ):
        self.num_time_steps = int(num_time_steps)
        self.chunk_stride = _resolve_chunk_stride(self.num_time_steps, chunk_stride)
        self.audio_files = list(audio_files)
        self.annotation_files = [] if annotation_files is None else list(annotation_files)
        self.class_names = [] if class_names is None else list(class_names)
        self.class_types = _normalize_class_types(class_names=self.class_names, class_types=class_types)
        self.return_targets = return_targets
        self.hop_s = float(hop_s)
        self.min_annotation_duration_s = max(float(min_annotation_duration_s), 0.0)
        self.target_samplerate = None if target_samplerate is None else int(target_samplerate)
        self.audio_dataset = audio_dataset
        self.data_samplerate_hz = data_samplerate_hz

        if annotations is not None:
            self.annotations = [annotation.copy() for annotation in annotations]
        elif self.return_targets:
            self.annotations = [read_annotation_file(annot_file) for annot_file in self.annotation_files]
        else:
            self.annotations = []

        self.num_classes = len(self.class_names)
        nb_samples_in_file = []
        samplerate_per_file = []
        source_samplerate_per_file = []
        for audio_file in self.audio_files:
            if self.audio_dataset is None and self.data_samplerate_hz is None:
                audio = open_audio_file(audio_file)
            else:
                audio = open_audio_file(
                    audio_file,
                    audio_dataset=self.audio_dataset,
                    data_samplerate_hz=self.data_samplerate_hz,
                )
            try:
                source_samplerate = int(audio.samplerate)
                samplerate = int(self.target_samplerate or source_samplerate)
                nb_samples = (
                    len(audio)
                    if samplerate == source_samplerate
                    else _resampled_sample_count(len(audio), source_samplerate, samplerate)
                )
                nb_samples_in_file.append(nb_samples)
                samplerate_per_file.append(samplerate)
                source_samplerate_per_file.append(source_samplerate)
            finally:
                audio.close()

        self.nb_samples_in_file = np.array(nb_samples_in_file, dtype=np.int64)
        self.samplerate_per_file = np.array(samplerate_per_file, dtype=np.int64)
        self.source_samplerate_per_file = np.array(source_samplerate_per_file, dtype=np.int64)

        if chunk_starts is None:
            self.file_chunk_starts = [
                _compute_chunk_starts(int(num_samples), self.num_time_steps, self.chunk_stride)
                for num_samples in self.nb_samples_in_file
            ]
        else:
            self.file_chunk_starts = [np.asarray(starts, dtype=np.int64) for starts in chunk_starts]
        if chunk_input_lengths is None:
            self.file_chunk_input_lengths = [
                np.asarray(
                    [
                        min(self.num_time_steps, max(int(num_samples) - int(start), 0))
                        for start in starts
                    ],
                    dtype=np.int64,
                )
                for num_samples, starts in zip(self.nb_samples_in_file, self.file_chunk_starts, strict=True)
            ]
        else:
            self.file_chunk_input_lengths = [np.asarray(lengths, dtype=np.int64) for lengths in chunk_input_lengths]
        self.nb_chunks_in_file = np.array([len(starts) for starts in self.file_chunk_starts], dtype=np.int64)
        self.chunk_borders = np.concatenate(([0], np.cumsum(self.nb_chunks_in_file)))
        self.nb_total = int(self.nb_chunks_in_file.sum())

        if self.nb_total > 0:
            self.chunk_file_indices = np.concatenate(
                [
                    np.full(len(starts), file_idx, dtype=np.int64)
                    for file_idx, starts in enumerate(self.file_chunk_starts)
                    if len(starts) > 0
                ]
            )
            self.chunk_start_samples = np.concatenate([starts for starts in self.file_chunk_starts if len(starts) > 0])
            self.chunk_input_lengths = np.concatenate(
                [lengths for lengths in self.file_chunk_input_lengths if len(lengths) > 0]
            )
        else:
            self.chunk_file_indices = np.zeros((0,), dtype=np.int64)
            self.chunk_start_samples = np.zeros((0,), dtype=np.int64)
            self.chunk_input_lengths = np.zeros((0,), dtype=np.int64)

    def __len__(self):
        return self.nb_total

    def _read_chunk(self, idx: int):
        file_idx = int(self.chunk_file_indices[idx])
        start = int(self.chunk_start_samples[idx])
        input_length = int(self.chunk_input_lengths[idx])
        stop = start + input_length

        if self.audio_dataset is None and self.data_samplerate_hz is None:
            audio_file = open_audio_file(self.audio_files[file_idx])
        else:
            audio_file = open_audio_file(
                self.audio_files[file_idx],
                audio_dataset=self.audio_dataset,
                data_samplerate_hz=self.data_samplerate_hz,
            )
        try:
            samplerate = int(self.samplerate_per_file[file_idx])
            chunk = _read_resampled_audio_chunk(audio_file, start, input_length, self.target_samplerate)
            chunk = _pad_normalized_chunk(chunk, self.num_time_steps)
        finally:
            audio_file.close()

        return chunk, file_idx, start, stop, input_length, samplerate

    def _build_labels(self, file_idx: int, start: int, stop: int, samplerate: int, input_length: int):
        hop_samples = max(int(round(samplerate * self.hop_s)), 1)
        total_frames = _label_frame_count(self.num_time_steps, hop_samples)
        target_length = _label_frame_count(input_length, hop_samples)
        start_sec = start / samplerate
        stop_sec = stop / samplerate

        labels = np.zeros((total_frames, self.num_classes), dtype=np.float32)
        for row in self.annotations[file_idx].itertuples(index=False):
            if row.name not in self.class_names:
                continue

            annot_start_sec, annot_stop_sec = _annotation_bounds(row, self.min_annotation_duration_s)
            if annot_stop_sec <= start_sec or annot_start_sec >= stop_sec:
                continue

            label_start_sec = max(annot_start_sec, start_sec) - start_sec
            label_stop_sec = min(annot_stop_sec, stop_sec) - start_sec
            label_start = int(np.floor(label_start_sec * samplerate / hop_samples + 1e-9))
            label_stop = int(np.ceil(label_stop_sec * samplerate / hop_samples - 1e-9))
            if label_stop > label_start:
                labels[label_start:label_stop, self.class_names.index(row.name)] = 1.0

        if self.num_classes > 0:
            labels[:, 0] = 1.0 - np.sum(labels[:, 1:], axis=1)
        return labels, target_length

    def __getitem__(self, idx):
        chunk, file_idx, start, stop, input_length, samplerate = self._read_chunk(idx)

        if not self.return_targets:
            return (
                chunk,
                input_length,
            )

        labels, target_length = self._build_labels(
            file_idx=file_idx,
            start=start,
            stop=stop,
            samplerate=samplerate,
            input_length=input_length,
        )

        return (
            chunk,
            input_length,
            labels,
            target_length,
        )


class AudioDirDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        data_dir: Optional[str] = None,
        audio_files: Optional[list] = None,
        batch_size: int = 32,
        num_time_steps: int = 512,
        chunk_stride: int | None = None,
        hop_s: float = 0.001,
        class_names: Optional[Sequence[str]] = None,
        num_workers: Optional[int] = 2,
        persistent_workers: bool = True,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        split_within_files: bool = False,
        min_annotation_duration_s: float = 4e-3,
        ignore_class_names: bool = False,
        include_labels: Optional[Sequence[str]] = None,
        target_samplerate: int | None = None,
        audio_dataset: str | None = None,
        data_samplerate_hz: float | None = None,
        split_seed: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = int(batch_size)
        self.num_time_steps = int(num_time_steps)
        self.chunk_stride = _resolve_chunk_stride(self.num_time_steps, chunk_stride)
        self.hop_s = float(hop_s)
        self.num_workers = int(num_workers) if num_workers is not None else 0
        self.persistent_workers = bool(persistent_workers)
        self.data_dir = data_dir
        self.target_samplerate = None if target_samplerate is None else int(target_samplerate)
        self.audio_dataset = audio_dataset
        self.data_samplerate_hz = data_samplerate_hz

        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.train_ratio = 1.0 - self.val_ratio - self.test_ratio
        self.split_within_files = bool(split_within_files)
        self.min_annotation_duration_s = max(float(min_annotation_duration_s), 0.0)
        self.ignore_class_names = bool(ignore_class_names)
        self.include_labels = _normalize_include_labels(include_labels)
        if self.val_ratio < 0 or self.test_ratio < 0 or self.train_ratio <= 0:
            raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")

        if audio_files is not None:
            all_files = audio_files
            annotation_tables = {}
        elif data_dir is not None:
            self.data_dir = Path(self.data_dir)
            if self.data_dir.is_file() or _is_zarr_store_path(self.data_dir):
                all_files = [self.data_dir]
                annotation_files = _paired_annotation_files(self.data_dir)
            else:
                all_files = iter_audio_candidate_paths(self.data_dir)
                annotation_files = sorted(path for path in all_files if path.suffix.lower() in {".csv", ".json"})
            annotation_tables = _read_annotation_tables(annotation_files)
            all_files = [*all_files, *_audio_files_from_filepath_annotations(annotation_tables)]
        else:
            raise ValueError("Need to provide either audio_files or data_dir, both were None.")

        self.all_audio_files = []
        self.annotated_audio_files = []
        self.annotation_files = []
        self.annotations = []
        self.annotation_intervals_samples = []
        self.annotation_chunk_starts = []
        self.annotation_chunk_input_lengths = []
        self.audio_file_info_by_path = {}
        self.source_audio_file_info_by_path = {}
        self.source_samplerates = set()
        available_class_names = set()
        event_label_evidence: dict[str, bool] = {}
        channel_counts = []
        seen_audio_files = set()
        for audio_file in all_files:
            try:
                audio_file = Path(audio_file)
                audio_file_key = audio_file.expanduser().resolve()
                if audio_file_key in seen_audio_files:
                    continue
                source_info = audio_file_info(
                    audio_file,
                    audio_dataset=self.audio_dataset,
                    data_samplerate_hz=self.data_samplerate_hz,
                )
                source_samplerate = int(source_info["samplerate"])
                samplerate = int(self.target_samplerate or source_samplerate)
                info = dict(source_info)
                if samplerate != source_samplerate:
                    info["frames"] = _resampled_sample_count(int(source_info["frames"]), source_samplerate, samplerate)
                    info["samplerate"] = samplerate
                seen_audio_files.add(audio_file_key)
                self.source_audio_file_info_by_path[audio_file_key] = source_info
                self.audio_file_info_by_path[audio_file_key] = info
                self.source_samplerates.add(source_samplerate)
                self.all_audio_files.append(audio_file)
                channel_counts.append(int(info["channels"]))
                annotation_result = _collect_annotations_for_audio(audio_file, annotation_tables)
                if annotation_result is not None:
                    annot_files, annotations = annotation_result
                    intervals = _annotation_intervals_samples(
                        annotations,
                        num_samples=int(info["frames"]),
                        samplerate=int(info["samplerate"]),
                    )
                    if not intervals:
                        continue
                    chunk_starts, chunk_input_lengths = _chunk_starts_and_lengths_for_intervals(
                        intervals,
                        self.num_time_steps,
                        self.chunk_stride,
                    )
                    annotations = _drop_annotation_boundary_rows(annotations)
                    annotations = _annotations_clipped_to_intervals(
                        annotations,
                        [
                            (float(start) / float(info["samplerate"]), float(stop) / float(info["samplerate"]))
                            for start, stop in intervals
                        ],
                    )
                    available_class_names.update(str(name) for name in annotations["name"].dropna().tolist())
                    _update_event_label_evidence(event_label_evidence, annotations)
                    annotations = _filter_annotations_for_include_labels(annotations, self.include_labels)
                    if self.ignore_class_names:
                        annotations = _collapse_annotation_class_names(annotations)
                    self.annotated_audio_files.append(audio_file)
                    self.annotation_files.append(annot_files[0])
                    self.annotations.append(annotations)
                    self.annotation_intervals_samples.append(intervals)
                    self.annotation_chunk_starts.append(chunk_starts)
                    self.annotation_chunk_input_lengths.append(chunk_input_lengths)
            except Exception:
                pass

        self.input_num_channels = max(channel_counts, default=1)
        if len(set(channel_counts)) > 1 and self.batch_size != 1:
            # ponytail: pad and mask channels if mixed-count batches larger than one become necessary.
            raise ValueError("Mixed channel counts require batch_size=1.")

        _validate_include_labels(self.include_labels, available_class_names)
        if self.ignore_class_names:
            self.class_names = ["noise", "song"]
        elif self.include_labels:
            self.class_names = list(self.include_labels)
        elif class_names is None:
            inferred_class_names = sorted({str(name) for annot in self.annotations for name in annot["name"].to_list()})
            self.class_names = inferred_class_names
        else:
            self.class_names = list(class_names)

        if self.class_names and self.class_names[0] == "noise":
            pass
        elif "noise" in self.class_names:
            self.class_names = ["noise"] + [name for name in self.class_names if name != "noise"]
        elif self.annotations or class_names is not None:
            self.class_names.insert(0, "noise")
        self.class_types = (
            ["segment"] * len(self.class_names)
            if self.ignore_class_names
            else _class_types_from_event_evidence(self.class_names, event_label_evidence)
        )
        self.num_classes = len(self.class_names)
        self.subsets = {}
        if self.annotated_audio_files:
            if self.split_within_files:
                self._build_within_file_subsets()
            else:
                nb_files = len(self.annotated_audio_files)
                split_lengths = _split_lengths(nb_files, [self.train_ratio, self.val_ratio, self.test_ratio])

                generator = None if split_seed is None else torch.Generator().manual_seed(int(split_seed))
                shuffled_indices = torch.randperm(len(self.annotated_audio_files), generator=generator).tolist()
                start = 0
                for split_length, name in zip(split_lengths, ["train", "val", "test"]):
                    split_indices = shuffled_indices[start : start + split_length]
                    start += split_length
                    self.subsets[name] = {
                        "audio": [self.annotated_audio_files[i] for i in split_indices],
                        "annotations": [self.annotation_files[i] for i in split_indices],
                        "annotation_tables": [self.annotations[i] for i in split_indices],
                        "chunk_starts": [self.annotation_chunk_starts[i] for i in split_indices],
                        "chunk_input_lengths": [self.annotation_chunk_input_lengths[i] for i in split_indices],
                    }
        else:
            for name in ["train", "val", "test"]:
                self.subsets[name] = {
                    "audio": [],
                    "annotations": [],
                    "annotation_tables": [],
                    "chunk_starts": [],
                    "chunk_input_lengths": [],
                }

    def _build_within_file_subsets(self) -> None:
        self.subsets = {
            name: {"audio": [], "annotations": [], "annotation_tables": [], "chunk_starts": [], "chunk_input_lengths": []}
            for name in ("train", "val", "test")
        }
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        for audio_file, annotation_file, annotations, starts, input_lengths in zip(
            self.annotated_audio_files,
            self.annotation_files,
            self.annotations,
            self.annotation_chunk_starts,
            self.annotation_chunk_input_lengths,
            strict=True,
        ):
            info = self.audio_file_info_by_path[Path(audio_file).expanduser().resolve()]
            samplerate = int(info["samplerate"])
            split_lengths = _split_lengths(len(starts), ratios)
            start_idx = 0
            split_start_values = []
            split_input_length_values = []
            for split_length in split_lengths:
                split_start_values.append(starts[start_idx : start_idx + split_length])
                split_input_length_values.append(input_lengths[start_idx : start_idx + split_length])
                start_idx += split_length
            split_start_values, split_input_length_values = _drop_split_boundary_overlaps_with_lengths(
                split_start_values,
                split_input_length_values,
            )

            for split_starts, split_input_lengths, name in zip(
                split_start_values,
                split_input_length_values,
                ("train", "val", "test"),
                strict=True,
            ):
                if len(split_starts) == 0:
                    continue
                intervals_seconds = [
                    (float(start) / samplerate, float(start + length) / samplerate)
                    for start, length in zip(split_starts, split_input_lengths, strict=True)
                ]
                self.subsets[name]["audio"].append(audio_file)
                self.subsets[name]["annotations"].append(annotation_file)
                self.subsets[name]["annotation_tables"].append(
                    _annotations_clipped_to_intervals(annotations, intervals_seconds)
                )
                self.subsets[name]["chunk_starts"].append(split_starts)
                self.subsets[name]["chunk_input_lengths"].append(split_input_lengths)

    @property
    def has_test(self):
        return len(self.subsets["test"]["audio"]) > 0

    @property
    def has_val(self):
        return len(self.subsets["val"]["audio"]) > 0

    def setup(self, stage: str):
        pass

    def _audio_duration_seconds(self, audio_files: Sequence[Path]) -> float:
        total_seconds = 0.0
        for audio_file in audio_files:
            info = self.audio_file_info_by_path.get(Path(audio_file).expanduser().resolve())
            if info is None:
                continue
            total_seconds += float(info["frames"]) / float(info["samplerate"])
        return total_seconds

    def _chunk_duration_seconds(
        self,
        audio_files: Sequence[Path],
        chunk_starts: Sequence[Sequence[int]] | None,
        chunk_input_lengths: Sequence[Sequence[int]] | None,
    ) -> float:
        if chunk_starts is None:
            return self._audio_duration_seconds(audio_files)

        total_seconds = 0.0
        if chunk_input_lengths is None:
            chunk_input_lengths = [
                np.full(len(starts), self.num_time_steps, dtype=np.int64)
                for starts in chunk_starts
            ]
        for audio_file, starts, lengths in zip(audio_files, chunk_starts, chunk_input_lengths, strict=True):
            info = self.audio_file_info_by_path.get(Path(audio_file).expanduser().resolve())
            if info is None:
                continue
            intervals = [
                (int(start), min(int(start) + int(length), int(info["frames"])))
                for start, length in zip(np.asarray(starts, dtype=np.int64), np.asarray(lengths, dtype=np.int64), strict=True)
            ]
            if not intervals:
                continue
            intervals.sort()
            merged = []
            for start, stop in intervals:
                if not merged or start > merged[-1][1]:
                    merged.append([start, stop])
                else:
                    merged[-1][1] = max(merged[-1][1], stop)
            total_seconds += sum(stop - start for start, stop in merged) / float(info["samplerate"])
        return total_seconds

    def split_stats(self) -> list[dict[str, float | int | str]]:
        stats = []
        for split in ("train", "val", "test"):
            subset = self.subsets[split]
            annotation_files = {Path(path).expanduser().resolve() for path in subset["annotations"]}
            annotation_minutes = (
                sum(_annotation_duration_seconds(annotations) for annotations in subset["annotation_tables"]) / 60.0
            )
            stats.append(
                {
                    "split": split,
                    "audio_file_count": len(subset["audio"]),
                    "audio_minutes": self._chunk_duration_seconds(
                        subset["audio"],
                        subset["chunk_starts"],
                        subset["chunk_input_lengths"],
                    )
                    / 60.0,
                    "annotation_file_count": len(annotation_files),
                    "annotation_count": sum(len(annotations) for annotations in subset["annotation_tables"]),
                    "annotation_minutes": annotation_minutes,
                }
            )
        return stats

    def _dataloader(self, stage: str, *, shuffle: bool | None = None):
        if stage == "predict":
            if not self.all_audio_files:
                raise ValueError("No readable audio files were found for prediction.")
            dataset = AudioDirGenerator(
                num_time_steps=self.num_time_steps,
                chunk_stride=self.chunk_stride,
                audio_files=self.all_audio_files,
                return_targets=False,
                hop_s=self.hop_s,
                target_samplerate=self.target_samplerate,
                audio_dataset=self.audio_dataset,
                data_samplerate_hz=self.data_samplerate_hz,
            )
        else:
            if not self.annotated_audio_files:
                raise ValueError(f"No annotated audio files were found for the '{stage}' stage.")
            dataset = AudioDirGenerator(
                num_time_steps=self.num_time_steps,
                chunk_stride=self.chunk_stride,
                audio_files=self.subsets[stage]["audio"],
                annotation_files=self.subsets[stage]["annotations"],
                annotations=self.subsets[stage]["annotation_tables"],
                chunk_starts=self.subsets[stage]["chunk_starts"],
                chunk_input_lengths=self.subsets[stage]["chunk_input_lengths"],
                class_names=self.class_names,
                class_types=self.class_types,
                return_targets=True,
                hop_s=self.hop_s,
                min_annotation_duration_s=self.min_annotation_duration_s,
                target_samplerate=self.target_samplerate,
                audio_dataset=self.audio_dataset,
                data_samplerate_hz=self.data_samplerate_hz,
            )

        if stage == "predict":
            annotation_files = self.annotation_files
            annotation_tables = self.annotations
            audio_files = self.annotated_audio_files
        else:
            annotation_files = dataset.annotation_files
            annotation_tables = dataset.annotations
            audio_files = dataset.audio_files

        dataset.annotation_files_by_audio = {
            Path(audio_file).expanduser().resolve(): annotation_file
            for audio_file, annotation_file in zip(audio_files, annotation_files, strict=True)
        }
        dataset.annotations_by_audio = {
            Path(audio_file).expanduser().resolve(): annotation_table
            for audio_file, annotation_table in zip(audio_files, annotation_tables, strict=True)
        }

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(stage == "train" if shuffle is None else bool(shuffle)),
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

    def predict_dataloader(self):
        return self._dataloader("predict")
