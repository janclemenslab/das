from pathlib import Path
from typing import Any, Optional, Sequence

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .audio_dir import (
    _compute_chunk_starts,
    _label_frame_count,
    _normalize_class_types,
    _normalize_chunk,
    _normalize_include_labels,
    _resolve_chunk_stride,
    _validate_include_labels,
)


def is_npy_dir(path: str | Path) -> bool:
    root = Path(path)
    if not root.is_dir():
        return False
    if not (root / "attrs.npy").exists():
        return False
    return (root / "train" / "x.npy").exists() and (root / "train" / "y.npy").exists()


def load_npy_dir_attrs(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    attrs_path = root / "attrs.npy"
    if not attrs_path.exists():
        raise ValueError(f"No attrs.npy file was found in '{root}'.")

    attrs = np.load(attrs_path, allow_pickle=True).item()
    if not isinstance(attrs, dict):
        raise ValueError(f"Expected attrs.npy in '{root}' to contain a dict, got {type(attrs).__name__}.")
    return attrs


class NPYDirGenerator(Dataset):
    def __init__(
        self,
        *,
        x_path: str | Path,
        y_path: str | Path,
        num_time_steps: int,
        chunk_stride: int | None,
        hop_s: float,
        samplerate_x_hz: float,
        samplerate_y_hz: float,
        ignore_class_names: bool = False,
        include_label_indices: Optional[Sequence[int]] = None,
    ):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.num_time_steps = int(num_time_steps)
        self.chunk_stride = _resolve_chunk_stride(self.num_time_steps, chunk_stride)
        self.hop_s = float(hop_s)
        self.samplerate_x_hz = float(samplerate_x_hz)
        self.samplerate_y_hz = float(samplerate_y_hz)
        self.ignore_class_names = bool(ignore_class_names)
        self.include_label_indices = None if include_label_indices is None else [int(index) for index in include_label_indices]

        x_shape = np.load(self.x_path, mmap_mode="r").shape
        y_shape = np.load(self.y_path, mmap_mode="r").shape
        if len(x_shape) not in {1, 2}:
            raise ValueError(f"Expected x.npy to have shape [time] or [time, channels], got {x_shape}.")
        if len(y_shape) != 2:
            raise ValueError(f"Expected y.npy to have shape [time, classes], got {y_shape}.")

        self.num_samples = int(x_shape[0])
        if self.ignore_class_names:
            self.num_classes = 2
        elif self.include_label_indices is not None:
            self.num_classes = 1 + len(self.include_label_indices)
        else:
            self.num_classes = int(y_shape[1])
        self.chunk_start_samples = _compute_chunk_starts(self.num_samples, self.num_time_steps, self.chunk_stride)
        self.chunk_input_lengths = np.array(
            [
                min(self.num_time_steps, max(self.num_samples - int(start), 0))
                for start in self.chunk_start_samples
            ],
            dtype=np.int64,
        )
        self.nb_total = int(len(self.chunk_start_samples))

        self._x = None
        self._y = None

    def __len__(self):
        return self.nb_total

    def _load_arrays(self):
        if self._x is None:
            self._x = np.load(self.x_path, mmap_mode="r")
            self._y = np.load(self.y_path, mmap_mode="r")

    def _read_chunk(self, idx: int) -> tuple[np.ndarray, int, int]:
        self._load_arrays()

        start = int(self.chunk_start_samples[idx])
        input_length = int(self.chunk_input_lengths[idx])
        stop = start + self.num_time_steps
        chunk = np.asarray(self._x[start:stop], dtype=np.float32)

        return _normalize_chunk(chunk, self.num_time_steps), start, input_length

    def _build_labels(self, start: int, input_length: int) -> tuple[np.ndarray, int]:
        self._load_arrays()

        hop_x = max(int(round(self.hop_s * self.samplerate_x_hz)), 1)
        total_frames = _label_frame_count(self.num_time_steps, hop_x)
        target_length = _label_frame_count(input_length, hop_x)

        labels = np.zeros((total_frames, self.num_classes), dtype=np.float32)
        if self.num_classes > 0:
            labels[:, 0] = 1.0
        if target_length == 0:
            return labels, target_length

        frame_positions_x = start + (np.arange(target_length, dtype=np.int64) * hop_x)
        frame_positions_y = np.rint(
            frame_positions_x.astype(np.float64) * (self.samplerate_y_hz / self.samplerate_x_hz)
        ).astype(np.int64)
        valid = np.logical_and(frame_positions_y >= 0, frame_positions_y < int(self._y.shape[0]))
        if not np.any(valid):
            return labels, target_length

        sampled = np.asarray(self._y[frame_positions_y[valid]], dtype=np.float32)
        sampled = np.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
        if self.include_label_indices is None:
            selected_positive = sampled[:, 1:] if sampled.shape[1] > 1 else np.zeros((sampled.shape[0], 0), dtype=np.float32)
        elif self.include_label_indices:
            selected_positive = sampled[:, self.include_label_indices]
        else:
            selected_positive = np.zeros((sampled.shape[0], 0), dtype=np.float32)

        if self.ignore_class_names:
            positive = (
                np.max(selected_positive, axis=1)
                if selected_positive.shape[1] > 0
                else np.zeros(sampled.shape[0], dtype=np.float32)
            )
            sampled = np.stack([1.0 - positive, positive], axis=1).astype(np.float32)
        elif self.include_label_indices is not None:
            positive_sum = np.sum(selected_positive, axis=1)
            noise = np.clip(1.0 - positive_sum, 0.0, 1.0)
            sampled = np.concatenate([noise[:, None], selected_positive], axis=1).astype(np.float32)
        labels[np.flatnonzero(valid)[:total_frames]] = sampled[:total_frames]
        return labels, target_length

    def __getitem__(self, idx):
        chunk, start, input_length = self._read_chunk(idx)
        labels, target_length = self._build_labels(start, input_length)
        return chunk, input_length, labels, target_length


class NPYDirDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int = 32,
        num_time_steps: int = 512,
        chunk_stride: int | None = None,
        hop_s: float = 0.001,
        class_names: Optional[Sequence[str]] = None,
        num_workers: Optional[int] = 2,
        persistent_workers: bool = True,
        ignore_class_names: bool = False,
        include_labels: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        if not is_npy_dir(self.data_dir):
            raise ValueError(f"'{self.data_dir}' is not a supported npy_dir dataset.")

        self.attrs = load_npy_dir_attrs(self.data_dir)
        self.batch_size = int(batch_size)
        self.num_time_steps = int(num_time_steps)
        self.chunk_stride = _resolve_chunk_stride(self.num_time_steps, chunk_stride)
        self.hop_s = float(hop_s)
        self.num_workers = int(num_workers) if num_workers is not None else 0
        self.persistent_workers = bool(persistent_workers)
        self.ignore_class_names = bool(ignore_class_names)
        self.include_labels = _normalize_include_labels(include_labels)
        self.samplerate_x_hz = float(self.attrs["samplerate_x_Hz"])
        self.samplerate_y_hz = float(self.attrs.get("samplerate_y_Hz", self.samplerate_x_hz))
        train_x_shape = np.load(self.data_dir / "train" / "x.npy", mmap_mode="r").shape
        self.input_num_channels = 1 if len(train_x_shape) == 1 else int(train_x_shape[1])

        train_y = np.load(self.data_dir / "train" / "y.npy", mmap_mode="r")
        if train_y.ndim != 2:
            raise ValueError(f"Expected train/y.npy to have shape [time, classes], got {train_y.shape}.")
        inferred_class_names = [str(name) for name in self.attrs.get("class_names", [])]
        if not inferred_class_names:
            inferred_class_names = [f"class_{idx}" for idx in range(train_y.shape[1])]
        if len(inferred_class_names) != int(train_y.shape[1]):
            raise ValueError(
                f"class_names has length {len(inferred_class_names)}, but train/y.npy has {train_y.shape[1]} columns."
            )
        inferred_class_types = _normalize_class_types(
            class_names=inferred_class_names,
            class_types=self.attrs.get("class_types"),
        )
        label_to_index = {
            label: index
            for index, label in enumerate(inferred_class_names)
            if index != 0 and label != "noise"
        }
        _validate_include_labels(self.include_labels, set(label_to_index))
        self.include_label_indices = (
            None
            if not self.include_labels
            else [label_to_index[label] for label in self.include_labels]
        )
        if self.ignore_class_names:
            self.class_names = ["noise", "song"]
            self.class_types = ["segment", "segment"]
        elif self.include_labels:
            self.class_names = ["noise", *self.include_labels]
            self.class_types = [
                "segment",
                *[inferred_class_types[label_to_index[label]] for label in self.include_labels],
            ]
        elif class_names is None:
            self.class_names = inferred_class_names
            self.class_types = inferred_class_types
        else:
            self.class_names = list(class_names)
            if len(self.class_names) != int(train_y.shape[1]):
                raise ValueError(
                    f"class_names has length {len(self.class_names)}, but train/y.npy has {train_y.shape[1]} columns."
                )
            self.class_types = _normalize_class_types(class_names=self.class_names, class_types=inferred_class_types)
        self.num_classes = len(self.class_names)

        self.split_paths = {}
        for split in ("train", "val", "test"):
            x_path = self.data_dir / split / "x.npy"
            y_path = self.data_dir / split / "y.npy"
            if x_path.exists() and y_path.exists():
                self.split_paths[split] = {"x": x_path, "y": y_path}

        if "train" not in self.split_paths:
            raise ValueError(f"'{self.data_dir}' does not contain a train split with x.npy and y.npy.")

    @property
    def has_val(self):
        return "val" in self.split_paths

    @property
    def has_test(self):
        return "test" in self.split_paths

    def split_stats(self) -> list[dict[str, float | int | str]]:
        stats = []
        for split in ("train", "val", "test"):
            paths = self.split_paths.get(split)
            if paths is None:
                continue
            x_samples = int(np.load(paths["x"], mmap_mode="r").shape[0])
            y_samples = int(np.load(paths["y"], mmap_mode="r").shape[0])
            stats.append(
                {
                    "split": split,
                    "audio_file_count": 1,
                    "audio_minutes": x_samples / self.samplerate_x_hz / 60.0,
                    "annotation_file_count": 1,
                    "annotation_minutes": y_samples / self.samplerate_y_hz / 60.0,
                }
            )
        return stats

    def setup(self, stage: str):
        pass

    def _dataloader(self, stage: str):
        split = self.split_paths.get(stage)
        if split is None:
            raise ValueError(f"No '{stage}' split was found in '{self.data_dir}'.")

        dataset = NPYDirGenerator(
            x_path=split["x"],
            y_path=split["y"],
            num_time_steps=self.num_time_steps,
            chunk_stride=self.chunk_stride,
            hop_s=self.hop_s,
            samplerate_x_hz=self.samplerate_x_hz,
            samplerate_y_hz=self.samplerate_y_hz,
            ignore_class_names=self.ignore_class_names,
            include_label_indices=self.include_label_indices,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=stage == "train",
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
