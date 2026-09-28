from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading

import lightning as L
import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from .config import Config, SplitName
from .data import (
    AudioDirDataModule,
    NPYDirDataModule,
    audio_file_info,
    iter_audio_candidate_paths,
    is_npy_dir,
    load_audio_array,
    load_npy_dir_attrs,
    resolve_training_data_dir,
)
from .data.audio_dir import (
    _compute_chunk_starts,
    _normalize_chunk,
    _resample_audio_array,
    _resolve_chunk_stride,
)
from .data.legacy_store import is_legacy_store, materialize_legacy_store
from .das_legacy import is_legacy_model_source, legacy_to_das_model, load_legacy_predictor
from .models import DASModel
from .models.frontends import frontend_hop_seconds, normalize_frontend_config
from .models.encoders import model_hop_seconds
from . import prediction_results
from .prediction_results import (
    PredictionEvaluation,
    PredictionResultContext,
)
from .progress import TrainingLogProgress

ConfigLike = Config | Mapping[str, object] | None
RawAudioLike = np.ndarray | list[float] | list[np.ndarray] | tuple[np.ndarray, ...]


@dataclass
class PredictRuntime:
    sr: float
    hop_seconds: float
    num_time_steps: int
    chunk_stride: int | None
    class_names: list[str]
    class_types: list[str] | None
    batch_size: int | None = None
    fill_gap_ms: float | None = None
    min_syllable_ms: float | None = None
    segment_threshold_low: float | None = None
    segment_threshold_high: float | None = None
    event_threshold: float | None = None
    event_dist_min_ms: float | None = None
    event_dist_max_ms: float | None = None
    syllable_postprocessor: str | None = None
    legacy_data_padding: int = 0


@dataclass
class TrainingRuntime:
    input_num_channels: int
    num_time_steps: int
    chunk_stride: int | None


@dataclass
class InferenceRuntime:
    model: object
    runtime: PredictRuntime
    datamodule: AudioDirDataModule
    trainer: object
    device: torch.device
    context: PredictionResultContext


class _RawAudioDataset(Dataset):
    def __init__(
        self,
        *,
        audio_arrays: list[np.ndarray],
        samplerate: int,
        num_time_steps: int,
        chunk_stride: int | None,
        hop_s: float,
        class_names: list[str],
    ):
        self.audio_arrays = [_normalize_raw_audio_array(audio) for audio in audio_arrays]
        self.audio_files = [f"audio_{idx}" for idx in range(len(self.audio_arrays))]
        self.samplerate = int(samplerate)
        self.num_time_steps = int(num_time_steps)
        self.chunk_stride = _resolve_chunk_stride(self.num_time_steps, chunk_stride)
        self.hop_s = float(hop_s)
        self.class_names = list(class_names)

        self.nb_samples_in_file = np.array([len(audio) for audio in self.audio_arrays], dtype=np.int64)
        self.samplerate_per_file = np.full(len(self.audio_arrays), self.samplerate, dtype=np.int64)
        self.file_chunk_starts = [
            _compute_chunk_starts(int(num_samples), self.num_time_steps, self.chunk_stride)
            for num_samples in self.nb_samples_in_file
        ]
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
            self.chunk_input_lengths = np.array(
                [
                    min(self.num_time_steps, max(int(self.nb_samples_in_file[file_idx]) - int(start), 0))
                    for file_idx, starts in enumerate(self.file_chunk_starts)
                    for start in starts
                ],
                dtype=np.int64,
            )
        else:
            self.chunk_file_indices = np.zeros((0,), dtype=np.int64)
            self.chunk_start_samples = np.zeros((0,), dtype=np.int64)
            self.chunk_input_lengths = np.zeros((0,), dtype=np.int64)

    def __len__(self):
        return self.nb_total

    def __getitem__(self, idx):
        file_idx = int(self.chunk_file_indices[idx])
        start = int(self.chunk_start_samples[idx])
        input_length = int(self.chunk_input_lengths[idx])
        chunk = self.audio_arrays[file_idx][start : start + self.num_time_steps]
        return _normalize_chunk(chunk, self.num_time_steps), input_length


class _StopOnEventCallback(Callback):
    def __init__(self, stop_event: threading.Event, *, verbose: bool):
        super().__init__()
        self.stop_event = stop_event
        self.verbose = verbose
        self._logged = False

    def _request_stop(self, trainer) -> None:
        if not self.stop_event.is_set():
            return
        if self.verbose and not self._logged:
            print("Cancellation requested. Stopping training after the current step.")
            self._logged = True
        trainer.should_stop = True

    def on_fit_start(self, trainer, pl_module) -> None:
        del pl_module
        self._request_stop(trainer)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        del pl_module
        self._request_stop(trainer)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        del pl_module, batch, batch_idx
        self._request_stop(trainer)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:
        del pl_module, batch, batch_idx, dataloader_idx
        self._request_stop(trainer)


class _PredictionFileProgressCallback(Callback):
    def __init__(self, chunk_borders, batch_size: int):
        super().__init__()
        self.chunk_borders = np.asarray(chunk_borders[1:])
        self.batch_size = int(batch_size)
        self.progress = None

    def on_predict_start(self, trainer, pl_module) -> None:
        del pl_module
        if trainer.is_global_zero:
            self.progress = tqdm(total=len(self.chunk_borders), desc="Predicting files", unit="file")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        del trainer, pl_module, outputs, batch, dataloader_idx
        if self.progress is not None:
            completed = int(np.searchsorted(self.chunk_borders, (batch_idx + 1) * self.batch_size, side="right"))
            self.progress.update(completed - self.progress.n)

    def on_predict_end(self, trainer, pl_module) -> None:
        del trainer, pl_module
        if self.progress is not None:
            self.progress.update(self.progress.total - self.progress.n)
            self.progress.close()


def _print_prediction_summary(annotations: pd.DataFrame | list[pd.DataFrame]) -> None:
    frames = [annotations] if isinstance(annotations, pd.DataFrame) else list(annotations)
    counts = Counter(
        str(name)
        for frame in frames
        if "name" in frame.columns
        for name in frame["name"].dropna()
    )
    total = sum(counts.values())
    print("Prediction summary:")
    print(f"Total instances: {total}")
    for name in sorted(counts):
        print(f"{name}: {counts[name]}")


class _EpochLogCallback(Callback):
    def _has_validation(self, trainer) -> bool:
        num_val_batches = getattr(trainer, "num_val_batches", 0)
        if isinstance(num_val_batches, int):
            return num_val_batches > 0
        try:
            return sum(int(value) for value in num_val_batches) > 0
        except TypeError:
            return bool(num_val_batches)

    def _format_metric(self, value) -> str:
        if hasattr(value, "item"):
            value = value.item()
        return f"{float(value):.4f}"

    def _summary_text(self, trainer) -> str:
        callback_metrics = getattr(trainer, "callback_metrics", {})
        parts = []
        for name in ("train_loss", "val_loss", "val_acc", "lr"):
            if name in callback_metrics:
                parts.append(f"{name}={self._format_metric(callback_metrics[name])}")
        return ", ".join(parts)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        del pl_module
        current_epoch = int(trainer.current_epoch) + 1
        max_epochs = int(getattr(trainer, "max_epochs", 0) or 0)
        print(f"Epoch {current_epoch}/{max_epochs}" if max_epochs > 0 else f"Epoch {current_epoch}")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        del pl_module
        if getattr(trainer, "sanity_checking", False) or self._has_validation(trainer):
            return
        summary = self._summary_text(trainer)
        if summary:
            print(f"Epoch {int(trainer.current_epoch) + 1} complete: {summary}")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        del pl_module
        if getattr(trainer, "sanity_checking", False):
            return
        summary = self._summary_text(trainer)
        if summary:
            print(f"Epoch {int(trainer.current_epoch) + 1} complete: {summary}")


def _checkpoint_class_names(model) -> list[str]:
    class_names = model.hparams.get("class_names")
    if class_names is None:
        return [f"class_{idx}" for idx in range(model.num_classes)]
    return list(class_names)


def _training_start_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _checkpoint_filename(prefix: str | None, timestamp: str) -> str:
    prefix_text = f"{prefix.strip()}_" if prefix else ""
    return f"{prefix_text}{timestamp}_model"


def _normalize_raw_audio_array(audio) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError(f"Expected raw audio with 1 or 2 dimensions, got shape {array.shape}.")
    if array.shape[0] <= 32 < array.shape[1]:
        return array.T
    return array


def _resample_raw_audio_array(audio, source_samplerate: int, target_samplerate: int) -> np.ndarray:
    array = _normalize_raw_audio_array(audio)
    return _resample_audio_array(array, int(source_samplerate), int(target_samplerate), axis=0)


def _format_samplerate(samplerate: float) -> str:
    samplerate = float(samplerate)
    if samplerate.is_integer():
        return f"{int(samplerate)} Hz"
    return f"{samplerate:g} Hz"


def _log_prediction_samplerates(source_samplerates, model_samplerate: float) -> None:
    rates = sorted({int(round(float(rate))) for rate in source_samplerates})
    if not rates:
        return
    source_text = ", ".join(_format_samplerate(rate) for rate in rates)
    model_text = _format_samplerate(model_samplerate)
    model_rate = int(round(float(model_samplerate)))
    action = "resampling to model rate" if any(rate != model_rate for rate in rates) else "no resampling needed"
    label = "Audio sample rate" if len(rates) == 1 else "Audio sample rates"
    print(f"{label}: {source_text}; model sample rate: {model_text}; {action}.")


def _raw_audio_inputs(audio) -> tuple[list[np.ndarray], bool]:
    if isinstance(audio, (list, tuple)) and audio and all(hasattr(item, "shape") for item in audio):
        if all(np.asarray(item).ndim in (1, 2) for item in audio):
            return [np.asarray(item, dtype=np.float32) for item in audio], False
    return [np.asarray(audio, dtype=np.float32)], True


def _whisper_normalize_audio_array(audio) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError(f"Expected audio with 1 or 2 dimensions, got shape {array.shape}.")
    if array.shape[0] <= 32 < array.shape[1]:
        array = array.T
    return np.mean(array, axis=1).astype(np.float32)


def _whisper_prediction_to_dataframe(prediction: dict, *, filename: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "name": [str(value) for value in prediction.get("cluster", [])],
            "start_seconds": [float(value) for value in prediction.get("onset", [])],
            "stop_seconds": [float(value) for value in prediction.get("offset", [])],
        }
    )
    if filename is not None:
        frame.insert(0, "filename", filename)
    return frame


def _whisper_effective_samplerate(samplerate: int, frequency_scale: float) -> int:
    if samplerate <= 0:
        raise ValueError("samplerate must be positive.")
    scaled = int(round(samplerate * frequency_scale))
    if scaled <= 0:
        raise ValueError("frequency_scale produces an effective samplerate below 1 Hz.")
    return scaled


def _whisper_model_samplerate(segmenter) -> int | None:
    value = getattr(segmenter, "default_segmentation_config", {}).get("sr")
    if value is None:
        return None
    return int(round(float(value)))


def _whisper_scale_prediction_time(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
    if scale == 1.0:
        return frame
    scaled = frame.copy()
    scaled["start_seconds"] = scaled["start_seconds"] * scale
    scaled["stop_seconds"] = scaled["stop_seconds"] * scale
    return scaled


def _whisper_audio_paths(
    data_dir: str,
    *,
    audio_dataset: str | None = None,
    data_samplerate_hz: float | None = None,
) -> list[Path]:
    data_path = Path(data_dir).expanduser()
    if not data_path.exists():
        raise ValueError(f"data_dir '{data_dir}' is neither a file nor a directory.")
    paths = []
    for path in iter_audio_candidate_paths(data_path):
        try:
            audio_file_info(path, audio_dataset=audio_dataset, data_samplerate_hz=data_samplerate_hz)
        except Exception:
            continue
        paths.append(path)
    if not paths:
        raise ValueError(f"No readable audio files found in data_dir '{data_dir}'.")
    return paths


def _whisper_audio_paths_for_config(config: Config) -> list[Path]:
    try:
        return _whisper_audio_paths(
            config.data_dir,
            audio_dataset=config.audio_dataset,
            data_samplerate_hz=config.data_samplerate_hz,
        )
    except TypeError:
        # Some tests monkeypatch _whisper_audio_paths with the historical one-argument signature.
        return _whisper_audio_paths(config.data_dir)


def _resolve_whisper_device(accelerator: str) -> str:
    if accelerator not in {"auto", "tpu"}:
        return accelerator
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _whisper_segmenter(config: Config):
    from .whisperseg.model import WhisperSegmenter

    return WhisperSegmenter(config.checkpoint, device=_resolve_whisper_device(str(config.accelerator)))


def _whisper_segment_audio(segmenter, config: Config, audio, samplerate: int, *, filename: str | None = None) -> pd.DataFrame:
    frequency_scale = (
        float(config.frequency_scale)
        if config.frequency_scale is not None
        else float(getattr(segmenter, "default_segmentation_config", {}).get("frequency_scale", 1.0))
    )
    if frequency_scale <= 0:
        raise ValueError("frequency_scale must be positive.")
    original_sr = int(samplerate)
    model_sr = _whisper_model_samplerate(segmenter)
    if model_sr is None:
        model_input_sr = original_sr
        effective_sr = _whisper_effective_samplerate(original_sr, frequency_scale)
    else:
        model_input_sr = _whisper_effective_samplerate(model_sr, 1.0 / frequency_scale)
        effective_sr = model_sr
    normalized_audio = _whisper_normalize_audio_array(audio)
    if model_input_sr != original_sr:
        normalized_audio = _resample_raw_audio_array(normalized_audio, original_sr, model_input_sr)
    time_scale = effective_sr / model_input_sr
    prediction = segmenter.segment(
        normalized_audio,
        effective_sr,
        min_frequency=config.min_frequency,
        spec_time_step=config.spec_time_step,
        max_length=int(config.generation_max_length),
        batch_size=int(config.batch_size),
        num_trials=int(config.num_trials),
        num_beams=int(config.num_beams),
        top_k=int(config.top_k),
        top_p=float(config.top_p),
        length_penalty=float(config.length_penalty),
    )
    frame = _whisper_prediction_to_dataframe(prediction, filename=filename)
    return _whisper_scale_prediction_time(frame, time_scale)


def _build_whisperseg_report_datamodule(config: Config, data_dir: str | Path) -> AudioDirDataModule:
    return AudioDirDataModule(
        data_dir=data_dir,
        batch_size=int(config.batch_size),
        num_time_steps=int(config.num_time_steps),
        val_ratio=float(config.validation_fraction),
        test_ratio=float(config.test_fraction),
        split_within_files=bool(config.split_within_files),
        min_annotation_duration_s=float(config.min_annotation_duration_ms) / 1000.0,
        ignore_class_names=bool(config.ignore_class_names),
        include_labels=config.include_labels,
        num_workers=0,
        persistent_workers=False,
        audio_dataset=config.audio_dataset,
        data_samplerate_hz=config.data_samplerate_hz,
        split_seed=config.seed,
    )


def _whisperseg_report_frame_rate_hz(config: Config, segmenter) -> float:
    spec_time_step = config.spec_time_step
    if spec_time_step is None:
        spec_time_step = getattr(segmenter, "default_segmentation_config", {}).get("spec_time_step", None)
    if spec_time_step is None:
        spec_time_step = 0.0025
    spec_time_step = float(spec_time_step)
    if spec_time_step <= 0:
        raise ValueError("spec_time_step must be positive for WhisperSeg evaluation.")
    return 1.0 / spec_time_step


def _whisperseg_report_class_names(segmenter, annotation_tables: list[pd.DataFrame], prediction_tables: list[pd.DataFrame]) -> list[str]:
    codebook = getattr(segmenter, "cluster_codebook", {}) or {}
    ordered_names = [
        str(name)
        for name, _cluster_id in sorted(codebook.items(), key=lambda item: int(item[1]))
        if str(name) != "noise"
    ]
    seen = set(ordered_names)
    for table in [*annotation_tables, *prediction_tables]:
        for name in table.get("name", pd.Series(dtype=str)).dropna().map(str):
            if name != "noise" and name not in seen:
                ordered_names.append(name)
                seen.add(name)
    if not ordered_names:
        ordered_names.append("Vocal")
    return ["noise", *ordered_names]


def _whisperseg_report_annotation_table(annotations: pd.DataFrame, *, ignore_class_names: bool) -> pd.DataFrame:
    table = annotations.copy()
    if ignore_class_names and "name" in table.columns:
        table["name"] = "Vocal"
    return table


def _annotation_table_to_probabilities(
    annotation_table: pd.DataFrame,
    *,
    class_names: list[str],
    frame_count: int,
    frame_rate_hz: float,
) -> np.ndarray:
    probabilities = np.zeros((max(int(frame_count), 0), len(class_names)), dtype=np.float32)
    if probabilities.shape[0] == 0:
        return probabilities
    probabilities[:, 0] = 1.0
    name_to_index = {str(name): index for index, name in enumerate(class_names)}
    for row in prediction_results.load_reference_syllables(annotation_file=annotation_table, class_names=class_names):
        label_index = name_to_index[str(row["name"])]
        start_frame = max(int(np.floor(float(row["start_seconds"]) * frame_rate_hz)), 0)
        stop_frame = min(int(np.ceil(float(row["stop_seconds"]) * frame_rate_hz)), probabilities.shape[0])
        if stop_frame > start_frame:
            probabilities[start_frame:stop_frame, :] = 0.0
            probabilities[start_frame:stop_frame, label_index] = 1.0
    return probabilities


def _whisperseg_evaluation_components(
    config: Config,
    *,
    checkpoint_path: str,
    datamodule: AudioDirDataModule,
    split: str | None,
) -> tuple[np.ndarray, str, np.ndarray, str, float, dict[str, int], list[str]]:
    subset = (
        {
            "audio": datamodule.annotated_audio_files,
            "annotation_tables": datamodule.annotations,
        }
        if split is None
        else datamodule.subsets[str(split)]
    )
    audio_files = list(subset["audio"])
    annotation_tables = [
        _whisperseg_report_annotation_table(annotations, ignore_class_names=bool(config.ignore_class_names))
        for annotations in subset["annotation_tables"]
    ]
    if not audio_files:
        raise ValueError("Evaluation requires at least one annotated audio file.")

    segmenter_config = config.copy(mode="predict", checkpoint=checkpoint_path)
    segmenter = _whisper_segmenter(segmenter_config)
    frame_rate_hz = _whisperseg_report_frame_rate_hz(config, segmenter)
    prediction_tables: list[pd.DataFrame] = []
    loaded_audio: list[tuple[np.ndarray, int]] = []
    for audio_file in audio_files:
        audio, sr = load_audio_array(
            audio_file,
            audio_dataset=config.audio_dataset,
            data_samplerate_hz=config.data_samplerate_hz,
        )
        audio = _whisper_normalize_audio_array(audio)
        loaded_audio.append((audio, int(sr)))
        prediction_tables.append(_whisper_segment_audio(segmenter, segmenter_config, audio, int(sr), filename=Path(audio_file).name))

    class_names = _whisperseg_report_class_names(segmenter, annotation_tables, prediction_tables)
    annotated_entries = []
    syllable_pairs = []
    for audio_file, annotation_table, prediction_table, (audio, sr) in zip(
        audio_files,
        annotation_tables,
        prediction_tables,
        loaded_audio,
        strict=True,
    ):
        duration_seconds = len(audio) / float(sr)
        max_stop_seconds = duration_seconds
        if not annotation_table.empty:
            max_stop_seconds = max(max_stop_seconds, float(annotation_table["stop_seconds"].max()))
        if not prediction_table.empty:
            max_stop_seconds = max(max_stop_seconds, float(prediction_table["stop_seconds"].max()))
        frame_count = int(np.ceil(max_stop_seconds * frame_rate_hz))
        probabilities = _annotation_table_to_probabilities(
            prediction_table,
            class_names=class_names,
            frame_count=frame_count,
            frame_rate_hz=frame_rate_hz,
        )
        annotated_entries.append((Path(audio_file), annotation_table, probabilities))
        syllable_pairs.append(
            (
                prediction_results.load_reference_syllables(annotation_file=annotation_table, class_names=class_names),
                prediction_results.load_reference_syllables(annotation_file=prediction_table, class_names=class_names),
            )
        )

    dense_matrix, dense_report = prediction_results.classification_summary_from_files(
        annotated_entries=annotated_entries,
        class_names=class_names,
        frame_rate_hz=frame_rate_hz,
    )
    syllable_matrix, syllable_report, syllable_wer = prediction_results.syllable_classification_summary_from_syllable_pairs(
        syllable_pairs=syllable_pairs,
        class_names=class_names,
        tolerance_seconds=float(config.syllable_tolerance_ms) / 1000.0,
    )
    summary = {"evaluated_file_count": len(annotated_entries), "skipped_file_count": 0}
    return dense_matrix, dense_report, syllable_matrix, syllable_report, syllable_wer, summary, class_names


def _print_whisperseg_evaluation_report(
    config: Config,
    *,
    checkpoint_path: str,
    datamodule: AudioDirDataModule,
    split: str | None,
) -> dict[str, int]:
    dense_matrix, dense_report, syllable_matrix, syllable_report, syllable_wer, summary, class_names = (
        _whisperseg_evaluation_components(
            config,
            checkpoint_path=checkpoint_path,
            datamodule=datamodule,
            split=split,
        )
    )
    prediction_results.print_evaluation_report(
        PredictionEvaluation(
            summary=summary,
            class_names=class_names,
            dense_matrix=dense_matrix,
            dense_report=dense_report,
            syllable_matrix=syllable_matrix,
            syllable_report=syllable_report,
            syllable_wer=syllable_wer,
        )
    )
    return summary


def _train_whisperseg(
    config: Config,
    *,
    verbose: bool,
    stop_event: threading.Event | None,
    emit_epoch_logs: bool,
) -> str:
    from .whisperseg.train import train as train_whisperseg

    _seed_if_requested(config, verbose=verbose)
    data_dir = resolve_training_data_dir(config.data_dir)
    report_datamodule = _build_whisperseg_report_datamodule(config, data_dir) if verbose else None
    if verbose:
        _log_datamodule_split_stats(report_datamodule)
        print("Data prepared.")

    training_started_at = _training_start_timestamp()
    if verbose:
        print("Training starts.")
    checkpoint_path = train_whisperseg(
        model_folder=config.output_dir,
        train_dataset_folder=data_dir,
        initial_model_path=config.initial_model,
        device="auto" if config.accelerator == "tpu" else str(config.accelerator),
        n_device=int(config.num_devices or 1),
        num_epochs=int(config.num_epochs),
        validation_fraction=float(config.validation_fraction),
        test_fraction=float(config.test_fraction),
        max_length=int(config.max_length),
        total_spec_columns=int(config.total_spec_columns),
        batch_size=int(config.batch_size),
        learning_rate=float(config.learning_rate),
        linear_lr_schedule=bool(config.linear_lr_schedule),
        reduce_lr=bool(config.reduce_lr),
        reduce_lr_patience=int(config.reduce_lr_patience),
        reduce_lr_factor=float(config.reduce_lr_factor),
        reduce_lr_min=float(config.reduce_lr_min),
        early_stopping=bool(config.early_stopping),
        early_stopping_patience=int(config.early_stopping_patience),
        seed=config.seed,
        weight_decay=float(config.weight_decay),
        warmup_steps=int(config.warmup_steps),
        freeze_encoder=bool(config.freeze_encoder),
        encoder_dropout=float(config.encoder_dropout),
        decoder_dropout=float(config.decoder_dropout),
        num_workers=int(config.num_workers),
        ignore_class_names=bool(config.ignore_class_names),
        include_labels=config.include_labels,
        audio_dataset=config.audio_dataset,
        data_samplerate_hz=config.data_samplerate_hz,
        max_num_steps_per_epoch=config.max_num_steps_per_epoch,
        min_frequency=config.min_frequency,
        frequency_scale=1.0 if config.frequency_scale is None else float(config.frequency_scale),
        spec_time_step=config.spec_time_step,
        stop_event=stop_event,
        emit_epoch_logs=emit_epoch_logs and verbose,
        verbose=verbose,
        checkpoint_filename=_checkpoint_filename(config.checkpoint_prefix, training_started_at),
        checkpoint_metadata=_whisperseg_checkpoint_metadata(config=config, timestamp=training_started_at),
    )
    if verbose:
        print(f"Training ended: checkpoint={checkpoint_path}")
    if verbose and report_datamodule is not None and report_datamodule.has_test:
        print("Evaluating.")
        _print_whisperseg_evaluation_report(
            config,
            checkpoint_path=str(checkpoint_path),
            datamodule=report_datamodule,
            split="test",
        )
    return str(checkpoint_path)


def _predict_whisperseg(
    config: Config,
    *,
    audio: RawAudioLike | None,
    samplerate: int | None,
    verbose: bool,
    stop_event: threading.Event | None = None,
) -> list[str] | pd.DataFrame:
    if not config.checkpoint:
        raise ValueError("Provide checkpoint for prediction.")

    _seed_if_requested(config, verbose=verbose)
    segmenter = _whisper_segmenter(config)
    model_samplerate = _whisper_model_samplerate(segmenter)
    if audio is not None:
        if config.evaluate:
            raise ValueError("Evaluation is only supported for data_dir prediction.")
        if samplerate is None:
            raise ValueError("samplerate is required when predicting from raw audio.")
        if verbose:
            _log_prediction_samplerates([int(samplerate)], model_samplerate or int(samplerate))
        frame = _whisper_segment_audio(segmenter, config, audio, int(samplerate))
        if verbose:
            _print_prediction_summary(frame)
        return frame

    if not config.data_dir:
        raise ValueError("Provide data_dir for prediction.")
    write_outputs = config.output_dir != ""
    output_dir = Path(config.output_dir).expanduser() if config.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    written_files: list[str] = []
    source_samplerates = []
    audio_paths = _pending_prediction_audio_paths(config, _whisper_audio_paths_for_config(config))
    for audio_path in tqdm(
        audio_paths,
        desc="Predicting files",
        unit="file",
        disable=not verbose or len(audio_paths) < 2,
    ):
        if stop_event is not None and stop_event.is_set():
            break
        audio_data, sr = load_audio_array(
            audio_path,
            audio_dataset=config.audio_dataset,
            data_samplerate_hz=config.data_samplerate_hz,
        )
        audio_data = _whisper_normalize_audio_array(audio_data)
        source_samplerates.append(int(sr))
        frame = _whisper_segment_audio(segmenter, config, audio_data, int(sr), filename=audio_path.name)
        frames.append(frame)
        if write_outputs:
            output_path = prediction_results.prediction_output_path(
                audio_path,
                output_dir=output_dir,
                output_suffix=str(config.output_suffix),
            )
            if config.existing_annotations == "skip" and output_path.exists():
                continue
            prediction_results.write_annotation_file(
                output_path,
                frame.drop(columns=["filename"], errors="ignore"),
                merge=config.existing_annotations == "merge",
            )
            written_files.append(str(output_path))

    result: list[str] | pd.DataFrame
    if write_outputs:
        result = written_files
    else:
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filename", "name", "start_seconds", "stop_seconds"])
    if verbose:
        log_model_samplerate = model_samplerate or (source_samplerates[0] if source_samplerates else 0)
        _log_prediction_samplerates(source_samplerates, log_model_samplerate)
        if config.evaluate and not bool(stop_event is not None and stop_event.is_set()):
            report_datamodule = _build_whisperseg_report_datamodule(config, resolve_training_data_dir(config.data_dir))
            _print_whisperseg_evaluation_report(
                config,
                checkpoint_path=str(config.checkpoint),
                datamodule=report_datamodule,
                split=config.split,
            )
        else:
            _print_prediction_summary(frames)
    return result


def _is_whisperseg_predict_checkpoint(source: str) -> bool:
    if not source:
        return False
    path = Path(source).expanduser()
    if not path.is_file() or path.suffix != ".ckpt":
        return False
    try:
        metadata = _load_checkpoint_metadata(str(path))
    except Exception:
        return False
    if metadata.get("backend") != "whisperseg":
        return False

    from .whisperseg.model import is_model_checkpoint_path

    return is_model_checkpoint_path(path)


def _is_whisperseg_predict_source(source: str) -> bool:
    return _is_whisperseg_predict_checkpoint(source)


def _predict_checkpoint_metadata(
    *,
    config: Config,
    num_time_steps: int,
    chunk_stride: int | None,
    timestamp: str,
) -> dict[str, object]:
    return {
        "version": 1,
        "predict": {
            "num_time_steps": int(num_time_steps),
            "chunk_stride": None if chunk_stride is None else int(chunk_stride),
            "batch_size": int(config.batch_size),
            "fill_gap_ms": float(config.fill_gap_ms),
            "min_syllable_ms": float(config.min_syllable_ms),
            "segment_threshold_low": float(config.segment_threshold_low),
            "segment_threshold_high": float(config.segment_threshold_high),
            "event_threshold": float(config.event_threshold),
            "event_dist_min_ms": float(config.event_dist_min_ms),
            "event_dist_max_ms": None if config.event_dist_max_ms is None else float(config.event_dist_max_ms),
            "syllable_postprocessor": str(config.syllable_postprocessor),
        },
        "train": {
            "started_at": timestamp,
            "checkpoint_prefix": config.checkpoint_prefix,
            "initial_model": None,
            "freeze_encoder": bool(config.freeze_encoder),
        },
    }


def _whisperseg_checkpoint_metadata(
    *,
    config: Config,
    timestamp: str,
) -> dict[str, object]:
    return {
        "version": 1,
        "backend": "whisperseg",
        "train": {
            "started_at": timestamp,
            "checkpoint_prefix": config.checkpoint_prefix,
            "initial_model": config.initial_model,
            "freeze_encoder": bool(config.freeze_encoder),
            "linear_lr_schedule": bool(config.linear_lr_schedule),
            "reduce_lr": bool(config.reduce_lr),
            "reduce_lr_patience": int(config.reduce_lr_patience),
            "reduce_lr_factor": float(config.reduce_lr_factor),
            "reduce_lr_min": float(config.reduce_lr_min),
            "early_stopping": bool(config.early_stopping),
            "early_stopping_patience": int(config.early_stopping_patience),
            "weight_decay": float(config.weight_decay),
            "warmup_steps": int(config.warmup_steps),
        },
    }


def _load_checkpoint_metadata(source: str) -> dict[str, object]:
    checkpoint = torch.load(source, map_location="cpu")
    metadata = checkpoint.get("das")
    return metadata if isinstance(metadata, dict) else {}


def convert_legacy_checkpoint(source: str, destination: str) -> str:
    metadata = {
        "version": 1,
        "converted_from": "legacy_das",
    }
    model, params = legacy_to_das_model(source, checkpoint_metadata=metadata)
    metadata["predict"] = {
        "num_time_steps": int(params["nb_hist"]),
        "chunk_stride": int(params["stride"]),
        "batch_size": None,
        "fill_gap_ms": None,
        "min_syllable_ms": None,
    }
    model.checkpoint_metadata = metadata

    output_path = Path(destination).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, object] = {
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "das": dict(model.checkpoint_metadata),
        "pytorch-lightning_version": L.__version__,
    }
    torch.save(checkpoint, output_path)
    return str(output_path)


def _reject_npy_dir_for_command(data_dir: str, command: str) -> None:
    if is_npy_dir(data_dir):
        raise ValueError(f"Legacy npy_dir datasets are only supported for `das train`, not `das {command}`.")


def _infer_samplerate_for_config(data_dir: str, config: Config) -> int:
    if config.target_samplerate_hz is not None:
        return int(round(config.target_samplerate_hz))
    return _infer_samplerate(
        data_dir,
        audio_dataset=config.audio_dataset,
        data_samplerate_hz=config.data_samplerate_hz,
    )


def _build_train_datamodule(
    config: Config,
) -> tuple[AudioDirDataModule | NPYDirDataModule, float, bool]:
    num_workers = int(config.num_workers)
    frontend_config = config.frontend_mapping(include_raw_num_channels=True)
    data_dir = resolve_training_data_dir(config.data_dir)

    legacy_directory = None
    if is_legacy_store(data_dir):
        legacy_directory, data_dir = materialize_legacy_store(data_dir)

    if is_npy_dir(data_dir):
        attrs = load_npy_dir_attrs(data_dir)
        samplerate = float(attrs["samplerate_x_Hz"])
        datamodule = NPYDirDataModule(
            data_dir=data_dir,
            batch_size=int(config.batch_size),
            num_time_steps=int(config.num_time_steps),
            chunk_stride=config.chunk_stride,
            hop_s=frontend_hop_seconds(frontend_config, sr=samplerate),
            ignore_class_names=bool(config.ignore_class_names),
            include_labels=config.include_labels,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        if legacy_directory is not None:
            datamodule._legacy_directory = legacy_directory
        return datamodule, samplerate, False

    samplerate = _infer_samplerate_for_config(data_dir, config)
    datamodule = AudioDirDataModule(
        data_dir=data_dir,
        batch_size=int(config.batch_size),
        num_time_steps=int(config.num_time_steps),
        chunk_stride=config.chunk_stride,
        hop_s=model_hop_seconds(frontend_config, config.encoder_mapping(), sr=samplerate),
        val_ratio=float(config.validation_fraction),
        test_ratio=float(config.test_fraction),
        split_within_files=bool(config.split_within_files),
        min_annotation_duration_s=float(config.min_annotation_duration_ms) / 1000.0,
        ignore_class_names=bool(config.ignore_class_names),
        include_labels=config.include_labels,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        target_samplerate=samplerate,
        audio_dataset=config.audio_dataset,
        data_samplerate_hz=config.data_samplerate_hz,
        split_seed=config.seed,
    )
    if not datamodule.annotated_audio_files:
        raise ValueError("Training requires at least one annotated audio file.")
    return datamodule, samplerate, bool(getattr(datamodule, "has_test", False))


def _api_config(config: ConfigLike = None, *, mode: str, overrides: Mapping[str, object], validate: bool = True) -> Config:
    if config is None:
        resolved = Config()
    elif isinstance(config, Config):
        resolved = config.copy()
    elif isinstance(config, Mapping):
        resolved = Config.from_mapping(config)
    else:
        raise TypeError("config must be a Config, mapping, or None.")

    payload = dict(overrides)
    payload["mode"] = mode
    resolved = Config.from_mapping(payload, base=resolved)
    if validate:
        resolved.validate()
    return resolved


def _seed_if_requested(config: Config, *, verbose: bool) -> None:
    if config.seed is not None:
        L.seed_everything(config.seed, workers=True, verbose=verbose)


def _log_datamodule_split_stats(datamodule) -> None:
    split_stats = getattr(datamodule, "split_stats", None)
    if split_stats is None:
        return

    stats = split_stats()
    if not stats:
        return

    table = Table(title="Dataset split summary", box=box.ASCII, show_header=True)
    table.add_column("split", no_wrap=True)
    table.add_column("audio files", justify="right", no_wrap=True)
    table.add_column("audio min", justify="right", no_wrap=True)
    table.add_column("annotation files", justify="right", no_wrap=True)
    table.add_column("annotations", justify="right", no_wrap=True)
    table.add_column("annotation min", justify="right", no_wrap=True)
    for row in stats:
        table.add_row(
            str(row["split"]),
            str(row["audio_file_count"]),
            f"{float(row['audio_minutes']):.2f}",
            str(row["annotation_file_count"]),
            "" if row.get("annotation_count") is None else str(row["annotation_count"]),
            f"{float(row['annotation_minutes']):.2f}",
        )

    Console(force_terminal=False, color_system=None, width=512).print(table)


def train(
    config: ConfigLike = None,
    *,
    verbose: bool = False,
    stop_event: threading.Event | None = None,
    emit_epoch_logs: bool = False,
    **overrides,
) -> str:
    config = _api_config(config, mode="train", overrides=overrides)
    if config.encoder_type == "whisperseg":
        return _train_whisperseg(
            config,
            verbose=verbose,
            stop_event=stop_event,
            emit_epoch_logs=emit_epoch_logs,
        )
    return _train_supervised(
        config,
        verbose=verbose,
        stop_event=stop_event,
        emit_epoch_logs=emit_epoch_logs,
    )


def _training_runtime(config: Config, datamodule) -> TrainingRuntime:
    input_num_channels = int(getattr(datamodule, "input_num_channels", 1))
    if input_num_channels > 1 and config.frontend_type not in {"raw", "conv_resnet", "sinc"}:
        raise ValueError("Multi-channel waveform inputs require the raw, ConvResNet, or Sinc frontend.")

    raw_chunk_stride = getattr(datamodule, "chunk_stride", config.chunk_stride)
    return TrainingRuntime(
        input_num_channels=input_num_channels,
        num_time_steps=int(getattr(datamodule, "num_time_steps", config.num_time_steps)),
        chunk_stride=None if raw_chunk_stride is None else int(raw_chunk_stride),
    )


def _checkpoint_callback(
    config: Config,
    *,
    output_dir: Path,
    monitor: str,
    timestamp: str,
    verbose: bool,
) -> ModelCheckpoint:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=False,
        filename=_checkpoint_filename(config.checkpoint_prefix, timestamp),
        enable_version_counter=False,
        verbose=verbose,
    )


def _training_callbacks(
    checkpoint_callback: ModelCheckpoint,
    *,
    extra_callbacks: list[object] | None = None,
    stop_event: threading.Event | None,
    verbose: bool,
    emit_epoch_logs: bool,
) -> list[object]:
    callbacks: list[object] = [checkpoint_callback, *(extra_callbacks or [])]
    if verbose and emit_epoch_logs:
        callbacks.extend((_EpochLogCallback(), TrainingLogProgress()))
    elif verbose:
        callbacks.append(TQDMProgressBar())
    if stop_event is not None:
        callbacks.append(_StopOnEventCallback(stop_event, verbose=verbose))
    return callbacks


def _training_trainer(
    config: Config,
    *,
    output_dir: Path,
    callbacks: list[object],
    train_loader,
    verbose: bool,
    emit_epoch_logs: bool,
):
    trainer_kwargs = dict(
        max_epochs=int(config.num_epochs),
        accelerator=str(config.accelerator),
        devices=_trainer_devices(config.num_devices),
        logger=CSVLogger(save_dir=str(output_dir), name="logs"),
        callbacks=callbacks,
        enable_progress_bar=verbose and not emit_epoch_logs,
        enable_model_summary=verbose,
    )
    if config.max_num_steps_per_epoch is not None:
        trainer_kwargs["limit_train_batches"] = min(int(config.max_num_steps_per_epoch), len(train_loader))
    return L.Trainer(**trainer_kwargs)


def _validation_loader_for_plateau(loader: DataLoader, max_batches: int | None) -> DataLoader:
    if max_batches is None or len(loader) <= int(max_batches):
        return loader
    count = min(len(loader.dataset), int(max_batches) * int(loader.batch_size))
    # ponytail: sampled validation detects plateau; use full validation when fit time permits.
    indices = np.linspace(0, len(loader.dataset) - 1, num=count, dtype=np.int64).tolist()
    return DataLoader(
        Subset(loader.dataset, indices),
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        persistent_workers=loader.persistent_workers and loader.num_workers > 0,
        pin_memory=loader.pin_memory,
    )


def _train_supervised(
    config: Config,
    *,
    verbose: bool,
    stop_event: threading.Event | None,
    emit_epoch_logs: bool,
) -> str:
    _seed_if_requested(config, verbose=verbose)

    datamodule, samplerate, auto_evaluate = _build_train_datamodule(config)
    if verbose:
        source_samplerates = sorted(getattr(datamodule, "source_samplerates", ()))
        if any(rate != samplerate for rate in source_samplerates):
            rates = ", ".join(_format_samplerate(rate) for rate in source_samplerates)
            label = "Audio sample rate" if len(source_samplerates) == 1 else "Audio sample rates"
            target = "median" if config.target_samplerate_hz is None else "configured"
            print(f"{label}: {rates}; resampling on the fly to {target} {_format_samplerate(samplerate)}.")
        _log_datamodule_split_stats(datamodule)
        print("Data prepared.")
    training_started_at = _training_start_timestamp()
    runtime = _training_runtime(config, datamodule)

    model = DASModel(
        **config.model_kwargs(
            num_classes=datamodule.num_classes,
            class_names=datamodule.class_names,
            class_types=getattr(datamodule, "class_types", None),
            sr=samplerate,
            frontend_num_channels=runtime.input_num_channels if config.frontend_type == "raw" else None,
            num_time_steps=runtime.num_time_steps,
            chunk_stride=runtime.chunk_stride,
        ),
        reduce_lr_monitor="val_loss" if datamodule.has_val else "train_loss",
        checkpoint_metadata=_predict_checkpoint_metadata(
            config=config,
            num_time_steps=runtime.num_time_steps,
            chunk_stride=runtime.chunk_stride,
            timestamp=training_started_at,
        ),
    )

    output_dir = Path(config.output_dir)
    monitor = "val_loss" if datamodule.has_val else "train_loss"
    checkpoint_callback = _checkpoint_callback(
        config,
        output_dir=output_dir,
        monitor=monitor,
        timestamp=training_started_at,
        verbose=verbose,
    )
    extra_callbacks = []
    if config.early_stopping:
        extra_callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode="min",
                patience=int(config.early_stopping_patience),
                min_delta=float(config.early_stopping_min_delta),
                verbose=verbose,
            )
        )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader() if datamodule.has_val else None
    if val_loader is not None:
        val_loader = _validation_loader_for_plateau(val_loader, config.max_num_steps_per_epoch)
    trainer = _training_trainer(
        config,
        output_dir=output_dir,
        callbacks=_training_callbacks(
            checkpoint_callback,
            extra_callbacks=extra_callbacks,
            stop_event=stop_event,
            verbose=verbose,
            emit_epoch_logs=emit_epoch_logs,
        ),
        train_loader=train_loader,
        verbose=verbose,
        emit_epoch_logs=emit_epoch_logs,
    )

    if verbose:
        print("Training starts.")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    checkpoint_path = checkpoint_callback.best_model_path
    if verbose:
        print(f"Training ended: checkpoint={checkpoint_path}")
    if auto_evaluate:
        if verbose:
            print("Evaluating.")
        evaluate(config.copy(mode="predict", checkpoint=checkpoint_path, evaluate=True, split="test"), verbose=verbose)
    return checkpoint_path


def _build_inference_datamodule(*, data_dir: str, config: Config, runtime: PredictRuntime) -> AudioDirDataModule:
    num_workers = int(config.num_workers)
    data_dir = resolve_training_data_dir(data_dir)
    return AudioDirDataModule(
        data_dir=data_dir,
        batch_size=int(config.batch_size),
        num_time_steps=int(runtime.num_time_steps),
        chunk_stride=runtime.chunk_stride,
        hop_s=float(runtime.hop_seconds),
        class_names=list(runtime.class_names),
        val_ratio=float(config.validation_fraction),
        test_ratio=float(config.test_fraction),
        split_within_files=bool(config.split_within_files),
        ignore_class_names=bool(config.ignore_class_names),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        target_samplerate=int(round(runtime.sr)),
        audio_dataset=config.audio_dataset,
        data_samplerate_hz=config.data_samplerate_hz,
        split_seed=config.seed,
    )


def _build_inference_trainer(config: Config, *, verbose: bool = False):
    return L.Trainer(
        accelerator=str(config.accelerator),
        devices=_trainer_devices(config.num_devices),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
    )


def _prepare_inference(config: Config, *, verbose: bool = False) -> InferenceRuntime:
    model, runtime = _load_predict_model(config.checkpoint, config=config)
    datamodule = _build_inference_datamodule(data_dir=config.data_dir, config=config, runtime=runtime)
    if verbose:
        _log_prediction_samplerates(getattr(datamodule, "source_samplerates", []), runtime.sr)
    trainer = _build_inference_trainer(config, verbose=verbose)
    device = getattr(getattr(trainer, "strategy", None), "root_device", torch.device("cpu"))
    return InferenceRuntime(
        model=model,
        runtime=runtime,
        datamodule=datamodule,
        trainer=trainer,
        device=device,
        context=_prediction_result_context(config, runtime),
    )


def _load_stage_dataloader(datamodule, stage: SplitName | str, *, shuffle: bool | None = None):
    if shuffle is not None and hasattr(datamodule, "_dataloader"):
        loader = datamodule._dataloader(stage, shuffle=shuffle)
    else:
        loader = getattr(datamodule, f"{stage}_dataloader")()
    empty_messages = {
        "train": "No training chunks were created. Increase audio duration or use a checkpoint trained with a smaller chunk length.",
        "val": "No validation chunks were created. Increase audio duration or use a checkpoint trained with a smaller chunk length.",
        "test": "No test chunks were created. Increase audio duration or use a checkpoint trained with a smaller chunk length.",
        "predict": "No prediction chunks were created. Increase audio duration or use a checkpoint trained with a smaller chunk length.",
    }
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "__len__") and len(dataset) == 0:
        raise ValueError(empty_messages[stage])
    return loader


def _add_prediction_file_progress(trainer, loader, *, verbose: bool) -> None:
    dataset = getattr(loader, "dataset", None)
    audio_files = getattr(dataset, "audio_files", [])
    if not verbose or len(audio_files) < 2 or not hasattr(dataset, "chunk_borders"):
        return
    progress_bar = getattr(trainer, "progress_bar_callback", None)
    if progress_bar is not None:
        progress_bar.disable()
    trainer.callbacks.append(
        _PredictionFileProgressCallback(dataset.chunk_borders, getattr(loader, "batch_size", 1))
    )


def evaluate(config: ConfigLike = None, *, verbose: bool = False, **overrides) -> dict[str, int]:
    config = _api_config(config, mode="predict", overrides=overrides)
    _seed_if_requested(config, verbose=verbose)
    _reject_npy_dir_for_command(config.data_dir, "predict")
    if verbose:
        print("Evaluating.")

    inference = _prepare_inference(config, verbose=verbose)
    context = inference.context

    if config.split is None:
        loader = _load_stage_dataloader(inference.datamodule, "predict")
        predictions = inference.trainer.predict(inference.model, dataloaders=loader)
        report = prediction_results.evaluate_file_predictions(
            dataset=loader.dataset,
            predictions=predictions,
            context=context,
        )
    else:
        loader = _load_stage_dataloader(inference.datamodule, config.split)
        report = prediction_results.evaluate_supervised_predictions(
            model=inference.model,
            dataloader=loader,
            device=inference.device,
            context=context,
        )

    if verbose:
        prediction_results.print_evaluation_report(report)
    return report.summary


def predict(
    config: ConfigLike = None,
    *,
    audio: RawAudioLike | None = None,
    samplerate: int | None = None,
    verbose: bool = False,
    stop_event: threading.Event | None = None,
    **overrides,
) -> list[str] | pd.DataFrame | list[pd.DataFrame]:
    config = _api_config(config, mode="predict", overrides=overrides, validate=audio is None)
    if _is_whisperseg_predict_source(config.checkpoint):
        return _predict_whisperseg(
            config,
            audio=audio,
            samplerate=samplerate,
            verbose=verbose,
            stop_event=stop_event,
        )
    if audio is not None:
        return _predict_raw_audio(
            config=config,
            audio=audio,
            samplerate=samplerate,
            verbose=verbose,
        )

    _seed_if_requested(config, verbose=verbose)
    _reject_npy_dir_for_command(config.data_dir, "predict")

    inference = _prepare_inference(config, verbose=verbose)
    stage = config.split if (config.evaluate and config.split is not None) else "predict"
    all_audio_files = getattr(inference.datamodule, "all_audio_files", None)
    if stage == "predict" and all_audio_files is not None:
        inference.datamodule.all_audio_files = _pending_prediction_audio_paths(config, all_audio_files)
        if not inference.datamodule.all_audio_files:
            return []
    loader = _load_stage_dataloader(inference.datamodule, stage)
    if stop_event is not None and len(getattr(loader.dataset, "audio_files", [])) > 1:
        return _predict_files_until_stopped(
            config,
            inference=inference,
            audio_files=loader.dataset.audio_files,
            stop_event=stop_event,
            verbose=verbose,
        )
    _add_prediction_file_progress(inference.trainer, loader, verbose=verbose)

    predictions = inference.trainer.predict(inference.model, dataloaders=loader)
    context = inference.context
    prediction_result, prediction_annotations = _prediction_outputs(
        config,
        dataset=loader.dataset,
        predictions=predictions,
        context=context,
        verbose=verbose,
    )
    if not config.evaluate:
        if verbose:
            if prediction_annotations is not None:
                _print_prediction_summary(prediction_annotations)
        return prediction_result

    if verbose:
        print("Evaluating.")
        _print_prediction_evaluation(
            dataset=loader.dataset,
            predictions=predictions,
            context=context,
            written_file_count=len(prediction_result) if config.output_dir != "" else 0,
        )
    return prediction_result


def _prediction_outputs(
    config: Config,
    *,
    dataset,
    predictions,
    context: PredictionResultContext,
    verbose: bool,
) -> tuple[list[str] | list[pd.DataFrame], list[pd.DataFrame] | None]:
    annotations = None
    if config.output_dir == "" or (verbose and not config.evaluate):
        annotations = prediction_results.prediction_annotations(
            dataset=dataset,
            predictions=predictions,
            context=context,
        )
    if config.output_dir == "":
        return annotations or [], annotations
    return (
        prediction_results.write_prediction_outputs(
            output_dir=Path(config.output_dir) if config.output_dir else None,
            dataset=dataset,
            predictions=predictions,
            context=context,
            output_suffix=str(config.output_suffix),
            existing_annotations=str(config.existing_annotations),
            annotations=annotations,
        ),
        annotations,
    )


def _pending_prediction_audio_paths(config: Config, audio_paths) -> list[Path]:
    paths = [Path(path) for path in audio_paths]
    if config.existing_annotations != "skip" or config.output_dir == "":
        return paths
    output_dir = Path(config.output_dir).expanduser() if config.output_dir else None
    return [
        path
        for path in paths
        if not prediction_results.prediction_output_path(
            path,
            output_dir=output_dir,
            output_suffix=str(config.output_suffix),
        ).exists()
    ]


def _print_prediction_evaluation(*, dataset, predictions, context: PredictionResultContext, written_file_count: int) -> None:
    report = prediction_results.evaluate_file_predictions(
        dataset=dataset,
        predictions=predictions,
        context=context,
    )
    prediction_results.print_evaluation_report(
        PredictionEvaluation(
            summary={**report.summary, "written_file_count": written_file_count},
            class_names=report.class_names,
            dense_matrix=report.dense_matrix,
            dense_report=report.dense_report,
            syllable_matrix=report.syllable_matrix,
            syllable_report=report.syllable_report,
            syllable_wer=report.syllable_wer,
        )
    )


def _predict_files_until_stopped(
    config: Config,
    *,
    inference: InferenceRuntime,
    audio_files,
    stop_event: threading.Event,
    verbose: bool,
) -> list[str] | list[pd.DataFrame]:
    progress_bar = getattr(inference.trainer, "progress_bar_callback", None)
    if progress_bar is not None:
        progress_bar.disable()
    outputs: list[str] | list[pd.DataFrame] = []
    summary_annotations: list[pd.DataFrame] = []
    for audio_file in tqdm(audio_files, desc="Predicting files", unit="file", disable=not verbose):
        if stop_event.is_set():
            break
        datamodule = _build_inference_datamodule(data_dir=str(audio_file), config=config, runtime=inference.runtime)
        loader = _load_stage_dataloader(datamodule, "predict")
        predictions = inference.trainer.predict(inference.model, dataloaders=loader)
        file_outputs, annotations = _prediction_outputs(
            config,
            dataset=loader.dataset,
            predictions=predictions,
            context=inference.context,
            verbose=verbose,
        )
        outputs.extend(file_outputs)
        summary_annotations.extend(annotations or [])
        if config.evaluate and verbose:
            print(f"Evaluating {Path(audio_file).name}.")
            _print_prediction_evaluation(
                dataset=loader.dataset,
                predictions=predictions,
                context=inference.context,
                written_file_count=len(file_outputs) if config.output_dir != "" else 0,
            )
    if verbose and not config.evaluate:
        _print_prediction_summary(summary_annotations)
    return outputs


def _predict_raw_audio(
    *,
    config: Config,
    audio,
    samplerate: int | None,
    verbose: bool,
) -> pd.DataFrame | list[pd.DataFrame]:
    if not config.checkpoint:
        raise ValueError("Provide checkpoint for prediction.")
    if config.evaluate:
        raise ValueError("Evaluation is only supported for data_dir prediction.")
    if config.split is not None:
        raise ValueError("split is only supported for data_dir prediction.")

    _seed_if_requested(config, verbose=verbose)
    model, runtime = _load_predict_model(config.checkpoint, config=config)
    model_samplerate = int(round(runtime.sr))
    input_samplerate = model_samplerate if samplerate is None else int(samplerate)
    if verbose:
        _log_prediction_samplerates([input_samplerate], model_samplerate)
    audio_arrays, single_input = _raw_audio_inputs(audio)
    audio_arrays = [_resample_raw_audio_array(item, input_samplerate, model_samplerate) for item in audio_arrays]
    dataset = _RawAudioDataset(
        audio_arrays=audio_arrays,
        samplerate=model_samplerate,
        num_time_steps=int(runtime.num_time_steps),
        chunk_stride=runtime.chunk_stride,
        hop_s=float(runtime.hop_seconds),
        class_names=list(runtime.class_names),
    )
    if len(dataset) == 0:
        raise ValueError(
            "No prediction chunks were created. Increase audio duration or use a checkpoint trained with a smaller chunk length."
        )

    loader = DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        persistent_workers=int(config.num_workers) > 0,
        pin_memory=torch.cuda.is_available(),
    )
    trainer = _build_inference_trainer(config, verbose=verbose)
    predictions = trainer.predict(model, dataloaders=loader)

    annotations = prediction_results.prediction_annotations(
        dataset=dataset,
        predictions=predictions,
        context=_prediction_result_context(config, runtime),
    )
    if verbose:
        _print_prediction_summary(annotations)
    return annotations[0] if single_input else annotations


def _trainer_devices(num_devices: int | None) -> str | int:
    return "auto" if num_devices is None else int(num_devices)


def _resolve_postprocessing_config(config: Config, runtime: PredictRuntime) -> tuple[float, float, float, float | None, str]:
    defaults = Config(mode="predict", output_dir="")
    use_runtime_fill_gap = runtime.fill_gap_ms is not None and float(config.fill_gap_ms) == float(defaults.fill_gap_ms)
    use_runtime_min_syllable = runtime.min_syllable_ms is not None and float(config.min_syllable_ms) == float(
        defaults.min_syllable_ms
    )
    use_runtime_event_dist_min = runtime.event_dist_min_ms is not None and float(config.event_dist_min_ms) == float(
        defaults.event_dist_min_ms
    )
    use_runtime_event_dist_max = (
        runtime.event_dist_max_ms is not None and config.event_dist_max_ms == defaults.event_dist_max_ms
    )
    fill_gap_ms = float(runtime.fill_gap_ms) if use_runtime_fill_gap else float(config.fill_gap_ms)
    min_syllable_ms = float(runtime.min_syllable_ms) if use_runtime_min_syllable else float(config.min_syllable_ms)
    event_dist_min_ms = (
        float(runtime.event_dist_min_ms) if use_runtime_event_dist_min else float(config.event_dist_min_ms)
    )
    event_dist_max_ms = (
        float(runtime.event_dist_max_ms) if use_runtime_event_dist_max else config.event_dist_max_ms
    )
    syllable_postprocessor = (
        str(runtime.syllable_postprocessor)
        if runtime.syllable_postprocessor is not None
        and str(config.syllable_postprocessor) == str(defaults.syllable_postprocessor)
        and use_runtime_fill_gap
        and use_runtime_min_syllable
        else str(config.syllable_postprocessor)
    )
    return fill_gap_ms, min_syllable_ms, event_dist_min_ms, event_dist_max_ms, syllable_postprocessor


def _prediction_result_context(config: Config, runtime: PredictRuntime) -> PredictionResultContext:
    fill_gap_ms, min_syllable_ms, event_dist_min_ms, event_dist_max_ms, postprocessor = _resolve_postprocessing_config(
        config, runtime
    )
    defaults = Config(mode="predict")

    def threshold(name: str) -> float:
        value = getattr(config, name)
        checkpoint_value = getattr(runtime, name)
        return float(checkpoint_value if checkpoint_value is not None and value == getattr(defaults, name) else value)

    return PredictionResultContext(
        class_names=list(runtime.class_names),
        class_types=runtime.class_types,
        frame_rate_hz=1.0 / float(runtime.hop_seconds),
        fill_gap_seconds=fill_gap_ms / 1000.0,
        min_syllable_seconds=min_syllable_ms / 1000.0,
        event_dist_min_seconds=event_dist_min_ms / 1000.0,
        event_dist_max_seconds=None if event_dist_max_ms is None else event_dist_max_ms / 1000.0,
        tolerance_seconds=float(config.syllable_tolerance_ms) / 1000.0,
        postprocessor=postprocessor,
        segment_threshold_low=threshold("segment_threshold_low"),
        segment_threshold_high=threshold("segment_threshold_high"),
        event_threshold=threshold("event_threshold"),
        legacy_data_padding=int(runtime.legacy_data_padding),
    )


def _load_predict_model(source: str, *, config: Config) -> tuple[object, PredictRuntime]:
    if is_legacy_model_source(source):
        model, runtime = load_legacy_predictor(source)
        return model, PredictRuntime(
            sr=float(runtime["sr"]),
            hop_seconds=float(runtime["hop_seconds"]),
            num_time_steps=int(runtime["num_time_steps"]),
            chunk_stride=None if runtime["chunk_stride"] is None else int(runtime["chunk_stride"]),
            class_names=list(runtime["class_names"]),
            class_types=runtime.get("class_types"),
            syllable_postprocessor=runtime.get("syllable_postprocessor"),
            legacy_data_padding=int(runtime.get("data_padding", 0) or 0),
        )

    checkpoint_metadata = _load_checkpoint_metadata(source)
    predict_metadata = checkpoint_metadata.get("predict", {})
    if not isinstance(predict_metadata, dict):
        predict_metadata = {}
    model = DASModel.load_from_checkpoint(source)
    model_hparams = getattr(model, "hparams", {})
    frontend_config = normalize_frontend_config(model_hparams["frontend"])
    chunk_stride = predict_metadata.get("chunk_stride", model_hparams.get("chunk_stride", config.chunk_stride))
    if chunk_stride is not None:
        chunk_stride = int(chunk_stride)
    return model, PredictRuntime(
        sr=float(model_hparams["sr"]),
        hop_seconds=model_hop_seconds(frontend_config, model_hparams["encoder"], sr=float(model_hparams["sr"])),
        num_time_steps=int(predict_metadata.get("num_time_steps", model_hparams.get("num_time_steps", config.num_time_steps))),
        chunk_stride=chunk_stride,
        class_names=_checkpoint_class_names(model),
        class_types=model_hparams.get("class_types"),
        batch_size=None if predict_metadata.get("batch_size") is None else int(predict_metadata["batch_size"]),
        fill_gap_ms=None if predict_metadata.get("fill_gap_ms") is None else float(predict_metadata["fill_gap_ms"]),
        min_syllable_ms=(
            None if predict_metadata.get("min_syllable_ms") is None else float(predict_metadata["min_syllable_ms"])
        ),
        segment_threshold_low=(
            None if predict_metadata.get("segment_threshold_low") is None else float(predict_metadata["segment_threshold_low"])
        ),
        segment_threshold_high=(
            None if predict_metadata.get("segment_threshold_high") is None else float(predict_metadata["segment_threshold_high"])
        ),
        event_threshold=(
            None if predict_metadata.get("event_threshold") is None else float(predict_metadata["event_threshold"])
        ),
        event_dist_min_ms=(
            None
            if predict_metadata.get("event_dist_min_ms") is None
            else float(predict_metadata["event_dist_min_ms"])
        ),
        event_dist_max_ms=(
            None
            if predict_metadata.get("event_dist_max_ms") is None
            else float(predict_metadata["event_dist_max_ms"])
        ),
        syllable_postprocessor=(
            None if predict_metadata.get("syllable_postprocessor") is None else str(predict_metadata["syllable_postprocessor"])
        ),
    )


def _infer_samplerate(
    data_dir: str,
    *,
    audio_dataset: str | None = None,
    data_samplerate_hz: float | None = None,
) -> int:
    samplerates = []
    for path in iter_audio_candidate_paths(data_dir):
        try:
            info = audio_file_info(path, audio_dataset=audio_dataset, data_samplerate_hz=data_samplerate_hz)
        except Exception:
            continue
        samplerates.append(int(info["samplerate"]))
    if not samplerates:
        raise ValueError(f"No readable audio files found in '{data_dir}'.")
    return int(round(float(np.median(samplerates))))
