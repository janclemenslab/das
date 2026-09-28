from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .config import Config, format_config_yaml, load_yaml_config_mapping, save_config
from .data.audio_dir import (
    _audio_files_from_filepath_annotations,
    _collect_annotations_for_audio,
    _drop_annotation_boundary_rows,
    _read_annotation_tables,
    audio_file_info,
)
from .data.npy_dir import is_npy_dir, load_npy_dir_attrs

FRONTEND_OPTIONS = ["raw", "stft", "mel", "conv", "conv_resnet", "whisperseg"]
DAS_FRONTEND_OPTIONS = ["raw", "stft", "mel", "conv", "conv_resnet"]
PAD_MODE_OPTIONS = ["constant", "reflect", "replicate", "circular"]
ENCODER_OPTIONS = ["conformer", "tcn", "tweetynet", "whisperseg"]
DAS_ENCODER_OPTIONS = ["conformer", "tcn", "tweetynet"]
TCN_PADDING_OPTIONS = ["same", "causal"]
DECODER_OPTIONS = ["linear", "lstm", "conv", "attention", "whisperseg"]
DAS_DECODER_OPTIONS = ["linear", "lstm", "conv", "attention"]
ACCELERATOR_OPTIONS = ["auto", "cpu", "cuda", "mps", "tpu"]
PREDICT_SPLIT_OPTIONS = ["", "train", "val", "test"]
_DEFAULT_CONFIG = Config()


def browser_initial_path(path: str, *, selection_mode: str = "directory") -> str:
    candidate_value = _optional_str(path)
    if candidate_value is None:
        return str(Path.cwd())

    candidate = Path(candidate_value).expanduser()
    if selection_mode == "directory":
        if candidate.is_dir():
            return str(candidate)
    else:
        if candidate.is_file():
            return str(candidate.parent)
        if candidate.is_dir():
            return str(candidate)

    current = candidate
    while not current.exists() and current != current.parent:
        current = current.parent

    if current.exists() and current.is_file():
        return str(current.parent)
    if current.exists() and current.is_dir():
        return str(current)
    return str(Path.cwd())


def _split_csv_lines(value: str) -> list[str]:
    parts: list[str] = []
    for line in value.splitlines():
        for piece in line.split(","):
            piece = piece.strip()
            if piece:
                parts.append(piece)
    return parts


def _optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value = value.strip()
    if not value:
        return None
    return int(value)


def _optional_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    value = value.strip()
    if not value:
        return None
    return float(value)


def _optional_str(value: str) -> str | None:
    value = value.strip()
    return value or None


def _format_optional_number(value: int | float | None) -> str:
    if value is None:
        return ""
    return str(value)


def _format_sequence_inline(values: list[int] | list[bool]) -> str:
    return ", ".join(str(value).lower() if isinstance(value, bool) else str(value) for value in values)


def _parse_bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected a boolean value, got '{value}'.")


def _parse_bool_or_list(value: str | bool) -> bool | list[bool]:
    if isinstance(value, bool):
        return value
    parts = _split_csv_lines(value)
    if not parts:
        return False
    parsed = [_parse_bool_value(part) for part in parts]
    if len(parsed) == 1:
        return parsed[0]
    return parsed


def _parse_dilations(value: str) -> list[int]:
    parts = _split_csv_lines(value)
    if not parts:
        raise ValueError("TCN dilations cannot be empty.")
    return [int(part) for part in parts]


def _format_bool_or_list(value: bool | list[bool]) -> str:
    if isinstance(value, list):
        return _format_sequence_inline(value)
    return str(bool(value)).lower()


def _type_default_or_none(value: int | float, default: int | float) -> int | float | None:
    return None if value == default else value


@dataclass
class TrainPathsState:
    data_dir: str = ""
    output_dir: str = "./"
    initial_model: str = ""
    checkpoint_prefix: str = ""


@dataclass
class PredictPathsState:
    data_dir: str = ""
    checkpoint: str = ""
    output_dir: str = ""


@dataclass
class DataSectionState:
    batch_size: int = 8
    num_time_steps: int = 4096
    chunk_stride: str = ""
    num_workers: int = 0
    validation_fraction: float = _DEFAULT_CONFIG.validation_fraction
    test_fraction: float = _DEFAULT_CONFIG.test_fraction
    split_within_files: bool = _DEFAULT_CONFIG.split_within_files
    ignore_class_names: bool = _DEFAULT_CONFIG.ignore_class_names
    include_labels: list[str] = field(default_factory=list)


@dataclass
class PredictDataSectionState:
    batch_size: int = _DEFAULT_CONFIG.batch_size
    num_workers: int = _DEFAULT_CONFIG.num_workers


@dataclass
class ModelSelectionState:
    model_class: Literal["das", "whisperseg"] = "das"
    frontend_type: Literal["raw", "stft", "mel", "conv", "conv_resnet", "whisperseg"] = "mel"
    encoder_type: Literal["conformer", "tcn", "tweetynet", "whisperseg"] = "conformer"
    decoder_type: Literal["linear", "lstm", "conv", "attention", "whisperseg"] = "linear"


@dataclass
class EncoderSettingsState:
    freeze_encoder: bool = _DEFAULT_CONFIG.freeze_encoder


@dataclass
class STFTFrontendState:
    num_channels: int = 128
    kernel_size: int = 1024
    hop_seconds: float = 0.004
    fmin: float = 100.0
    fmax: str = ""
    trainable: bool = True


@dataclass
class MelFrontendState:
    num_channels: int = 128
    kernel_size: int = 1024
    hop_seconds: float = 0.004
    fmin: float = 100.0
    fmax: str = ""
    trainable: bool = True


@dataclass
class ConvFrontendState:
    num_channels: int = 128
    kernel_size: int = 9
    hop_seconds: float = 0.004
    pad_mode: Literal["constant", "reflect", "replicate", "circular"] = "constant"


@dataclass
class WhisperSegFrontendState:
    min_frequency: str = _format_optional_number(_DEFAULT_CONFIG.min_frequency)
    frequency_scale: str = str(_DEFAULT_CONFIG.frequency_scale)
    spec_time_step: str = ""


@dataclass
class ConformerEncoderState:
    num_heads: int = 4
    hidden_size: int = 128
    num_layers: int = 3
    kernel_size: int = 31


@dataclass
class TCNEncoderState:
    hidden_size: int = 128
    num_layers: int = 3
    dilations: str = "1, 2, 4, 8, 16"
    kernel_size: int = 16
    dropout: float = 0.1
    use_skip_connections: bool = True
    use_separable: str = "false"
    padding: Literal["same", "causal"] = "same"


@dataclass
class TweetynetEncoderState:
    hidden_size: int = 512
    num_layers: int = 1
    kernel_size: int = 5
    dropout: float = 0.1


@dataclass
class LSTMDecoderState:
    hidden_size: int = 64


@dataclass
class ConvDecoderState:
    kernel_size: int = 8


@dataclass
class AttentionDecoderState:
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class WhisperSegDecoderState:
    decoder_dropout: float = _DEFAULT_CONFIG.decoder_dropout
    max_length: int = _DEFAULT_CONFIG.max_length
    generation_max_length: int = _DEFAULT_CONFIG.generation_max_length
    num_trials: int = _DEFAULT_CONFIG.num_trials
    num_beams: int = _DEFAULT_CONFIG.num_beams
    top_k: int = _DEFAULT_CONFIG.top_k
    top_p: float = _DEFAULT_CONFIG.top_p
    length_penalty: float = _DEFAULT_CONFIG.length_penalty


@dataclass
class ModelHyperparametersState:
    cross_entropy_weight: float = _DEFAULT_CONFIG.cross_entropy_weight
    learning_rate: float = _DEFAULT_CONFIG.learning_rate
    early_stopping: bool = _DEFAULT_CONFIG.early_stopping
    early_stopping_patience: int = _DEFAULT_CONFIG.early_stopping_patience
    reduce_lr: bool = _DEFAULT_CONFIG.reduce_lr
    reduce_lr_patience: int = _DEFAULT_CONFIG.reduce_lr_patience
    reduce_lr_factor: float = _DEFAULT_CONFIG.reduce_lr_factor
    reduce_lr_min: float = _DEFAULT_CONFIG.reduce_lr_min
    linear_lr_schedule: bool = _DEFAULT_CONFIG.linear_lr_schedule
    weight_decay: float = _DEFAULT_CONFIG.weight_decay
    warmup_steps: int = _DEFAULT_CONFIG.warmup_steps


@dataclass
class TrainerSectionState:
    seed: str = ""
    accelerator: Literal["auto", "cpu", "cuda", "mps", "tpu"] = "auto"
    num_devices: str = ""
    num_epochs: int = 100
    max_num_steps_per_epoch: str = ""


@dataclass
class PredictTrainerSectionState:
    seed: str = ""
    accelerator: Literal["auto", "cpu", "cuda", "mps", "tpu"] = _DEFAULT_CONFIG.accelerator
    num_devices: str = ""


@dataclass
class TrainPredictionSectionState:
    output_suffix: str = _DEFAULT_CONFIG.output_suffix
    syllable_postprocessor: Literal["binary_mask", "label_aware_dense"] = _DEFAULT_CONFIG.syllable_postprocessor
    fill_gap_ms: float = _DEFAULT_CONFIG.fill_gap_ms
    min_syllable_ms: float = _DEFAULT_CONFIG.min_syllable_ms
    segment_threshold_low: float = _DEFAULT_CONFIG.segment_threshold_low
    segment_threshold_high: float = _DEFAULT_CONFIG.segment_threshold_high
    event_threshold: float = _DEFAULT_CONFIG.event_threshold
    event_dist_min_ms: float = _DEFAULT_CONFIG.event_dist_min_ms
    event_dist_max_ms: str = ""


@dataclass
class PredictionSectionState:
    output_suffix: str = _DEFAULT_CONFIG.output_suffix
    existing_annotations: str = _DEFAULT_CONFIG.existing_annotations
    fill_gap_ms: float = _DEFAULT_CONFIG.fill_gap_ms
    min_syllable_ms: float = _DEFAULT_CONFIG.min_syllable_ms
    syllable_postprocessor: str = _DEFAULT_CONFIG.syllable_postprocessor
    segment_threshold_low: float = _DEFAULT_CONFIG.segment_threshold_low
    segment_threshold_high: float = _DEFAULT_CONFIG.segment_threshold_high
    event_threshold: float = _DEFAULT_CONFIG.event_threshold
    event_dist_min_ms: float = _DEFAULT_CONFIG.event_dist_min_ms
    event_dist_max_ms: str = ""
    evaluate: bool = _DEFAULT_CONFIG.evaluate
    split: str = ""
    syllable_tolerance_ms: float = _DEFAULT_CONFIG.syllable_tolerance_ms


@dataclass
class EvaluationSectionState:
    syllable_tolerance_ms: float = 10.0


@dataclass
class TrainGuiState:
    paths: TrainPathsState = field(default_factory=TrainPathsState)
    data: DataSectionState = field(default_factory=DataSectionState)
    model_selection: ModelSelectionState = field(default_factory=ModelSelectionState)
    encoder_settings: EncoderSettingsState = field(default_factory=EncoderSettingsState)
    stft_frontend: STFTFrontendState = field(default_factory=STFTFrontendState)
    mel_frontend: MelFrontendState = field(default_factory=MelFrontendState)
    conv_frontend: ConvFrontendState = field(default_factory=ConvFrontendState)
    whisperseg_frontend: WhisperSegFrontendState = field(default_factory=WhisperSegFrontendState)
    conformer_encoder: ConformerEncoderState = field(default_factory=ConformerEncoderState)
    tcn_encoder: TCNEncoderState = field(default_factory=TCNEncoderState)
    tweetynet_encoder: TweetynetEncoderState = field(default_factory=TweetynetEncoderState)
    lstm_decoder: LSTMDecoderState = field(default_factory=LSTMDecoderState)
    conv_decoder: ConvDecoderState = field(default_factory=ConvDecoderState)
    attention_decoder: AttentionDecoderState = field(default_factory=AttentionDecoderState)
    whisperseg_decoder: WhisperSegDecoderState = field(default_factory=WhisperSegDecoderState)
    model_hyperparameters: ModelHyperparametersState = field(default_factory=ModelHyperparametersState)
    trainer: TrainerSectionState = field(default_factory=TrainerSectionState)
    prediction: TrainPredictionSectionState = field(default_factory=TrainPredictionSectionState)
    evaluation: EvaluationSectionState = field(default_factory=EvaluationSectionState)


@dataclass
class PredictGuiState:
    paths: PredictPathsState = field(default_factory=PredictPathsState)
    data: PredictDataSectionState = field(default_factory=PredictDataSectionState)
    trainer: PredictTrainerSectionState = field(default_factory=PredictTrainerSectionState)
    prediction: PredictionSectionState = field(default_factory=PredictionSectionState)
    whisperseg_decoder: WhisperSegDecoderState = field(default_factory=WhisperSegDecoderState)


def default_train_gui_state() -> TrainGuiState:
    return TrainGuiState()


def default_predict_gui_state() -> PredictGuiState:
    return PredictGuiState()


def train_gui_state_to_config(state: TrainGuiState) -> Config:
    frontend_type = str(state.model_selection.frontend_type)
    encoder_type = str(state.model_selection.encoder_type)
    decoder_type = str(state.model_selection.decoder_type)
    if (
        state.model_selection.model_class == "whisperseg"
        or frontend_type == "whisperseg"
        or encoder_type == "whisperseg"
        or decoder_type == "whisperseg"
    ):
        frontend_type = "whisperseg"
        encoder_type = "whisperseg"
        decoder_type = "whisperseg"
    frontend_num_channels = (
        None
        if frontend_type in {"raw", "whisperseg"}
        else _type_default_or_none(
            int(
                state.stft_frontend.num_channels
                if frontend_type == "stft"
                else state.mel_frontend.num_channels
                if frontend_type == "mel"
                else state.conv_frontend.num_channels
            ),
            128,
        )
    )
    frontend_kernel_size = (
        None
        if frontend_type in {"raw", "whisperseg"}
        else _type_default_or_none(
            int(
                state.stft_frontend.kernel_size
                if frontend_type == "stft"
                else state.mel_frontend.kernel_size
                if frontend_type == "mel"
                else state.conv_frontend.kernel_size
            ),
            1024 if frontend_type in {"stft", "mel"} else 9,
        )
    )
    frontend_hop_seconds = (
        None
        if frontend_type in {"raw", "whisperseg"}
        else _type_default_or_none(
            float(
                state.stft_frontend.hop_seconds
                if frontend_type == "stft"
                else state.mel_frontend.hop_seconds
                if frontend_type == "mel"
                else state.conv_frontend.hop_seconds
            ),
            0.004,
        )
    )
    if encoder_type == "conformer":
        encoder_hidden_size = int(state.conformer_encoder.hidden_size)
        encoder_num_layers = int(state.conformer_encoder.num_layers)
        encoder_kernel_size = _type_default_or_none(int(state.conformer_encoder.kernel_size), 31)
        encoder_dropout = float(state.tcn_encoder.dropout)
    elif encoder_type == "tweetynet":
        encoder_hidden_size = int(state.tweetynet_encoder.hidden_size)
        encoder_num_layers = int(state.tweetynet_encoder.num_layers)
        encoder_kernel_size = _type_default_or_none(int(state.tweetynet_encoder.kernel_size), 5)
        encoder_dropout = float(state.tweetynet_encoder.dropout)
    else:
        encoder_hidden_size = int(state.tcn_encoder.hidden_size)
        encoder_num_layers = int(state.tcn_encoder.num_layers)
        encoder_kernel_size = _type_default_or_none(int(state.tcn_encoder.kernel_size), 16)
        encoder_dropout = float(state.tcn_encoder.dropout)
    if decoder_type == "whisperseg":
        decoder_dropout = float(state.whisperseg_decoder.decoder_dropout)
    else:
        decoder_dropout = float(state.attention_decoder.dropout)
    return Config(
        mode="train",
        data_dir=state.paths.data_dir.strip(),
        output_dir=state.paths.output_dir.strip(),
        initial_model=_optional_str(state.paths.initial_model) or _DEFAULT_CONFIG.initial_model,
        checkpoint_prefix=_optional_str(state.paths.checkpoint_prefix),
        batch_size=int(state.data.batch_size),
        num_time_steps=int(state.data.num_time_steps),
        chunk_stride=_optional_int(state.data.chunk_stride),
        num_workers=int(state.data.num_workers),
        validation_fraction=float(state.data.validation_fraction),
        test_fraction=float(state.data.test_fraction),
        split_within_files=bool(state.data.split_within_files),
        ignore_class_names=bool(state.data.ignore_class_names),
        include_labels=list(state.data.include_labels),
        frontend_type=frontend_type,
        frontend_num_channels=frontend_num_channels,
        frontend_kernel_size=frontend_kernel_size,
        frontend_hop_seconds=frontend_hop_seconds,
        frontend_pad_mode=str(state.conv_frontend.pad_mode),
        frontend_fmin=float(state.stft_frontend.fmin if frontend_type == "stft" else state.mel_frontend.fmin),
        frontend_fmax=_optional_float(state.stft_frontend.fmax if frontend_type == "stft" else state.mel_frontend.fmax),
        frontend_trainable=bool(
            state.stft_frontend.trainable if frontend_type == "stft" else state.mel_frontend.trainable
        ),
        min_frequency=_optional_int(state.whisperseg_frontend.min_frequency),
        frequency_scale=_optional_float(state.whisperseg_frontend.frequency_scale),
        spec_time_step=_optional_float(state.whisperseg_frontend.spec_time_step),
        encoder_type=encoder_type,
        encoder_num_heads=int(state.conformer_encoder.num_heads),
        encoder_hidden_size=encoder_hidden_size,
        encoder_num_layers=encoder_num_layers,
        encoder_kernel_size=encoder_kernel_size,
        encoder_dilations=_parse_dilations(state.tcn_encoder.dilations),
        encoder_dropout=encoder_dropout,
        encoder_use_skip_connections=bool(state.tcn_encoder.use_skip_connections),
        encoder_use_separable=_parse_bool_or_list(state.tcn_encoder.use_separable),
        encoder_padding=str(state.tcn_encoder.padding),
        decoder_type=decoder_type,
        decoder_hidden_size=int(state.lstm_decoder.hidden_size),
        decoder_kernel_size=int(state.conv_decoder.kernel_size),
        decoder_num_heads=int(state.attention_decoder.num_heads),
        decoder_num_layers=int(state.attention_decoder.num_layers),
        decoder_dropout=decoder_dropout,
        max_length=int(state.whisperseg_decoder.max_length),
        generation_max_length=int(state.whisperseg_decoder.generation_max_length),
        num_trials=int(state.whisperseg_decoder.num_trials),
        num_beams=int(state.whisperseg_decoder.num_beams),
        top_k=int(state.whisperseg_decoder.top_k),
        top_p=float(state.whisperseg_decoder.top_p),
        length_penalty=float(state.whisperseg_decoder.length_penalty),
        cross_entropy_weight=float(state.model_hyperparameters.cross_entropy_weight),
        learning_rate=float(state.model_hyperparameters.learning_rate),
        early_stopping=bool(state.model_hyperparameters.early_stopping),
        early_stopping_patience=int(state.model_hyperparameters.early_stopping_patience),
        reduce_lr=bool(state.model_hyperparameters.reduce_lr),
        reduce_lr_patience=int(state.model_hyperparameters.reduce_lr_patience),
        reduce_lr_factor=float(state.model_hyperparameters.reduce_lr_factor),
        reduce_lr_min=float(state.model_hyperparameters.reduce_lr_min),
        linear_lr_schedule=bool(state.model_hyperparameters.linear_lr_schedule),
        weight_decay=float(state.model_hyperparameters.weight_decay),
        warmup_steps=int(state.model_hyperparameters.warmup_steps),
        freeze_encoder=bool(state.encoder_settings.freeze_encoder),
        seed=_optional_int(state.trainer.seed),
        accelerator=str(state.trainer.accelerator),
        num_devices=_optional_int(state.trainer.num_devices),
        num_epochs=int(state.trainer.num_epochs),
        max_num_steps_per_epoch=_optional_int(state.trainer.max_num_steps_per_epoch),
        output_suffix=state.prediction.output_suffix,
        syllable_postprocessor=str(state.prediction.syllable_postprocessor),
        fill_gap_ms=float(state.prediction.fill_gap_ms),
        min_syllable_ms=float(state.prediction.min_syllable_ms),
        segment_threshold_low=float(state.prediction.segment_threshold_low),
        segment_threshold_high=float(state.prediction.segment_threshold_high),
        event_threshold=float(state.prediction.event_threshold),
        event_dist_min_ms=float(state.prediction.event_dist_min_ms),
        event_dist_max_ms=_optional_float(state.prediction.event_dist_max_ms),
        syllable_tolerance_ms=float(state.evaluation.syllable_tolerance_ms),
    )


def predict_gui_state_to_config(state: PredictGuiState) -> Config:
    return Config(
        mode="predict",
        data_dir=state.paths.data_dir.strip(),
        checkpoint=state.paths.checkpoint.strip(),
        output_dir=_optional_str(state.paths.output_dir),
        batch_size=int(state.data.batch_size),
        num_workers=int(state.data.num_workers),
        seed=_optional_int(state.trainer.seed),
        accelerator=str(state.trainer.accelerator),
        num_devices=_optional_int(state.trainer.num_devices),
        output_suffix=state.prediction.output_suffix,
        existing_annotations=str(state.prediction.existing_annotations),
        fill_gap_ms=float(state.prediction.fill_gap_ms),
        min_syllable_ms=float(state.prediction.min_syllable_ms),
        segment_threshold_low=float(state.prediction.segment_threshold_low),
        segment_threshold_high=float(state.prediction.segment_threshold_high),
        event_threshold=float(state.prediction.event_threshold),
        event_dist_min_ms=float(state.prediction.event_dist_min_ms),
        event_dist_max_ms=_optional_float(state.prediction.event_dist_max_ms),
        generation_max_length=int(state.whisperseg_decoder.generation_max_length),
        syllable_postprocessor=str(state.prediction.syllable_postprocessor),
        evaluate=bool(state.prediction.evaluate),
        split=_optional_str(state.prediction.split),
        syllable_tolerance_ms=float(state.prediction.syllable_tolerance_ms),
        num_trials=int(state.whisperseg_decoder.num_trials),
        num_beams=int(state.whisperseg_decoder.num_beams),
        top_k=int(state.whisperseg_decoder.top_k),
        top_p=float(state.whisperseg_decoder.top_p),
        length_penalty=float(state.whisperseg_decoder.length_penalty),
    )


def train_config_to_gui_state(config: Config) -> TrainGuiState:
    frontend = config.frontend_mapping()
    encoder = config.encoder_mapping()
    decoder = config.decoder_mapping()
    state = default_train_gui_state()
    state.paths.data_dir = str(config.data_dir)
    state.paths.output_dir = str(config.output_dir)
    state.paths.initial_model = "" if config.initial_model == _DEFAULT_CONFIG.initial_model else str(config.initial_model)
    state.paths.checkpoint_prefix = "" if config.checkpoint_prefix is None else str(config.checkpoint_prefix)
    state.data.batch_size = int(config.batch_size)
    state.data.num_time_steps = int(config.num_time_steps)
    state.data.chunk_stride = _format_optional_number(config.chunk_stride)
    state.data.num_workers = int(config.num_workers)
    state.data.validation_fraction = float(config.validation_fraction)
    state.data.test_fraction = float(config.test_fraction)
    state.data.split_within_files = bool(config.split_within_files)
    state.data.ignore_class_names = bool(config.ignore_class_names)
    state.data.include_labels = list(config.include_labels)

    if "whisperseg" in {frontend["type"], encoder["type"], decoder["type"]}:
        state.model_selection.model_class = "whisperseg"
    state.model_selection.frontend_type = str(frontend["type"])
    state.encoder_settings.freeze_encoder = bool(config.freeze_encoder)
    if frontend["type"] == "raw":
        pass
    elif frontend["type"] == "stft":
        state.stft_frontend.num_channels = int(frontend["num_channels"])
        state.stft_frontend.kernel_size = int(frontend["kernel_size"])
        state.stft_frontend.hop_seconds = float(frontend["hop_seconds"])
        state.stft_frontend.fmin = float(frontend["fmin"])
        state.stft_frontend.fmax = _format_optional_number(frontend.get("fmax"))
        state.stft_frontend.trainable = bool(frontend.get("trainable", True))
    elif frontend["type"] == "mel":
        state.mel_frontend.num_channels = int(frontend["num_channels"])
        state.mel_frontend.kernel_size = int(frontend["kernel_size"])
        state.mel_frontend.hop_seconds = float(frontend["hop_seconds"])
        state.mel_frontend.fmin = float(frontend["fmin"])
        state.mel_frontend.fmax = _format_optional_number(frontend.get("fmax"))
        state.mel_frontend.trainable = bool(frontend.get("trainable", True))
    elif frontend["type"] in {"conv", "conv_resnet"}:
        state.conv_frontend.num_channels = int(frontend["num_channels"])
        state.conv_frontend.kernel_size = int(frontend["kernel_size"])
        state.conv_frontend.hop_seconds = float(frontend["hop_seconds"])
        state.conv_frontend.pad_mode = str(frontend["pad_mode"])
    elif frontend["type"] == "whisperseg":
        state.whisperseg_frontend.min_frequency = _format_optional_number(frontend.get("min_frequency"))
        state.whisperseg_frontend.frequency_scale = _format_optional_number(frontend.get("frequency_scale"))
        state.whisperseg_frontend.spec_time_step = _format_optional_number(frontend.get("spec_time_step"))

    state.model_selection.encoder_type = str(encoder["type"])
    if encoder["type"] == "conformer":
        state.conformer_encoder.num_heads = int(encoder["num_heads"])
        state.conformer_encoder.hidden_size = int(encoder["hidden_size"])
        state.conformer_encoder.num_layers = int(encoder["num_layers"])
        state.conformer_encoder.kernel_size = int(encoder["kernel_size"])
    elif encoder["type"] == "tweetynet":
        state.tweetynet_encoder.hidden_size = int(encoder["hidden_size"])
        state.tweetynet_encoder.num_layers = int(encoder["num_layers"])
        state.tweetynet_encoder.kernel_size = int(encoder["kernel_size"])
        state.tweetynet_encoder.dropout = float(encoder["dropout"])
    elif encoder["type"] == "tcn":
        state.tcn_encoder.hidden_size = int(encoder["hidden_size"])
        state.tcn_encoder.num_layers = int(encoder["num_layers"])
        state.tcn_encoder.dilations = _format_sequence_inline([int(value) for value in encoder["dilations"]])
        state.tcn_encoder.kernel_size = int(encoder["kernel_size"])
        state.tcn_encoder.dropout = float(encoder["dropout"])
        state.tcn_encoder.use_skip_connections = bool(encoder["use_skip_connections"])
        state.tcn_encoder.use_separable = _format_bool_or_list(encoder["use_separable"])
        state.tcn_encoder.padding = str(encoder["padding"])

    state.model_selection.decoder_type = str(decoder["type"])
    if decoder["type"] == "lstm":
        state.lstm_decoder.hidden_size = int(decoder["hidden_size"])
    elif decoder["type"] == "conv":
        state.conv_decoder.kernel_size = int(decoder["kernel_size"])
    elif decoder["type"] == "attention":
        state.attention_decoder.num_heads = int(decoder["num_heads"])
        state.attention_decoder.num_layers = int(decoder["num_layers"])
        state.attention_decoder.dropout = float(decoder["dropout"])
    elif decoder["type"] == "whisperseg":
        state.whisperseg_decoder.decoder_dropout = float(decoder["dropout"])
        state.whisperseg_decoder.max_length = int(decoder["max_length"])
        state.whisperseg_decoder.generation_max_length = int(decoder["generation_max_length"])
        state.whisperseg_decoder.num_trials = int(decoder["num_trials"])
        state.whisperseg_decoder.num_beams = int(decoder["num_beams"])
        state.whisperseg_decoder.top_k = int(decoder["top_k"])
        state.whisperseg_decoder.top_p = float(decoder["top_p"])
        state.whisperseg_decoder.length_penalty = float(decoder["length_penalty"])

    state.model_hyperparameters.cross_entropy_weight = float(config.cross_entropy_weight)
    state.model_hyperparameters.learning_rate = float(config.learning_rate)
    state.model_hyperparameters.early_stopping = bool(config.early_stopping)
    state.model_hyperparameters.early_stopping_patience = int(config.early_stopping_patience)
    state.model_hyperparameters.reduce_lr = bool(config.reduce_lr)
    state.model_hyperparameters.reduce_lr_patience = int(config.reduce_lr_patience)
    state.model_hyperparameters.reduce_lr_factor = float(config.reduce_lr_factor)
    state.model_hyperparameters.reduce_lr_min = float(config.reduce_lr_min)
    state.model_hyperparameters.linear_lr_schedule = bool(config.linear_lr_schedule)
    state.model_hyperparameters.weight_decay = float(config.weight_decay)
    state.model_hyperparameters.warmup_steps = int(config.warmup_steps)
    state.trainer.seed = _format_optional_number(config.seed)
    state.trainer.accelerator = str(config.accelerator)
    state.trainer.num_devices = _format_optional_number(config.num_devices)
    state.trainer.num_epochs = int(config.num_epochs)
    state.trainer.max_num_steps_per_epoch = _format_optional_number(config.max_num_steps_per_epoch)
    state.prediction.output_suffix = str(config.output_suffix)
    state.prediction.syllable_postprocessor = str(config.syllable_postprocessor)
    state.prediction.fill_gap_ms = float(config.fill_gap_ms)
    state.prediction.min_syllable_ms = float(config.min_syllable_ms)
    state.prediction.segment_threshold_low = float(config.segment_threshold_low)
    state.prediction.segment_threshold_high = float(config.segment_threshold_high)
    state.prediction.event_threshold = float(config.event_threshold)
    state.prediction.event_dist_min_ms = float(config.event_dist_min_ms)
    state.prediction.event_dist_max_ms = _format_optional_number(config.event_dist_max_ms)
    state.evaluation.syllable_tolerance_ms = float(config.syllable_tolerance_ms)
    return state


def predict_config_to_gui_state(config: Config) -> PredictGuiState:
    state = default_predict_gui_state()
    state.paths.data_dir = str(config.data_dir)
    state.paths.checkpoint = str(config.checkpoint)
    state.paths.output_dir = "" if config.output_dir is None else str(config.output_dir)
    state.data.batch_size = int(config.batch_size)
    state.data.num_workers = int(config.num_workers)
    state.trainer.seed = _format_optional_number(config.seed)
    state.trainer.accelerator = str(config.accelerator)
    state.trainer.num_devices = _format_optional_number(config.num_devices)
    state.prediction.output_suffix = str(config.output_suffix)
    state.prediction.existing_annotations = str(config.existing_annotations)
    state.prediction.fill_gap_ms = float(config.fill_gap_ms)
    state.prediction.min_syllable_ms = float(config.min_syllable_ms)
    state.prediction.syllable_postprocessor = str(config.syllable_postprocessor)
    state.prediction.segment_threshold_low = float(config.segment_threshold_low)
    state.prediction.segment_threshold_high = float(config.segment_threshold_high)
    state.prediction.event_threshold = float(config.event_threshold)
    state.prediction.event_dist_min_ms = float(config.event_dist_min_ms)
    state.prediction.event_dist_max_ms = _format_optional_number(config.event_dist_max_ms)
    state.prediction.evaluate = bool(config.evaluate)
    state.prediction.split = "" if config.split is None else str(config.split)
    state.prediction.syllable_tolerance_ms = float(config.syllable_tolerance_ms)
    state.whisperseg_decoder.generation_max_length = int(config.generation_max_length)
    state.whisperseg_decoder.num_trials = int(config.num_trials)
    state.whisperseg_decoder.num_beams = int(config.num_beams)
    state.whisperseg_decoder.top_k = int(config.top_k)
    state.whisperseg_decoder.top_p = float(config.top_p)
    state.whisperseg_decoder.length_penalty = float(config.length_penalty)
    return state


def _read_yaml_mapping(path: str) -> tuple[Path, Mapping[str, object]]:
    path_value = _optional_str(path)
    if path_value is None:
        raise ValueError("Provide a YAML config path to load.")
    config_path = Path(path_value).expanduser()
    return config_path, load_yaml_config_mapping(path_value)


def load_train_gui_state_from_mapping(payload: Mapping[str, object]) -> TrainGuiState:
    return train_config_to_gui_state(Config.from_mapping(payload, base=Config(mode="train")))


def load_train_gui_state_from_yaml(path: str) -> TrainGuiState:
    _, payload = _read_yaml_mapping(path)
    return load_train_gui_state_from_mapping(payload)


def load_predict_gui_state_from_mapping(payload: Mapping[str, object]) -> PredictGuiState:
    return predict_config_to_gui_state(Config.from_mapping(payload, base=Config(mode="predict")))


def load_predict_gui_state_from_yaml(path: str) -> PredictGuiState:
    _, payload = _read_yaml_mapping(path)
    return load_predict_gui_state_from_mapping(payload)


def format_train_config_yaml(config: Config) -> str:
    return format_config_yaml(config)


def format_predict_config_yaml(config: Config) -> str:
    return format_config_yaml(config)


def save_train_config(config: Config, destination: str) -> Path:
    return save_config(config, str(_optional_str(destination) or Path(config.output_dir) / "train-config.yaml"))


def save_predict_config(config: Config, destination: str) -> Path:
    return save_config(config, str(_optional_str(destination) or Path(config.output_dir or ".") / "predict-config.yaml"))


def _format_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def describe_dataset_path(data_dir: str) -> dict[str, object]:
    data_dir_value = _optional_str(data_dir)
    if data_dir_value is None:
        return {"status": "empty", "message": "Set a data directory to inspect the dataset."}

    root = Path(data_dir_value).expanduser()
    if not root.exists():
        return {"status": "error", "error": f"Data path '{root}' does not exist."}
    if not root.is_dir():
        return {"status": "error", "error": f"Data path '{root}' is not a directory."}

    if is_npy_dir(root):
        attrs = load_npy_dir_attrs(root)
        split_names: list[str] = []
        total_samples = 0
        for split in ("train", "val", "test"):
            x_path = root / split / "x.npy"
            if x_path.exists():
                split_names.append(split)
                total_samples += int(np.load(x_path, mmap_mode="r").shape[0])
        samplerate = float(attrs.get("samplerate_x_Hz", 1.0))
        classes = [str(name) for name in attrs.get("class_names", [])]
        return {
            "status": "ok",
            "dataset_type": "npy_dir",
            "path": str(root),
            "splits": split_names,
            "classes": classes,
            "total_duration_seconds": total_samples / samplerate if samplerate > 0 else 0.0,
            "total_duration_human": _format_duration(total_samples / samplerate if samplerate > 0 else 0.0),
            "note": "Instance counts are not available for npy_dir datasets.",
        }

    audio_file_count = 0
    annotated_audio_file_count = 0
    total_duration_seconds = 0.0
    class_counts: Counter[str] = Counter()
    annotation_errors: list[str] = []
    annotation_files = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".csv", ".json"})
    annotation_tables = _read_annotation_tables(annotation_files)
    audio_paths = [*sorted(root.rglob("*")), *_audio_files_from_filepath_annotations(annotation_tables)]
    seen_audio_paths = set()
    annotation_file_sources = set()

    for path in audio_paths:
        try:
            audio_path = Path(path).expanduser().resolve()
            if audio_path in seen_audio_paths:
                continue
            info = audio_file_info(path)
            seen_audio_paths.add(audio_path)
        except Exception:
            continue

        audio_file_count += 1
        total_duration_seconds += float(info["frames"]) / float(info["samplerate"])

        annotation_result = _collect_annotations_for_audio(path, annotation_tables)
        if annotation_result is None:
            continue

        annotation_sources, annotations = annotation_result
        annotations = _drop_annotation_boundary_rows(annotations)
        annotated_audio_file_count += 1
        annotation_file_sources.update(annotation_sources)

        if "name" not in annotations.columns:
            source_names = ", ".join(source.name for source in annotation_sources)
            annotation_errors.append(f"{source_names}: missing 'name' column")
            continue

        class_counts.update(str(value) for value in annotations["name"].dropna().tolist())

    if audio_file_count == 0:
        return {
            "status": "error",
            "error": f"No readable audio files were found in '{root}'.",
        }

    classes = sorted(class_counts)
    if annotation_file_sources and "noise" not in classes:
        classes = ["noise", *classes]

    return {
        "status": "ok",
        "dataset_type": "audio_dir",
        "path": str(root),
        "audio_file_count": audio_file_count,
        "annotation_file_count": len(annotation_file_sources),
        "unannotated_audio_file_count": audio_file_count - annotated_audio_file_count,
        "total_duration_seconds": total_duration_seconds,
        "total_duration_human": _format_duration(total_duration_seconds),
        "classes": classes,
        "class_counts": dict(sorted(class_counts.items())),
        "annotation_errors": annotation_errors,
    }
