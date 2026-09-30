from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Literal

import yaml

from .models.decoders import serialize_decoder_config
from .models.encoders import TcnPadding, serialize_encoder_config
from .models.frontends import PadMode, serialize_frontend_config

ModeName = Literal["train", "predict", "gui", "convert-legacy"]
SplitName = Literal["train", "val", "test"]
TrainerAccelerator = Literal["auto", "cpu", "cuda", "mps", "tpu"]
SyllablePostprocessor = Literal["binary_mask", "label_aware_dense"]
ExistingAnnotations = Literal["skip", "overwrite", "merge"]


@dataclass(frozen=True)
class _BackendDefault:
    native: int | float
    whisperseg: int | float

    def __repr__(self) -> str:
        return repr(self.native)


TRAIN_DATASET_HF_OPTIONS = [
    "nccratliri/bengalese-finch-subset-with-csv-label",
    "nccratliri/vad-animals",
    "nccratliri/vad-multi-species",
    "nccratliri/vad-zebra-finch",
    "nccratliri/vad-bengalese-finch",
    "nccratliri/vad-marmoset",
    "nccratliri/vad-mouse",
    "nccratliri/vad-human-ava-speech",
]

BUILTIN_CONFIG_FILES: dict[str, dict[str, str]] = {
    "fly": {"train": "fly-train.yaml", "predict": "fly-predict.yaml"},
    "fly-pulse": {
        "train": "fly-pulse-train.yaml",
        "predict": "fly-predict.yaml",
    },
    "zebra-finch": {
        "train": "zebra-finch-train.yaml",
        "predict": "zebra-finch-predict.yaml",
    },
    "tweetynet": {
        "train": "tweetynet-train.yaml",
        "predict": "tweetynet-predict.yaml",
    },
}
BUILTIN_CONFIG_OPTIONS = [
    ("fly-pulse", "Fly pulse"),
    ("fly", "Fly (classic)"),
    ("zebra-finch", "Zebra Finch"),
    ("tweetynet", "TweetyNet"),
]


def _bool_parser(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got '{value}'.")


def _split_inline_values(value: str | list[object] | tuple[object, ...]) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]

    normalized = str(value).strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    normalized = normalized.replace("\n", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _optional_int_parser(value: int | str | None) -> int | None:
    if value is None or isinstance(value, int):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(value)


def _optional_float_parser(value: float | int | str | None) -> float | None:
    if value is None or isinstance(value, (float, int)):
        return None if value is None else float(value)
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return float(value)


def _optional_string_parser(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


def _int_list_parser(value: str | list[object] | tuple[object, ...]) -> list[int]:
    return [int(part) for part in _split_inline_values(value)]


def _string_list_parser(value: str | list[object] | tuple[object, ...] | None) -> list[str]:
    if value is None:
        return []
    labels = []
    seen = set()
    for part in _split_inline_values(value):
        if part == "noise" or part in seen:
            continue
        labels.append(part)
        seen.add(part)
    return labels


def _bool_or_list_parser(value: bool | str | list[object] | tuple[object, ...]) -> bool | list[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_bool_parser(part) for part in value]

    parts = _split_inline_values(value)
    is_list_like = "," in str(value) or str(value).strip().startswith("[")
    if is_list_like:
        return [_bool_parser(part) for part in parts]
    if len(parts) != 1:
        raise ValueError(f"Expected a boolean or comma-separated list of booleans, got '{value}'.")
    return _bool_parser(parts[0])


def _mode_parser(value: str) -> ModeName:
    normalized = str(value).strip()
    if normalized not in {"train", "predict", "gui", "convert-legacy"}:
        raise ValueError(f"Expected mode to be one of train, predict, gui, convert-legacy, got '{value}'.")
    return normalized


def _split_parser(value: str | None) -> SplitName | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if normalized not in {"train", "val", "test"}:
        raise ValueError(f"Expected split to be one of train, val, test, got '{value}'.")
    return normalized


def _existing_annotations_parser(value: str) -> ExistingAnnotations:
    normalized = str(value).strip().lower()
    if normalized not in {"skip", "overwrite", "merge"}:
        raise ValueError(f"Expected existing_annotations to be one of skip, overwrite, merge, got '{value}'.")
    return normalized


def _initial_model_parser(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        return ""
    local_path = Path(normalized).expanduser()
    if local_path.is_file() and local_path.suffix == ".ckpt":
        return str(local_path)
    raise ValueError("initial_model must be a local DAS .ckpt file.")


def _identity(value):
    return value


def _config_field(
    default,
    *,
    help: str,
    parser: Callable[[object], object] = _identity,
    flags: list[str] | None = None,
    choices: list[str] | None = None,
    bool_flag: bool = False,
):
    return field(
        default=default,
        metadata={
            "help": help,
            "parser": parser,
            "flags": flags,
            "choices": choices,
            "bool_flag": bool_flag,
        },
    )


def _backend_default_field(
    native: int | float,
    whisperseg: int | float,
    *,
    help: str,
    parser: Callable[[object], object],
):
    return _config_field(_BackendDefault(native, whisperseg), help=help, parser=parser)


def _config_factory(
    factory: Callable[[], object],
    *,
    help: str,
    parser: Callable[[object], object] = _identity,
    flags: list[str] | None = None,
    choices: list[str] | None = None,
    bool_flag: bool = False,
):
    return field(
        default_factory=factory,
        metadata={
            "help": help,
            "parser": parser,
            "flags": flags,
            "choices": choices,
            "bool_flag": bool_flag,
        },
    )


@dataclass
class Config:
    mode: ModeName = _config_field(
        "train",
        help="Command mode.",
        parser=_mode_parser,
        choices=["train", "predict", "gui", "convert-legacy"],
    )
    data_dir: str = _config_field(
        "",
        help=(
            "Directory with audio files or npy_dir data. For training, this may also be a Hugging Face "
            "owner/dataset ID. Presets: " + ", ".join(TRAIN_DATASET_HF_OPTIONS) + "."
        ),
    )
    audio_dataset: str | None = _config_field(
        None,
        help="Optional dataset/key inside h5, hdf5, mat, zarr, or npz audio containers.",
        parser=_optional_string_parser,
    )
    data_samplerate_hz: float | None = _config_field(
        None,
        help="Optional sample rate for raw audio files or containers without embedded sample-rate metadata.",
        parser=_optional_float_parser,
    )
    target_samplerate_hz: float | None = _config_field(
        None,
        help="Optional sample rate in Hz to resample every input audio file to before training.",
        parser=_optional_float_parser,
    )
    output_dir: str | None = _config_field(
        None,
        help="Output directory. Prediction defaults to writing each annotation file next to its audio file.",
    )
    checkpoint_prefix: str | None = _config_field(
        None,
        help="Optional prefix for checkpoint filenames written during training.",
        parser=_optional_string_parser,
    )
    checkpoint: str = _config_field(
        "",
        help="DAS .ckpt checkpoint or legacy DAS H5/YAML model trunk for inference.",
    )
    initial_model: str = _config_field(
        "",
        help="DAS WhisperSeg .ckpt checkpoint used to start WhisperSeg training.",
        parser=_initial_model_parser,
        flags=["--initial-model"],
    )
    converted_checkpoint: str = _config_field(
        "",
        help="Destination checkpoint path for converted legacy DAS models.",
        flags=["--output"],
    )
    evaluate: bool = _config_field(
        False,
        help="Evaluate predictions when annotation CSV files are available.",
        parser=_bool_parser,
        bool_flag=True,
    )
    split: SplitName | None = _config_field(
        None,
        help="Optional dataset split to use for predict-side evaluation.",
        parser=_split_parser,
        choices=["train", "val", "test"],
    )
    batch_size: int = _backend_default_field(8, 4, help="Number of chunks per batch.", parser=int)
    num_time_steps: int = _config_field(4096, help="Chunk length in samples for training.", parser=int)
    chunk_stride: int | None = _config_field(
        None,
        help="Step size between chunks in samples. Leave unset for overlap defaults.",
        parser=_optional_int_parser,
    )
    num_workers: int = _config_field(0, help="Number of dataloader worker processes.", parser=int)
    validation_fraction: float = _backend_default_field(
        0.2, 0.1, help="Fraction of annotated audio files used for validation.", parser=float
    )
    test_fraction: float = _backend_default_field(
        0.2, 0.0, help="Fraction of annotated audio files held out as test data.", parser=float
    )
    split_within_files: bool = _config_field(
        False,
        help="Split train/validation/test chunks within each annotated audio file instead of assigning whole files to splits.",
        parser=_bool_parser,
        bool_flag=True,
    )
    min_annotation_duration_ms: float = _config_field(
        4.0,
        help="Minimum duration of audio_dir training labels in milliseconds. Shorter annotations are centered in this window.",
        parser=float,
    )
    ignore_class_names: bool = _config_field(
        False,
        help="Treat all annotations as a single segmentation class.",
        parser=_bool_parser,
        bool_flag=True,
    )
    include_labels: list[str] = _config_factory(
        list,
        help="Comma-separated non-noise label names to include during training. Leave empty to train on all labels.",
        parser=_string_list_parser,
    )
    frontend_type: Literal["raw", "stft", "mel", "conv", "conv_resnet", "sinc", "legacy_stft", "whisperseg"] = _config_field(
        "mel",
        help="Frontend type.",
        parser=str,
        flags=["--frontend"],
        choices=["raw", "stft", "mel", "conv", "conv_resnet", "sinc", "legacy_stft", "whisperseg"],
    )
    frontend_num_channels: int | None = _config_field(
        None,
        help="Frontend feature dimension or filter count.",
        parser=_optional_int_parser,
    )
    frontend_kernel_size: int | None = _config_field(
        None,
        help="Frontend kernel or FFT size.",
        parser=_optional_int_parser,
    )
    frontend_hop_seconds: float | None = _config_field(
        None,
        help="Seconds between frontend frames.",
        parser=_optional_float_parser,
    )
    frontend_pad_mode: PadMode = _config_field(
        "constant",
        help="Padding mode for convolutional frontends.",
        parser=str,
        choices=["constant", "reflect", "replicate", "circular"],
    )
    frontend_fmin: float = _config_field(100.0, help="Minimum STFT, mel, or Sinc frequency.", parser=float)
    frontend_fmax: float | None = _config_field(
        None,
        help="Maximum STFT, mel, or Sinc frequency. Leave unset to derive from sample rate.",
        parser=_optional_float_parser,
    )
    frontend_trainable: bool = _config_field(
        True,
        help="Whether STFT or mel frontend parameters are trainable.",
        parser=_bool_parser,
        bool_flag=True,
    )
    encoder_type: Literal["conformer", "tcn", "tweetynet", "whisperseg"] = _config_field(
        "conformer",
        help="Encoder type.",
        parser=str,
        flags=["--encoder"],
        choices=["conformer", "tcn", "tweetynet", "whisperseg"],
    )
    encoder_num_heads: int = _config_field(4, help="Number of conformer attention heads.", parser=int)
    encoder_hidden_size: int = _config_field(128, help="Encoder hidden size or filter count.", parser=int)
    encoder_num_layers: int = _config_field(3, help="Number of encoder layers or stacks.", parser=int)
    encoder_kernel_size: int | None = _config_field(
        None,
        help="Encoder kernel size.",
        parser=_optional_int_parser,
    )
    encoder_dilations: list[int] = _config_factory(
        lambda: [1, 2, 4, 8, 16],
        help="TCN dilation schedule.",
        parser=_int_list_parser,
    )
    encoder_dropout: float = _backend_default_field(
        0.1, 0.0, help="Encoder dropout for TCN, TweetyNet, or WhisperSeg.", parser=float
    )
    encoder_use_skip_connections: bool = _config_field(
        True,
        help="Whether TCN uses skip connections.",
        parser=_bool_parser,
        bool_flag=True,
    )
    encoder_use_separable: bool | list[bool] = _config_field(
        False,
        help="Whether TCN blocks use separable convolutions.",
        parser=_bool_or_list_parser,
    )
    encoder_padding: TcnPadding = _config_field(
        "same",
        help="TCN padding mode.",
        parser=str,
        choices=["same", "causal"],
    )
    decoder_type: Literal["linear", "lstm", "conv", "attention", "legacy_linear_upsample", "whisperseg"] = _config_field(
        "linear",
        help="Decoder type.",
        parser=str,
        flags=["--decoder"],
        choices=["linear", "lstm", "conv", "attention", "legacy_linear_upsample", "whisperseg"],
    )
    decoder_hidden_size: int = _config_field(64, help="LSTM decoder hidden size.", parser=int)
    decoder_kernel_size: int = _config_field(8, help="Conv decoder kernel size.", parser=int)
    decoder_num_heads: int = _config_field(4, help="Attention decoder number of heads.", parser=int)
    decoder_num_layers: int = _config_field(2, help="Attention decoder number of layers.", parser=int)
    decoder_dropout: float = _backend_default_field(
        0.1, 0.0, help="Decoder dropout for attention or WhisperSeg.", parser=float
    )
    cross_entropy_weight: float = _config_field(0.9, help="Cross-entropy loss weight.", parser=float)
    positive_class_weight: float = _config_field(
        1.0,
        help="Relative cross-entropy weight for every non-noise class.",
        parser=float,
    )
    boundary_weight: float = _config_field(
        1.0,
        help="Relative dense cross-entropy weight near label transitions.",
        parser=float,
    )
    boundary_width_ms: float = _config_field(
        20.0,
        help="Width on each side of a transition receiving boundary_weight.",
        parser=float,
    )
    learning_rate: float = _backend_default_field(0.0001, 3e-6, help="Optimizer learning rate.", parser=float)
    early_stopping: bool = _config_field(
        True,
        help="Use early stopping during supervised training.",
        parser=_bool_parser,
        bool_flag=True,
    )
    early_stopping_patience: int = _backend_default_field(
        10,
        3,
        help="Early stopping patience in epochs.",
        parser=int,
    )
    early_stopping_min_delta: float = _config_field(
        0.0,
        help="Minimum validation-loss improvement that resets early stopping.",
        parser=float,
    )
    reduce_lr: bool = _config_field(
        True,
        help="Use ReduceLROnPlateau for native supervised training.",
        parser=_bool_parser,
        bool_flag=True,
    )
    reduce_lr_patience: int = _config_field(
        5,
        help="Native supervised ReduceLROnPlateau patience in epochs.",
        parser=int,
    )
    reduce_lr_factor: float = _config_field(
        0.1,
        help="Native supervised ReduceLROnPlateau learning-rate reduction factor.",
        parser=float,
    )
    reduce_lr_min: float = _config_field(
        1e-8,
        help="Native supervised ReduceLROnPlateau minimum learning rate.",
        parser=float,
    )
    linear_lr_schedule: bool = _config_field(
        True,
        help="Use a linear learning-rate scheduler for WhisperSeg.",
        parser=_bool_parser,
        bool_flag=True,
    )
    weight_decay: float = _config_field(0.01, help="WhisperSeg optimizer weight decay.", parser=float)
    warmup_steps: int = _config_field(100, help="WhisperSeg linear scheduler warmup steps.", parser=int)
    freeze_encoder: bool = _config_field(
        False,
        help="Freeze the WhisperSeg encoder during training.",
        parser=_bool_parser,
        bool_flag=True,
    )
    max_length: int = _config_field(100, help="Maximum WhisperSeg decoder token length during training.", parser=int)
    generation_max_length: int = _config_field(128, help="Maximum WhisperSeg decoder token length during prediction.", parser=int)
    total_spec_columns: int = _config_field(1000, help="WhisperSeg spectrogram time columns per training clip.", parser=int)
    seed: int | None = _config_field(None, help="Optional random seed. Leave unset for nondeterministic runs.", parser=_optional_int_parser)
    accelerator: TrainerAccelerator = _config_field(
        "auto",
        help="Lightning accelerator.",
        parser=str,
        choices=["auto", "cpu", "cuda", "mps", "tpu"],
    )
    num_devices: int | None = _config_field(
        None,
        help="Number of devices to use.",
        parser=_optional_int_parser,
    )
    num_epochs: int = _backend_default_field(100, 10, help="Maximum number of training epochs.", parser=int)
    max_num_steps_per_epoch: int | None = _config_field(
        None,
        help="Maximum number of training batches per epoch. Leave unset to use the full epoch.",
        parser=_optional_int_parser,
    )
    output_suffix: str = _config_field("_annotations.csv", help="Suffix appended to prediction CSV files.")
    existing_annotations: ExistingAnnotations = _config_field(
        "overwrite",
        help="How prediction handles existing output annotation CSV files.",
        parser=_existing_annotations_parser,
        choices=["skip", "overwrite", "merge"],
    )
    merge: bool = _config_field(
        False,
        help="Merge prediction rows with existing output annotation CSV files instead of overwriting them.",
        parser=_bool_parser,
        bool_flag=True,
    )
    fill_gap_ms: float = _config_field(10.0, help="Merge gaps shorter than this many milliseconds.", parser=float)
    min_syllable_ms: float = _config_field(10.0, help="Drop detections shorter than this many milliseconds.", parser=float)
    segment_threshold_low: float = _config_field(
        0.5,
        help="Low probability threshold used to retain binary segment boundaries.",
        parser=float,
    )
    segment_threshold_high: float = _config_field(
        0.5,
        help="High probability threshold a binary segment must reach to be retained.",
        parser=float,
    )
    event_threshold: float = _config_field(
        0.5,
        help="Minimum peak probability used to retain event detections.",
        parser=float,
    )
    event_dist_min_ms: float = _config_field(
        0.0,
        help="Drop event detections closer than this many milliseconds to a neighboring event of the same class.",
        parser=float,
    )
    event_dist_max_ms: float | None = _config_field(
        None,
        help="Drop event detections farther than this many milliseconds from both neighboring events of the same class.",
        parser=_optional_float_parser,
    )
    syllable_postprocessor: SyllablePostprocessor = _config_field(
        "label_aware_dense",
        help="Syllable postprocessor used to convert frame logits into annotation rows.",
        parser=str,
        choices=["binary_mask", "label_aware_dense"],
    )
    syllable_tolerance_ms: float = _config_field(
        10.0,
        help="Maximum onset or offset error still counted as a syllable match.",
        parser=float,
    )
    min_frequency: int | None = _config_field(
        None,
        help="Minimum WhisperSeg spectrogram frequency. Leave unset to use the model default.",
        parser=_optional_int_parser,
    )
    frequency_scale: float | None = _config_field(
        1.0,
        help="Scale applied to the effective WhisperSeg sample rate.",
        parser=_optional_float_parser,
    )
    spec_time_step: float | None = _config_field(
        None,
        help="WhisperSeg spectrogram time step. Leave unset to use model defaults.",
        parser=_optional_float_parser,
    )
    num_trials: int = _config_field(1, help="Number of WhisperSeg prediction trials.", parser=int)
    num_beams: int = _config_field(4, help="Number of WhisperSeg generation beams.", parser=int)
    top_k: int = _config_field(1, help="WhisperSeg generation top-k value.", parser=int)
    top_p: float = _config_field(1.0, help="WhisperSeg generation top-p value.", parser=float)
    length_penalty: float = _config_field(1.0, help="WhisperSeg generation length penalty.", parser=float)
    _default_fields: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def __post_init__(self) -> None:
        default_fields = set(self._default_fields)
        if self.output_dir is None:
            default_fields.add("output_dir")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, _BackendDefault):
                default_fields.add(item.name)
        self._default_fields = frozenset(default_fields)
        if "output_dir" in default_fields:
            self.output_dir = None if self.mode == "predict" else "./"
        for item in fields(self):
            if item.name not in default_fields or not isinstance(item.default, _BackendDefault):
                continue
            value = item.default.whisperseg if self.encoder_type == "whisperseg" else item.default.native
            setattr(self, item.name, value)
        if self.encoder_type == "whisperseg":
            self.frontend_type = "whisperseg"
            self.decoder_type = "whisperseg"
            if self.reduce_lr and self.linear_lr_schedule:
                self.reduce_lr = False

    @classmethod
    def field_names(cls) -> set[str]:
        return {item.name for item in fields(cls) if not item.name.startswith("_")}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object] | None = None, *, base: Config | None = None) -> Config:
        config = base.copy() if base is not None else cls()
        if payload is None:
            return config
        if not isinstance(payload, Mapping):
            raise ValueError("Expected config payload to be a mapping.")
        payload = dict(payload)
        unknown = sorted(set(payload) - cls.field_names())
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")

        updates = {}
        for item in fields(cls):
            if item.name not in payload:
                continue
            parser = item.metadata.get("parser", _identity)
            updates[item.name] = parser(payload[item.name])
        return config.copy(**updates)

    @classmethod
    def from_config_sources(
        cls,
        sources: list[str] | tuple[str, ...],
        *,
        base: Config | None = None,
        mode: ModeName | None = None,
    ) -> Config:
        config = base.copy() if base is not None else cls()
        for source in sources:
            config = cls.from_mapping(load_config_mapping(source, mode=mode or config.mode), base=config)
        return config

    def copy(self, **changes) -> Config:
        default_fields = self._default_fields.difference(changes)
        return replace(self, **changes, _default_fields=default_fields)

    def to_mapping(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("_default_fields")
        return payload

    def frontend_mapping(self, *, include_raw_num_channels: bool = True) -> dict[str, object]:
        return serialize_frontend_config(
            {
                "type": str(self.frontend_type),
                "num_channels": self.frontend_num_channels,
                "kernel_size": self.frontend_kernel_size,
                "hop_seconds": self.frontend_hop_seconds,
                "pad_mode": str(self.frontend_pad_mode),
                "fmin": float(self.frontend_fmin),
                "fmax": self.frontend_fmax,
                "trainable": bool(self.frontend_trainable),
                "min_frequency": self.min_frequency,
                "frequency_scale": self.frequency_scale,
                "spec_time_step": self.spec_time_step,
            },
            include_raw_num_channels=include_raw_num_channels,
        )

    def encoder_mapping(self) -> dict[str, object]:
        return serialize_encoder_config(
            {
                "type": str(self.encoder_type),
                "num_heads": int(self.encoder_num_heads),
                "hidden_size": int(self.encoder_hidden_size),
                "num_layers": int(self.encoder_num_layers),
                "kernel_size": self.encoder_kernel_size,
                "dilations": list(self.encoder_dilations),
                "dropout": float(self.encoder_dropout),
                "use_skip_connections": bool(self.encoder_use_skip_connections),
                "use_separable": self.encoder_use_separable,
                "padding": str(self.encoder_padding),
            }
        )

    def decoder_mapping(self) -> dict[str, object]:
        return serialize_decoder_config(
            {
                "type": str(self.decoder_type),
                "hidden_size": int(self.decoder_hidden_size),
                "kernel_size": int(self.decoder_kernel_size),
                "num_heads": int(self.decoder_num_heads),
                "num_layers": int(self.decoder_num_layers),
                "dropout": float(self.decoder_dropout),
                "max_length": int(self.max_length),
                "generation_max_length": int(self.generation_max_length),
                "num_trials": int(self.num_trials),
                "num_beams": int(self.num_beams),
                "top_k": int(self.top_k),
                "top_p": float(self.top_p),
                "length_penalty": float(self.length_penalty),
            }
        )

    def model_kwargs(
        self,
        *,
        num_classes: int,
        sr: float,
        class_names: list[str] | None = None,
        class_types: list[str] | None = None,
        num_time_steps: int | None = None,
        chunk_stride: int | None = None,
        frontend_num_channels: int | None = None,
    ) -> dict[str, object]:
        frontend = self.frontend_mapping(include_raw_num_channels=True)
        if frontend_num_channels is not None and frontend["type"] == "raw":
            frontend["num_channels"] = int(frontend_num_channels)
        return {
            "num_classes": int(num_classes),
            "sr": float(sr),
            "class_names": None if class_names is None else list(class_names),
            "class_types": None if class_types is None else list(class_types),
            "frontend": frontend,
            "encoder": self.encoder_mapping(),
            "decoder": self.decoder_mapping(),
            "cross_entropy_weight": float(self.cross_entropy_weight),
            "positive_class_weight": float(self.positive_class_weight),
            "boundary_weight": float(self.boundary_weight),
            "boundary_width_ms": float(self.boundary_width_ms),
            "learning_rate": float(self.learning_rate),
            "reduce_lr": bool(self.reduce_lr),
            "reduce_lr_patience": int(self.reduce_lr_patience),
            "reduce_lr_factor": float(self.reduce_lr_factor),
            "reduce_lr_min": float(self.reduce_lr_min),
            "num_time_steps": None if num_time_steps is None else int(num_time_steps),
            "chunk_stride": None if chunk_stride is None else int(chunk_stride),
        }

    def validate(self) -> None:
        if self.encoder_type not in {"conformer", "tcn", "tweetynet", "whisperseg"}:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")
        if self.decoder_type not in {"linear", "lstm", "conv", "attention", "legacy_linear_upsample", "whisperseg"}:
            raise ValueError(f"Unsupported decoder_type: {self.decoder_type}")
        if self.mode == "train" and self.encoder_type == "whisperseg" and not self.initial_model:
            raise ValueError("WhisperSeg training requires a DAS .ckpt initial_model.")
        if self.mode == "train" and self.encoder_type == "whisperseg" and Path(self.initial_model).suffix != ".ckpt":
            raise ValueError("WhisperSeg training requires a DAS .ckpt initial_model.")
        if self.mode == "train" and self.encoder_type != "whisperseg" and self.initial_model:
            raise ValueError("initial_model is only supported for WhisperSeg training in this release.")
        if self.mode == "train" and self.encoder_type != "whisperseg" and self.freeze_encoder:
            raise ValueError("freeze_encoder is only supported for WhisperSeg training in this release.")
        if self.positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive.")
        if self.boundary_weight <= 0 or self.boundary_width_ms < 0:
            raise ValueError("boundary_weight must be positive and boundary_width_ms must be non-negative.")
        if self.existing_annotations not in {"skip", "overwrite", "merge"}:
            raise ValueError("existing_annotations must be one of skip, overwrite, merge.")
        if self.validation_fraction < 0:
            raise ValueError("validation_fraction must be non-negative.")
        if self.test_fraction < 0:
            raise ValueError("test_fraction must be non-negative.")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation_fraction + test_fraction must be less than 1.")
        if self.max_num_steps_per_epoch is not None and self.max_num_steps_per_epoch <= 0:
            raise ValueError("max_num_steps_per_epoch must be positive when set.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative.")
        if self.reduce_lr_patience < 1:
            raise ValueError("reduce_lr_patience must be at least 1.")
        if not 0 < self.reduce_lr_factor < 1:
            raise ValueError("reduce_lr_factor must be greater than 0 and less than 1.")
        if self.reduce_lr_min < 0:
            raise ValueError("reduce_lr_min must be non-negative.")
        if self.frequency_scale is not None and self.frequency_scale <= 0:
            raise ValueError("frequency_scale must be positive.")
        if self.target_samplerate_hz is not None and self.target_samplerate_hz <= 0:
            raise ValueError("target_samplerate_hz must be positive.")
        if self.spec_time_step is not None and self.spec_time_step <= 0:
            raise ValueError("spec_time_step must be positive.")
        if self.frontend_type == "whisperseg" and self.encoder_type != "whisperseg":
            raise ValueError("frontend_type=whisperseg is only supported with encoder_type=whisperseg.")
        if self.decoder_type == "whisperseg" and self.encoder_type != "whisperseg":
            raise ValueError("decoder_type=whisperseg is only supported with encoder_type=whisperseg.")
        if self.encoder_type == "whisperseg" and self.frontend_type != "whisperseg":
            raise ValueError("encoder_type=whisperseg requires frontend_type=whisperseg.")
        if not 0 <= self.segment_threshold_low <= self.segment_threshold_high <= 1:
            raise ValueError("segment thresholds must satisfy 0 <= low <= high <= 1.")
        if not 0 <= self.event_threshold <= 1:
            raise ValueError("event_threshold must be between 0 and 1.")
        if self.event_dist_min_ms < 0:
            raise ValueError("event_dist_min_ms must be non-negative.")
        if self.event_dist_max_ms is not None and self.event_dist_max_ms < 0:
            raise ValueError("event_dist_max_ms must be non-negative.")
        if self.min_annotation_duration_ms < 0:
            raise ValueError("min_annotation_duration_ms must be non-negative.")
        if self.mode == "train" and not self.data_dir:
            raise ValueError("Provide data_dir.")
        if self.mode == "predict" and not self.data_dir:
            raise ValueError("Provide data_dir.")
        if self.mode == "train" and not self.output_dir:
            raise ValueError("Provide output_dir for training.")
        if self.mode == "predict":
            if not self.checkpoint:
                raise ValueError("Provide checkpoint for prediction.")
            if Path(self.checkpoint).suffix in {".pt", ".pth"} or self.checkpoint.startswith(("hf://", "nccratliri/")):
                raise ValueError("Prediction requires a DAS .ckpt or legacy DAS H5/YAML checkpoint.")
        if self.mode == "convert-legacy":
            if not self.checkpoint:
                raise ValueError("Provide checkpoint for legacy conversion.")
            if not self.converted_checkpoint:
                raise ValueError("Provide --output for legacy conversion.")
        if self.evaluate and self.mode != "predict":
            raise ValueError("evaluate=true is only supported for predict mode.")


def builtin_config_names() -> tuple[str, ...]:
    return tuple(BUILTIN_CONFIG_FILES)


def builtin_config_help() -> str:
    return ", ".join(builtin_config_names())


def _read_config_mapping_from_text(text: str, *, source: str) -> dict[str, object]:
    payload = yaml.safe_load(text)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected '{source}' to contain a YAML mapping.")
    return dict(payload)


def load_builtin_config_mapping(name: str, *, mode: ModeName | None = None) -> dict[str, object]:
    if name not in BUILTIN_CONFIG_FILES:
        raise ValueError(f"Unknown built-in config '{name}'. Available built-ins: {builtin_config_help()}.")
    if mode not in BUILTIN_CONFIG_FILES[name]:
        raise ValueError(f"Built-in config '{name}' is only available for train and predict commands.")

    filename = BUILTIN_CONFIG_FILES[name][mode]
    config_path = resource_files("das.presets").joinpath(filename)
    if not config_path.is_file():
        raise ValueError(f"Built-in config '{name}' points to missing file '{filename}'.")

    payload = _read_config_mapping_from_text(config_path.read_text(encoding="utf-8"), source=f"built-in config '{name}'")
    if payload.get("mode", mode) != mode:
        raise ValueError(f"Built-in config '{name}' is not valid for {mode} mode.")
    return payload


def load_config_mapping(source: str, *, mode: ModeName | None = None) -> dict[str, object]:
    source_value = str(source).strip()
    if source_value in BUILTIN_CONFIG_FILES:
        return load_builtin_config_mapping(source_value, mode=mode)
    return load_yaml_config_mapping(source_value)


def load_yaml_config_mapping(path: str) -> dict[str, object]:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise ValueError(f"Config file '{config_path}' does not exist.")
    if not config_path.is_file():
        raise ValueError(f"Config path '{config_path}' is not a file.")
    return _read_config_mapping_from_text(config_path.read_text(encoding="utf-8"), source=str(config_path))


def save_config(config: Config, destination: str) -> Path:
    output_path = Path(destination).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_config_yaml(config), encoding="utf-8")
    return output_path


def format_config_yaml(config: Config) -> str:
    return yaml.safe_dump(config.to_mapping(), sort_keys=False)
