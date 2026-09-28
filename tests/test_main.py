from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

import das.cli as cli
import das.api as api
from das.config import Config


def test_validation_sample_spans_the_full_validation_set():
    loader = DataLoader(TensorDataset(torch.arange(9_000)), batch_size=4)

    sampled = api._validation_loader_for_plateau(loader, max_batches=1_000)
    values = torch.cat([batch[0] for batch in sampled])

    assert len(sampled) == 1_000
    assert values[0].item() == 0
    assert values[-1].item() == 8_999


def test_package_exposes_train_and_predict(monkeypatch: pytest.MonkeyPatch):
    import das

    monkeypatch.setattr(api, "train", lambda *args, **kwargs: ("train", args, kwargs))
    monkeypatch.setattr(api, "predict", lambda *args, **kwargs: ("predict", args, kwargs))

    assert das.train(data_dir="data", output_dir="run") == (
        "train",
        (),
        {"data_dir": "data", "output_dir": "run"},
    )
    assert not hasattr(das, "pretrain")
    assert das.predict(data_dir="data", checkpoint="model.ckpt", output_dir="pred") == (
        "predict",
        (),
        {"data_dir": "data", "checkpoint": "model.ckpt", "output_dir": "pred"},
    )
    assert not hasattr(das, "embed")
    assert not hasattr(das, "label")
    assert not hasattr(das, "tune_postprocessing")


def _write_legacy_fixture(tmp_path: Path, name: str = "test") -> Path:
    trunk = tmp_path / name
    trunk.with_name(f"{name}_model.h5").write_bytes(b"legacy")
    trunk.with_name(f"{name}_params.yaml").write_text("model_name: tcn\n", encoding="utf-8")
    return trunk


def _write_npy_dir_fixture(
    tmp_path: Path,
    name: str = "legacy.npy",
    samplerate: int = 16_000,
    channels: int = 1,
) -> Path:
    root = tmp_path / name
    root.mkdir()
    np.save(
        root / "attrs.npy",
        {
            "class_names": ["noise", "song"],
            "class_types": ["segment", "segment"],
            "samplerate_x_Hz": samplerate,
            "samplerate_y_Hz": samplerate,
        },
        allow_pickle=True,
    )

    for split, samples in {"train": 128, "val": 64, "test": 64}.items():
        split_dir = root / split
        split_dir.mkdir()
        if channels == 1:
            x = np.zeros((samples,), dtype=np.float16)
            x[8:16] = 0.5
        else:
            x = np.zeros((samples, channels), dtype=np.float16)
            x[8:16, 0] = 0.5
        y = np.zeros((samples, 2), dtype=np.float16)
        y[:, 0] = 1.0
        y[24:40, 0] = 0.0
        y[24:40, 1] = 1.0
        np.save(split_dir / "x.npy", x)
        np.save(split_dir / "y.npy", y)

    return root


def test_build_parser_has_train_predict_and_gui_subcommands():
    parser = cli.build_parser()

    train_args = parser.parse_args(["train", "--data-dir", "data", "--output-dir", "run"])
    predict_args = parser.parse_args(["predict", "--data-dir", "data", "--checkpoint", "model.ckpt", "--output-dir", "pred"])
    gui_args = parser.parse_args(["gui", "--config", "config.yaml"])
    version_args = parser.parse_args(["version"])

    assert train_args.command == "train"
    assert predict_args.command == "predict"
    assert gui_args.command == "gui"
    assert version_args.command == "version"
    assert not hasattr(train_args, "preset")


def test_train_defaults_output_dir_to_current_directory():
    config = cli.parse_command_config(["train", "--data-dir", "data"])

    assert config.output_dir == "./"


def test_predict_defaults_to_output_next_to_audio():
    config = cli.parse_command_config(["predict", "--data-dir", "data", "--checkpoint", "model.ckpt"])

    assert config.output_dir is None
    assert config.existing_annotations == "overwrite"


def test_checkpoint_detection_thresholds_are_prediction_defaults():
    runtime = api.PredictRuntime(
        sr=1_000,
        hop_seconds=0.01,
        num_time_steps=64,
        chunk_stride=None,
        class_names=["noise", "song"],
        class_types=["segment", "segment"],
        segment_threshold_low=0.2,
        segment_threshold_high=0.8,
        event_threshold=0.7,
    )
    context = api._prediction_result_context(Config(mode="predict"), runtime)
    assert (context.segment_threshold_low, context.segment_threshold_high, context.event_threshold) == (0.2, 0.8, 0.7)


def test_print_config_does_not_require_data_dir(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_command_config(["train", "--print-config"])

    assert exc_info.value.code == 0
    assert "mode: train" in capsys.readouterr().out


def test_print_config_includes_native_training_control_defaults(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_command_config(["train", "--print-config"])

    assert exc_info.value.code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["early_stopping"] is True
    assert payload["early_stopping_patience"] == 10
    assert payload["reduce_lr"] is True
    assert payload["reduce_lr_patience"] == 5
    assert payload["reduce_lr_factor"] == 0.1
    assert payload["reduce_lr_min"] == 1e-8


def test_train_help_shows_defaults_and_skips_usage(capsys: pytest.CaptureFixture[str]):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["train", "--help"])

    assert exc_info.value.code == 0

    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    assert help_text.startswith("Train a model on annotated audio.")
    assert "Load a YAML config path or built-in config name (fly, fly-pulse, zebra-finch, tweetynet)." in normalized_help
    assert "Can be provided multiple" in normalized_help
    assert "times. (default: [])" in help_text
    assert "Print the effective flat YAML config and exit." in help_text
    assert "(default: False)" in help_text
    assert "Number of chunks per batch. (default: 8)" in help_text
    assert "Frontend type. (default: 'mel')" in help_text
    assert "Whether STFT or mel frontend parameters are trainable." in help_text
    assert "Minimum WhisperSeg spectrogram frequency. Leave unset to use the model default. (default: None)" in normalized_help
    assert "Scale applied to the effective WhisperSeg sample rate. (default: 1.0)" in normalized_help
    assert "(default: True)" in help_text


def test_predict_help_describes_local_data_only(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["predict", "--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "Audio file or directory for prediction." in help_text
    assert "owner/dataset" not in help_text


def test_parse_command_config_accepts_native_training_controls():
    config = cli.parse_command_config(
        [
            "train",
            "--data-dir",
            "data",
            "--no-early-stopping",
            "--early-stopping-patience",
            "7",
            "--early-stopping-min-delta",
            "0.01",
            "--no-reduce-lr",
            "--reduce-lr-patience",
            "3",
            "--reduce-lr-factor",
            "0.25",
            "--reduce-lr-min",
            "1e-7",
        ]
    )

    assert config.early_stopping is False
    assert config.early_stopping_patience == 7
    assert config.early_stopping_min_delta == 0.01
    assert config.reduce_lr is False
    assert config.reduce_lr_patience == 3
    assert config.reduce_lr_factor == 0.25
    assert config.reduce_lr_min == 1e-7


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--early-stopping-patience", "0", "early_stopping_patience must be at least 1"),
        ("--early-stopping-min-delta", "-1", "early_stopping_min_delta must be non-negative"),
        ("--reduce-lr-patience", "0", "reduce_lr_patience must be at least 1"),
        ("--reduce-lr-factor", "1", "reduce_lr_factor must be greater than 0 and less than 1"),
        ("--reduce-lr-min", "-1", "reduce_lr_min must be non-negative"),
    ],
)
def test_parse_command_config_validates_native_training_controls(
    flag: str,
    value: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(SystemExit) as exc_info:
        cli.parse_command_config(["train", "--data-dir", "data", flag, value])

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_train_parse_errors_still_show_usage(capsys: pytest.CaptureFixture[str]):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["train", "--does-not-exist"])

    assert exc_info.value.code == 2

    error_text = capsys.readouterr().err
    assert "usage: das [-h] {train,predict,convert-legacy,gui,version} ..." in error_text
    assert "unrecognized arguments: --does-not-exist" in error_text


def test_cli_main_without_args_starts_gui(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    launch_args = {}

    def fake_launch_gui(argv, *, startup_config=None):
        launch_args["argv"] = argv
        launch_args["startup_config"] = startup_config
        return 23

    monkeypatch.setattr(cli, "_launch_gui", fake_launch_gui)

    result = cli.cli_main([])

    assert result == 23
    assert launch_args["argv"] == ["das"]
    assert isinstance(launch_args["startup_config"], Config)
    assert launch_args["startup_config"].mode == "train"

    banner = capsys.readouterr().out
    assert "DAS " in banner
    assert "Starting the GUI." in banner
    assert "das train" in banner
    assert "das predict" in banner
    assert "das version" in banner


def test_cli_main_with_config_starts_gui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config_path = tmp_path / "predict.yaml"
    config_path.write_text("mode: predict\nbatch_size: 23\n", encoding="utf-8")
    launch_args = {}

    def fake_launch_gui(argv, *, startup_config=None):
        launch_args["argv"] = argv
        launch_args["startup_config"] = startup_config
        return 23

    monkeypatch.setattr(cli, "_launch_gui", fake_launch_gui)

    result = cli.cli_main(["--config", str(config_path)])

    assert result == 23
    assert launch_args["argv"] == ["das", "gui", "--config", str(config_path)]
    assert launch_args["startup_config"].mode == "predict"
    assert launch_args["startup_config"].batch_size == 23


def test_top_level_help_lists_commands(capsys: pytest.CaptureFixture[str]):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])

    assert exc_info.value.code == 0

    help_text = capsys.readouterr().out
    assert help_text.startswith("CLI for DAS.")
    assert "train               Train a model on annotated audio." in help_text
    assert "label" not in help_text
    assert "embed" not in help_text
    assert "gui                 Launch the GUI." in help_text
    assert "version             Show the installed DAS version." in help_text


def test_cli_main_version_prints_package_version(capsys: pytest.CaptureFixture[str]):
    import das

    assert cli.cli_main(["version"]) == 0
    assert capsys.readouterr().out == f"DAS {das.__version__}\n"


def test_parse_command_config_version_exits_after_printing_version(capsys: pytest.CaptureFixture[str]):
    import das

    with pytest.raises(SystemExit) as exc_info:
        cli.parse_command_config(["version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"DAS {das.__version__}\n"


@pytest.mark.parametrize("removed_flag", ["--frontend-type", "--encoder-type", "--decoder-type"])
def test_build_parser_rejects_removed_type_aliases(removed_flag: str):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["train", removed_flag, "raw"])

    assert exc_info.value.code == 2


def test_train_parser_accepts_training_overrides(tmp_path: Path):
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "train",
            "--data-dir",
            "data",
            "--output-dir",
            "run",
            "--frontend",
            "raw",
            "--checkpoint-prefix",
            "zf",
            "--include-labels",
            "pulse,sine",
            "--event-dist-min-ms",
            "3",
            "--event-dist-max-ms",
            "40",
            "--num-epochs",
            "3",
        ]
    )

    assert args.frontend_type == "raw"
    assert args.checkpoint_prefix == "zf"
    assert args.include_labels == ["pulse", "sine"]
    assert args.event_dist_min_ms == 3.0
    assert args.event_dist_max_ms == 40.0
    assert args.num_epochs == 3


def test_train_parser_uses_whisperseg_encoder_as_backend(tmp_path: Path):
    checkpoint = tmp_path / "whisperseg.ckpt"
    checkpoint.touch()
    config = cli.parse_command_config(
        [
            "train",
            "--data-dir",
            "data",
            "--output-dir",
            "run",
            "--encoder",
            "whisperseg",
            "--initial-model",
            str(checkpoint),
            "--decoder",
            "linear",
            "--frontend",
            "mel",
            "--min-frequency",
            "250",
            "--frequency-scale",
            "0.5",
            "--spec-time-step",
            "0.001",
            "--decoder-dropout",
            "0.25",
            "--max-length",
            "77",
        ]
    )

    assert config.encoder_type == "whisperseg"
    assert config.frontend_type == "whisperseg"
    assert config.decoder_type == "whisperseg"
    assert config.min_frequency == 250
    assert config.frequency_scale == 0.5
    assert config.spec_time_step == 0.001
    assert config.decoder_dropout == 0.25
    assert config.max_length == 77
    assert config.batch_size == 4
    assert config.num_epochs == 10
    assert config.validation_fraction == 0.1
    assert config.test_fraction == 0.0
    assert config.learning_rate == 3e-6
    assert config.encoder_dropout == 0.0
    assert config.early_stopping_patience == 3


def test_train_parser_preserves_explicit_whisperseg_training_values(tmp_path: Path):
    checkpoint = tmp_path / "whisperseg.ckpt"
    checkpoint.touch()
    config = cli.parse_command_config(
        [
            "train",
            "--data-dir",
            "data",
            "--encoder",
            "whisperseg",
            "--initial-model",
            str(checkpoint),
            "--batch-size",
            "8",
            "--num-epochs",
            "100",
            "--validation-fraction",
            "0.2",
            "--test-fraction",
            "0.2",
            "--learning-rate",
            "0.0001",
            "--encoder-dropout",
            "0.1",
            "--decoder-dropout",
            "0.1",
            "--early-stopping-patience",
            "10",
        ]
    )

    assert config.batch_size == 8
    assert config.num_epochs == 100
    assert config.validation_fraction == 0.2
    assert config.test_fraction == 0.2
    assert config.learning_rate == 1e-4
    assert config.encoder_dropout == 0.1
    assert config.decoder_dropout == 0.1
    assert config.early_stopping_patience == 10


def test_train_parser_rejects_model_backend_flag():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["train", "--model", "whisperseg"])

    assert exc_info.value.code == 2


def test_train_parser_rejects_old_postprocessing_flag():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["train", "--no-optimize-postprocessing-after-train"])

    assert exc_info.value.code == 2


def test_train_parser_rejects_max_to_keep_flag():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["train", "--max-to-keep", "2"])

    assert exc_info.value.code == 2


def test_train_parser_rejects_global_dropout_flag():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["train", "--dropout", "0.2"])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "flag_and_value",
    [
        ["--pretrained-checkpoint", "ssl.ckpt"],
        ["--freeze-pretrained"],
        ["--pretrain"],
        ["--pretrain-only"],
        ["--pretrain-data-dir", "audio"],
        ["--pretrain-output-dir", "ssl-run"],
        ["--pretrain-num-epochs", "2"],
        ["--pretrain-max-num-steps-per-epoch", "3"],
        ["--ssl-mask-probability", "0.25"],
        ["--ssl-mask-span", "4"],
    ],
)
def test_train_parser_rejects_pretraining_flags(flag_and_value: list[str]):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["train", *flag_and_value])

    assert exc_info.value.code == 2


def test_parse_command_config_accepts_include_labels():
    config = cli.parse_command_config(
        [
            "train",
            "--data-dir",
            "labels",
            "--output-dir",
            "run",
            "--include-labels",
            "pulse,sine",
        ]
    )

    assert config.include_labels == ["pulse", "sine"]


def test_build_parser_rejects_pretrain_subcommand():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["pretrain", "--data-dir", "data", "--output-dir", "run"])

    assert exc_info.value.code == 2


def test_predict_parser_accepts_runtime_overrides():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "predict",
            "--data-dir",
            "data",
            "--checkpoint",
            "model.ckpt",
            "--output-dir",
            "pred",
            "--batch-size",
            "16",
            "--evaluate",
            "--existing-annotations",
            "merge",
            "--syllable-postprocessor",
            "binary_mask",
            "--event-dist-min-ms",
            "5",
            "--event-dist-max-ms",
            "none",
        ]
    )

    assert args.batch_size == 16
    assert args.evaluate is True
    assert args.existing_annotations == "merge"
    assert args.syllable_postprocessor == "binary_mask"
    assert args.event_dist_min_ms == 5.0
    assert args.event_dist_max_ms is None


def test_predict_parser_rejects_model_backend_flags():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["predict", "--data-dir", "data", "--checkpoint", "model.pt", "--model", "whisperseg"])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "flag_and_value",
    [["--checkpoint", "model.ckpt"], ["--evaluate"], ["--existing-annotations", "skip"]],
)
def test_train_parser_rejects_predict_only_flags(flag_and_value: list[str]):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["train", *flag_and_value])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "flag_and_value",
    [
        ["--frontend", "raw"],
        ["--num-epochs", "3"],
        ["--validation-fraction", "0.1"],
        ["--num-time-steps", "512"],
        ["--chunk-stride", "256"],
        ["--pretrained-checkpoint", "ssl.ckpt"],
        ["--freeze-pretrained"],
        ["--pretrain"],
        ["--pretrain-only"],
        ["--include-labels", "pulse"],
        ["--tune-postprocessing"],
        ["--postprocessing-tuning-step-ms", "4"],
        ["--postprocessing-tuning-max-fill-gap-ms", "40"],
        ["--postprocessing-tuning-max-min-syllable-ms", "40"],
    ],
)
def test_predict_parser_rejects_train_only_flags(flag_and_value: list[str]):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["predict", *flag_and_value])

    assert exc_info.value.code == 2


def test_gui_parser_rejects_command_override_flags():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["gui", "--data-dir", "data"])

    assert exc_info.value.code == 2


def test_parse_command_config_merges_flat_yaml_files_then_cli(tmp_path: Path):
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(
        yaml.safe_dump(
            {
                "mode": "train",
                "data_dir": "from-first",
                "output_dir": "run",
                "batch_size": 4,
                "frontend_type": "raw",
                "encoder_type": "tcn",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    second_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": "from-second",
                "batch_size": 8,
                "encoder_hidden_size": 24,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = cli.parse_command_config(
        [
            "train",
            "--config",
            str(first_path),
            "--config",
            str(second_path),
            "--batch-size",
            "16",
            "--num-epochs",
            "3",
        ]
    )

    assert config.mode == "train"
    assert config.data_dir == "from-second"
    assert config.output_dir == "run"
    assert config.batch_size == 16
    assert config.encoder_hidden_size == 24
    assert config.num_epochs == 3


def test_parse_command_config_loads_builtin_config_name():
    config = cli.parse_command_config(
        [
            "train",
            "--config",
            "zebra-finch",
            "--data-dir",
            "data",
            "--output-dir",
            "run",
            "--batch-size",
            "16",
        ]
    )

    assert config.mode == "train"
    assert config.data_dir == "data"
    assert config.output_dir == "run"
    assert config.batch_size == 16
    assert config.frontend_type == "stft"
    assert config.frontend_num_channels == 33
    assert config.encoder_type == "tcn"
    assert config.encoder_hidden_size == 32


def test_predict_builtin_config_uses_predict_yaml():
    config = cli.parse_command_config(
        [
            "predict",
            "--config",
            "tweetynet",
            "--data-dir",
            "data",
            "--checkpoint",
            "model.ckpt",
        ]
    )

    assert config.mode == "predict"
    assert config.batch_size == 16
    assert config.fill_gap_ms == 5.0
    assert config.min_syllable_ms == 20.0


def test_gui_config_can_load_builtin_config_name():
    config = cli.parse_command_config(["gui", "--config", "fly"])

    assert config.mode == "train"
    assert config.batch_size == 32
    assert config.frontend_type == "raw"


def test_predict_config_file_can_still_include_train_fields(tmp_path: Path):
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": "data",
                "checkpoint": "model.ckpt",
                "output_dir": "predictions",
                "existing_annotations": "merge",
                "event_dist_min_ms": 2.5,
                "event_dist_max_ms": 25.0,
                "frontend_type": "raw",
                "encoder_type": "tcn",
                "num_epochs": 7,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = cli.parse_command_config(["predict", "--config", str(config_path)])

    assert config.mode == "predict"
    assert config.existing_annotations == "merge"
    assert config.event_dist_min_ms == 2.5
    assert config.event_dist_max_ms == 25.0
    assert config.frontend_type == "raw"
    assert config.encoder_type == "tcn"
    assert config.num_epochs == 7


def test_predict_config_rejects_invalid_existing_annotations_value():
    with pytest.raises(ValueError, match="skip, overwrite, merge"):
        Config.from_mapping({"existing_annotations": "maybe"})
    with pytest.raises(ValueError, match="skip, overwrite, merge"):
        Config(existing_annotations="maybe").validate()


@pytest.mark.parametrize("field_name", ["event_dist_min_ms", "event_dist_max_ms"])
def test_config_rejects_negative_event_interval_postprocessing(field_name: str):
    with pytest.raises(ValueError, match=f"{field_name} must be non-negative"):
        Config(**{field_name: -1.0}).validate()


def test_config_rejects_whisperseg_decoder_without_whisperseg_encoder():
    with pytest.raises(ValueError, match="decoder_type=whisperseg"):
        Config(decoder_type="whisperseg").validate()


@pytest.mark.parametrize("field, value", [("encoder_type", "aves2"), ("decoder_type", "timestamp")])
def test_config_rejects_removed_model_options(field: str, value: str):
    with pytest.raises(ValueError, match=f"Unsupported {field}"):
        Config(**{field: value}).validate()


def test_config_rejects_whisperseg_frontend_without_whisperseg_encoder():
    with pytest.raises(ValueError, match="frontend_type=whisperseg"):
        Config(frontend_type="whisperseg").validate()


def test_parse_command_config_save_config_writes_flat_yaml(tmp_path: Path):
    config_path = tmp_path / "saved.yaml"

    config = cli.parse_command_config(
        [
            "predict",
            "--data-dir",
            "data",
            "--checkpoint",
            "model.ckpt",
            "--output-dir",
            "predictions",
            "--evaluate",
            "--existing-annotations",
            "merge",
            "--save-config",
            str(config_path),
        ]
    )

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config.mode == "predict"
    assert saved["mode"] == "predict"
    assert saved["evaluate"] is True
    assert saved["existing_annotations"] == "merge"
    assert saved["checkpoint"] == "model.ckpt"
    assert "predict" not in saved
    assert "data" not in saved


def test_cli_main_passes_config_object_to_gui(monkeypatch: pytest.MonkeyPatch):
    launch_args = {}

    def fake_launch_gui(argv, *, startup_config=None):
        launch_args["argv"] = argv
        launch_args["startup_config"] = startup_config
        return 23

    monkeypatch.setattr(cli, "_launch_gui", fake_launch_gui)

    result = cli.cli_main(["gui"])

    assert result == 23
    assert launch_args["argv"] == ["das", "gui"]
    assert isinstance(launch_args["startup_config"], Config)


def test_gui_launch_opens_dialog_only_for_explicit_config(monkeypatch: pytest.MonkeyPatch):
    import sys
    from types import ModuleType

    seen = []
    app = ModuleType("xarray_behave.gui.app")
    app.main_das = lambda **kwargs: seen.append(kwargs)
    monkeypatch.setitem(sys.modules, "xarray_behave.gui.app", app)

    config = Config(mode="train")
    cli._launch_gui(["das", "gui"], startup_config=config)
    cli._launch_gui(["das", "gui", "--config", "fly"], startup_config=config)

    assert seen == [{"das_startup_config": None}, {"das_startup_config": config}]


def test_trainer_devices_maps_none_to_auto():
    assert api._trainer_devices(None) == "auto"
    assert api._trainer_devices(2) == 2


def test_train_dispatches_to_whisperseg_backend(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_train_whisperseg(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs
        return "/tmp/run/checkpoints/model.ckpt"

    monkeypatch.setattr(api, "_train_whisperseg", fake_train_whisperseg)

    result = api.train(
        Config(
            mode="train",
            encoder_type="whisperseg",
            initial_model="model.ckpt",
            data_dir="data",
            output_dir="run",
            num_epochs=1,
        ),
        verbose=True,
    )

    assert result == "/tmp/run/checkpoints/model.ckpt"
    assert calls["config"].encoder_type == "whisperseg"
    assert calls["config"].frontend_type == "whisperseg"
    assert calls["config"].decoder_type == "whisperseg"
    assert calls["kwargs"]["verbose"] is True


def test_predict_dispatches_to_whisperseg_backend_from_checkpoint(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_predict_whisperseg(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs
        return pd.DataFrame([{"name": "song", "start_seconds": 0.1, "stop_seconds": 0.2}])

    monkeypatch.setattr(api, "_is_whisperseg_predict_checkpoint", lambda checkpoint: checkpoint == "model.ckpt")
    monkeypatch.setattr(api, "_predict_whisperseg", fake_predict_whisperseg)

    result = api.predict(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
        ),
        verbose=True,
    )

    assert result.to_dict("records") == [{"name": "song", "start_seconds": 0.1, "stop_seconds": 0.2}]
    assert calls["config"].checkpoint == "model.ckpt"
    assert calls["kwargs"]["verbose"] is True


def test_whisperseg_checkpoint_detection_requires_new_metadata(tmp_path: Path):
    from das.whisperseg.model import WHISPERSEG_CHECKPOINT_FORMAT, WHISPERSEG_CHECKPOINT_FORMAT_VERSION

    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "das": {"backend": "whisperseg"},
            "whisperseg": {
                "format": WHISPERSEG_CHECKPOINT_FORMAT,
                "format_version": WHISPERSEG_CHECKPOINT_FORMAT_VERSION,
            },
        },
        checkpoint_path,
    )
    missing_metadata_path = tmp_path / "native.ckpt"
    torch.save({"das": {"backend": "native"}}, missing_metadata_path)
    old_bundle_path = tmp_path / "model.pt"
    torch.save({"format": "das_whisper.model_bundle", "format_version": 1}, old_bundle_path)

    assert api._is_whisperseg_predict_checkpoint(str(checkpoint_path)) is True
    assert api._is_whisperseg_predict_checkpoint(str(missing_metadata_path)) is False
    assert api._is_whisperseg_predict_checkpoint(str(old_bundle_path)) is False


def test_predict_whisperseg_raw_audio_prints_prediction_summary(monkeypatch: pytest.MonkeyPatch, capsys):
    class FakeSegmenter:
        cluster_codebook = {"song": 0}

    monkeypatch.setattr(api, "_is_whisperseg_predict_checkpoint", lambda checkpoint: checkpoint == "model.ckpt")
    monkeypatch.setattr(api, "_whisper_segmenter", lambda config: FakeSegmenter())
    monkeypatch.setattr(api, "_whisper_model_samplerate", lambda segmenter: 100)
    monkeypatch.setattr(
        api,
        "_whisper_segment_audio",
        lambda segmenter, config, audio, samplerate, filename=None: pd.DataFrame(
            [{"name": "song", "start_seconds": 0.1, "stop_seconds": 0.2}]
        ),
    )

    result = api.predict(
        Config(mode="predict", checkpoint="model.ckpt"),
        audio=np.zeros(8, dtype=np.float32),
        samplerate=100,
        verbose=True,
    )

    assert result.to_dict("records") == [{"name": "song", "start_seconds": 0.1, "stop_seconds": 0.2}]
    output = capsys.readouterr().out
    assert "Prediction summary:" in output
    assert "Total instances: 1" in output
    assert "song: 1" in output


def test_whisperseg_segment_audio_passes_generation_settings():
    calls = {}

    class FakeSegmenter:
        default_segmentation_config = {"sr": 100, "frequency_scale": 1.0}

        def segment(self, audio, sr, **kwargs):
            calls["audio"] = audio
            calls["sr"] = sr
            calls.update(kwargs)
            return {"cluster": [], "onset": [], "offset": []}

    frame = api._whisper_segment_audio(
        FakeSegmenter(),
        Config(
            mode="predict",
            checkpoint="model.ckpt",
            batch_size=2,
            min_frequency=250,
            spec_time_step=0.001,
            generation_max_length=123,
            num_trials=3,
            num_beams=5,
            top_k=2,
            top_p=0.9,
            length_penalty=0.8,
        ),
        np.zeros(10, dtype=np.float32),
        100,
    )

    assert frame.empty
    assert calls["sr"] == 100
    assert calls["min_frequency"] == 250
    assert calls["spec_time_step"] == 0.001
    assert calls["max_length"] == 123
    assert calls["batch_size"] == 2
    assert calls["num_trials"] == 3
    assert calls["num_beams"] == 5
    assert calls["top_k"] == 2
    assert calls["top_p"] == 0.9
    assert calls["length_penalty"] == 0.8


def test_train_whisperseg_uses_shared_evaluation_report(monkeypatch: pytest.MonkeyPatch):
    import das.whisperseg.train as whisper_train

    class FakeDataModule:
        has_test = True

        def split_stats(self):
            return [
                {
                    "split": "test",
                    "audio_file_count": 1,
                    "audio_minutes": 0.1,
                    "annotation_file_count": 1,
                    "annotation_minutes": 0.01,
                }
            ]

    report_calls = {}
    train_calls = {}
    fake_datamodule = FakeDataModule()

    monkeypatch.setattr(api, "resolve_training_data_dir", lambda data_dir: data_dir)
    monkeypatch.setattr(api, "_build_whisperseg_report_datamodule", lambda config, data_dir: fake_datamodule)

    def fake_train_whisperseg(**kwargs):
        train_calls.update(kwargs)
        return "/tmp/run/checkpoints/zf_20260423_203741_model.ckpt"

    monkeypatch.setattr(whisper_train, "train", fake_train_whisperseg)
    monkeypatch.setattr(api, "_training_start_timestamp", lambda: "20260423_203741")

    def fake_print_report(config, **kwargs):
        report_calls["config"] = config
        report_calls.update(kwargs)
        return {"evaluated_file_count": 1, "skipped_file_count": 0}

    monkeypatch.setattr(api, "_print_whisperseg_evaluation_report", fake_print_report)

    result = api._train_whisperseg(
        Config(
            mode="train",
            encoder_type="whisperseg",
            initial_model="model.ckpt",
            data_dir="data",
            output_dir="run",
            checkpoint_prefix="zf",
            num_epochs=20,
            encoder_dropout=0.2,
            decoder_dropout=0.3,
            min_frequency=200,
            frequency_scale=0.75,
            spec_time_step=0.002,
            max_length=120,
        ),
        verbose=True,
        stop_event=None,
        emit_epoch_logs=False,
    )

    assert result == "/tmp/run/checkpoints/zf_20260423_203741_model.ckpt"
    assert train_calls["num_epochs"] == 20
    assert train_calls["encoder_dropout"] == 0.2
    assert train_calls["decoder_dropout"] == 0.3
    assert train_calls["min_frequency"] == 200
    assert train_calls["frequency_scale"] == 0.75
    assert train_calls["spec_time_step"] == 0.002
    assert train_calls["max_length"] == 120
    assert train_calls["linear_lr_schedule"] is True
    assert train_calls["reduce_lr"] is False
    assert train_calls["early_stopping"] is True
    assert train_calls["early_stopping_patience"] == 3
    assert train_calls["checkpoint_filename"] == "zf_20260423_203741_model"
    assert train_calls["checkpoint_metadata"] == {
        "version": 1,
        "backend": "whisperseg",
        "train": {
            "started_at": "20260423_203741",
            "checkpoint_prefix": "zf",
            "initial_model": "model.ckpt",
            "freeze_encoder": False,
            "linear_lr_schedule": True,
            "reduce_lr": False,
            "reduce_lr_patience": 5,
            "reduce_lr_factor": 0.1,
            "reduce_lr_min": 1e-08,
            "early_stopping": True,
            "early_stopping_patience": 3,
            "weight_decay": 0.01,
            "warmup_steps": 100,
        },
    }
    assert report_calls["checkpoint_path"] == "/tmp/run/checkpoints/zf_20260423_203741_model.ckpt"
    assert report_calls["datamodule"] is fake_datamodule
    assert report_calls["split"] == "test"


def test_whisperseg_train_uses_regular_checkpoint_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import das.whisperseg.train as whisper_train

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(species_codebook={"unknown": "<|unknown|>"})

    checkpoint_kwargs = {}
    trainer_kwargs = {}
    early_stopping_kwargs = {}

    class FakeCheckpoint:
        def __init__(self, **kwargs):
            checkpoint_kwargs.update(kwargs)
            self.best_model_path = str(Path(kwargs["dirpath"]) / f"{kwargs['filename']}.ckpt")

    class FakeTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)

        def fit(self, model, train_dataloaders, val_dataloaders):
            del model, train_dataloaders, val_dataloaders

    monkeypatch.setattr(whisper_train, "load_model", lambda **kwargs: (FakeModel(), object()))
    monkeypatch.setattr(whisper_train, "resolve_training_data_dir", lambda data_dir: data_dir)
    monkeypatch.setattr(whisper_train, "get_audio_and_label_paths", lambda folder: (["clip.wav"], ["clip.csv"]))
    monkeypatch.setattr(
        whisper_train,
        "determine_default_config",
        lambda *args, **kwargs: {
            "species": "unknown",
            "sr": 1000,
            "input_sr": 1000,
            "min_frequency": 0,
            "frequency_scale": 1.0,
            "spec_time_step": 0.001,
        },
    )
    monkeypatch.setattr(whisper_train, "get_cluster_codebook", lambda *args, **kwargs: {"song": 0})
    monkeypatch.setattr(whisper_train, "build_vocalseg_clip_records", lambda *args, **kwargs: ([object()], [], []))
    monkeypatch.setattr(whisper_train, "VocalSegDataset", lambda *args, **kwargs: [0])
    monkeypatch.setattr(whisper_train, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(whisper_train, "LearningRateMonitor", lambda **kwargs: ("lr", kwargs))
    monkeypatch.setattr(
        whisper_train,
        "EarlyStopping",
        lambda **kwargs: early_stopping_kwargs.update(kwargs) or ("early", kwargs),
    )
    monkeypatch.setattr(whisper_train, "CSVLogger", lambda **kwargs: ("logger", kwargs))
    monkeypatch.setattr(whisper_train.pl, "Trainer", FakeTrainer)

    result = whisper_train.train(
        str(tmp_path / "run"),
        "data",
        initial_model_path="model.ckpt",
        validation_fraction=0.0,
        num_epochs=1,
        checkpoint_filename="zf_20260423_203741_model",
        checkpoint_metadata={"version": 1, "backend": "whisperseg"},
    )

    assert result == str(tmp_path / "run" / "checkpoints" / "zf_20260423_203741_model.ckpt")
    assert checkpoint_kwargs["dirpath"] == str(tmp_path / "run" / "checkpoints")
    assert checkpoint_kwargs["filename"] == "zf_20260423_203741_model"
    assert checkpoint_kwargs["monitor"] == "train_loss_epoch"
    assert checkpoint_kwargs["save_top_k"] == 1
    assert checkpoint_kwargs["save_last"] is False
    assert checkpoint_kwargs["enable_version_counter"] is False
    assert trainer_kwargs["logger"] == ("logger", {"save_dir": str(tmp_path / "run"), "name": "logs"})
    assert early_stopping_kwargs["monitor"] == "train_loss_epoch"
    assert early_stopping_kwargs["patience"] == 3
    assert any(isinstance(callback, tuple) and callback[0] == "early" for callback in trainer_kwargs["callbacks"])
    assert "enable_checkpointing" not in trainer_kwargs

    trainer_kwargs.clear()
    early_stopping_kwargs.clear()
    whisper_train.train(
        str(tmp_path / "run"),
        "data",
        initial_model_path="model.ckpt",
        validation_fraction=0.0,
        num_epochs=1,
        early_stopping=False,
    )

    assert early_stopping_kwargs == {}
    assert all(not (isinstance(callback, tuple) and callback[0] == "early") for callback in trainer_kwargs["callbacks"])


def test_whisperseg_lightning_checkpoint_embeds_metadata(monkeypatch: pytest.MonkeyPatch):
    import das.whisperseg.train as whisper_train

    monkeypatch.setattr(
        whisper_train,
        "checkpoint_payload",
        lambda model, tokenizer, current_step: {
            "model": model,
            "tokenizer": tokenizer,
            "current_step": current_step,
        },
    )
    metadata = {"version": 1, "backend": "whisperseg"}
    model = object()
    tokenizer = object()
    module = whisper_train.WhisperSegLightningModule(
        model,
        learning_rate=1e-4,
        weight_decay=0.0,
        linear_lr_schedule=False,
        warmup_steps=0,
        reduce_lr=False,
        reduce_lr_patience=5,
        reduce_lr_factor=0.1,
        reduce_lr_min=1e-8,
        lr_monitor="val_loss",
        total_training_steps=None,
        tokenizer=tokenizer,
        checkpoint_metadata=metadata,
    )

    checkpoint = {}
    module.on_save_checkpoint(checkpoint)

    assert checkpoint["das"] == metadata
    assert checkpoint["whisperseg"] == {
        "model": model,
        "tokenizer": tokenizer,
        "current_step": 0,
    }


def test_whisperseg_train_rejects_multiple_lr_schedulers(tmp_path: Path):
    import das.whisperseg.train as whisper_train

    with pytest.raises(ValueError, match="either linear_lr_schedule or reduce_lr"):
        whisper_train.train(
            str(tmp_path / "run"),
            "data",
            initial_model_path="model.ckpt",
            linear_lr_schedule=True,
            reduce_lr=True,
        )


def test_whisperseg_lightning_module_configures_reduce_lr_scheduler():
    import das.whisperseg.train as whisper_train

    module = whisper_train.WhisperSegLightningModule(
        torch.nn.Linear(2, 2),
        learning_rate=1e-4,
        weight_decay=0.0,
        linear_lr_schedule=False,
        warmup_steps=0,
        reduce_lr=True,
        reduce_lr_patience=3,
        reduce_lr_factor=0.25,
        reduce_lr_min=1e-7,
        lr_monitor="train_loss_epoch",
        total_training_steps=None,
        tokenizer=object(),
    )

    config = module.configure_optimizers()

    assert config["lr_scheduler"]["monitor"] == "train_loss_epoch"
    scheduler = config["lr_scheduler"]["scheduler"]
    assert scheduler.patience == 3
    assert scheduler.factor == 0.25
    assert scheduler.min_lrs == [1e-7, 1e-7]


def test_whisperseg_evaluation_uses_shared_report_printer(monkeypatch: pytest.MonkeyPatch):
    annotation_table = pd.DataFrame(
        [{"name": "song", "start_seconds": 0.0, "stop_seconds": 0.5}]
    )

    class FakeSegmenter:
        cluster_codebook = {"song": 0}
        default_segmentation_config = {"spec_time_step": 0.25}

    datamodule = SimpleNamespace(
        subsets={"test": {"audio": [Path("clip.wav")], "annotation_tables": [annotation_table]}},
    )
    captured = {}

    monkeypatch.setattr(api, "_whisper_segmenter", lambda config: FakeSegmenter())
    monkeypatch.setattr(api, "load_audio_array", lambda audio_file, **kwargs: (np.zeros(4, dtype=np.float32), 4))
    monkeypatch.setattr(
        api,
        "_whisper_segment_audio",
        lambda segmenter, config, audio, samplerate, filename=None: pd.DataFrame(
            [{"filename": filename, "name": "song", "start_seconds": 0.0, "stop_seconds": 0.5}]
        ),
    )
    monkeypatch.setattr(api.prediction_results, "print_evaluation_report", lambda report: captured.update(report.__dict__))

    summary = api._print_whisperseg_evaluation_report(
        Config(mode="predict", data_dir="data", checkpoint="model.ckpt"),
        checkpoint_path="model.ckpt",
        datamodule=datamodule,
        split="test",
    )

    assert summary == {"evaluated_file_count": 1, "skipped_file_count": 0}
    assert captured["class_names"] == ["noise", "song"]
    assert captured["summary"] == summary
    assert captured["syllable_wer"] == 0.0


def test_predict_whisperseg_evaluate_prints_report_not_predictions(monkeypatch: pytest.MonkeyPatch, capsys):
    class FakeSegmenter:
        cluster_codebook = {"song": 0}

    report_calls = {}
    fake_datamodule = object()

    monkeypatch.setattr(api, "_is_whisperseg_predict_checkpoint", lambda checkpoint: checkpoint == "model.ckpt")
    monkeypatch.setattr(api, "_whisper_segmenter", lambda config: FakeSegmenter())
    monkeypatch.setattr(api, "_whisper_audio_paths", lambda data_dir: [Path("clip.wav")])
    monkeypatch.setattr(api, "load_audio_array", lambda audio_file, **kwargs: (np.zeros(4, dtype=np.float32), 4))
    monkeypatch.setattr(
        api,
        "_whisper_segment_audio",
        lambda segmenter, config, audio, samplerate, filename=None: pd.DataFrame(
            [{"filename": filename, "name": "song", "start_seconds": 0.0, "stop_seconds": 0.5}]
        ),
    )
    monkeypatch.setattr(api, "resolve_training_data_dir", lambda data_dir: data_dir)
    monkeypatch.setattr(api, "_build_whisperseg_report_datamodule", lambda config, data_dir: fake_datamodule)

    def fake_print_report(config, **kwargs):
        report_calls["config"] = config
        report_calls.update(kwargs)
        print("Dense confusion matrix:")
        return {"evaluated_file_count": 1, "skipped_file_count": 0}

    monkeypatch.setattr(api, "_print_whisperseg_evaluation_report", fake_print_report)

    api.predict(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir="",
            evaluate=True,
            split="test",
        ),
        verbose=True,
    )

    output = capsys.readouterr().out
    assert "Dense confusion matrix:" in output
    assert '"name": "song"' not in output
    assert report_calls["datamodule"] is fake_datamodule
    assert report_calls["split"] == "test"


def test_predict_whisperseg_passes_merge_to_annotation_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeSegmenter:
        cluster_codebook = {"song": 0}

    calls = []

    monkeypatch.setattr(api, "_is_whisperseg_predict_checkpoint", lambda checkpoint: checkpoint == "model.ckpt")
    monkeypatch.setattr(api, "_whisper_segmenter", lambda config: FakeSegmenter())
    monkeypatch.setattr(api, "_whisper_audio_paths", lambda data_dir: [Path("clip.wav")])
    monkeypatch.setattr(api, "load_audio_array", lambda audio_file, **kwargs: (np.zeros(4, dtype=np.float32), 4))
    monkeypatch.setattr(
        api,
        "_whisper_segment_audio",
        lambda segmenter, config, audio, samplerate, filename=None: pd.DataFrame(
            [{"filename": filename, "name": "song", "start_seconds": 0.0, "stop_seconds": 0.5}]
        ),
    )
    monkeypatch.setattr(
        api.prediction_results,
        "write_annotation_file",
        lambda output_path, annotation, *, merge=False: calls.append(
            {"output_path": output_path, "annotation": annotation.copy(), "merge": merge}
        ),
    )

    written_files = api.predict(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir=str(tmp_path),
            existing_annotations="merge",
        )
    )

    assert written_files == [str(tmp_path / "clip_annotations.csv")]
    assert calls[0]["output_path"] == tmp_path / "clip_annotations.csv"
    assert calls[0]["merge"] is True
    assert "filename" not in calls[0]["annotation"].columns


def test_predict_whisperseg_skips_existing_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    existing_path = tmp_path / "existing_annotations.csv"
    pd.DataFrame([{"name": "old"}]).to_csv(existing_path, index=False)
    loaded_paths = []

    monkeypatch.setattr(api, "_is_whisperseg_predict_source", lambda checkpoint: True)
    monkeypatch.setattr(api, "_whisper_segmenter", lambda config: object())
    monkeypatch.setattr(
        api,
        "_whisper_audio_paths_for_config",
        lambda config: [Path("existing.wav"), Path("new.wav")],
    )

    def fake_load_audio_array(audio_path, **kwargs):
        loaded_paths.append(audio_path)
        return np.zeros(4, dtype=np.float32), 4

    monkeypatch.setattr(api, "load_audio_array", fake_load_audio_array)
    monkeypatch.setattr(
        api,
        "_whisper_segment_audio",
        lambda *args, filename=None, **kwargs: pd.DataFrame(
            [{"filename": filename, "name": "song", "start_seconds": 0.0, "stop_seconds": 0.5}]
        ),
    )

    written = api.predict(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir=str(tmp_path),
            existing_annotations="skip",
        )
    )

    assert loaded_paths == [Path("new.wav")]
    assert written == [str(tmp_path / "new_annotations.csv")]
    assert pd.read_csv(existing_path).to_dict("records") == [{"name": "old"}]


def test_predict_whisperseg_shows_file_progress_for_multiple_files(monkeypatch: pytest.MonkeyPatch):
    progress_kwargs = {}

    monkeypatch.setattr(api, "_whisper_segmenter", lambda config: object())
    monkeypatch.setattr(api, "_whisper_audio_paths_for_config", lambda config: [Path("one.wav"), Path("two.wav")])
    monkeypatch.setattr(api, "load_audio_array", lambda audio_file, **kwargs: (np.zeros(4, dtype=np.float32), 4))
    monkeypatch.setattr(api, "_whisper_segment_audio", lambda *args, **kwargs: pd.DataFrame())

    def fake_tqdm(iterable, **kwargs):
        progress_kwargs.update(kwargs)
        return iterable

    monkeypatch.setattr(api, "tqdm", fake_tqdm)

    api._predict_whisperseg(
        Config(mode="predict", data_dir="data", checkpoint="model.ckpt", output_dir=""),
        audio=None,
        samplerate=None,
        verbose=True,
    )

    assert progress_kwargs == {"desc": "Predicting files", "unit": "file", "disable": False}


def test_predict_whisperseg_stop_event_finishes_current_file(monkeypatch: pytest.MonkeyPatch):
    stop_event = threading.Event()
    predicted_files = []

    monkeypatch.setattr(api, "_whisper_segmenter", lambda config: object())
    monkeypatch.setattr(api, "_whisper_audio_paths_for_config", lambda config: [Path("one.wav"), Path("two.wav")])
    monkeypatch.setattr(api, "load_audio_array", lambda audio_file, **kwargs: (np.zeros(4, dtype=np.float32), 4))

    def fake_segment(*args, filename=None, **kwargs):
        predicted_files.append(filename)
        stop_event.set()
        return pd.DataFrame([{"filename": filename, "name": "song", "start_seconds": 0.0, "stop_seconds": 0.5}])

    monkeypatch.setattr(api, "_whisper_segment_audio", fake_segment)

    result = api._predict_whisperseg(
        Config(mode="predict", data_dir="data", checkpoint="model.ckpt", output_dir=""),
        audio=None,
        samplerate=None,
        verbose=False,
        stop_event=stop_event,
    )

    assert predicted_files == ["one.wav"]
    assert result["filename"].tolist() == ["one.wav"]


def test_prediction_file_progress_tracks_completed_file_boundaries(monkeypatch: pytest.MonkeyPatch):
    class FakeProgress:
        def __init__(self, *, total, **kwargs):
            self.total = total
            self.n = 0
            self.closed = False

        def update(self, amount):
            self.n += amount

        def close(self):
            self.closed = True

    monkeypatch.setattr(api, "tqdm", FakeProgress)
    default_progress_disabled = []
    trainer = SimpleNamespace(
        is_global_zero=True,
        callbacks=[],
        progress_bar_callback=SimpleNamespace(disable=lambda: default_progress_disabled.append(True)),
    )
    loader = SimpleNamespace(
        batch_size=2,
        dataset=SimpleNamespace(audio_files=["one.wav", "two.wav", "three.wav"], chunk_borders=[0, 2, 5, 6]),
    )
    api._add_prediction_file_progress(trainer, loader, verbose=True)
    callback = trainer.callbacks[0]

    callback.on_predict_start(trainer, None)
    callback.on_predict_batch_end(trainer, None, None, None, 0)
    assert callback.progress.n == 1
    callback.on_predict_batch_end(trainer, None, None, None, 1)
    assert callback.progress.n == 1
    callback.on_predict_batch_end(trainer, None, None, None, 2)
    callback.on_predict_end(trainer, None)

    assert callback.progress.n == 3
    assert callback.progress.closed is True
    assert default_progress_disabled == [True]


def test_conformer_stop_event_finishes_current_file(monkeypatch: pytest.MonkeyPatch):
    stop_event = threading.Event()
    predicted_files = []
    progress_bar_disabled = []

    class FakeTrainer:
        progress_bar_callback = SimpleNamespace(disable=lambda: progress_bar_disabled.append(True))

        def predict(self, model, dataloaders):
            del model
            predicted_files.extend(dataloaders.dataset.audio_files)
            stop_event.set()
            return ["predictions"]

    monkeypatch.setattr(api, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(
        api,
        "_build_inference_datamodule",
        lambda data_dir, **kwargs: SimpleNamespace(
            predict_dataloader=lambda: SimpleNamespace(dataset=SimpleNamespace(audio_files=[data_dir]))
        ),
    )
    monkeypatch.setattr(
        api,
        "_prediction_outputs",
        lambda config, *, dataset, **kwargs: ([f"{dataset.audio_files[0]}.csv"], None),
    )

    outputs = api._predict_files_until_stopped(
        Config(mode="predict", data_dir="data", checkpoint="model.ckpt", output_dir="predictions"),
        inference=SimpleNamespace(trainer=FakeTrainer(), model=object(), runtime=object(), context=object()),
        audio_files=["one.wav", "two.wav"],
        stop_event=stop_event,
        verbose=False,
    )

    assert outputs == ["one.wav.csv"]
    assert predicted_files == ["one.wav"]
    assert progress_bar_disabled == [True]


def test_infer_samplerate_finds_nested_audio(tmp_path: Path):
    nested = tmp_path / "train"
    nested.mkdir()
    sf.write(nested / "clip.wav", np.zeros(128, dtype=np.float32), 16_000)

    assert api._infer_samplerate(str(tmp_path)) == 16_000


def test_train_resamples_mixed_rate_folder_to_median(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    for name, rate in (("low", 8_000), ("high-a", 16_000), ("high-b", 16_000)):
        sf.write(tmp_path / f"{name}.wav", np.zeros(rate // 10, dtype=np.float32), rate)
        pd.DataFrame([{"name": "song", "start_seconds": 0.02, "stop_seconds": 0.04}]).to_csv(
            tmp_path / f"{name}_annotations.csv", index=False
        )

    config = Config(mode="train", data_dir=str(tmp_path), frontend_type="raw", num_time_steps=512,
                    batch_size=1, validation_fraction=0, test_fraction=0, num_workers=0)
    datamodule, samplerate, _ = api._build_train_datamodule(config)

    assert samplerate == 16_000
    assert datamodule.target_samplerate == 16_000
    dataset = datamodule.train_dataloader().dataset
    low_idx = dataset.audio_files.index(tmp_path / "low.wav")
    assert dataset.source_samplerate_per_file[low_idx] == 8_000
    assert dataset.samplerate_per_file[low_idx] == 16_000
    assert dataset.nb_samples_in_file[low_idx] == 1_600
    assert dataset[int(dataset.chunk_borders[low_idx])][0].shape == (512,)

    monkeypatch.setattr(api, "_build_train_datamodule", lambda config: (datamodule, samplerate, False))

    def stop_before_model():
        raise RuntimeError("stop before model")

    monkeypatch.setattr(api, "_training_start_timestamp", stop_before_model)
    with pytest.raises(RuntimeError, match="stop before model"):
        api._train_supervised(config, verbose=True, stop_event=None, emit_epoch_logs=False)
    assert "resampling on the fly to median 16000 Hz" in capsys.readouterr().out


def test_train_uses_target_samplerate_without_inferring_input_rates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.annotated_audio_files = [Path("annotated.wav")]
            self.has_test = False

    monkeypatch.setattr(api, "AudioDirDataModule", FakeDataModule)
    monkeypatch.setattr(api, "_infer_samplerate", lambda data_dir, **kwargs: pytest.fail("should not infer input rates"))

    _, samplerate, _ = api._build_train_datamodule(
        Config(mode="train", data_dir=str(tmp_path), target_samplerate_hz=32_000, num_workers=0)
    )

    assert samplerate == 32_000
    assert captured["target_samplerate"] == 32_000


def test_train_uses_config_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    datamodule_kwargs = {}
    model_kwargs = {}
    trainer_kwargs = {}
    checkpoint_kwargs = {}

    monkeypatch.setattr(api, "_infer_samplerate", lambda data_dir, **kwargs: 16_000)
    monkeypatch.setattr(api, "_training_start_timestamp", lambda: "20260423_203741")

    class FakeDataModule:
        def __init__(self, **kwargs):
            datamodule_kwargs.update(kwargs)
            self.annotated_audio_files = [Path("annotated.wav")]
            self.num_classes = 2
            self.class_names = ["noise", "cfg_song"]
            self.class_types = ["segment", "event"]
            self.input_num_channels = 1
            self.num_time_steps = kwargs["num_time_steps"]
            self.chunk_stride = kwargs["chunk_stride"]
            self.has_test = False

        @property
        def has_val(self):
            return False

        def train_dataloader(self):
            return SimpleNamespace(dataset=[0])

    class FakeModel:
        def __init__(self, **kwargs):
            model_kwargs.update(kwargs)

    class FakeCheckpoint:
        def __init__(self, **kwargs):
            checkpoint_kwargs.update(kwargs)
            self.best_model_path = "/tmp/model.ckpt"
            self.last_model_path = ""

    class FakeTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)

        def fit(self, model, train_dataloaders, val_dataloaders):
            del model, train_dataloaders, val_dataloaders

    monkeypatch.setattr(api, "AudioDirDataModule", FakeDataModule)
    monkeypatch.setattr(api, "DASModel", FakeModel)
    monkeypatch.setattr(api, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(api, "CSVLogger", lambda **kwargs: ("logger", kwargs))
    monkeypatch.setattr(api.L, "Trainer", FakeTrainer)

    checkpoint_path = api.train(
        data_dir="data",
        output_dir=str(tmp_path / "run"),
        checkpoint_prefix="zf",
        batch_size=5,
        num_time_steps=2048,
        chunk_stride=1024,
        num_workers=2,
        include_labels=["pulse"],
        frontend_type="raw",
        encoder_type="conformer",
        num_epochs=1,
    )

    assert checkpoint_path == "/tmp/model.ckpt"
    assert datamodule_kwargs["batch_size"] == 5
    assert datamodule_kwargs["num_time_steps"] == 2048
    assert datamodule_kwargs["chunk_stride"] == 1024
    assert datamodule_kwargs["include_labels"] == ["pulse"]
    assert "class_names" not in datamodule_kwargs
    assert model_kwargs["frontend"]["type"] == "raw"
    assert model_kwargs["class_names"] == ["noise", "cfg_song"]
    assert model_kwargs["class_types"] == ["segment", "event"]
    assert model_kwargs["checkpoint_metadata"] == {
        "version": 1,
        "predict": {
            "num_time_steps": 2048,
            "chunk_stride": 1024,
            "batch_size": 5,
            "fill_gap_ms": 10.0,
            "min_syllable_ms": 10.0,
            "segment_threshold_low": 0.5,
            "segment_threshold_high": 0.5,
            "event_threshold": 0.5,
            "event_dist_min_ms": 0.0,
            "event_dist_max_ms": None,
            "syllable_postprocessor": "label_aware_dense",
        },
        "train": {
            "started_at": "20260423_203741",
            "checkpoint_prefix": "zf",
            "initial_model": None,
            "freeze_encoder": False,
        },
    }
    assert checkpoint_kwargs["filename"] == "zf_20260423_203741_model"
    assert checkpoint_kwargs["enable_version_counter"] is False
    assert trainer_kwargs["max_epochs"] == 1


def test_log_datamodule_split_stats_formats_summary(capsys: pytest.CaptureFixture[str]):
    class FakeDataModule:
        def split_stats(self):
            return [
                {
                    "split": "train",
                    "audio_file_count": 3,
                    "audio_minutes": 12.345,
                    "annotation_file_count": 3,
                    "annotation_count": 42,
                    "annotation_minutes": 1.234,
                }
            ]

    api._log_datamodule_split_stats(FakeDataModule())

    output = capsys.readouterr().out
    assert "Dataset split summary" in output
    assert "train" in output
    assert "12.35" in output
    assert "42" in output
    assert "1.23" in output


def test_train_supports_npy_dir_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_kwargs = {}
    data_dir = _write_npy_dir_fixture(tmp_path, channels=2)

    class FakeModel:
        def __init__(self, **kwargs):
            model_kwargs.update(kwargs)

    class FakeCheckpoint:
        def __init__(self, **kwargs):
            del kwargs
            self.best_model_path = "/tmp/model.ckpt"
            self.last_model_path = ""

    class FakeTrainer:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, model, train_dataloaders, val_dataloaders):
            del model, train_dataloaders, val_dataloaders

    monkeypatch.setattr(api, "DASModel", FakeModel)
    monkeypatch.setattr(api, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(api, "EarlyStopping", lambda **kwargs: ("early", kwargs))
    monkeypatch.setattr(api, "CSVLogger", lambda **kwargs: ("logger", kwargs))
    monkeypatch.setattr(api.L, "Trainer", FakeTrainer)

    checkpoint_path = api.train(
        Config(
            mode="train",
            data_dir=str(data_dir),
            output_dir=str(tmp_path / "run"),
            batch_size=2,
            num_time_steps=32,
            chunk_stride=16,
            num_workers=0,
            frontend_type="raw",
            num_epochs=1,
        )
    )

    assert checkpoint_path == "/tmp/model.ckpt"
    assert model_kwargs["frontend"]["type"] == "raw"
    assert model_kwargs["frontend"]["num_channels"] == 2
    assert model_kwargs["class_names"] == ["noise", "song"]
    assert model_kwargs["class_types"] == ["segment", "segment"]


def test_train_configures_native_early_stopping_patience(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trainer_kwargs = {}
    early_stopping_kwargs = {}
    data_dir = _write_npy_dir_fixture(tmp_path)

    class FakeModel:
        def __init__(self, **kwargs):
            del kwargs

    class FakeCheckpoint:
        def __init__(self, **kwargs):
            del kwargs
            self.best_model_path = "/tmp/model.ckpt"
            self.last_model_path = ""

    class FakeTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)

        def fit(self, model, train_dataloaders, val_dataloaders):
            del model, train_dataloaders, val_dataloaders

    def fake_early_stopping(**kwargs):
        early_stopping_kwargs.update(kwargs)
        return ("early", kwargs)

    monkeypatch.setattr(api, "DASModel", FakeModel)
    monkeypatch.setattr(api, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(api, "EarlyStopping", fake_early_stopping)
    monkeypatch.setattr(api, "CSVLogger", lambda **kwargs: ("logger", kwargs))
    monkeypatch.setattr(api.L, "Trainer", FakeTrainer)

    api.train(
        Config(
            mode="train",
            data_dir=str(data_dir),
            output_dir=str(tmp_path / "run"),
            batch_size=2,
            num_time_steps=32,
            chunk_stride=16,
            num_workers=0,
            frontend_type="raw",
            num_epochs=1,
            early_stopping_patience=4,
        )
    )

    assert early_stopping_kwargs["patience"] == 4
    assert ("early", early_stopping_kwargs) in trainer_kwargs["callbacks"]


def test_train_can_disable_native_early_stopping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trainer_kwargs = {}
    early_stopping_calls = []
    data_dir = _write_npy_dir_fixture(tmp_path)

    class FakeModel:
        def __init__(self, **kwargs):
            del kwargs

    class FakeCheckpoint:
        def __init__(self, **kwargs):
            del kwargs
            self.best_model_path = "/tmp/model.ckpt"
            self.last_model_path = ""

    class FakeTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)

        def fit(self, model, train_dataloaders, val_dataloaders):
            del model, train_dataloaders, val_dataloaders

    def fake_early_stopping(**kwargs):
        early_stopping_calls.append(kwargs)
        return ("early", kwargs)

    monkeypatch.setattr(api, "DASModel", FakeModel)
    monkeypatch.setattr(api, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(api, "EarlyStopping", fake_early_stopping)
    monkeypatch.setattr(api, "CSVLogger", lambda **kwargs: ("logger", kwargs))
    monkeypatch.setattr(api.L, "Trainer", FakeTrainer)

    api.train(
        Config(
            mode="train",
            data_dir=str(data_dir),
            output_dir=str(tmp_path / "run"),
            batch_size=2,
            num_time_steps=32,
            chunk_stride=16,
            num_workers=0,
            frontend_type="raw",
            num_epochs=1,
            early_stopping=False,
        )
    )

    assert early_stopping_calls == []
    assert all(not (isinstance(callback, tuple) and callback[0] == "early") for callback in trainer_kwargs["callbacks"])


def test_predict_uses_checkpoint_metadata_and_predict_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    datamodule_kwargs = {}
    writer_kwargs = {}

    class FakePredictDataset:
        audio_files = []
        chunk_borders = [0]

        def __len__(self):
            return 1

    class FakeDataModule:
        def __init__(self, **kwargs):
            datamodule_kwargs.update(kwargs)

        def predict_dataloader(self):
            return SimpleNamespace(dataset=FakePredictDataset())

    class FakeModel:
        num_classes = 2
        hparams = {
            "sr": 16_000,
            "encoder": {"type": "conformer"},
            "frontend": {
                "type": "mel",
                "num_channels": 64,
                "kernel_size": 64,
                "hop_seconds": 8 / 16_000,
                "fmin": 100.0,
                "fmax": None,
                "trainable": True,
            },
            "num_time_steps": 2048,
            "chunk_stride": 512,
            "class_names": ["noise", "ckpt_song"],
        }

    class FakeTrainer:
        def __init__(self, **kwargs):
            del kwargs

        def predict(self, model, dataloaders):
            del model, dataloaders
            return []

    monkeypatch.setattr(api, "AudioDirDataModule", FakeDataModule)
    monkeypatch.setattr(
        api,
        "_predict_whisperseg",
        lambda *args, **kwargs: pytest.fail("encoder_type must not force WhisperSeg prediction"),
    )
    monkeypatch.setattr(
        api,
        "_load_checkpoint_metadata",
        lambda checkpoint: {
            "predict": {
                "num_time_steps": 1024,
                "chunk_stride": 256,
                "batch_size": 7,
                "fill_gap_ms": 12.0,
                "min_syllable_ms": 34.0,
                "event_dist_min_ms": 3.0,
                "event_dist_max_ms": 45.0,
                "syllable_postprocessor": "binary_mask",
            }
        },
    )
    monkeypatch.setattr(api.DASModel, "load_from_checkpoint", staticmethod(lambda checkpoint: FakeModel()))
    monkeypatch.setattr(api.L, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        api.prediction_results,
        "write_prediction_outputs",
        lambda **kwargs: writer_kwargs.update(kwargs) or ["/tmp/output_frames.csv"],
    )

    written_files = api.predict(
        Config(
            mode="predict",
            encoder_type="whisperseg",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir="predictions",
            batch_size=2,
            num_time_steps=256,
            chunk_stride=128,
            num_workers=0,
            output_suffix="_frames.csv",
            existing_annotations="merge",
            fill_gap_ms=15,
            min_syllable_ms=25,
            event_dist_min_ms=7,
            event_dist_max_ms=8,
        )
    )

    assert written_files == ["/tmp/output_frames.csv"]
    assert datamodule_kwargs["num_time_steps"] == 1024
    assert datamodule_kwargs["chunk_stride"] == 256
    assert writer_kwargs["output_suffix"] == "_frames.csv"
    assert writer_kwargs["existing_annotations"] == "merge"
    assert writer_kwargs["context"].fill_gap_seconds == pytest.approx(0.015)
    assert writer_kwargs["context"].min_syllable_seconds == pytest.approx(0.025)
    assert writer_kwargs["context"].event_dist_min_seconds == pytest.approx(0.007)
    assert writer_kwargs["context"].event_dist_max_seconds == pytest.approx(0.008)
    assert writer_kwargs["context"].postprocessor == "label_aware_dense"
    assert capsys.readouterr().out == ""

    writer_kwargs.clear()
    api.predict(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir="predictions",
            batch_size=2,
            num_workers=0,
            output_suffix="_frames.csv",
        ),
        verbose=True,
    )
    assert writer_kwargs["context"].fill_gap_seconds == pytest.approx(0.012)
    assert writer_kwargs["context"].min_syllable_seconds == pytest.approx(0.034)
    assert writer_kwargs["context"].event_dist_min_seconds == pytest.approx(0.003)
    assert writer_kwargs["context"].event_dist_max_seconds == pytest.approx(0.045)
    assert writer_kwargs["context"].postprocessor == "binary_mask"
    assert writer_kwargs["existing_annotations"] == "overwrite"
    output = capsys.readouterr().out
    assert "Prediction summary:" in output
    assert "Total instances: 0" in output


def test_predict_rejects_npy_dir_data_dir(tmp_path: Path):
    data_dir = _write_npy_dir_fixture(tmp_path)

    with pytest.raises(ValueError, match="Legacy npy_dir datasets are only supported for `das train`"):
        api.predict(
            Config(
                mode="predict",
                data_dir=str(data_dir),
                checkpoint="model.ckpt",
                output_dir="predictions",
            )
        )


def test_predict_accepts_raw_audio_and_returns_annotations(monkeypatch: pytest.MonkeyPatch, capsys):
    class FakeModel:
        pass

    class FakeTrainer:
        def predict(self, model, dataloaders):
            del model
            inputs, input_lengths = next(iter(dataloaders))
            assert inputs.shape == (1, 8)
            assert input_lengths.tolist() == [8]

            logits = torch.full((1, 8, 2), fill_value=-8.0)
            logits[:, :, 0] = 8.0
            logits[:, 2:5, 0] = -8.0
            logits[:, 2:5, 1] = 8.0
            return [(logits, input_lengths)]

    monkeypatch.setattr(
        api,
        "_load_predict_model",
        lambda checkpoint, config: (
            FakeModel(),
            api.PredictRuntime(
                sr=100,
                hop_seconds=0.01,
                num_time_steps=8,
                chunk_stride=None,
                class_names=["noise", "song"],
                class_types=["segment", "segment"],
            ),
        ),
    )
    monkeypatch.setattr(api, "_build_inference_trainer", lambda config, **kwargs: FakeTrainer())
    monkeypatch.setattr(
        api.prediction_results,
        "write_prediction_outputs",
        lambda **kwargs: pytest.fail("raw audio prediction should not write files"),
    )

    annotations = api.predict(
        audio=np.zeros(8, dtype=np.float32),
        samplerate=100,
        checkpoint="model.ckpt",
        output_dir="predictions",
        batch_size=1,
        num_workers=0,
        fill_gap_ms=0,
        min_syllable_ms=0,
        verbose=True,
    )

    assert annotations.to_dict("records") == [
        {"name": "song", "start_seconds": 0.025, "stop_seconds": 0.045},
    ]
    output = capsys.readouterr().out
    assert "Prediction summary:" in output
    assert "Total instances: 1" in output
    assert "song: 1" in output


def test_predict_resamples_raw_audio_to_model_samplerate(monkeypatch: pytest.MonkeyPatch, capsys):
    class FakeModel:
        pass

    class FakeTrainer:
        def predict(self, model, dataloaders):
            del model
            dataset = dataloaders.dataset
            assert dataset.samplerate_per_file.tolist() == [100]
            assert dataset.nb_samples_in_file.tolist() == [8]
            inputs, input_lengths = next(iter(dataloaders))
            assert inputs.shape == (1, 8)
            assert input_lengths.tolist() == [8]

            logits = torch.full((1, 8, 2), fill_value=-8.0)
            logits[:, :, 0] = 8.0
            return [(logits, input_lengths)]

    monkeypatch.setattr(
        api,
        "_load_predict_model",
        lambda checkpoint, config: (
            FakeModel(),
            api.PredictRuntime(
                sr=100,
                hop_seconds=0.01,
                num_time_steps=8,
                chunk_stride=None,
                class_names=["noise", "song"],
                class_types=["segment", "segment"],
            ),
        ),
    )
    monkeypatch.setattr(api, "_build_inference_trainer", lambda config, **kwargs: FakeTrainer())

    api.predict(
        audio=np.zeros(16, dtype=np.float32),
        samplerate=200,
        checkpoint="model.ckpt",
        batch_size=1,
        num_workers=0,
        verbose=True,
    )

    output = capsys.readouterr().out
    assert "Audio sample rate: 200 Hz; model sample rate: 100 Hz; resampling to model rate." in output


def test_predict_without_output_dir_returns_annotations(monkeypatch: pytest.MonkeyPatch, capsys):
    class FakeDataModule:
        def predict_dataloader(self):
            return SimpleNamespace(dataset=SimpleNamespace(audio_files=["clip.wav"]))

    class FakeTrainer:
        def predict(self, model, dataloaders):
            del model, dataloaders
            return ["predictions"]

    expected = pd.DataFrame([{"name": "song", "start_seconds": 0.01, "stop_seconds": 0.02}])

    monkeypatch.setattr(api, "AudioDirDataModule", lambda **kwargs: FakeDataModule())
    monkeypatch.setattr(api, "_build_inference_trainer", lambda config, **kwargs: FakeTrainer())
    monkeypatch.setattr(
        api,
        "_load_predict_model",
        lambda checkpoint, config: (
            object(),
            api.PredictRuntime(
                sr=100,
                hop_seconds=0.01,
                num_time_steps=8,
                chunk_stride=None,
                class_names=["noise", "song"],
                class_types=["segment", "segment"],
            ),
        ),
    )
    monkeypatch.setattr(api.prediction_results, "prediction_annotations", lambda **kwargs: [expected])
    monkeypatch.setattr(
        api.prediction_results,
        "write_prediction_outputs",
        lambda **kwargs: pytest.fail("should not write prediction files"),
    )

    annotations = api.predict(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir="",
        ),
        verbose=True,
    )

    assert annotations == [expected]
    output = capsys.readouterr().out
    assert "Prediction summary:" in output
    assert "Total instances: 1" in output
    assert "song: 1" in output


def test_predict_accepts_single_audio_file_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, np.zeros(16, dtype=np.float32), 1_000)
    captured = {}

    class FakeTrainer:
        def predict(self, model, dataloaders):
            del model
            captured["audio_files"] = list(dataloaders.dataset.audio_files)
            captured["dataset_length"] = len(dataloaders.dataset)
            return ["predictions"]

    expected = pd.DataFrame([{"name": "song", "start_seconds": 0.01, "stop_seconds": 0.02}])

    monkeypatch.setattr(api, "_build_inference_trainer", lambda config, **kwargs: FakeTrainer())
    monkeypatch.setattr(
        api,
        "_load_predict_model",
        lambda checkpoint, config: (
            object(),
            api.PredictRuntime(
                sr=1_000,
                hop_seconds=0.001,
                num_time_steps=8,
                chunk_stride=None,
                class_names=["noise", "song"],
                class_types=["segment", "segment"],
            ),
        ),
    )
    monkeypatch.setattr(api.prediction_results, "prediction_annotations", lambda **kwargs: [expected])

    annotations = api.predict(
        Config(
            mode="predict",
            data_dir=str(audio_path),
            checkpoint="model.ckpt",
            output_dir="",
        )
    )

    assert annotations == [expected]
    assert captured["audio_files"] == [audio_path]
    assert captured["dataset_length"] > 0


def test_predict_logs_file_and_model_samplerates(monkeypatch: pytest.MonkeyPatch, capsys):
    class FakeDataModule:
        source_samplerates = {200}

        def predict_dataloader(self):
            return SimpleNamespace(dataset=SimpleNamespace(audio_files=["clip.wav"]))

    class FakeTrainer:
        def predict(self, model, dataloaders):
            del model, dataloaders
            return ["predictions"]

    expected = pd.DataFrame([{"name": "song", "start_seconds": 0.01, "stop_seconds": 0.02}])

    monkeypatch.setattr(api, "_build_inference_datamodule", lambda **kwargs: FakeDataModule())
    monkeypatch.setattr(api, "_build_inference_trainer", lambda config, **kwargs: FakeTrainer())
    monkeypatch.setattr(
        api,
        "_load_predict_model",
        lambda checkpoint, config: (
            object(),
            api.PredictRuntime(
                sr=100,
                hop_seconds=0.01,
                num_time_steps=8,
                chunk_stride=None,
                class_names=["noise", "song"],
                class_types=["segment", "segment"],
            ),
        ),
    )
    monkeypatch.setattr(api.prediction_results, "prediction_annotations", lambda **kwargs: [expected])

    api.predict(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir="",
        ),
        verbose=True,
    )

    output = capsys.readouterr().out
    assert "Audio sample rate: 200 Hz; model sample rate: 100 Hz; resampling to model rate." in output


def test_predict_evaluation_summary_only_runs_when_verbose(monkeypatch: pytest.MonkeyPatch):
    class FakeDataModule:
        def predict_dataloader(self):
            return SimpleNamespace(dataset=SimpleNamespace(audio_files=["clip.wav"]))

    class FakeTrainer:
        def predict(self, model, dataloaders):
            del model, dataloaders
            return ["predictions"]

    monkeypatch.setattr(api, "AudioDirDataModule", lambda **kwargs: FakeDataModule())
    monkeypatch.setattr(api, "_build_inference_trainer", lambda config, **kwargs: FakeTrainer())
    monkeypatch.setattr(
        api,
        "_load_predict_model",
        lambda checkpoint, config: (
            object(),
            api.PredictRuntime(
                sr=100,
                hop_seconds=0.01,
                num_time_steps=8,
                chunk_stride=None,
                class_names=["noise", "song"],
                class_types=["segment", "segment"],
            ),
        ),
    )
    monkeypatch.setattr(api.prediction_results, "write_prediction_outputs", lambda **kwargs: ["/tmp/clip_annotations.csv"])
    monkeypatch.setattr(
        api.prediction_results,
        "evaluate_file_predictions",
        lambda **kwargs: pytest.fail("should only evaluate summaries when verbose"),
    )

    written_files = api.predict(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir="predictions",
            evaluate=True,
        )
    )

    assert written_files == ["/tmp/clip_annotations.csv"]


def test_predict_uses_legacy_model_hparams_for_datamodule(monkeypatch: pytest.MonkeyPatch):
    datamodule_kwargs = {}
    writer_kwargs = {}

    class FakePredictDataset:
        audio_files = []
        chunk_borders = [0]

        def __len__(self):
            return 1

    dataset = FakePredictDataset()

    class FakeDataModule:
        def __init__(self, **kwargs):
            datamodule_kwargs.update(kwargs)

        def predict_dataloader(self):
            return SimpleNamespace(dataset=dataset)

    class FakeLegacyModel:
        num_classes = 3

    class FakeTrainer:
        def __init__(self, **kwargs):
            del kwargs

        def predict(self, model, dataloaders):
            del model, dataloaders
            return []

    monkeypatch.setattr(api, "AudioDirDataModule", FakeDataModule)
    monkeypatch.setattr(
        api,
        "_load_predict_model",
        lambda checkpoint, config: (
            FakeLegacyModel(),
            api.PredictRuntime(
                sr=10_000,
                hop_seconds=1 / 10_000,
                num_time_steps=1024,
                chunk_stride=928,
                class_names=["noise", "pulse", "sine"],
                class_types=["segment", "event", "segment"],
            ),
        ),
    )
    monkeypatch.setattr(api.L, "Trainer", FakeTrainer)
    monkeypatch.setattr(
        api.prediction_results,
        "write_prediction_outputs",
        lambda **kwargs: writer_kwargs.update(kwargs) or ["/tmp/legacy_annotations.csv"],
    )

    written_files = api.predict(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="legacy",
            output_dir="predictions",
            batch_size=4,
            num_time_steps=256,
            chunk_stride=64,
            num_workers=0,
        )
    )

    assert written_files == ["/tmp/legacy_annotations.csv"]
    assert datamodule_kwargs["num_time_steps"] == 1024
    assert datamodule_kwargs["chunk_stride"] == 928
    assert writer_kwargs["context"].frame_rate_hz == pytest.approx(10_000)


def test_evaluate_uses_predict_loader_when_split_is_missing(monkeypatch: pytest.MonkeyPatch, capsys):
    class FakePredictDataset:
        audio_files = ["clip.wav"]
        chunk_borders = [0, 0]

        def __len__(self):
            return 1

    class FakeDataModule:
        def __init__(self, **kwargs):
            del kwargs

        def predict_dataloader(self):
            return SimpleNamespace(dataset=FakePredictDataset())

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.strategy = SimpleNamespace(root_device=torch.device("cpu"))

        def predict(self, model, dataloaders):
            del model, dataloaders
            return []

    monkeypatch.setattr(api, "AudioDirDataModule", FakeDataModule)
    monkeypatch.setattr(api, "_build_inference_trainer", lambda config, **kwargs: FakeTrainer())
    monkeypatch.setattr(
        api,
        "_load_predict_model",
        lambda source, config: (
            object(),
            api.PredictRuntime(
                sr=32_000,
                hop_seconds=1 / 32_000,
                num_time_steps=128,
                chunk_stride=64,
                class_names=["noise", "song"],
                class_types=["segment", "segment"],
            ),
        ),
    )
    monkeypatch.setattr(
        api.prediction_results,
        "evaluate_file_predictions",
        lambda **kwargs: api.PredictionEvaluation(
            summary={"evaluated_file_count": 1, "skipped_file_count": 0},
            class_names=kwargs["context"].class_names,
            dense_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
            dense_report="dense report",
            syllable_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
            syllable_report="syllable report",
            syllable_wer=0.125,
        ),
    )

    summary = api.evaluate(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir="predictions",
            evaluate=True,
        ),
        verbose=True,
    )

    assert summary == {"evaluated_file_count": 1, "skipped_file_count": 0}
    output = capsys.readouterr().out
    assert '"evaluated_file_count": 1' in output
    assert "Syllable WER: 0.1250" in output


def test_evaluate_uses_split_loader_when_requested(monkeypatch: pytest.MonkeyPatch):
    calls = {"loader": None}

    class FakeDataModule:
        def __init__(self, **kwargs):
            del kwargs

        def test_dataloader(self):
            calls["loader"] = "test"
            return SimpleNamespace(dataset=SimpleNamespace(audio_files=["clip.wav"], annotation_files=["ann.csv"]))

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.strategy = SimpleNamespace(root_device=torch.device("cpu"))

    monkeypatch.setattr(api, "AudioDirDataModule", FakeDataModule)
    monkeypatch.setattr(api, "_build_inference_trainer", lambda config, **kwargs: FakeTrainer())
    monkeypatch.setattr(
        api,
        "_load_predict_model",
        lambda source, config: (
            object(),
            api.PredictRuntime(
                sr=32_000,
                hop_seconds=1 / 32_000,
                num_time_steps=128,
                chunk_stride=64,
                class_names=["noise", "song"],
                class_types=["segment", "segment"],
            ),
        ),
    )
    monkeypatch.setattr(
        api.prediction_results,
        "evaluate_supervised_predictions",
        lambda **kwargs: api.PredictionEvaluation(
            summary={"evaluated_file_count": 1, "skipped_file_count": 0},
            class_names=kwargs["context"].class_names,
            dense_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
            dense_report="dense",
            syllable_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
            syllable_report="syllable",
            syllable_wer=0.25,
        ),
    )

    summary = api.evaluate(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir="predictions",
            evaluate=True,
            split="test",
        )
    )

    assert calls["loader"] == "test"
    assert summary == {"evaluated_file_count": 1, "skipped_file_count": 0}


def test_build_inference_datamodule_resolves_hf_data_and_uses_split_fractions(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class FakeDataModule:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(api, "resolve_training_data_dir", lambda data_dir: "/tmp/hf-snapshot")
    monkeypatch.setattr(api, "AudioDirDataModule", FakeDataModule)

    api._build_inference_datamodule(
        data_dir="nccratliri/bengalese-finch-subset-with-csv-label",
        config=Config(
            mode="predict",
            data_dir="nccratliri/bengalese-finch-subset-with-csv-label",
            checkpoint="model.ckpt",
            validation_fraction=0.2,
            test_fraction=0.3,
            ignore_class_names=True,
        ),
        runtime=api.PredictRuntime(
            sr=32_000,
            hop_seconds=1 / 32_000,
            num_time_steps=128,
            chunk_stride=64,
            class_names=["noise", "song"],
            class_types=["segment", "segment"],
        ),
    )

    assert captured["data_dir"] == "/tmp/hf-snapshot"
    assert captured["val_ratio"] == 0.2
    assert captured["test_ratio"] == 0.3
    assert captured["ignore_class_names"] is True
    assert captured["target_samplerate"] == 32_000


def test_train_stop_event_adds_callback_and_requests_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trainer_kwargs = {}
    fit_call = {}
    stop_event = threading.Event()

    monkeypatch.setattr(api, "_infer_samplerate", lambda data_dir, **kwargs: 16_000)

    class FakeDataModule:
        def __init__(self, **kwargs):
            self.num_time_steps = kwargs["num_time_steps"]
            self.chunk_stride = kwargs["chunk_stride"]
            self.annotated_audio_files = [Path("annotated.wav")]
            self.num_classes = 2
            self.class_names = ["noise", "song"]
            self.input_num_channels = 1
            self.has_test = False

        @property
        def has_val(self):
            return False

        def train_dataloader(self):
            return SimpleNamespace(dataset=[0])

    class FakeModel:
        def __init__(self, **kwargs):
            del kwargs

    class FakeCheckpoint:
        def __init__(self, **kwargs):
            del kwargs
            self.best_model_path = "/tmp/model.ckpt"
            self.last_model_path = ""

    class FakeTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)
            self.should_stop = False

        def fit(self, model, train_dataloaders, val_dataloaders):
            del model, train_dataloaders, val_dataloaders
            stop_event.set()
            cancel_callback = trainer_kwargs["callbacks"][-1]
            cancel_callback.on_train_batch_start(self, None, None, 0)
            fit_call["should_stop"] = self.should_stop

    monkeypatch.setattr(api, "AudioDirDataModule", FakeDataModule)
    monkeypatch.setattr(api, "DASModel", FakeModel)
    monkeypatch.setattr(api, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(api, "CSVLogger", lambda **kwargs: ("logger", kwargs))
    monkeypatch.setattr(api.L, "Trainer", FakeTrainer)

    checkpoint_path = api.train(
        Config(
            mode="train",
            data_dir="data",
            output_dir=str(tmp_path / "run"),
            frontend_type="raw",
            num_epochs=1,
        ),
        stop_event=stop_event,
    )

    assert checkpoint_path == "/tmp/model.ckpt"
    assert fit_call["should_stop"] is True
