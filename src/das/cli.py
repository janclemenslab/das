from __future__ import annotations

import argparse
from dataclasses import MISSING, fields
import sys

from . import __version__
from .api import convert_legacy_checkpoint, predict, train
from .config import Config, builtin_config_help, format_config_yaml, save_config


_COMMAND_CONFIG_FIELDS: dict[str, tuple[str, ...]] = {
    "train": (
        "data_dir",
        "audio_dataset",
        "data_samplerate_hz",
        "output_dir",
        "checkpoint_prefix",
        "initial_model",
        "batch_size",
        "num_time_steps",
        "chunk_stride",
        "num_workers",
        "validation_fraction",
        "test_fraction",
        "split_within_files",
        "min_annotation_duration_ms",
        "ignore_class_names",
        "include_labels",
        "frontend_type",
        "frontend_num_channels",
        "frontend_kernel_size",
        "frontend_hop_seconds",
        "frontend_pad_mode",
        "frontend_fmin",
        "frontend_fmax",
        "frontend_trainable",
        "encoder_type",
        "encoder_num_heads",
        "encoder_hidden_size",
        "encoder_num_layers",
        "encoder_kernel_size",
        "encoder_dilations",
        "encoder_dropout",
        "encoder_use_skip_connections",
        "encoder_use_separable",
        "encoder_padding",
        "decoder_type",
        "decoder_hidden_size",
        "decoder_kernel_size",
        "decoder_num_heads",
        "decoder_num_layers",
        "decoder_dropout",
        "cross_entropy_weight",
        "positive_class_weight",
        "boundary_weight",
        "boundary_width_ms",
        "learning_rate",
        "early_stopping",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "reduce_lr",
        "reduce_lr_patience",
        "reduce_lr_factor",
        "reduce_lr_min",
        "linear_lr_schedule",
        "weight_decay",
        "warmup_steps",
        "freeze_encoder",
        "max_length",
        "total_spec_columns",
        "seed",
        "accelerator",
        "num_devices",
        "num_epochs",
        "max_num_steps_per_epoch",
        "fill_gap_ms",
        "min_syllable_ms",
        "segment_threshold_low",
        "segment_threshold_high",
        "event_threshold",
        "event_dist_min_ms",
        "event_dist_max_ms",
        "syllable_postprocessor",
        "syllable_tolerance_ms",
        "min_frequency",
        "frequency_scale",
        "spec_time_step",
    ),
    "predict": (
        "data_dir",
        "audio_dataset",
        "data_samplerate_hz",
        "output_dir",
        "checkpoint",
        "evaluate",
        "split",
        "batch_size",
        "num_workers",
        "seed",
        "accelerator",
        "num_devices",
        "output_suffix",
        "existing_annotations",
        "generation_max_length",
        "min_frequency",
        "frequency_scale",
        "spec_time_step",
        "num_trials",
        "num_beams",
        "top_k",
        "top_p",
        "length_penalty",
        "fill_gap_ms",
        "min_syllable_ms",
        "segment_threshold_low",
        "segment_threshold_high",
        "event_threshold",
        "event_dist_min_ms",
        "event_dist_max_ms",
        "syllable_postprocessor",
        "syllable_tolerance_ms",
    ),
    "convert-legacy": (
        "checkpoint",
        "converted_checkpoint",
    ),
    "gui": (),
}


_ASCII_ICON = r"""
    |   |   |
  | | | | | | |
| | | | | | | | |
  | | | | | | |
    |   |   |
""".strip("\n")


class _HelpOnlyArgumentParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        formatter = self._get_formatter()
        formatter.add_text(self.description)

        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()

        formatter.add_text(self.epilog)
        return formatter.format_help()


def _command_descriptions() -> dict[str, str]:
    return {
        "train": "Train a model on annotated audio.",
        "predict": "Run inference and optionally evaluate against annotation CSV files.",
        "convert-legacy": "Convert a legacy DAS H5/YAML model to a native Torch checkpoint.",
        "gui": "Launch the GUI.",
        "version": "Show the installed DAS version.",
    }


def _field_default(item) -> object:
    if item.default is not MISSING:
        return item.default
    if item.default_factory is not MISSING:
        return item.default_factory()
    raise ValueError(f"Config field '{item.name}' is missing a default.")


def _help_with_default(help_text: str | None, default: object) -> str:
    suffix = f"(default: {repr(default).replace('%', '%%')})"
    if not help_text:
        return suffix
    return f"{help_text} {suffix}"


def _config_help() -> str:
    return f"Load a YAML config path or built-in config name ({builtin_config_help()}). Can be provided multiple times."


def build_parser() -> argparse.ArgumentParser:
    parser = _HelpOnlyArgumentParser(
        prog="das",
        description="CLI for DAS.",
        allow_abbrev=False,
    )
    subcommands = parser.add_subparsers(dest="command", parser_class=_HelpOnlyArgumentParser)
    subcommands.required = True

    for command, description in _command_descriptions().items():
        subparser = subcommands.add_parser(command, description=description, help=description, allow_abbrev=False)
        if command == "version":
            continue
        subparser.add_argument(
            "--config",
            action="append",
            default=[],
            help=_help_with_default(_config_help(), []),
        )
        subparser.add_argument(
            "--print-config",
            action="store_true",
            help=_help_with_default("Print the effective flat YAML config and exit.", False),
        )
        subparser.add_argument(
            "--save-config",
            help=_help_with_default("Write the effective flat YAML config to a file.", None),
        )
        _add_config_arguments(subparser, _COMMAND_CONFIG_FIELDS[command], command=command)

    return parser


def _add_config_arguments(parser: argparse.ArgumentParser, field_names: tuple[str, ...], *, command: str) -> None:
    allowed_fields = set(field_names)
    for item in fields(Config):
        if item.name == "mode" or item.name not in allowed_fields:
            continue
        flags = list(item.metadata.get("flags") or [f"--{item.name.replace('_', '-')}"])
        help_text = item.metadata.get("help")
        if command == "predict" and item.name == "data_dir":
            help_text = "Audio file or directory for prediction."
        kwargs: dict[str, object] = {
            "dest": item.name,
            "default": argparse.SUPPRESS,
            "help": _help_with_default(help_text, _field_default(item)),
        }
        choices = item.metadata.get("choices")
        if item.metadata.get("bool_flag", False):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = item.metadata.get("parser")
        if choices is not None:
            kwargs["choices"] = choices
        parser.add_argument(*flags, **kwargs)


def _collect_cli_overrides(namespace: argparse.Namespace) -> dict[str, object]:
    payload = {}
    for item in fields(Config):
        if item.name == "mode":
            continue
        if hasattr(namespace, item.name):
            payload[item.name] = getattr(namespace, item.name)
    return payload


def _print_startup_banner() -> None:
    print(
        f"{_ASCII_ICON}\n\n"
        f"DAS {__version__}\n"
        "Starting the GUI.\n"
        "Other commands: das train, das predict, das convert-legacy, das version. "
        "Use das <command> --help for options.\n"
    )


def _print_version() -> None:
    print(f"DAS {__version__}")


def parse_command_config(argv: list[str] | None = None) -> Config:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        parser.print_help()
        raise SystemExit(0)
    parsed = parser.parse_args(raw_argv)
    if parsed.command == "version":
        _print_version()
        raise SystemExit(0)

    base = Config()
    config_mode = "train" if parsed.command == "gui" else parsed.command
    config = Config.from_config_sources(getattr(parsed, "config", []), base=base, mode=config_mode)
    config = Config.from_mapping(_collect_cli_overrides(parsed), base=config)
    if parsed.command != "gui":
        config = config.copy(mode=parsed.command)

    if getattr(parsed, "print_config", False):
        sys.stdout.write(format_config_yaml(config))
        raise SystemExit(0)

    if parsed.command != "gui":
        try:
            config.validate()
        except ValueError as exc:
            parser.error(str(exc))

    destination = getattr(parsed, "save_config", None)
    if destination:
        save_config(config, destination)
        print(f"Saved config to {destination}")

    return config


def _launch_gui(argv: list[str] | None = None, *, startup_config: Config | None = None):
    try:
        from xarray_behave.gui.app import main_das
    except ImportError as exc:
        raise RuntimeError("The GUI requires xarray-behave. Install the package with `.[gui]`.") from exc
    open_config = bool(argv and any(arg == "--config" or arg.startswith("--config=") for arg in argv))
    return main_das(das_startup_config=startup_config if open_config else None)


def cli_main(argv: list[str] | None = None):
    called_from_console = argv is None
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        _print_startup_banner()
        return _launch_gui(["das"], startup_config=Config(mode="train"))
    if raw_argv[0] == "version":
        build_parser().parse_args(raw_argv)
        _print_version()
        return 0
    if raw_argv[0] == "--config" or raw_argv[0].startswith("--config="):
        raw_argv.insert(0, "gui")
    config = parse_command_config(raw_argv)

    if raw_argv and raw_argv[0] == "gui":
        return _launch_gui(["das", *raw_argv], startup_config=config)
    if config.mode == "train":
        result = train(config, verbose=True)
        return 0 if called_from_console else result
    if config.mode == "predict":
        result = predict(config, verbose=True)
        return 0 if called_from_console else result
    if config.mode == "convert-legacy":
        output = convert_legacy_checkpoint(config.checkpoint, config.converted_checkpoint)
        print(output)
        return output
    raise ValueError(f"Unsupported config mode '{config.mode}'.")


if __name__ == "__main__":
    cli_main()
