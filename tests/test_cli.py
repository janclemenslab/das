import inspect

import defopt
import pytest
from xarray_behave.gui.app import main_das

from das import cli, evaluate, predict, train


COMMANDS = {
    "train": train.train,
    "predict": predict.cli_predict,
    "evaluate": evaluate.cli_evaluate,
    "version": cli.version,
    "gui": main_das,
}


def _assert_all_arguments_bind(function, argv):
    selected, bound = defopt.bind(function, argv=argv, show_defaults=False)
    public_parameters = {name for name in inspect.signature(function).parameters if not name.startswith("_")}
    assert selected is function
    assert set(bound.arguments) == public_parameters


@pytest.mark.parametrize("argv", [["--help"], *[[command, "--help"] for command in COMMANDS]])
def test_cli_help(argv):
    with pytest.raises(SystemExit) as exit_info:
        defopt.bind(COMMANDS, argv=argv, show_defaults=False)
    assert exit_info.value.code == 0


def test_all_train_arguments_bind():
    _assert_all_arguments_bind(
        train.train,
        [
            "--data-dir", "data",
            "--x-suffix", "x",
            "--y-suffix", "y",
            "--save-dir", "out",
            "--save-prefix", "prefix",
            "--save-name", "name",
            "--model-name", "tcn",
            "--nb-filters", "4",
            "--nb-kernels", "2",
            "--kernel-size", "8",
            "--nb-conv", "2",
            "--use-separable", "True", "False",
            "--nb-hist", "128",
            "--no-ignore-boundaries",
            "--no-batch-norm",
            "--nb-pre-conv", "1",
            "--pre-nb-conv", "2",
            "--pre-nb-dft", "32",
            "--pre-kernel-size", "3",
            "--pre-nb-filters", "4",
            "--pre-nb-kernels", "2",
            "--no-upsample",
            "--dilations", "1", "2",
            "--nb-lstm-units", "2",
            "--verbose", "0",
            "--batch-size", "2",
            "--nb-epoch", "1",
            "--learning-rate", "0.001",
            "--reduce-lr",
            "--reduce-lr-patience", "2",
            "--fraction-data", "0.5",
            "--first-sample-train", "1",
            "--last-sample-train", "10",
            "--first-sample-val", "2",
            "--last-sample-val", "8",
            "--seed", "1",
            "--batch-level-subsampling",
            "--augmentations", "{}",
            "--tensorboard",
            "--wandb-api-token", "token",
            "--wandb-project", "project",
            "--wandb-entity", "entity",
            "--log-messages",
            "--nb-stacks", "1",
            "--no-with-y-hist",
            "--balance",
            "--no-version-data",
            "--post-opt",
            "--post-opt-nb-workers", "1",
            "--post-opt-fill-gaps-min", "0.001",
            "--post-opt-fill-gaps-max", "0.1",
            "--post-opt-fill-gaps-steps", "2",
            "--post-opt-min-len-min", "0.001",
            "--post-opt-min-len-max", "0.1",
            "--post-opt-min-len-steps", "2",
            "--resnet-compute",
            "--resnet-train",
            "--tmse-weight", "0.1",
        ],
    )


def test_all_predict_arguments_bind():
    _assert_all_arguments_bind(
        predict.cli_predict,
        [
            "audio.wav", "model",
            "--save-filename", "out.csv",
            "--save-format", "csv",
            "--verbose", "0",
            "--batch-size", "2",
            "--event-thres", "0.4",
            "--event-dist", "0.02",
            "--event-dist-min", "0.01",
            "--event-dist-max", "1",
            "--segment-thres", "0.4",
            "--no-segment-use-optimized",
            "--segment-minlen", "0.01",
            "--segment-fillgap", "0.02",
            "--bandpass-low-freq", "100",
            "--bandpass-up-freq", "4000",
            "--no-resample",
        ],
    )


def test_all_evaluate_arguments_bind():
    _assert_all_arguments_bind(evaluate.cli_evaluate, ["model", "0"])


def test_all_gui_arguments_bind():
    _assert_all_arguments_bind(
        main_das,
        [
            "audio.wav",
            "--song-types-string", "pulse,event",
            "--spec-freq-min", "100",
            "--spec-freq-max", "5000",
            "--skip-dialog",
        ],
    )


def test_version_command_runs():
    cli.version()
