from pathlib import Path
import runpy
import sys

import pytest

import das.cli as cli


@pytest.mark.parametrize("entrypoint", ["das_app_entry.py", "das_pyinstaller_entry.py"])
@pytest.mark.parametrize("args", [[], ["gui", "recording.wav", "--skip-dialog"]])
def test_packaged_gui_uses_cli_parser(monkeypatch, entrypoint, args):
    calls = []
    monkeypatch.setattr(cli, "_launch_gui", lambda argv, **kwargs: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["DAS", *args])
    entry = Path(__file__).resolve().parents[1] / "scripts" / entrypoint

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(entry), run_name="__main__")

    assert exit_info.value.code == 0
    assert calls == [["das", *args]]


@pytest.mark.parametrize("entrypoint", ["das_app_entry.py", "das_pyinstaller_entry.py"])
@pytest.mark.parametrize(
    "args, expected_output",
    [
        (["--help"], "CLI for DAS."),
        (["version"], f"DAS {cli.__version__}"),
        (["train", "--config", "fly", "--print-config"], "mode: train"),
        (["predict", "--print-config"], "mode: predict"),
    ],
)
def test_packaged_cli_commands(monkeypatch, capsys, entrypoint, args, expected_output):
    def unexpected_gui(*args, **kwargs):
        pytest.fail("CLI command launched the GUI")

    monkeypatch.setattr(cli, "_launch_gui", unexpected_gui)
    monkeypatch.setattr(sys, "argv", ["DAS", *args])
    entry = Path(__file__).resolve().parents[1] / "scripts" / entrypoint

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(entry), run_name="__main__")

    assert exit_info.value.code == 0
    assert expected_output in capsys.readouterr().out
