from pathlib import Path
import runpy
import sys

import pytest


@pytest.mark.parametrize("platform, windowed", [("darwin", True), ("win32", False), ("linux", False)])
def test_app_build_keeps_cli_output_outside_macos(monkeypatch, platform, windowed):
    build = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/build_executable.py"))
    monkeypatch.setattr(sys, "platform", platform)

    args = build["build_pyinstaller_args"]("DAS", False, app=True)

    assert ("--windowed" in args) == windowed
    assert ("--console" in args) == (not windowed)
