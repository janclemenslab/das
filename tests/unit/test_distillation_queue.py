"""A queue handoff must wait for its existing trainer or verified remote artifacts."""
import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


def test_training_ownership(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/whisperseg_large_distillation.py"
    spec = importlib.util.spec_from_file_location("distillation_queue", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "student"
    output.mkdir()
    target = output / "training_result.json"
    owner = output / "training-owner.json"
    runner.wait_for_owned_training(tmp_path, output, target)
    owner.write_text(json.dumps({"host": "pan.uol"}))
    target.write_text("{}")
    waits = []

    def finish_transfer(seconds):
        waits.append(seconds)
        (output / "transfer-verified.json").write_text("{}")
        target.write_text("{}")

    monkeypatch.setattr(runner.time, "sleep", finish_transfer)
    runner.wait_for_owned_training(tmp_path, output, target)
    assert waits == [30]  # A result JSON alone cannot release the remote stage.
    owner.write_text(json.dumps({"local_pid": 123}))
    target.unlink()
    commands = iter([f"python train_evaluate_hybrid.py --output-dir {output}", ""])
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=next(commands)))
    runner.wait_for_owned_training(tmp_path, output, target)
    assert waits == [30, 30]
    target.unlink()
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    with pytest.raises(RuntimeError, match="did not finish"):
        runner.wait_for_owned_training(tmp_path, output, target)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="unrelated process"))
    with pytest.raises(RuntimeError, match="no longer matches"):
        runner.wait_for_owned_training(tmp_path, output, target)


def test_middle_waits_for_successful_largest(tmp_path, monkeypatch):
    import subprocess
    import time

    source = Path(__file__).resolve().parents[2] / "scripts/whisperseg_train_middle.py"
    script = tmp_path / "run-middle.py"
    script.write_text(source.read_text())
    output = tmp_path / "large-distilled-1.92m"
    output.mkdir()
    (output / "dependency.json").write_text('{"pid": 123}')
    (output / "train-command.json").write_text('["python", "trainer.py"]')
    largest = tmp_path / "large-distilled-4.01m"
    largest.mkdir()
    (largest / "training_result.json").write_text('{}')
    commands = iter([f"python {tmp_path / 'run-largest.py'}", ""])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=next(commands)))
    monkeypatch.setattr(time, "sleep", lambda seconds: (tmp_path / "status.json").write_text('{"state": "complete", "exit_code": 0}'))
    launched = []

    def launch(command, **kwargs):
        assert json.loads((tmp_path / "status.json").read_text())["state"] == "complete"
        launched.append(command)
        (output / "training_result.json").write_text('{}')
        return SimpleNamespace(pid=456, wait=lambda: 0)

    monkeypatch.setattr(subprocess, "Popen", launch)
    runpy.run_path(str(script))
    assert launched == [["python", "trainer.py"]]
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    with pytest.raises(RuntimeError, match="already attempted"):
        runpy.run_path(str(script))
    (tmp_path / "status.json").write_text('{"state": "failed", "exit_code": 1}')
    with pytest.raises(RuntimeError, match="did not finish"):
        runpy.run_path(str(script))
    assert len(launched) == 1
    assert json.loads((output / "status.json").read_text())["state"] == "failed"
