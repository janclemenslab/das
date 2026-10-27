import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from das import gui_app as gui
from das.config import Config


@pytest.mark.parametrize("available", [False, True])
def test_train_current_audio_source(qtbot, tmp_path, available):
    widget = gui.TrainCommandWidget(current_audio_available=available)
    qtbot.addWidget(widget)
    paths = widget.paths_form
    assert paths.uses_current_audio() is available
    assert (paths.data_source_combo.findText("Current audio") >= 0) is available
    if available:
        assert paths.data_dir_field.isHidden()
        assert paths.hf_dataset_combo.isHidden()
        widget.load_state(widget.state())
        assert paths.uses_current_audio()
        assert widget.build_config().data_dir == ""
        assert widget.build_config().split_within_files

    widget.model_section.chunking_form.split_within_files = False
    paths.data_dir = str(tmp_path)
    assert not paths.uses_current_audio()
    assert widget.build_config().data_dir == str(tmp_path)
    assert not widget.build_config().split_within_files


@pytest.mark.parametrize("in_process", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_train_worker_stages_current_audio_and_cleans_up(qtbot, monkeypatch, tmp_path, in_process, fail):
    audio = np.linspace(-2, 2, 100, dtype=np.float32)
    annotations = pd.DataFrame({
        "name": ["annotation_start", "pulse", "annotation_end"],
        "start_seconds": [0.01, 0.02, 0.09],
        "stop_seconds": [0.01, 0.04, 0.09],
    })
    staged = []

    def check_dataset(config, **kwargs):
        data_dir = Path(config.data_dir)
        staged.append(data_dir)
        samples, samplerate = sf.read(data_dir / "current.wav", dtype="float32")
        np.testing.assert_array_equal(samples, audio)
        assert samplerate == 1_000
        pd.testing.assert_frame_equal(pd.read_csv(data_dir / "current_annotations.csv"), annotations)
        assert config.output_dir == str(tmp_path)
        assert config.split_within_files
        if fail:
            raise RuntimeError("training failed")
        return "model.ckpt"

    monkeypatch.setattr(gui, "train", check_dataset)
    monkeypatch.setattr(gui.JobWorker, "_run_train_in_process", lambda self: {
        "ok": True, "command": "train", "result": check_dataset(self.config),
    })
    worker = gui.JobWorker(
        "train", Config(mode="train", output_dir=str(tmp_path)),
        audio=audio, samplerate=1_000, annotations=annotations, run_train_in_process=in_process,
    )
    payloads = []
    worker.finished.connect(payloads.append)
    worker.run()

    assert payloads[0]["ok"] is not fail
    assert len(staged) == 1
    assert not staged[0].exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("annotations_provider", [None, pd.DataFrame])
def test_train_current_audio_requires_annotations(qtbot, monkeypatch, annotations_provider):
    window = gui.DASConformerWindow(
        current_audio_provider=lambda *_: (np.zeros(100), 1_000, 0.0),
        current_annotations_provider=annotations_provider,
    )
    qtbot.addWidget(window)
    errors = []
    monkeypatch.setattr(window, "_show_error", lambda *args: errors.append(args))

    window.run_active_command()

    assert window._thread is None
    assert len(errors) == 1
    assert "annotations provider" in errors[0][1] or "Annotate the current audio" in errors[0][1]
