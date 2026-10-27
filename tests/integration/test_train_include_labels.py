import os

import numpy as np
import pytest
import xarray as xr

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xarray_behave import annot
from xarray_behave.gui.app import MainWindow


@pytest.mark.integration
def test_viewer_train_dialog_includes_unsaved_annotation_labels(qtbot):
    viewer = MainWindow(is_das=True)
    qtbot.addWidget(viewer)
    viewer.ds = xr.Dataset(
        {"song_raw": (("sampletime", "channels"), np.zeros((1_000, 1)))},
        coords={"sampletime": np.arange(1_000) / 1_000},
    )
    viewer.event_times = annot.Events({"pulse": [[0.1, 0.2]], "sine": [[0.3, 0.4]]})

    dialog = viewer._open_das_window("train", use_current_audio=True)
    qtbot.addWidget(dialog)
    assert dialog.train_widget.paths_form.uses_current_audio()
    widget = dialog.train_widget.include_labels_widget

    assert widget.available_labels == ["pulse", "sine"]
    assert widget.combo.lineEdit().text() == "All labels"
    widget.combo._toggle_activated_index(1)
    assert dialog.train_widget.build_config().include_labels == ["pulse"]


@pytest.mark.integration
def test_current_audio_gui_training_creates_checkpoint(qtbot, tmp_path, monkeypatch):
    import pandas as pd
    import torch

    from das.config import Config
    from das.gui_app import DASConformerWindow

    audio = np.sin(np.arange(2_000, dtype=np.float32) / 10)
    annotations = pd.DataFrame({
        "name": ["annotation_start", "pulse", "annotation_end"],
        "start_seconds": [0.1, 0.2, 1.9],
        "stop_seconds": [0.1, 0.3, 1.9],
    })
    requests, payloads, errors = [], [], []
    window = DASConformerWindow(
        startup_config=Config(
            mode="train", output_dir=str(tmp_path), frontend_type="raw",
            frontend_num_channels=1, encoder_type="tcn", decoder_type="linear",
            encoder_hidden_size=4, encoder_num_layers=1, encoder_dilations=[1, 2],
            encoder_kernel_size=3, num_time_steps=64, batch_size=2,
            accelerator="cpu", num_devices=1, num_epochs=1, max_num_steps_per_epoch=1,
            cross_entropy_weight=1.0,
        ),
        current_audio_provider=lambda *args: requests.append(args) or (audio, 1_000, 0.0),
        current_annotations_provider=lambda: annotations,
    )
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "_show_error", lambda *args: errors.append(args))
    original_finished = window._handle_job_finished

    def finished(payload):
        payloads.append(payload)
        original_finished(payload)

    monkeypatch.setattr(window, "_handle_job_finished", finished)
    window.run_active_command()
    qtbot.waitUntil(lambda: window._thread is None, timeout=60_000)

    assert not errors
    assert requests == [(0.0, None)]
    assert len(payloads) == 1
    assert payloads[0]["ok"], payloads[0].get("error")
    checkpoint = torch.load(payloads[0]["result"], map_location="cpu", weights_only=False)
    assert checkpoint["das"]["predict"]["num_time_steps"] == 64
