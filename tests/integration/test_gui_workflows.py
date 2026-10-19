"""Exercise button/worker/result wiring with a real Qt event loop."""

import os
import threading

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy")
pytest.importorskip("magicgui")

from qtpy.QtCore import QThread, Qt

import das.gui_app as gui

pytestmark = pytest.mark.integration


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    widget = gui.DASConformerWindow(
        current_audio_provider=lambda start, stop: (np.arange(64, dtype=np.float32), 1_000, start),
        current_duration_provider=lambda: 3.0,
    )
    qtbot.addWidget(widget)
    errors = []
    monkeypatch.setattr(widget, "_show_error", lambda title, message: errors.append((title, message)))
    widget.train_widget.paths_form.data_dir = str(tmp_path)
    widget.train_widget.paths_form.output_dir = str(tmp_path / "run")
    widget.predict_widget.paths_form.checkpoint = str(tmp_path / "model.ckpt")
    widget.show()
    return widget, errors


@pytest.mark.parametrize("command", ["train", "predict"])
@pytest.mark.parametrize("outcome", ["success", "failure", "stop", "cancel", "callback_failure"])
def test_run_button_worker_lifecycle(window, qtbot, monkeypatch, command, outcome):
    widget, errors = window
    predictions = pd.DataFrame([{"name": "song", "start_seconds": 0.01, "stop_seconds": 0.02}])
    callbacks = []
    calls = []
    started = threading.Event()
    release = threading.Event()

    def on_predictions(rows, offset):
        callbacks.append((rows, offset, QThread.currentThread()))
        if outcome == "callback_failure":
            raise RuntimeError("annotation view failed")

    def run_process(worker):
        calls.append(worker)
        worker.log.emit("backend progress\n")
        started.set()
        assert release.wait(10), "Test did not release the worker"
        if outcome == "failure" and len(calls) == 1:
            raise RuntimeError("backend failed")
        payload = {
            "ok": True,
            "command": worker.command_name,
            "result": "model.ckpt" if command == "train" else predictions,
            "time_offset_seconds": worker.time_offset_seconds,
            "cancelled": worker.stop_event.is_set(),
        }
        if command == "train" and not worker.stop_event.is_set():
            payload["prediction_after_train"] = {"result": predictions, "time_offset_seconds": 2.0}
        return payload

    monkeypatch.setattr(gui.JobWorker, "_run_train_in_process", run_process)
    widget.on_predictions = on_predictions
    widget.select_tab(command)
    if command == "train":
        prediction = widget.train_widget.prediction_form
        prediction.predict_after_train_combo.setCurrentText("Current file")
        audio_range = prediction.time_range_widget
    else:
        audio_range = widget.predict_widget.current_audio_widget
    audio_range.start_seconds_field.setText("2.0")
    audio_range.stop_seconds_field.setText("3.0")

    controls = [widget.command_tabs, widget.preset_dropdown, widget.load_button, widget.save_button, widget.run_button]
    try:
        qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(started.is_set)
        qtbot.waitUntil(lambda: "backend progress" in widget.log_output.toPlainText())
        assert all(not control.isEnabled() for control in controls)
        assert widget.stop_early_button.isEnabled()
        assert widget.cancel_button.isEnabled()
        worker = calls[0]
        assert worker.config.mode == command
        assert worker.run_train_in_process
        assert worker.thread() is not widget.thread()
        request = worker.predict_after_train if command == "train" else vars(worker)
        np.testing.assert_array_equal(request["audio"], np.arange(64, dtype=np.float32))
        assert request["samplerate"] == 1_000
        assert request["time_offset_seconds"] == 2.0

        if outcome == "stop":
            qtbot.mouseClick(widget.stop_early_button, Qt.MouseButton.LeftButton)
            assert worker.stop_event.is_set()
            assert not worker.cancel_event.is_set()
            assert not widget.stop_early_button.isEnabled()
            assert widget.cancel_button.isEnabled()
        elif outcome == "cancel":
            qtbot.mouseClick(widget.cancel_button, Qt.MouseButton.LeftButton)
            assert worker.cancel_event.is_set()
            assert not widget.stop_early_button.isEnabled()
            assert not widget.cancel_button.isEnabled()
    finally:
        release.set()
        qtbot.waitUntil(lambda: widget._thread is None, timeout=10_000)

    assert all(control.isEnabled() for control in controls)
    assert not widget.stop_early_button.isEnabled()
    assert not widget.cancel_button.isEnabled()
    assert widget._worker is widget._stop_event is widget._cancel_event is widget._running_command is None
    log = widget.log_output.toPlainText()
    if outcome == "failure":
        assert len(errors) == 1
        assert errors[0][0] == "Command Failed"
        assert "RuntimeError: backend failed" in errors[0][1]
        assert not callbacks
        errors.clear()
        qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: widget._thread is None, timeout=10_000)
        assert len(calls) == 2
        assert len(callbacks) == 1
        assert not errors
    elif outcome == "cancel":
        assert "cancelled." in log
        assert not callbacks
        assert not errors
    elif outcome == "stop" and command == "train":
        assert "Training stopped early. Checkpoint: model.ckpt" in log
        assert not callbacks
        assert not errors
    else:
        assert len(callbacks) == 1
        pd.testing.assert_frame_equal(callbacks[0][0], predictions)
        assert callbacks[0][1] == 2.0
        assert callbacks[0][2] is widget.thread()
        if outcome == "callback_failure":
            assert len(errors) == 1
            assert errors[0][0] == "Prediction Callback Failed"
            assert "annotation view failed" in errors[0][1]
            assert "Returned 1 predictions." not in log
        else:
            assert not errors
            assert "Returned 1 predictions." in log
            if outcome == "stop":
                assert "Prediction stopped early." in log


@pytest.mark.parametrize("command", ["train", "predict"])
def test_invalid_form_does_not_start_worker(window, qtbot, command):
    widget, errors = window
    widget.select_tab(command)
    form = widget.train_widget.model_section.hyperparameters_form if command == "train" else widget.predict_widget.runtime_form
    field = "learning_rate" if command == "train" else "num_devices"
    setattr(form, field, "not a number")

    qtbot.mouseClick(widget.run_button, Qt.MouseButton.LeftButton)

    assert len(errors) == 1
    assert errors[0][0] == "Invalid Configuration"
    assert "not a number" in errors[0][1]
    assert widget._thread is widget._worker is None
    assert widget.run_button.isEnabled()
    assert widget.command_tabs.isEnabled()


@pytest.mark.parametrize("contents", ["", "- list\n- instead of mapping\n", "mode: [broken", "mode: predict\nbatch_size: bad\n"])
def test_load_button_reports_bad_yaml_and_recovers(window, qtbot, monkeypatch, tmp_path, contents):
    widget, errors = window
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), "yaml"))

    qtbot.mouseClick(widget.load_button, Qt.MouseButton.LeftButton)

    assert len(errors) == 1
    assert errors[0][0] == "Load Config Failed"
    assert widget.run_button.isEnabled()

    path.write_text("mode: predict\ncheckpoint: model.ckpt\nbatch_size: 3\n", encoding="utf-8")
    errors.clear()
    qtbot.mouseClick(widget.load_button, Qt.MouseButton.LeftButton)

    assert not errors
    assert widget.active_command_name() == "predict"
    assert widget.run_button.text() == "Start Prediction"
    assert widget.predict_widget.build_config().batch_size == 3
