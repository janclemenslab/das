"""Real GUI startup and CPU jobs, without replacing the backend or process boundary."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy")
pytest.importorskip("magicgui")

from qtpy.QtCore import Qt

import das.gui_app as gui
from das.config import Config

pytestmark = pytest.mark.integration


def test_gui_import_and_accelerator_exit_cleanly(tmp_path):
    probe = textwrap.dedent("""\
        import das.gui_app
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        actual = torch.arange(8, device=device, dtype=torch.float32).square().cpu()
        torch.testing.assert_close(actual, torch.arange(8, dtype=torch.float32).square())
        """)
    source = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "PYTHONPATH": str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, f"Accelerator process exited with {result.returncode}:\n{result.stdout}{result.stderr}"


@pytest.mark.parametrize("argv", [[], ["gui"], ["gui", "--config", "fly"]])
def test_cli_opens_annotation_window_and_das_menu(argv, tmp_path):
    pytest.importorskip("xarray_behave")
    probe = textwrap.dedent("""\
        import sys
        import traceback
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication
        from das.cli import cli_main
        from das.gui_app import DASConformerWindow
        from xarray_behave.gui.app import MainWindow

        app = QApplication([])
        checked = []

        def check_windows():
            try:
                parents = [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]
                assert len(parents) == 1, "Annotation window did not open"
                parent = parents[0]
                dialogs = [w for w in app.topLevelWidgets() if isinstance(w, DASConformerWindow)]
                assert len(dialogs) == (1 if "--config" in sys.argv else 0)
                if dialogs:
                    assert dialogs[0].active_command_name() == "train"
                    assert dialogs[0].train_widget.build_config().encoder_type == "tcn"
                actions = {action.text(): action for action in parent.das_menu.actions()}
                for name, mode in [("Train", "train"), ("Predict", "predict")]:
                    actions[name].trigger()
                    dialog = parent._das_windows[-1]
                    assert isinstance(dialog, DASConformerWindow)
                    assert dialog.isVisible()
                    assert dialog.active_command_name() == mode
                checked.append(True)
            except Exception:
                traceback.print_exc()
            finally:
                app.closeAllWindows()
                app.quit()

        QTimer.singleShot(0, check_windows)
        cli_main(sys.argv[1:])
        assert checked == [True], "GUI startup or menu checks failed"
        """)
    source = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "PYTHONPATH": str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(
        [sys.executable, "-c", probe, *argv], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, f"GUI process exited with {result.returncode}:\n{result.stdout}{result.stderr}"


def test_train_and_predict_from_buttons_with_real_processes(qtbot, monkeypatch, tmp_path):
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    audio = np.sin(np.arange(256, dtype=np.float32) / 10) * 0.25
    for index in range(5):
        sf.write(data_dir / f"clip{index}.wav", audio, 1_000)
        (data_dir / f"clip{index}_annotations.csv").write_text(
            "name,start_seconds,stop_seconds\nsong,0.032,0.080\nsong,0.160,0.208\n", encoding="utf-8",
        )
    output_dir = tmp_path / "run"
    callbacks = []
    errors = []
    window = gui.DASConformerWindow(
        startup_config=Config(
            mode="train", data_dir=str(data_dir), output_dir=str(output_dir),
            frontend_type="raw", frontend_num_channels=1, encoder_type="tcn", decoder_type="linear",
            encoder_hidden_size=4, encoder_num_layers=1, encoder_dilations=[1, 2], encoder_kernel_size=3,
            encoder_dropout=0.0, num_time_steps=64, batch_size=2, num_workers=0, num_epochs=1,
            accelerator="cpu", num_devices=1, seed=0, cross_entropy_weight=1.0,
        ),
        current_audio_provider=lambda start, stop: (audio, 1_000, start),
        on_predictions=lambda rows, offset: callbacks.append((rows, offset)),
    )
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "_show_error", lambda title, message: errors.append((title, message)))
    window.show()

    def run_job():
        try:
            qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
            qtbot.waitUntil(lambda: window._thread is None, timeout=60_000)
        finally:
            if window._cancel_event is not None:
                window._cancel_event.set()
                qtbot.waitUntil(lambda: window._thread is None, timeout=10_000)
        assert not errors, "\n\n".join(message for _title, message in errors)
        assert window.run_button.isEnabled()
        assert window._worker is None

    run_job()

    checkpoints = list((output_dir / "checkpoints").glob("*.ckpt"))
    assert checkpoints
    assert "Training finished. Checkpoint:" in window.log_output.toPlainText()

    window.select_tab("predict")
    prediction = window.predict_widget
    prediction.paths_form.checkpoint = str(checkpoints[0])
    prediction.trainer_form.accelerator = "cpu"
    prediction.runtime_form.num_devices = "1"
    prediction.current_audio_widget.start_seconds_field.setText("2.0")
    run_job()

    assert len(callbacks) == 1
    assert isinstance(callbacks[0][0], pd.DataFrame)
    assert {"name", "start_seconds", "stop_seconds"}.issubset(callbacks[0][0].columns)
    assert callbacks[0][1] == 2.0

    prediction.source_combo.setCurrentText("File or folder")
    prediction.paths_form.data_dir = str(data_dir / "clip0.wav")
    prediction.prediction_form.output_suffix = "_gui_predictions.csv"
    run_job()

    output = data_dir / "clip0_gui_predictions.csv"
    assert output.exists()
    assert {"name", "start_seconds", "stop_seconds"}.issubset(pd.read_csv(output).columns)
    assert len(callbacks) == 1
    assert str(output) in window.log_output.toPlainText()


def test_loaded_audio_connects_prediction_dialog_to_annotations(qtbot, monkeypatch, tmp_path):
    pytest.importorskip("xarray_behave")
    from xarray_behave.gui.app import MainWindow

    audio_path = tmp_path / "clip.wav"
    audio = np.sin(np.arange(8_000, dtype=np.float32) / 10) * 0.25
    sf.write(audio_path, audio, 8_000)
    parent = MainWindow.from_file(filename=str(audio_path), skip_dialog=True, is_das=True)
    qtbot.addWidget(parent)
    errors = []
    monkeypatch.setattr(gui.DASConformerWindow, "_show_error", lambda self, title, message: errors.append((title, message)))
    rows = pd.DataFrame([{"name": "song", "start_seconds": 0.01, "stop_seconds": 0.02}])

    def run_process(worker):
        assert worker.audio.shape[0] == 2_000
        assert worker.samplerate == 8_000
        return {"ok": True, "command": "predict", "result": rows, "time_offset_seconds": worker.time_offset_seconds}

    monkeypatch.setattr(gui.JobWorker, "_run_train_in_process", run_process)
    actions = {action.text(): action for action in parent.das_menu.actions()}
    actions["Predict"].trigger()
    dialog = parent._das_windows[-1]
    qtbot.addWidget(dialog)
    assert dialog.predict_widget.uses_current_audio()
    assert dialog.current_duration_provider() == pytest.approx(1.0)
    dialog.predict_widget.paths_form.checkpoint = str(tmp_path / "model.ckpt")
    dialog.predict_widget.current_audio_widget.start_seconds_field.setText("0.25")
    dialog.predict_widget.current_audio_widget.stop_seconds_field.setText("0.5")

    try:
        qtbot.mouseClick(dialog.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: dialog._thread is None, timeout=10_000)
    finally:
        if dialog._cancel_event is not None:
            dialog._cancel_event.set()
            qtbot.waitUntil(lambda: dialog._thread is None, timeout=10_000)

    assert not errors, errors
    annotations = parent.event_times.to_df(preserve_empty=False, with_channels=False)
    predicted = annotations.loc[annotations["name"] == "song_proposals"]
    assert len(predicted) == 1
    assert predicted.iloc[0]["start_seconds"] == pytest.approx(0.26)
    assert predicted.iloc[0]["stop_seconds"] == pytest.approx(0.27)


def test_video_with_tracking_loads_and_seeks(qtbot, tmp_path):
    import av
    import xarray as xr
    from xarray_behave.gui.app import PSV
    from xarray_behave.gui.gui_config import GuiConfigManager
    from xarray_behave.gui.modern_video import PyAVVideoReader

    video_path = tmp_path / "clip.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for index in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    sampletime = np.arange(8_000) / 8_000
    ds = xr.Dataset(
        {
            "song_raw": (("sampletime", "channels"), np.sin(sampletime[:, None] * 100) * 0.25),
            "body_positions": (("time", "flies", "bodyparts", "coords"), np.full((10, 1, 2, 2), 16.0)),
        },
        coords={
            "sampletime": sampletime, "time": np.arange(10) / 10, "nearest_frame": ("time", np.arange(10)),
            "channels": [0], "flies": [0], "bodyparts": ["head", "thorax"], "coords": ["y", "x"],
        },
        attrs={"target_sampling_rate_Hz": 10},
    )
    ds.song_raw.attrs["sampling_rate_Hz"] = 8_000
    ds.body_positions.attrs.update({"spatial_units": "pixels", "pixel_size_mm": 1.0})
    reader = PyAVVideoReader(video_path)
    viewer = PSV(ds, vr=reader, box_size=8, is_das=True, config_manager=GuiConfigManager(home=tmp_path / "home"))
    qtbot.addWidget(viewer)

    viewer.t0 = 2_400
    viewer.update_frame()

    assert viewer.movie_view.isVisible()
    assert viewer.index_other == 3
    assert viewer.framenumber == 3
    assert reader[viewer.framenumber].shape == (32, 32, 3)
