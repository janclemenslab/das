"""Opt-in real Whisper Large LoRA fitting and prediction through the GUI."""

import os
from pathlib import Path

import pytest
import soundfile as sf
import torch
from qtpy.QtCore import Qt

from das.gui_app import DASConformerWindow


@pytest.mark.slow
def test_whisper_large_gui_fit_and_predict(qtbot, tmp_path):
    source = os.environ.get("DAS_WHISPER_GUI_TRAIN_DATA")
    if not source:
        pytest.skip("Set DAS_WHISPER_GUI_TRAIN_DATA to a folder containing annotated speech.wav.")
    audio, samplerate = sf.read(Path(source) / "speech.wav", dtype="float32")
    results, errors, predictions = [], [], []
    window = DASConformerWindow(
        current_audio_provider=lambda *_: (audio, samplerate, 0.0),
        current_duration_provider=lambda: len(audio) / samplerate,
        on_predictions=lambda frame, offset: predictions.append(frame),
    )
    qtbot.addWidget(window)
    window._show_error = lambda title, message: errors.append(message)
    original_finished = window._handle_job_finished

    def finished(payload):
        results.append(payload)
        original_finished(payload)

    window._handle_job_finished = finished
    widget = window.train_widget
    widget.model_class.setCurrentText("whisperseg-aer")
    widget.model_section.initial_model_field.value = "whisper:large"
    widget.paths_form.data_dir = source
    widget.paths_form.output_dir = str(tmp_path / "run")
    widget.data_form.validation_fraction = 0.0
    widget.data_form.test_fraction = 0.0
    widget.trainer_form.num_epochs = 1
    widget.trainer_form.max_num_steps_per_epoch = "1"
    window.show()
    qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: bool(results or errors), timeout=900_000)
    qtbot.waitUntil(lambda: window._thread is None, timeout=30_000)
    assert not errors, "\n".join(errors)
    checkpoint = results[0]["result"]
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    assert saved["das"]["backend"] == "whisper_asr"
    assert saved["whisperseg"]["model_config"]["das_annotation_token_ids"]
    print("Trained checkpoint:", checkpoint)
    window.select_tab("predict")
    window.predict_widget.paths_form.checkpoint_field.value = checkpoint
    window.predict_widget.whisperseg_decoder_form.num_beams = 1
    assert window.predict_widget.model_class_label.text() == "Whisper ASR · word timestamps"
    qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: bool(predictions or errors), timeout=900_000)
    qtbot.waitUntil(lambda: window._thread is None, timeout=30_000)
    assert not errors, "\n".join(errors)
    assert not predictions[0].empty
    print(predictions[0].to_string(index=False))
