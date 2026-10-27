"""Opt-in original Whisper prediction through the GUI's real background process."""

import os
from difflib import SequenceMatcher
from pathlib import Path
import re

import numpy as np
import pytest
import soundfile as sf
from qtpy.QtCore import Qt

from das.gui_app import DASConformerWindow


@pytest.mark.slow
def test_original_whisper_gui_prediction(qtbot):
    filename = os.environ.get("DAS_WHISPER_GUI_AUDIO")
    if not filename:
        pytest.skip("Set DAS_WHISPER_GUI_AUDIO to a speech WAV with a matching reference .txt file.")
    audio, samplerate = sf.read(filename, dtype="float32")
    reference = re.findall(r"\w+", Path(filename).with_suffix(".txt").read_text().lower())
    predictions, errors = [], []
    window = DASConformerWindow(
        initial_tab="predict",
        current_audio_provider=lambda *_: (audio, samplerate, 0.0),
        current_duration_provider=lambda: len(audio) / samplerate,
        on_predictions=lambda frame, offset: predictions.append((frame, offset)),
    )
    qtbot.addWidget(window)
    window._show_error = lambda title, message: errors.append((title, message))
    widget = window.predict_widget
    checkpoint = os.environ.get("DAS_WHISPER_GUI_CHECKPOINT", "whisper:large")
    widget.paths_form.checkpoint_field.value = checkpoint
    widget.trainer_form.accelerator = os.environ.get("DAS_WHISPER_GUI_DEVICE", "auto")
    widget.whisperseg_decoder_form.num_beams = 1
    window.show()
    if checkpoint == "whisper:large":
        assert "Whisper Large" in widget.paths_form.checkpoint_field.combo.currentText()
    assert widget.model_class_label.text() == "Whisper ASR · word timestamps"
    qtbot.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: bool(predictions or errors), timeout=900_000)
    qtbot.waitUntil(lambda: window._thread is None, timeout=10_000)
    assert not errors, "\n".join(message for _, message in errors)
    frame, offset = predictions[0]
    assert offset == 0.0
    words = re.findall(r"\w+", " ".join(frame.name).lower())
    assert SequenceMatcher(None, reference, words).ratio() > 0.8
    assert np.all(frame.stop_seconds > frame.start_seconds)
    assert np.all(frame.start_seconds.to_numpy()[1:] >= frame.stop_seconds.to_numpy()[:-1])
    assert frame.start_seconds.min() >= 0
    assert frame.stop_seconds.max() <= len(audio) / samplerate
    assert frame.stop_seconds.max() > len(audio) / samplerate - 2
    print(frame.to_string(index=False))
