from __future__ import annotations

import os
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy")
pytest.importorskip("magicgui")

from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import QDialog, QGroupBox, QLabel, QLineEdit, QPlainTextEdit, QScrollArea, QTabWidget

import das.gui_app as gui
from das.config import Config


def _column_group_titles(container) -> list[list[str]]:
    layout = container.layout()
    columns: list[list[str]] = []
    for index in range(layout.count()):
        column = layout.itemAt(index).widget()
        if column is None:
            continue
        column_layout = column.layout()
        titles = []
        for column_index in range(column_layout.count()):
            group = column_layout.itemAt(column_index).widget()
            if isinstance(group, QGroupBox):
                titles.append(group.title())
        columns.append(titles)
    return columns


def _magicgui_label_text(widget) -> str:
    row = gui._magicgui_row(widget)
    labels = row.findChildren(QLabel) if row is not None else []
    return labels[0].text() if labels else ""


def test_window_builds_and_switches_primary_button(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    window.show()

    stop_early_pos = window.stop_early_button.mapTo(window, QPoint(0, 0))
    cancel_pos = window.cancel_button.mapTo(window, QPoint(0, 0))
    run_pos = window.run_button.mapTo(window, QPoint(0, 0))
    save_pos = window.save_button.mapTo(window, QPoint(0, 0))

    assert window.log_output.font().family() == QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
    assert window.stop_early_button.text() == "Stop early"
    assert window.cancel_button.text() == "Cancel"
    assert window.run_button.text() == "Start Training"
    assert cancel_pos.x() > save_pos.x()
    assert stop_early_pos.x() > cancel_pos.x()
    assert run_pos.x() > stop_early_pos.x()

    window.command_tabs.setCurrentWidget(window.predict_widget)

    assert window.run_button.text() == "Start Prediction"


def test_log_redraws_training_progress_in_place(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    window._append_log("Training starts.\n")
    window._append_log("\rEpoch 1/2: [##------------------] 1/10 batches (10%)")
    window._append_log("\rEpoch 1/2: [####----------------] 2/10 batches (20%)")

    assert window.log_output.toPlainText() == "Training starts.\nEpoch 1/2: [####----------------] 2/10 batches (20%)"

    window._append_log("\rEpoch 1/2: [####################] 10/10 batches (100%)")
    window._append_log("\nEpoch 1 complete.\n")
    assert window.log_output.toPlainText() == (
        "Training starts.\nEpoch 1/2: [####################] 10/10 batches (100%)\nEpoch 1 complete.\n"
    )


def test_error_dialog_keeps_trace_scrollable(qtbot, monkeypatch: pytest.MonkeyPatch):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(gui.QDialog, "exec", lambda self: 0)

    trace = "Traceback (most recent call last):\nValueError: bad input"
    window._show_error("Failed", trace)

    dialog = window.findChild(QDialog, "error_dialog")
    assert dialog is not None
    assert dialog.findChild(QLabel, "error_message").text() == "ValueError: bad input"
    details = dialog.findChild(QPlainTextEdit, "error_trace")
    assert details.isReadOnly()
    assert details.toPlainText() == trace
    assert details.minimumHeight() == details.maximumHeight() == 280


def test_stop_buttons_enable_while_training_or_predicting(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)

    assert not window.stop_early_button.isEnabled()
    assert not window.cancel_button.isEnabled()

    window._running_command = "train"
    window._set_running(True)

    assert window.stop_early_button.isEnabled()
    assert window.cancel_button.isEnabled()

    window._set_running(False)

    assert not window.stop_early_button.isEnabled()
    assert not window.cancel_button.isEnabled()

    window._running_command = "predict"
    window._set_running(True)

    assert window.stop_early_button.isEnabled()
    assert window.cancel_button.isEnabled()


def test_stop_early_sets_stop_event_but_leaves_cancel_available(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    window._running_command = "train"
    window._stop_event = threading.Event()
    window._set_running(True)

    window.stop_training_early()

    assert window._stop_event.is_set()
    assert not window.stop_early_button.isEnabled()
    assert window.cancel_button.isEnabled()
    assert "Stop requested. Waiting for the current training step to finish..." in window.log_output.toPlainText()

    window._running_command = "predict"
    window._stop_event = threading.Event()
    window._set_running(True)
    window.stop_training_early()

    assert window._stop_event.is_set()
    assert "Stop requested. Waiting for the current file to finish..." in window.log_output.toPlainText()


def test_cancel_training_terminates_training_process(qtbot):
    class FakeThread:
        def isRunning(self):
            return True

        def deleteLater(self):
            pass

    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    thread = FakeThread()
    window._running_command = "train"
    window._stop_event = threading.Event()
    window._cancel_event = threading.Event()
    window._thread = thread
    window._set_running(True)

    window.cancel_training()

    assert window._cancel_event.is_set()
    assert not window._stop_event.is_set()
    assert not window.stop_early_button.isEnabled()
    assert not window.cancel_button.isEnabled()
    assert "Cancel requested. Terminating training immediately..." in window.log_output.toPlainText()

    window._cleanup_worker()

    assert window.run_button.isEnabled()
    assert "Training cancelled." in window.log_output.toPlainText()


def test_cancel_prediction_requests_immediate_process_termination(qtbot):
    class FakeThread:
        def isRunning(self):
            return True

    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    window._running_command = "predict"
    window._stop_event = threading.Event()
    window._cancel_event = threading.Event()
    window._thread = FakeThread()
    window._set_running(True)

    window.cancel_training()

    assert window._cancel_event.is_set()
    assert not window._stop_event.is_set()
    assert "Cancel requested. Terminating prediction immediately..." in window.log_output.toPlainText()


def test_worker_cancel_only_signals_worker_thread(qtbot):
    class FakeProcess:
        def __init__(self):
            self.killed = False

        def is_alive(self):
            return True

        def kill(self):
            self.killed = True

    stop_event = threading.Event()
    worker = gui.JobWorker("train", Config(mode="train"), stop_event=stop_event)
    process = FakeProcess()
    worker._train_process = process

    worker.cancel_training_process()

    assert not stop_event.is_set()
    assert worker.cancel_event.is_set()
    assert not process.killed


def test_worker_thread_kills_training_process_after_cancel(qtbot):
    class FakeProcess:
        def __init__(self):
            self.killed = False

        def is_alive(self):
            return True

        def kill(self):
            self.killed = True

    stop_event = threading.Event()
    worker = gui.JobWorker("train", Config(mode="train"), stop_event=stop_event)
    process = FakeProcess()
    worker._train_process = process

    worker._kill_training_process()

    assert stop_event.is_set()
    assert process.killed


def test_gui_uses_cli_help_for_tooltips(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)

    assert window.train_widget.data_form.gui.batch_size.native.toolTip() == "Number of chunks per batch."
    assert window.train_widget.paths_form.data_dir_field.line_edit.toolTip().startswith(
        "Directory with audio files or npy_dir data."
    )
    assert window.train_widget.paths_form.checkpoint_prefix_field.toolTip() == (
        "Optional prefix for checkpoint filenames written during training."
    )
    assert "WhisperSeg .ckpt" in window.train_widget.model_section.initial_model_field.line_edit.toolTip()
    assert window.train_widget.model_section.frontend_type.toolTip() == "Frontend type."
    assert window.train_widget.evaluation_form.gui.syllable_tolerance_ms.native.toolTip() == (
        "Maximum onset or offset error still counted as a syllable match."
    )
    assert window.preset_dropdown.toolTip() == (
        "Built-in configs loaded like YAML configs: fly-pulse, fly, zebra-finch, tweetynet"
    )


def test_train_data_source_offers_hf_presets_and_custom_id(qtbot, tmp_path: Path):
    widget = gui.TrainCommandWidget()
    qtbot.addWidget(widget)
    paths = widget.paths_form

    datasets = [paths.hf_dataset_combo.itemText(index) for index in range(paths.hf_dataset_combo.count())]
    assert paths.data_source_combo.currentText() == gui.LOCAL_DATA_SOURCE
    assert paths.data_dir_field.isVisibleTo(paths)
    assert paths.data_dir_field.label.isHidden()
    assert paths.hf_dataset_combo.isHidden()
    assert "Hugging Face dataset" not in [label.text() for label in paths.findChildren(QLabel)]
    assert paths.hf_dataset_combo.isEditable()
    assert "nccratliri/whisperseg-conda-env" not in datasets

    paths.data_source_combo.setCurrentText(gui.HF_DATA_SOURCE)
    paths.hf_dataset_combo.setCurrentText("someone/custom-dataset")

    assert paths.data_dir_field.isHidden()
    assert not paths.hf_dataset_combo.isHidden()
    assert widget.build_config().data_dir == "someone/custom-dataset"

    paths.data_dir = str(tmp_path)

    assert paths.data_source_combo.currentText() == gui.LOCAL_DATA_SOURCE
    assert widget.build_config().data_dir == str(tmp_path)


def test_initial_tab_and_predict_source_defaults(qtbot):
    train_window = gui.DASConformerWindow(initial_tab="train")
    qtbot.addWidget(train_window)
    assert train_window.active_command_name() == "train"
    assert train_window.predict_widget.source_combo.currentText() == "File or folder"

    predict_window = gui.DASConformerWindow(
        initial_tab="predict",
        current_audio_provider=lambda start, stop: (np.zeros(10), 1_000, start),
    )
    qtbot.addWidget(predict_window)

    assert predict_window.active_command_name() == "predict"
    assert predict_window.predict_widget.source_combo.currentText() == "Current audio"
    assert predict_window.predict_widget.uses_current_audio()


def test_tab_layout_uses_expected_section_names(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    window.show()

    train_tabs = [window.train_widget.tabs.tabText(index) for index in range(window.train_widget.tabs.count())]
    train_setup = window.train_widget.tabs.widget(0)
    train_model = window.train_widget.tabs.widget(1)
    predict_content = window.predict_widget.layout().itemAt(0).widget()

    assert [window.command_tabs.tabText(index) for index in range(window.command_tabs.count())] == ["Train", "Predict"]
    assert train_tabs == ["Setup", "Model"]
    assert isinstance(train_setup, QScrollArea)
    assert _column_group_titles(train_setup.widget()) == [
        ["Paths", "Execution"],
        ["Data"],
        ["Prediction and Evaluation"],
    ]
    assert isinstance(train_model, QScrollArea)
    assert train_model.widget().isAncestorOf(window.train_widget.model_class)
    assert not train_setup.widget().isAncestorOf(window.train_widget.model_class)
    assert isinstance(predict_content, QScrollArea)
    assert predict_content.widget() is window.predict_widget.content_panel
    assert _column_group_titles(window.predict_widget.content_panel) == [
        ["Source", "Paths", "Execution"],
        ["Data", "Model"],
        ["Output", "Evaluation"],
    ]
    assert not window.predict_widget.findChildren(QTabWidget)
    assert _magicgui_field_is_hidden(window.train_widget.trainer_form.gui.seed)
    assert _magicgui_field_is_hidden(window.train_widget.trainer_form.gui.num_devices)


def test_train_predict_after_training_controls(qtbot):
    window = gui.DASConformerWindow(
        current_audio_provider=lambda start, stop: (np.zeros(10), 1_000, start),
        current_duration_provider=lambda: 12.5,
        annotated_region_provider=lambda: (2.0, 4.0),
    )
    qtbot.addWidget(window)
    window.show()

    prediction = window.train_widget.prediction_form
    labels = [prediction.predict_after_train_combo.itemText(index) for index in range(prediction.predict_after_train_combo.count())]

    assert labels == ["Do not predict", "Current file", "Annotated region", "File or folder"]
    assert prediction.predict_after_train_label.text() == "Predict after training"
    assert "Predict after training" not in [group.title() for group in prediction.findChildren(QGroupBox)]
    assert prediction.predict_after_train_combo.currentText() == "Do not predict"
    assert prediction.time_range_widget.isHidden()
    assert prediction.file_or_folder_field.isHidden()

    prediction.predict_after_train_combo.setCurrentText("Current file")

    assert not prediction.time_range_widget.isHidden()
    assert prediction.time_range_widget.start_seconds_field.text() == "0.0"
    assert prediction.time_range_widget.stop_seconds_field.text() == "12.5"

    prediction.predict_after_train_combo.setCurrentText("File or folder")

    assert not prediction.file_or_folder_field.isHidden()
    assert prediction.file_or_folder_field.label.isHidden()
    assert window.predict_widget.paths_form.data_dir_field.label.text() == "Audio file or folder"


def test_file_or_folder_path_field_browses_file_or_directory(qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    selected = {"file": str(tmp_path / "clip.wav"), "folder": str(tmp_path)}

    class FakeFileDialog:
        @staticmethod
        def getOpenFileName(*args, **kwargs):
            return selected["file"], "audio"

        @staticmethod
        def getExistingDirectory(*args, **kwargs):
            return selected["folder"]

    monkeypatch.setattr(gui, "QFileDialog", FakeFileDialog)
    field = gui.PathField(label="Audio file or folder", selection_mode="file_or_directory")
    qtbot.addWidget(field)

    assert field.browse_button.text() == "File"
    assert field.folder_browse_button is not None
    assert field.folder_browse_button.text() == "Folder"

    field.browse()
    assert field.value == str(tmp_path / "clip.wav")

    field.browse_folder()

    assert field.value == str(tmp_path)


def test_predict_data_selection_keeps_adjacent_output_default(qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    selected = {"file": str(tmp_path / "clip.wav"), "folder": str(tmp_path)}

    class FakeFileDialog:
        @staticmethod
        def getOpenFileName(*args, **kwargs):
            return selected["file"], "audio"

        @staticmethod
        def getExistingDirectory(*args, **kwargs):
            return selected["folder"]

    monkeypatch.setattr(gui, "QFileDialog", FakeFileDialog)
    widget = gui.PredictPathsWidget()
    qtbot.addWidget(widget)

    widget.data_dir_field.browse()
    assert widget.output_dir == ""

    widget.data_dir_field.browse_folder()
    assert widget.output_dir == ""


def test_train_include_labels_dropdown_populates_from_data_dir(qtbot, tmp_path: Path):
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, np.zeros(1_000, dtype=np.float32), 1_000)
    pd.DataFrame(
        [
            {"name": "pulse", "start_seconds": 0.1, "stop_seconds": 0.2},
            {"name": "sine", "start_seconds": 0.3, "stop_seconds": 0.4},
        ]
    ).to_csv(tmp_path / "clip_annotations.csv", index=False)
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)

    window.train_widget.paths_form.data_dir = str(tmp_path)
    labels = [
        window.train_widget.include_labels_widget.combo.itemText(index)
        for index in range(window.train_widget.include_labels_widget.combo.count())
    ]
    window.train_widget.include_labels_widget.set_labels(labels, selected_labels=["pulse"])

    assert labels == ["pulse", "sine"]
    assert window.train_widget.build_config().include_labels == ["pulse"]


def test_include_labels_dropdown_toggles_on_activation(qtbot):
    widget = gui.IncludeLabelsWidget()
    qtbot.addWidget(widget)
    widget.set_labels(["pulse", "sine"])

    widget.combo._toggle_activated_index(0)

    assert widget.include_labels() == ["sine"]


def test_include_labels_dropdown_ignores_activation_after_pressed_toggle(qtbot):
    widget = gui.IncludeLabelsWidget()
    qtbot.addWidget(widget)
    widget.set_labels(["pulse", "sine"])

    widget.combo._toggle_index(widget.combo.model().index(0, 0))
    widget.combo._toggle_activated_index(0)

    assert widget.include_labels() == ["sine"]


def test_train_and_predict_dialogs_expose_detection_thresholds(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    window.show()
    window.train_widget.tabs.setCurrentWidget(window.train_widget.tabs.widget(1))

    postprocessing = window.train_widget.model_section.postprocessing_widget

    assert postprocessing.manual_form.gui.native.parent() is postprocessing
    postprocessing.manual_form.segment_threshold_low = 0.2
    postprocessing.manual_form.segment_threshold_high = 0.8
    postprocessing.manual_form.event_threshold = 0.7
    postprocessing.manual_form.event_dist_min_ms = 3.0
    postprocessing.manual_form.event_dist_max_ms = "40"
    config = window.train_widget.build_config()
    assert (config.segment_threshold_low, config.segment_threshold_high) == (0.2, 0.8)
    assert (config.event_threshold, config.event_dist_min_ms, config.event_dist_max_ms) == (0.7, 3.0, 40.0)

    window.predict_widget.postprocessing_form.segment_threshold_low = 0.1
    window.predict_widget.postprocessing_form.segment_threshold_high = 0.9
    window.predict_widget.postprocessing_form.syllable_postprocessor = "binary_mask"
    window.predict_widget.evaluation_form.evaluate = True
    window.predict_widget.evaluation_form.split = "test"
    window.predict_widget.evaluation_form.syllable_tolerance_ms = 15.0
    predict_config = window.predict_widget.build_config()
    assert (predict_config.segment_threshold_low, predict_config.segment_threshold_high) == (0.1, 0.9)
    assert predict_config.syllable_postprocessor == "binary_mask"
    assert (predict_config.evaluate, predict_config.split, predict_config.syllable_tolerance_ms) == (True, "test", 15.0)


def test_learning_rate_field_preserves_decimal_precision(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)

    learning_rate = "0.000123456789"
    field = window.train_widget.model_section.hyperparameters_form.gui.learning_rate.native
    assert isinstance(field, QLineEdit)

    window.train_widget.model_section.hyperparameters_form.learning_rate = learning_rate

    assert field.text() == learning_rate
    assert window.train_widget.build_config().learning_rate == float(learning_rate)


def test_whisperseg_training_uses_only_local_das_checkpoint(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    selector = window.train_widget.model_section.initial_model_field

    assert window.train_widget.model_class.currentText() == "DAS"
    assert window.train_widget.model_section.isAncestorOf(selector)
    assert not window.train_widget.paths_form.isAncestorOf(selector)
    assert selector.combo.itemText(0) == "Local checkpoint"
    assert selector.isHidden()
    assert window.train_widget.build_config().initial_model == ""

    window.train_widget.model_class.setCurrentText("WhisperSeg")

    assert selector.combo.itemText(0) == "Local checkpoint"
    assert not selector.isHidden()
    assert selector.combo.count() == 1
    assert selector.combo.currentText() == "Local checkpoint"

    selector.value = "/tmp/local.ckpt"

    assert selector.combo.currentText() == "Local checkpoint"
    assert window.train_widget.build_config().initial_model == "/tmp/local.ckpt"

    window.train_widget.model_class.setCurrentText("DAS")

    assert selector.isHidden()
    assert selector.combo.currentText() == "Local checkpoint"


def test_predict_checkpoint_selector_accepts_local_models(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    selector = window.predict_widget.paths_form.checkpoint_field

    assert selector.combo.itemText(0) == "Local checkpoint"
    assert selector.combo.count() == 1

    selector.value = "/tmp/model.ckpt"

    assert selector.combo.currentText() == "Local checkpoint"
    assert selector.line_edit.text() == "/tmp/model.ckpt"
    assert window.predict_widget.build_config().checkpoint == "/tmp/model.ckpt"


def test_predict_model_pane_switches_controls_from_checkpoint(qtbot, tmp_path: Path):
    widget = gui.PredictCommandWidget()
    qtbot.addWidget(widget)

    assert widget.model_class_label.text() == "DAS"
    assert widget.model_stack.currentWidget() is widget.das_model_panel
    assert [box.title() for box in widget.das_model_panel.findChildren(QGroupBox)] == ["Runtime", "Postprocessing"]

    widget.runtime_form.num_workers = 3
    widget.runtime_form.num_devices = "2"
    widget.postprocessing_form.fill_gap_ms = 4.5
    widget.whisperseg_decoder_form.num_beams = 7
    whisperseg_checkpoint = tmp_path / "whisperseg.ckpt"
    torch.save(
        {
            "das": {"backend": "whisperseg"},
            "whisperseg": {
                "format": "das_whisperseg.checkpoint",
                "format_version": 1,
            },
        },
        whisperseg_checkpoint,
    )
    widget.paths_form.checkpoint = str(whisperseg_checkpoint)

    assert widget.model_class_label.text() == "WhisperSeg"
    assert widget.model_stack.currentWidget() is widget.whisperseg_model_panel
    assert [box.title() for box in widget.whisperseg_model_panel.findChildren(QGroupBox)] == ["Generation"]

    widget.paths_form.checkpoint = str(tmp_path / "das.ckpt")
    config = widget.build_config()

    assert widget.model_class_label.text() == "DAS"
    assert widget.runtime_form.num_workers == 3
    assert widget.postprocessing_form.fill_gap_ms == 4.5
    assert widget.whisperseg_decoder_form.num_beams == 7
    assert config.num_workers == 3
    assert config.num_devices == 2
    assert config.fill_gap_ms == 4.5
    assert config.num_beams == 7


def _magicgui_field_row(widget):
    return widget.native.parent()


def _magicgui_field_is_hidden(widget):
    row = _magicgui_field_row(widget)
    return widget.native.isHidden() or row.isHidden()


def test_training_control_fields_build_config_and_expose_whisperseg_options(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    model_section = window.train_widget.model_section
    form = model_section.hyperparameters_form
    encoder_form = model_section.encoder_settings_form

    assert _magicgui_field_is_hidden(form.gui.reduce_lr_min)
    assert _magicgui_field_is_hidden(form.gui.linear_lr_schedule)
    assert _magicgui_field_is_hidden(form.gui.weight_decay)
    assert _magicgui_field_is_hidden(form.gui.warmup_steps)
    assert not form.gui.native.isHidden()
    assert not _magicgui_field_row(form.gui.cross_entropy_weight).isHidden()
    assert not _magicgui_field_row(form.gui.learning_rate).isHidden()
    assert not _magicgui_field_row(form.gui.reduce_lr_factor).isHidden()

    form.early_stopping = False
    form.early_stopping_patience = 4
    form.reduce_lr = False
    form.reduce_lr_patience = 3
    form.reduce_lr_factor = "0.25"
    config = window.train_widget.build_config()

    assert config.early_stopping is False
    assert config.early_stopping_patience == 4
    assert config.reduce_lr is False
    assert config.reduce_lr_patience == 3
    assert config.reduce_lr_factor == 0.25

    window.train_widget.model_class.setCurrentText("WhisperSeg")

    assert form.gui.early_stopping.native.isEnabled()
    assert _magicgui_field_is_hidden(form.gui.reduce_lr_min)
    assert not _magicgui_field_is_hidden(form.gui.linear_lr_schedule)
    assert not _magicgui_field_is_hidden(form.gui.weight_decay)
    assert not _magicgui_field_is_hidden(form.gui.warmup_steps)
    assert form.gui.linear_lr_schedule.native.isEnabled()
    assert encoder_form.gui.freeze_encoder.native.isEnabled()
    assert form.gui.learning_rate.native.isEnabled()

    form.linear_lr_schedule = True
    form.weight_decay = "0.02"
    form.warmup_steps = 12
    form.reduce_lr = False
    encoder_form.freeze_encoder = True
    config = window.train_widget.build_config()
    assert config.linear_lr_schedule is True
    assert config.weight_decay == 0.02
    assert config.warmup_steps == 12
    assert config.freeze_encoder is True


@pytest.mark.parametrize(
    ("frontend_type", "form_name"),
    [
        ("stft", "stft_frontend_form"),
        ("mel", "mel_frontend_form"),
        ("conv", "conv_frontend_form"),
        ("conv_resnet", "conv_frontend_form"),
    ],
)
def test_frontend_hop_seconds_fields_preserve_decimal_precision(qtbot, frontend_type, form_name):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)

    hop_seconds = "0.000123456789"
    model_section = window.train_widget.model_section
    form = getattr(model_section, form_name)
    field = form.gui.hop_seconds.native
    assert isinstance(field, QLineEdit)

    model_section.frontend_type.setCurrentText(frontend_type)
    form.hop_seconds = hop_seconds

    assert model_section.frontend_stack.currentWidget() is form.gui.native
    assert field.text() == hop_seconds
    assert window.train_widget.build_config().frontend_hop_seconds == float(hop_seconds)


def test_stft_frontend_exposes_frequency_bounds(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)

    model_section = window.train_widget.model_section
    model_section.frontend_type.setCurrentText("stft")
    model_section.stft_frontend_form.fmin = 250.0
    model_section.stft_frontend_form.fmax = "9000"

    assert hasattr(model_section.stft_frontend_form.gui, "fmin")
    assert isinstance(model_section.stft_frontend_form.gui.fmax.native, QLineEdit)
    config = window.train_widget.build_config()
    assert config.frontend_fmin == 250.0
    assert config.frontend_fmax == 9000.0


def _visible_model_section_titles(model_section) -> list[str]:
    sections = (
        model_section.chunking_section,
        model_section.frontend_section,
        model_section.encoder_section,
        model_section.decoder_section,
        model_section.optimization_section,
        model_section.postprocessing_section,
    )
    return [section.title() for section in sections if not section.isHidden()]


def _model_section_column_titles(model_section) -> list[list[str]]:
    columns = []
    for column in model_section.model_columns:
        column_layout = column.layout()
        titles = []
        for index in range(column_layout.count()):
            widget = column_layout.itemAt(index).widget()
            if isinstance(widget, QGroupBox):
                titles.append(widget.title())
        columns.append(titles)
    return columns


def test_model_class_switches_between_das_and_whisperseg_sections(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)

    model_section = window.train_widget.model_section
    assert window.train_widget.model_class.currentText() == "DAS"
    assert not model_section.findChildren(QTabWidget)
    assert _visible_model_section_titles(model_section) == [
        "Chunking",
        "Frontend",
        "Encoder",
        "Decoder",
        "Optimization",
        "Postprocessing",
    ]
    assert _model_section_column_titles(model_section) == [
        ["Chunking", "Frontend"],
        ["Encoder"],
        ["Decoder", "Optimization", "Postprocessing"],
    ]
    assert model_section.frontend_type.findText("whisperseg") == -1
    assert model_section.encoder_type.findText("whisperseg") == -1
    assert model_section.decoder_type.findText("whisperseg") == -1

    model_section.chunking_form.num_time_steps = 2048
    model_section.chunking_form.chunk_stride = "1024"
    model_section.chunking_form.split_within_files = True
    model_section.postprocessing_widget.manual_form.fill_gap_ms = 6.5
    model_section.frontend_type.setCurrentText("mel")
    model_section.decoder_type.setCurrentText("linear")
    window.train_widget.model_class.setCurrentText("WhisperSeg")

    assert _visible_model_section_titles(model_section) == [
        "Frontend",
        "Decoder",
        "Optimization",
    ]
    assert _model_section_column_titles(model_section) == [
        ["Frontend"],
        ["Decoder"],
        ["Optimization"],
    ]
    assert _magicgui_field_is_hidden(model_section.hyperparameters_form.gui.cross_entropy_weight)
    assert not _magicgui_field_is_hidden(model_section.encoder_settings_form.gui.freeze_encoder)
    assert hasattr(model_section.whisperseg_frontend_form.gui, "min_frequency")
    assert hasattr(model_section.whisperseg_decoder_form.gui, "generation_max_length")
    assert model_section.whisperseg_frontend_form.min_frequency == ""
    assert model_section.whisperseg_frontend_form.frequency_scale == "1.0"
    assert window.train_widget.data_form.batch_size == 4
    assert window.train_widget.data_form.validation_fraction == 0.1
    assert window.train_widget.data_form.test_fraction == 0.0
    assert window.train_widget.trainer_form.num_epochs == 10
    assert model_section.tcn_encoder_form.dropout == 0.0
    assert model_section.whisperseg_decoder_form.decoder_dropout == 0.0
    assert float(model_section.hyperparameters_form.learning_rate) == 3e-6
    assert model_section.hyperparameters_form.early_stopping_patience == 3

    model_section.whisperseg_frontend_form.min_frequency = "250"
    model_section.whisperseg_frontend_form.frequency_scale = "0.5"
    model_section.whisperseg_frontend_form.spec_time_step = "0.001"
    model_section.whisperseg_decoder_form.decoder_dropout = 0.25
    model_section.hyperparameters_form.learning_rate = 1e-4
    model_section.whisperseg_decoder_form.generation_max_length = 222
    config = window.train_widget.build_config()
    assert config.frontend_type == "whisperseg"
    assert config.min_frequency == 250
    assert config.frequency_scale == 0.5
    assert config.spec_time_step == 0.001
    assert config.decoder_dropout == 0.25
    assert config.learning_rate == 1e-4
    assert config.generation_max_length == 222

    window.train_widget.model_class.setCurrentText("DAS")

    assert model_section.frontend_type.currentText() == "mel"
    assert model_section.decoder_type.currentText() == "linear"
    config = window.train_widget.build_config()
    assert model_section.chunking_form.num_time_steps == 2048
    assert model_section.chunking_form.chunk_stride == "1024"
    assert model_section.postprocessing_widget.manual_form.fill_gap_ms == 6.5
    assert config.frontend_type == "mel"
    assert config.num_time_steps == 2048
    assert config.chunk_stride == 1024
    assert config.split_within_files is True
    assert config.fill_gap_ms == 6.5


def test_loading_whisperseg_state_selects_whisperseg_model_class(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)

    state = gui.TrainGuiState()
    state.model_selection.frontend_type = "whisperseg"
    state.model_selection.encoder_type = "whisperseg"
    state.model_selection.decoder_type = "whisperseg"
    window.train_widget.load_state(state)

    assert window.train_widget.model_class.currentText() == "WhisperSeg"
    assert window.train_widget.build_config().encoder_type == "whisperseg"


def test_save_and_load_use_flat_yaml(qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    window.show()

    train_path = tmp_path / "train.yaml"
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(train_path), "yaml"))
    window.train_widget.paths_form.data_dir = "/tmp/audio"
    window.train_widget.paths_form.output_dir = "/tmp/run"
    window.train_widget.model_section.initial_model = "/tmp/initial.ckpt"
    window.train_widget.paths_form.checkpoint_prefix = "zf"
    qtbot.mouseClick(window.save_button, Qt.MouseButton.LeftButton)

    train_payload = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    assert train_payload["data_dir"] == "/tmp/audio"
    assert train_payload["initial_model"] == "/tmp/initial.ckpt"
    assert train_payload["checkpoint_prefix"] == "zf"
    assert train_payload["mode"] == "train"
    assert train_payload["frontend_type"] == "mel"

    predict_path = tmp_path / "predict.yaml"
    window.command_tabs.setCurrentWidget(window.predict_widget)
    window.predict_widget.paths_form.data_dir = "/tmp/audio"
    window.predict_widget.paths_form.checkpoint = "/tmp/model.ckpt"
    window.predict_widget.paths_form.output_dir = "/tmp/predictions"
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(predict_path), "yaml"))
    qtbot.mouseClick(window.save_button, Qt.MouseButton.LeftButton)

    predict_payload = yaml.safe_load(predict_path.read_text(encoding="utf-8"))
    assert predict_payload["mode"] == "predict"
    assert predict_payload["checkpoint"] == "/tmp/model.ckpt"
    assert predict_payload["evaluate"] is False
    assert predict_payload["split"] is None

    load_path = tmp_path / "predict-load.yaml"
    load_path.write_text(
        yaml.safe_dump(
            {
                "mode": "predict",
                "data_dir": "/new/audio",
                "checkpoint": "/new/model.ckpt",
                "output_dir": "/new/predictions",
                "output_suffix": "_frames.csv",
                "existing_annotations": "skip",
                "evaluate": True,
                "split": "test",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(load_path), "yaml"))
    qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    assert window.predict_widget.paths_form.data_dir == "/new/audio"
    assert window.predict_widget.paths_form.checkpoint == "/new/model.ckpt"
    assert window.predict_widget.prediction_form.output_suffix == "_frames.csv"
    assert window.predict_widget.prediction_form.existing_annotations == "skip"
    loaded_predict_config = window.predict_widget.build_config()
    assert loaded_predict_config.evaluate is True
    assert loaded_predict_config.split == "test"
    assert loaded_predict_config.existing_annotations == "skip"


def test_startup_config_loads_predict_tab(qtbot):
    window = gui.DASConformerWindow(
        startup_config=Config(
            mode="predict",
            data_dir="/startup/audio",
            checkpoint="/startup/model.ckpt",
            output_dir="/startup/predictions",
            output_suffix="_frames.csv",
            existing_annotations="merge",
            evaluate=True,
            generation_max_length=333,
            num_trials=4,
            num_beams=6,
            top_k=3,
            top_p=0.8,
            length_penalty=0.7,
        )
    )
    qtbot.addWidget(window)
    window.show()

    assert window.active_command_name() == "predict"
    assert window.predict_widget.paths_form.data_dir == "/startup/audio"
    assert window.predict_widget.prediction_form.output_suffix == "_frames.csv"
    assert window.predict_widget.prediction_form.existing_annotations == "merge"
    assert window.predict_widget.whisperseg_decoder_form.generation_max_length == 333
    assert window.predict_widget.whisperseg_decoder_form.num_trials == 4
    assert window.predict_widget.whisperseg_decoder_form.num_beams == 6
    assert window.predict_widget.whisperseg_decoder_form.top_k == 3
    assert window.predict_widget.whisperseg_decoder_form.top_p == 0.8
    assert window.predict_widget.whisperseg_decoder_form.length_penalty == 0.7
    loaded_predict_config = window.predict_widget.build_config()
    assert loaded_predict_config.evaluate is True
    assert loaded_predict_config.split is None
    assert loaded_predict_config.existing_annotations == "merge"
    assert loaded_predict_config.generation_max_length == 333
    assert loaded_predict_config.num_trials == 4
    assert loaded_predict_config.num_beams == 6
    assert loaded_predict_config.top_k == 3
    assert loaded_predict_config.top_p == 0.8
    assert loaded_predict_config.length_penalty == 0.7


def test_predict_checkpoint_selection_loads_checkpoint_predict_defaults(qtbot, tmp_path: Path):
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "das": {
                "predict": {
                    "batch_size": 7,
                    "fill_gap_ms": 3.5,
                    "min_syllable_ms": 12.0,
                }
            }
        },
        checkpoint_path,
    )
    widget = gui.PredictCommandWidget()
    qtbot.addWidget(widget)
    widget.data_form.batch_size = 99
    widget.prediction_form.output_suffix = "_custom.csv"
    widget.prediction_form.existing_annotations = "merge"
    widget.postprocessing_form.fill_gap_ms = 99.0
    widget.postprocessing_form.min_syllable_ms = 99.0

    widget.paths_form.checkpoint = str(checkpoint_path)

    assert widget.data_form.batch_size == 7
    assert widget.prediction_form.output_suffix == "_custom.csv"
    assert widget.prediction_form.existing_annotations == "merge"
    assert widget.postprocessing_form.fill_gap_ms == 3.5
    assert widget.postprocessing_form.min_syllable_ms == 12.0


def test_predict_worker_passes_audio_only_for_current_audio(qtbot, monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_predict(config, *, audio=None, samplerate=None, verbose=False):
        calls.append({"config": config, "audio": audio, "samplerate": samplerate, "verbose": verbose})
        if audio is None:
            return ["/tmp/predictions/file_annotations.csv"]
        return pd.DataFrame([{"name": "song", "start_seconds": 0.1, "stop_seconds": 0.2}])

    monkeypatch.setattr(gui, "predict", fake_predict)
    config = Config(mode="predict", checkpoint="/tmp/model.ckpt")

    current_worker = gui.JobWorker(
        "predict",
        config,
        audio=np.arange(10),
        samplerate=1_000,
        time_offset_seconds=2.0,
    )
    current_payloads = []
    current_worker.finished.connect(current_payloads.append)
    current_worker.run()

    assert len(current_payloads) == 1
    assert current_payloads[0]["ok"] is True
    assert current_payloads[0]["time_offset_seconds"] == 2.0
    np.testing.assert_array_equal(calls[0]["audio"], np.arange(10))
    assert calls[0]["samplerate"] == 1_000

    folder_worker = gui.JobWorker("predict", config)
    folder_payloads = []
    folder_worker.finished.connect(folder_payloads.append)
    folder_worker.run()

    assert folder_payloads[0]["result"] == ["/tmp/predictions/file_annotations.csv"]
    assert calls[1]["audio"] is None
    assert calls[1]["samplerate"] is None


def test_train_worker_predicts_after_successful_training(qtbot, monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_train(config, *, verbose=False, stop_event=None, emit_epoch_logs=False):
        calls.append({"command": "train", "config": config, "verbose": verbose, "emit_epoch_logs": emit_epoch_logs})
        return "/tmp/model.ckpt"

    def fake_predict(config, *, audio=None, samplerate=None, verbose=False, stop_event=None):
        calls.append(
            {
                "command": "predict",
                "config": config,
                "audio": audio,
                "samplerate": samplerate,
                "verbose": verbose,
                "stop_event": stop_event,
            }
        )
        return pd.DataFrame([{"name": "song", "start_seconds": 0.1, "stop_seconds": 0.2}])

    monkeypatch.setattr(gui, "train", fake_train)
    monkeypatch.setattr(gui, "predict", fake_predict)
    worker = gui.JobWorker(
        "train",
        Config(mode="train", data_dir="/tmp/audio", output_dir="/tmp/run"),
        predict_after_train={
            "config": Config(mode="predict", data_dir="", checkpoint="", output_dir=""),
            "audio": np.arange(10),
            "samplerate": 1_000,
            "time_offset_seconds": 2.0,
        },
    )
    payloads = []
    worker.finished.connect(payloads.append)

    worker.run()

    assert payloads[0]["ok"] is True
    assert payloads[0]["result"] == "/tmp/model.ckpt"
    assert payloads[0]["prediction_after_train"]["time_offset_seconds"] == 2.0
    assert calls[1]["command"] == "predict"
    assert calls[1]["config"].checkpoint == "/tmp/model.ckpt"
    np.testing.assert_array_equal(calls[1]["audio"], np.arange(10))
    assert calls[1]["samplerate"] == 1_000


def test_current_file_predict_after_training_uses_audio_provider_without_duration_provider(qtbot):
    calls = []
    window = gui.DASConformerWindow(
        current_audio_provider=lambda start, stop: calls.append((start, stop)) or (np.zeros(10), 1_000, start),
    )
    qtbot.addWidget(window)
    window.train_widget.prediction_form.predict_after_train_combo.setCurrentText("Current file")

    request = window._build_predict_after_train_request(
        Config(mode="train", data_dir="/tmp/audio", output_dir="/tmp/run")
    )

    assert request["samplerate"] == 1_000
    assert calls == [(0.0, None)]


def test_annotated_region_predict_after_training_uses_provider_bounds(qtbot):
    calls = []
    window = gui.DASConformerWindow(
        current_audio_provider=lambda start, stop: calls.append((start, stop)) or (np.zeros(10), 1_000, start),
        annotated_region_provider=lambda: [(0.5, 1.0), (0.2, 0.4)],
    )
    qtbot.addWidget(window)
    window.train_widget.prediction_form.predict_after_train_combo.setCurrentText("Annotated region")

    request = window._build_predict_after_train_request(
        Config(mode="train", data_dir="/tmp/audio", output_dir="/tmp/run")
    )

    assert request["samplerate"] == 1_000
    assert calls == [(0.2, 1.0)]


def test_train_worker_skips_prediction_after_cancelled_or_failed_training(qtbot, monkeypatch: pytest.MonkeyPatch):
    predict_calls = []

    def fake_cancelled_train(config, *, verbose=False, stop_event=None, emit_epoch_logs=False):
        stop_event.set()
        return "/tmp/model.ckpt"

    monkeypatch.setattr(gui, "train", fake_cancelled_train)
    monkeypatch.setattr(gui, "predict", lambda *args, **kwargs: predict_calls.append((args, kwargs)))
    stop_event = threading.Event()
    worker = gui.JobWorker(
        "train",
        Config(mode="train", data_dir="/tmp/audio", output_dir="/tmp/run"),
        predict_after_train={"config": Config(mode="predict", data_dir="/tmp/audio", checkpoint="", output_dir="/tmp/pred")},
        stop_event=stop_event,
    )
    payloads = []
    worker.finished.connect(payloads.append)

    worker.run()

    assert payloads[0]["ok"] is True
    assert payloads[0]["cancelled"] is True
    assert "prediction_after_train" not in payloads[0]
    assert predict_calls == []

    def fake_failed_train(config, *, verbose=False, stop_event=None, emit_epoch_logs=False):
        raise RuntimeError("failed")

    monkeypatch.setattr(gui, "train", fake_failed_train)
    worker = gui.JobWorker(
        "train",
        Config(mode="train", data_dir="/tmp/audio", output_dir="/tmp/run"),
        predict_after_train={"config": Config(mode="predict", data_dir="/tmp/audio", checkpoint="", output_dir="/tmp/pred")},
    )
    payloads = []
    worker.finished.connect(payloads.append)

    worker.run()

    assert payloads[0]["ok"] is False
    assert predict_calls == []


def test_preset_dropdown_applies_active_tab_presets_and_preserves_paths(qtbot):
    window = gui.DASConformerWindow()
    qtbot.addWidget(window)
    window.show()

    window.train_widget.paths_form.data_dir = "/tmp/train-audio"
    window.train_widget.paths_form.output_dir = "/tmp/train-run"
    window.apply_selected_preset(window.preset_dropdown.findData("fly-pulse"))

    assert window.active_command_name() == "train"
    assert window.train_widget.paths_form.data_dir == "/tmp/train-audio"
    assert window.train_widget.paths_form.output_dir == "/tmp/train-run"
    assert window.train_widget.model_section.chunking_form.num_time_steps == 4096
    assert window.train_widget.model_section.chunking_form.chunk_stride == "2048"
    assert window.train_widget.model_section.frontend_type.currentText() == "conv_resnet"
    assert window.train_widget.model_section.encoder_type.currentText() == "tcn"
    assert window.train_widget.model_section.conv_frontend_form.hop_seconds == "0.001"
    assert window.train_widget.model_section.conv_frontend_form.pad_mode == "reflect"
    assert window.train_widget.model_section.tcn_encoder_form.hidden_size == 64
    assert window.train_widget.model_section.tcn_encoder_form.num_layers == 1
    assert window.train_widget.build_config().include_labels == ["pulse"]
    assert window.preset_dropdown.currentIndex() == 0

    window.apply_selected_preset(window.preset_dropdown.findData("tweetynet"))

    assert window.train_widget.paths_form.data_dir == "/tmp/train-audio"
    assert window.train_widget.paths_form.output_dir == "/tmp/train-run"
    assert window.train_widget.model_section.frontend_type.currentText() == "stft"
    assert window.train_widget.model_section.encoder_type.currentText() == "tweetynet"
    assert window.train_widget.model_section.tweetynet_encoder_form.hidden_size == 512
    assert window.train_widget.model_section.tweetynet_encoder_form.num_layers == 1
    assert window.train_widget.model_section.tweetynet_encoder_form.kernel_size == 5

    window.command_tabs.setCurrentWidget(window.predict_widget)
    window.predict_widget.paths_form.data_dir = "/tmp/predict-audio"
    window.predict_widget.paths_form.checkpoint = "/tmp/model.ckpt"
    window.predict_widget.paths_form.output_dir = "/tmp/predict-run"
    window.apply_selected_preset(window.preset_dropdown.findData("zebra-finch"))

    assert window.active_command_name() == "predict"
    assert window.predict_widget.paths_form.data_dir == "/tmp/predict-audio"
    assert window.predict_widget.paths_form.checkpoint == "/tmp/model.ckpt"
    assert window.predict_widget.paths_form.output_dir == "/tmp/predict-run"
    assert window.predict_widget.data_form.batch_size == 32
    assert window.predict_widget.postprocessing_form.fill_gap_ms == 5.0
    assert window.predict_widget.postprocessing_form.min_syllable_ms == 20.0
    assert window.predict_widget.build_config().syllable_tolerance_ms == 20.0

    log_text = window.log_output.toPlainText()
    assert "Applied Fly pulse train config." in log_text
    assert "Applied TweetyNet train config." in log_text
    assert "Applied Zebra Finch predict config." in log_text
