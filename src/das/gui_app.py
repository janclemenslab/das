import contextlib
from dataclasses import fields
import logging
import multiprocessing
from pathlib import Path
import sys
import threading
import traceback
from typing import Annotated, Callable, Literal, Sequence

from magicgui.experimental import guiclass
import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, QThread, QTimer, Signal, Qt
from qtpy.QtGui import QFontDatabase, QTextCursor
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import yaml

from .config import (
    BUILTIN_CONFIG_OPTIONS,
    TRAIN_DATASET_HF_OPTIONS,
)
from .gui_helpers import (
    AttentionDecoderState,
    DAS_DECODER_OPTIONS,
    DAS_ENCODER_OPTIONS,
    DAS_FRONTEND_OPTIONS,
    Config,
    ConformerEncoderState,
    ConvDecoderState,
    ConvFrontendState,
    DataSectionState,
    EncoderSettingsState,
    EvaluationSectionState,
    LSTMDecoderState,
    MelFrontendState,
    ModelHyperparametersState,
    ModelSelectionState,
    PredictDataSectionState,
    PredictGuiState,
    PredictPathsState,
    PredictTrainerSectionState,
    PredictionSectionState,
    STFTFrontendState,
    TCNEncoderState,
    TweetynetEncoderState,
    TrainGuiState,
    TrainPathsState,
    TrainPredictionSectionState,
    TrainerSectionState,
    WhisperSegDecoderState,
    WhisperSegFrontendState,
    browser_initial_path,
    describe_dataset_path,
    format_predict_config_yaml,
    format_train_config_yaml,
    load_predict_gui_state_from_yaml,
    load_train_gui_state_from_yaml,
    predict_config_to_gui_state,
    predict_gui_state_to_config,
    save_predict_config,
    save_train_config,
    train_config_to_gui_state,
    train_gui_state_to_config,
)
from .api import _is_whisperseg_predict_source, _load_checkpoint_metadata, predict, train

LineText = Annotated[str, {"widget_type": "LineEdit"}]
PositiveInt = Annotated[int, {"widget_type": "SpinBox", "min": 1, "max": 1_000_000}]
NonNegativeInt = Annotated[int, {"widget_type": "SpinBox", "min": 0, "max": 1_000_000}]
FloatValue = Annotated[float, {"widget_type": "FloatSpinBox", "min": 0.0, "max": 1_000_000.0}]
BoolValue = Annotated[bool, {"widget_type": "CheckBox"}]
CurrentAudioProvider = Callable[[float, float | None], tuple[np.ndarray, int, float]]
CurrentDurationProvider = Callable[[], float]
AnnotatedRegionProvider = Callable[[], tuple[float, float] | Sequence[tuple[float, float]]]
PredictionCallback = Callable[[pd.DataFrame, float], None]

MODEL_CLASS_LABELS = {"das": "DAS", "whisperseg": "WhisperSeg"}
MODEL_CLASS_VALUES = {label: value for value, label in MODEL_CLASS_LABELS.items()}

PREDICT_AFTER_TRAIN_NONE = "Do not predict"
PREDICT_AFTER_TRAIN_CURRENT_FILE = "Current file"
PREDICT_AFTER_TRAIN_ANNOTATED_REGION = "Annotated region"
PREDICT_AFTER_TRAIN_FILE_OR_FOLDER = "File or folder"
PREDICT_AFTER_TRAIN_OPTIONS = [
    PREDICT_AFTER_TRAIN_NONE,
    PREDICT_AFTER_TRAIN_CURRENT_FILE,
    PREDICT_AFTER_TRAIN_ANNOTATED_REGION,
    PREDICT_AFTER_TRAIN_FILE_OR_FOLDER,
]


@guiclass
class DataSectionForm:
    batch_size: PositiveInt = 8
    num_workers: NonNegativeInt = 0
    validation_fraction: FloatValue = DataSectionState.validation_fraction
    test_fraction: FloatValue = DataSectionState.test_fraction
    ignore_class_names: BoolValue = DataSectionState.ignore_class_names


@guiclass
class ChunkingSectionForm:
    num_time_steps: PositiveInt = DataSectionState.num_time_steps
    chunk_stride: LineText = DataSectionState.chunk_stride
    split_within_files: BoolValue = DataSectionState.split_within_files


@guiclass
class PredictDataSectionForm:
    batch_size: PositiveInt = PredictDataSectionState.batch_size


@guiclass
class PredictRuntimeSectionForm:
    num_workers: NonNegativeInt = PredictDataSectionState.num_workers
    num_devices: LineText = ""


@guiclass
class EncoderSettingsForm:
    freeze_encoder: BoolValue = EncoderSettingsState.freeze_encoder


@guiclass
class STFTFrontendForm:
    num_channels: PositiveInt = 128
    kernel_size: PositiveInt = 1024
    hop_seconds: LineText = "0.004"
    fmin: FloatValue = 100.0
    fmax: LineText = ""
    trainable: BoolValue = True


@guiclass
class MelFrontendForm:
    num_channels: PositiveInt = 128
    kernel_size: PositiveInt = 1024
    hop_seconds: LineText = "0.004"
    fmin: FloatValue = 100.0
    fmax: LineText = ""
    trainable: BoolValue = True


@guiclass
class ConvFrontendForm:
    num_channels: PositiveInt = 128
    kernel_size: PositiveInt = 9
    hop_seconds: LineText = "0.004"
    pad_mode: Literal["constant", "reflect", "replicate", "circular"] = "constant"


@guiclass
class WhisperSegFrontendForm:
    min_frequency: LineText = WhisperSegFrontendState.min_frequency
    frequency_scale: LineText = WhisperSegFrontendState.frequency_scale
    spec_time_step: LineText = WhisperSegFrontendState.spec_time_step


@guiclass
class ConformerEncoderForm:
    num_heads: PositiveInt = 4
    hidden_size: PositiveInt = 128
    num_layers: PositiveInt = 3
    kernel_size: PositiveInt = 31


@guiclass
class TCNEncoderForm:
    hidden_size: PositiveInt = 128
    num_layers: PositiveInt = 3
    dilations: LineText = "1, 2, 4, 8, 16"
    kernel_size: PositiveInt = 16
    dropout: FloatValue = 0.1
    use_skip_connections: BoolValue = True
    use_separable: LineText = "false"
    padding: Literal["same", "causal"] = "same"


@guiclass
class TweetynetEncoderForm:
    hidden_size: PositiveInt = 512
    num_layers: PositiveInt = 1
    kernel_size: PositiveInt = 5
    dropout: FloatValue = 0.1


@guiclass
class LSTMDecoderForm:
    hidden_size: PositiveInt = 64


@guiclass
class ConvDecoderForm:
    kernel_size: PositiveInt = 8


@guiclass
class AttentionDecoderForm:
    num_heads: PositiveInt = 4
    num_layers: PositiveInt = 2
    dropout: FloatValue = 0.1


@guiclass
class WhisperSegDecoderForm:
    decoder_dropout: FloatValue = WhisperSegDecoderState.decoder_dropout
    max_length: PositiveInt = WhisperSegDecoderState.max_length
    generation_max_length: PositiveInt = WhisperSegDecoderState.generation_max_length
    num_trials: PositiveInt = WhisperSegDecoderState.num_trials
    num_beams: PositiveInt = WhisperSegDecoderState.num_beams
    top_k: PositiveInt = WhisperSegDecoderState.top_k
    top_p: FloatValue = WhisperSegDecoderState.top_p
    length_penalty: FloatValue = WhisperSegDecoderState.length_penalty


@guiclass
class ModelHyperparametersForm:
    cross_entropy_weight: FloatValue = 0.9
    learning_rate: LineText = "0.0001"
    early_stopping: BoolValue = ModelHyperparametersState.early_stopping
    early_stopping_patience: PositiveInt = ModelHyperparametersState.early_stopping_patience
    reduce_lr: BoolValue = ModelHyperparametersState.reduce_lr
    reduce_lr_patience: PositiveInt = ModelHyperparametersState.reduce_lr_patience
    reduce_lr_factor: LineText = "0.1"
    reduce_lr_min: LineText = "1e-8"
    linear_lr_schedule: BoolValue = ModelHyperparametersState.linear_lr_schedule
    weight_decay: LineText = "0.01"
    warmup_steps: NonNegativeInt = ModelHyperparametersState.warmup_steps


@guiclass
class TrainerSectionForm:
    seed: LineText = ""
    accelerator: Literal["auto", "cpu", "cuda", "mps", "tpu"] = "auto"
    num_devices: LineText = ""
    num_epochs: PositiveInt = 100
    max_num_steps_per_epoch: LineText = ""


@guiclass
class PredictTrainerSectionForm:
    seed: LineText = PredictTrainerSectionState.seed
    accelerator: Literal["auto", "cpu", "cuda", "mps", "tpu"] = PredictTrainerSectionState.accelerator


@guiclass
class PredictionSectionForm:
    output_suffix: LineText = PredictionSectionState.output_suffix
    existing_annotations: Literal["skip", "overwrite", "merge"] = PredictionSectionState.existing_annotations


@guiclass
class PredictPostprocessingForm:
    fill_gap_ms: FloatValue = PredictionSectionState.fill_gap_ms
    min_syllable_ms: FloatValue = PredictionSectionState.min_syllable_ms
    syllable_postprocessor: Literal["binary_mask", "label_aware_dense"] = PredictionSectionState.syllable_postprocessor
    segment_threshold_low: FloatValue = PredictionSectionState.segment_threshold_low
    segment_threshold_high: FloatValue = PredictionSectionState.segment_threshold_high
    event_threshold: FloatValue = PredictionSectionState.event_threshold
    event_dist_min_ms: FloatValue = PredictionSectionState.event_dist_min_ms
    event_dist_max_ms: LineText = PredictionSectionState.event_dist_max_ms


@guiclass
class PredictEvaluationForm:
    evaluate: BoolValue = PredictionSectionState.evaluate
    split: Literal["", "train", "val", "test"] = PredictionSectionState.split
    syllable_tolerance_ms: FloatValue = PredictionSectionState.syllable_tolerance_ms


@guiclass
class TrainPredictionOutputForm:
    output_suffix: LineText = TrainPredictionSectionState.output_suffix


@guiclass
class TrainPostprocessingSettingsForm:
    syllable_postprocessor: Literal["binary_mask", "label_aware_dense"] = (
        TrainPredictionSectionState.syllable_postprocessor
    )


@guiclass
class ManualPostprocessingForm:
    fill_gap_ms: FloatValue = TrainPredictionSectionState.fill_gap_ms
    min_syllable_ms: FloatValue = TrainPredictionSectionState.min_syllable_ms
    segment_threshold_low: FloatValue = TrainPredictionSectionState.segment_threshold_low
    segment_threshold_high: FloatValue = TrainPredictionSectionState.segment_threshold_high
    event_threshold: FloatValue = TrainPredictionSectionState.event_threshold
    event_dist_min_ms: FloatValue = TrainPredictionSectionState.event_dist_min_ms
    event_dist_max_ms: LineText = TrainPredictionSectionState.event_dist_max_ms


@guiclass
class EvaluationSectionForm:
    syllable_tolerance_ms: FloatValue = 10.0


def _copy_from_form(form_cls, form) -> object:
    payload = {field.name: getattr(form, field.name) for field in fields(form_cls) if hasattr(form, field.name)}
    return form_cls(**payload)


def _assign_to_form(form, state) -> None:
    for field in fields(state):
        if hasattr(form, field.name):
            setattr(form, field.name, getattr(state, field.name))


def _set_placeholder(widget, text: str) -> None:
    native = getattr(widget, "native", None)
    if native is not None and hasattr(native, "setPlaceholderText"):
        native.setPlaceholderText(text)


def _config_help(field_name: str) -> str:
    for item in fields(Config):
        if item.name == field_name:
            return str(item.metadata.get("help") or "")
    return ""


def _set_tooltip(widget, tooltip: str) -> None:
    if not tooltip:
        return
    if hasattr(widget, "setToolTip"):
        widget.setToolTip(tooltip)
    if hasattr(widget, "tooltip"):
        try:
            widget.tooltip = tooltip
        except Exception:
            pass
    native = getattr(widget, "native", None)
    if native is not None and hasattr(native, "setToolTip"):
        native.setToolTip(tooltip)


def _set_magicgui_field_enabled(widget, enabled: bool) -> None:
    native = getattr(widget, "native", None)
    if native is not None and hasattr(native, "setEnabled"):
        native.setEnabled(enabled)


def _set_config_tooltip(widget, field_name: str) -> None:
    _set_tooltip(widget, _config_help(field_name))


def _set_widget_label(widget, label: str) -> None:
    native = getattr(widget, "native", None)
    row = _magicgui_row(widget)
    if not isinstance(native, QLineEdit) and hasattr(widget, "label"):
        try:
            widget.label = label
        except Exception:
            pass
    if row is not None:
        if isinstance(native, QCheckBox):
            native.setText(label)
            return
        labels = row.findChildren(QLabel)
        if labels:
            labels[0].setText(label)
            return
    if isinstance(native, (QLabel, QCheckBox, QPushButton)):
        native.setText(label)
        return
    if isinstance(widget, (QLabel, QCheckBox, QPushButton)):
        widget.setText(label)


def _set_form_tooltips(form, mapping: dict[str, str]) -> None:
    for widget_name, field_name in mapping.items():
        widget = getattr(form.gui, widget_name, None)
        if widget is not None:
            _set_config_tooltip(widget, field_name)


def _set_form_labels(form, mapping: dict[str, str]) -> None:
    for widget_name, label in mapping.items():
        widget = getattr(form.gui, widget_name, None)
        if widget is not None:
            _set_widget_label(widget, label)


def _magicgui_row(widget) -> QWidget | None:
    native = getattr(widget, "native", None)
    parent = native.parent() if native is not None else None
    return parent if isinstance(parent, QWidget) else None


def _set_magicgui_field_visible(widget, visible: bool) -> None:
    native = getattr(widget, "native", None)
    if isinstance(native, QCheckBox):
        native.setVisible(visible)
        return
    row = _magicgui_row(widget)
    if row is not None:
        row.setVisible(visible)
        return
    if native is not None and hasattr(native, "setVisible"):
        native.setVisible(visible)


def _hide_magicgui_field(widget) -> None:
    _set_magicgui_field_visible(widget, False)


def _cap_widget_width_to_hint(widget) -> None:
    native = getattr(widget, "native", None)
    if native is not None:
        native.setMaximumWidth(native.sizeHint().width())


def _compact_magicgui_form(form) -> None:
    native = form.gui.native
    layout = native.layout()
    if layout is not None:
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
    for row in native.findChildren(QWidget):
        row_layout = row.layout()
        if row_layout is not None:
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(3)


def _wrap_widget(widget: QWidget) -> QWidget:
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(widget)
    return scroll_area


def _group_box(title: str, child: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.addWidget(child)
    return box


def _stacked_sections(*sections: tuple[str, QWidget]) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    for title, widget in sections:
        layout.addWidget(_group_box(title, widget))
    layout.addStretch(1)
    return container


def _three_column_sections(
    left_sections: tuple[tuple[str, QWidget], ...],
    middle_sections: tuple[tuple[str, QWidget], ...],
    right_sections: tuple[tuple[str, QWidget], ...],
) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.addWidget(_stacked_sections(*left_sections), 1)
    layout.addWidget(_stacked_sections(*middle_sections), 1)
    layout.addWidget(_stacked_sections(*right_sections), 1)
    return container


def _set_combo_value(combo: QComboBox, value: str) -> None:
    index = combo.findText(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _normalize_checkpoint_selection(path: str) -> str:
    for suffix in ("_model.h5", "_params.yaml"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def _checkpoint_predict_metadata(path: str) -> dict[str, object]:
    path = path.strip()
    if not path:
        return {}

    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.suffix != ".ckpt" or not checkpoint_path.is_file():
        return {}

    metadata = _load_checkpoint_metadata(str(checkpoint_path))
    predict_metadata = metadata.get("predict", {})
    return predict_metadata if isinstance(predict_metadata, dict) else {}


class PathField(QWidget):
    path_selected = Signal(str)

    def __init__(
        self,
        *,
        label: str,
        selection_mode: Literal["file", "directory", "file_or_directory"],
        placeholder: str = "",
        dialog_filter: str = "All files (*)",
        config_field: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.selection_mode = selection_mode
        self.dialog_filter = dialog_filter

        self.label = QLabel(label)
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.browse_button = QPushButton("File" if selection_mode == "file_or_directory" else "...")
        self.browse_button.clicked.connect(self.browse)
        self.folder_browse_button: QPushButton | None = None
        if selection_mode == "file_or_directory":
            self.folder_browse_button = QPushButton("Folder")
            self.folder_browse_button.clicked.connect(self.browse_folder)
        if config_field is not None:
            self.set_config_tooltip(config_field)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.line_edit, 1)
        row.addWidget(self.browse_button)
        if self.folder_browse_button is not None:
            row.addWidget(self.folder_browse_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.label)
        layout.addLayout(row)

    def set_config_tooltip(self, field_name: str) -> None:
        tooltip = _config_help(field_name)
        _set_tooltip(self, tooltip)
        _set_tooltip(self.label, tooltip)
        _set_tooltip(self.line_edit, tooltip)
        _set_tooltip(self.browse_button, tooltip)
        if self.folder_browse_button is not None:
            _set_tooltip(self.folder_browse_button, tooltip)

    @property
    def value(self) -> str:
        return self.line_edit.text().strip()

    @value.setter
    def value(self, path: str) -> None:
        self.line_edit.setText(path)

    def browse(self) -> None:
        start_path = browser_initial_path(self.value, selection_mode=self.selection_mode)
        if self.selection_mode == "directory":
            selected = QFileDialog.getExistingDirectory(self, self.label.text(), start_path)
        elif self.selection_mode == "file_or_directory":
            selected, _ = QFileDialog.getOpenFileName(self, self.label.text(), start_path, self.dialog_filter)
        else:
            selected, _ = QFileDialog.getOpenFileName(self, self.label.text(), start_path, self.dialog_filter)
            selected = _normalize_checkpoint_selection(selected)
        if selected:
            self.value = selected
            self.path_selected.emit(selected)

    def browse_folder(self) -> None:
        start_path = browser_initial_path(self.value, selection_mode="directory")
        selected = QFileDialog.getExistingDirectory(self, self.label.text(), start_path)
        if selected:
            self.value = selected
            self.path_selected.emit(selected)


LOCAL_INITIAL_MODEL_OPTION = "Local checkpoint"
LOCAL_DATA_SOURCE = "Local data directory"
HF_DATA_SOURCE = "Hugging Face dataset"


class InitialModelField(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._model_class = "das"
        self._selected_by_model_class = {"das": "", "whisperseg": ""}
        self.label = QLabel("Initial model")
        self.combo = QComboBox()
        self.local_field = PathField(
            label="Local checkpoint",
            selection_mode="file",
            placeholder="optional supervised checkpoint",
            dialog_filter="DAS checkpoints (*.ckpt);;All files (*)",
            config_field="initial_model",
        )
        self.local_field.label.hide()

        tooltip = _config_help("initial_model")
        _set_tooltip(self, tooltip)
        _set_tooltip(self.label, tooltip)
        _set_tooltip(self.combo, tooltip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        layout.addWidget(self.local_field)

        self.combo.currentTextChanged.connect(self._sync_local_field)
        self._populate_options()

    @property
    def line_edit(self) -> QLineEdit:
        return self.local_field.line_edit

    @property
    def value(self) -> str:
        current = self.combo.currentText()
        if current == LOCAL_INITIAL_MODEL_OPTION:
            return self.local_field.value
        return current

    @value.setter
    def value(self, value: str) -> None:
        value = value.strip()
        index = self.combo.findText(value)
        if index >= 0:
            self.combo.setCurrentIndex(index)
        else:
            self.local_field.value = value
            _set_combo_value(self.combo, LOCAL_INITIAL_MODEL_OPTION)
        self._selected_by_model_class[self._model_class] = value

    def set_model_class(self, model_class: str) -> None:
        self._selected_by_model_class[self._model_class] = self.value
        self._model_class = model_class
        self._populate_options()

    def _hf_options(self) -> list[str]:
        return []

    def _populate_options(self) -> None:
        hf_options = self._hf_options()
        selected_value = self._selected_by_model_class.get(self._model_class, "")
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(LOCAL_INITIAL_MODEL_OPTION)
        self.combo.addItems(hf_options)
        self.combo.blockSignals(False)

        if selected_value in hf_options:
            _set_combo_value(self.combo, selected_value)
        else:
            self.local_field.value = selected_value
            _set_combo_value(self.combo, LOCAL_INITIAL_MODEL_OPTION)
        self._selected_by_model_class[self._model_class] = selected_value
        self._sync_local_field()

    def _sync_local_field(self, *_args) -> None:
        self.local_field.setVisible(self.combo.currentText() == LOCAL_INITIAL_MODEL_OPTION)
        self._selected_by_model_class[self._model_class] = self.value


class PredictCheckpointField(QWidget):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.label = QLabel("Checkpoint")
        self.combo = QComboBox()
        self.combo.addItem(LOCAL_INITIAL_MODEL_OPTION)
        self.local_field = PathField(
            label="DAS checkpoint or legacy DAS model",
            selection_mode="file",
            placeholder="/path/to/model.ckpt or legacy DAS model",
            dialog_filter="Model files (*.ckpt *.h5 *.yaml);;All files (*)",
            config_field="checkpoint",
        )
        self.local_field.label.hide()

        tooltip = _config_help("checkpoint")
        _set_tooltip(self, tooltip)
        _set_tooltip(self.label, tooltip)
        _set_tooltip(self.combo, tooltip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        layout.addWidget(self.local_field)

        self.combo.currentTextChanged.connect(self._sync_local_field)
        self.combo.currentTextChanged.connect(lambda *_args: self.changed.emit())
        self.local_field.line_edit.textChanged.connect(lambda *_args: self.changed.emit())
        self._sync_local_field()

    @property
    def line_edit(self) -> QLineEdit:
        return self.local_field.line_edit

    @property
    def value(self) -> str:
        current = self.combo.currentText()
        if current == LOCAL_INITIAL_MODEL_OPTION:
            return self.local_field.value
        return current

    @value.setter
    def value(self, value: str) -> None:
        value = value.strip()
        index = self.combo.findText(value)
        if index >= 0:
            self.combo.setCurrentIndex(index)
        else:
            self.local_field.value = value
            _set_combo_value(self.combo, LOCAL_INITIAL_MODEL_OPTION)
        self._sync_local_field()

    def _sync_local_field(self, *_args) -> None:
        self.local_field.setVisible(self.combo.currentText() == LOCAL_INITIAL_MODEL_OPTION)


class TrainPathsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.data_source_label = QLabel("Data source")
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems((LOCAL_DATA_SOURCE, HF_DATA_SOURCE))
        self.data_dir_field = PathField(
            label="Local data directory",
            selection_mode="directory",
            placeholder="/path/to/audio",
            config_field="data_dir",
        )
        self.data_dir_field.label.hide()
        self.hf_dataset_combo = QComboBox()
        self.hf_dataset_combo.setEditable(True)
        self.hf_dataset_combo.addItems(TRAIN_DATASET_HF_OPTIONS)
        self.hf_dataset_combo.lineEdit().setPlaceholderText("Select or type owner/dataset")
        data_tooltip = _config_help("data_dir")
        for widget in (
            self.data_source_label,
            self.data_source_combo,
            self.hf_dataset_combo,
        ):
            _set_tooltip(widget, data_tooltip)
        self.output_dir_field = PathField(
            label="Output directory",
            selection_mode="directory",
            placeholder="./",
            config_field="output_dir",
        )
        self.checkpoint_prefix_label = QLabel("Checkpoint prefix")
        self.checkpoint_prefix_field = QLineEdit()
        self.checkpoint_prefix_field.setPlaceholderText("optional")
        checkpoint_prefix_tooltip = _config_help("checkpoint_prefix")
        _set_tooltip(self.checkpoint_prefix_label, checkpoint_prefix_tooltip)
        _set_tooltip(self.checkpoint_prefix_field, checkpoint_prefix_tooltip)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.addWidget(self.data_source_label)
        layout.addWidget(self.data_source_combo)
        layout.addWidget(self.data_dir_field)
        layout.addWidget(self.hf_dataset_combo)
        layout.addWidget(self.output_dir_field)
        layout.addWidget(self.checkpoint_prefix_label)
        layout.addWidget(self.checkpoint_prefix_field)
        layout.addStretch(1)

        self.data_source_combo.currentTextChanged.connect(self._sync_data_source)
        self._sync_data_source()

    @property
    def data_dir(self) -> str:
        if self.data_source_combo.currentText() == HF_DATA_SOURCE:
            return self.hf_dataset_combo.currentText().strip()
        return self.data_dir_field.value

    @data_dir.setter
    def data_dir(self, path: str) -> None:
        path = path.strip()
        is_hf_dataset = path in TRAIN_DATASET_HF_OPTIONS or (
            not Path(path).expanduser().is_absolute()
            and path.count("/") == 1
            and not path.startswith((".", "~"))
        )
        if is_hf_dataset:
            self.hf_dataset_combo.setCurrentText(path)
            _set_combo_value(self.data_source_combo, HF_DATA_SOURCE)
        else:
            self.data_dir_field.value = path
            _set_combo_value(self.data_source_combo, LOCAL_DATA_SOURCE)
        self._sync_data_source()

    def _sync_data_source(self, *_args) -> None:
        local = self.data_source_combo.currentText() == LOCAL_DATA_SOURCE
        self.data_dir_field.setVisible(local)
        self.hf_dataset_combo.setVisible(not local)

    @property
    def output_dir(self) -> str:
        return self.output_dir_field.value

    @output_dir.setter
    def output_dir(self, path: str) -> None:
        self.output_dir_field.value = path

    @property
    def checkpoint_prefix(self) -> str:
        return self.checkpoint_prefix_field.text().strip()

    @checkpoint_prefix.setter
    def checkpoint_prefix(self, value: str) -> None:
        self.checkpoint_prefix_field.setText(value)


def _qt_value(name: str, enum_name: str):
    if hasattr(Qt, name):
        return getattr(Qt, name)
    return getattr(getattr(Qt, enum_name), name)


class CheckableLabelsComboBox(QComboBox):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._pressed_toggled_row: int | None = None
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        insert_policy = QComboBox.NoInsert if hasattr(QComboBox, "NoInsert") else QComboBox.InsertPolicy.NoInsert
        self.setInsertPolicy(insert_policy)
        self.view().pressed.connect(self._toggle_index)
        self.activated.connect(self._toggle_activated_index)

    def set_items(self, labels: list[str], selected: list[str]) -> None:
        previous_blocked = self.blockSignals(True)
        self.clear()
        selected_set = set(selected)
        check_role = _qt_value("CheckStateRole", "ItemDataRole")
        checked = _qt_value("Checked", "CheckState")
        unchecked = _qt_value("Unchecked", "CheckState")
        user_checkable = _qt_value("ItemIsUserCheckable", "ItemFlag")
        for label in labels:
            self.addItem(label)
            item = self.model().item(self.count() - 1)
            item.setFlags(item.flags() | user_checkable)
            item.setData(checked if label in selected_set else unchecked, check_role)
        self.blockSignals(previous_blocked)
        self._refresh_text()

    def checked_items(self) -> list[str]:
        check_role = _qt_value("CheckStateRole", "ItemDataRole")
        checked = _qt_value("Checked", "CheckState")
        labels = []
        for row in range(self.count()):
            item = self.model().item(row)
            if item.data(check_role) == checked:
                labels.append(self.itemText(row))
        return labels

    def _toggle_index(self, index) -> None:
        self._pressed_toggled_row = index.row()
        self._toggle_row(index.row())
        QTimer.singleShot(250, self._clear_pressed_toggled_row)

    def _toggle_activated_index(self, index) -> None:
        if not isinstance(index, int):
            index = self.currentIndex()
        if self._pressed_toggled_row == index:
            self._pressed_toggled_row = None
            return
        self._toggle_row(index)

    def _clear_pressed_toggled_row(self) -> None:
        self._pressed_toggled_row = None

    def _toggle_row(self, row: int) -> None:
        if row < 0:
            return
        index = self.model().index(row, 0)
        item = self.model().itemFromIndex(index)
        if item is None:
            return
        check_role = _qt_value("CheckStateRole", "ItemDataRole")
        checked = _qt_value("Checked", "CheckState")
        unchecked = _qt_value("Unchecked", "CheckState")
        item.setData(unchecked if item.data(check_role) == checked else checked, check_role)
        self._refresh_text()
        self.changed.emit()

    def _refresh_text(self) -> None:
        checked = self.checked_items()
        if self.count() == 0:
            text = "Set data directory"
        elif len(checked) == self.count():
            text = "All labels"
        elif not checked:
            text = "No labels"
        elif len(checked) <= 3:
            text = ", ".join(checked)
        else:
            text = f"{len(checked)} labels"
        self.lineEdit().setText(text)


class IncludeLabelsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.available_labels: list[str] = []
        self._explicit_selection = False
        self.label = QLabel("Include labels")
        self.combo = CheckableLabelsComboBox()
        tooltip = _config_help("include_labels")
        _set_tooltip(self, tooltip)
        _set_tooltip(self.label, tooltip)
        _set_tooltip(self.combo, tooltip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        self.combo.changed.connect(self._mark_explicit_selection)

    def _mark_explicit_selection(self) -> None:
        self._explicit_selection = True

    def set_labels(self, labels: list[str], selected_labels: list[str] | None = None) -> None:
        labels = [label for label in labels if label != "noise"]
        self.available_labels = list(dict.fromkeys(labels))
        if selected_labels is not None:
            selected = [label for label in selected_labels if label != "noise"]
            self._explicit_selection = bool(selected)
            if not selected:
                selected = list(self.available_labels)
        else:
            selected = self.include_labels()
            if not selected:
                selected = list(self.available_labels)

        display_labels = list(self.available_labels)
        display_labels.extend(label for label in selected if label not in display_labels)
        self.combo.set_items(display_labels, selected)

    def load_from_data_dir(self, data_dir: str, selected_labels: list[str] | None = None) -> None:
        labels: list[str] = []
        try:
            description = describe_dataset_path(data_dir)
            if description.get("status") == "ok":
                labels = [str(label) for label in description.get("classes", []) if str(label) != "noise"]
        except Exception:
            labels = []
        self.set_labels(labels, selected_labels=selected_labels)

    def include_labels(self) -> list[str]:
        selected = self.combo.checked_items()
        if not selected:
            return []
        if self.available_labels and selected == self.available_labels:
            return []
        if not self.available_labels and not self._explicit_selection:
            return []
        return selected


class PredictPathsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.data_dir_field = PathField(
            label="Audio file or folder",
            selection_mode="file_or_directory",
            placeholder="/path/to/audio-file-or-folder",
            dialog_filter="Audio files (*.wav *.flac *.aiff *.aif *.h5 *.hdf5 *.mat *.zarr *.npz *.npy *.mmap);;All files (*)",
            config_field="data_dir",
        )
        self.checkpoint_field = PredictCheckpointField()
        self.output_dir_field = PathField(
            label="Output directory (optional)",
            selection_mode="directory",
            placeholder="Next to each audio file",
            config_field="output_dir",
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.addWidget(self.data_dir_field)
        layout.addWidget(self.checkpoint_field)
        layout.addWidget(self.output_dir_field)
        layout.addStretch(1)

    @property
    def data_dir(self) -> str:
        return self.data_dir_field.value

    @data_dir.setter
    def data_dir(self, path: str) -> None:
        self.data_dir_field.value = path

    @property
    def checkpoint(self) -> str:
        return self.checkpoint_field.value

    @checkpoint.setter
    def checkpoint(self, path: str) -> None:
        self.checkpoint_field.value = path

    @property
    def output_dir(self) -> str:
        return self.output_dir_field.value

    @output_dir.setter
    def output_dir(self, path: str) -> None:
        self.output_dir_field.value = path

    def set_folder_fields_enabled(self, enabled: bool) -> None:
        self.data_dir_field.setEnabled(enabled)
        self.output_dir_field.setEnabled(enabled)


class CurrentAudioWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.start_seconds_field = QLineEdit()
        self.start_seconds_field.setPlaceholderText("0.0")
        self.start_seconds_field.setText("0.0")
        self.stop_seconds_field = QLineEdit()
        self.stop_seconds_field.setPlaceholderText("end")

        layout = QHBoxLayout(self)
        layout.addWidget(QLabel("Start"))
        layout.addWidget(self.start_seconds_field)
        layout.addWidget(QLabel("Stop"))
        layout.addWidget(self.stop_seconds_field)

    def request(self) -> tuple[float, float | None]:
        start_seconds = float(self.start_seconds_field.text().strip() or 0.0)
        stop_text = self.stop_seconds_field.text().strip()
        stop_seconds = None if not stop_text else float(stop_text)
        return start_seconds, stop_seconds


class TrainPostprocessingSectionWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.settings_form = TrainPostprocessingSettingsForm()
        self.manual_form = ManualPostprocessingForm()

        _set_form_tooltips(
            self.settings_form,
            {"syllable_postprocessor": "syllable_postprocessor"},
        )
        _set_form_tooltips(
            self.manual_form,
            {name: name for name in (
                "fill_gap_ms", "min_syllable_ms", "segment_threshold_low",
                "segment_threshold_high", "event_threshold", "event_dist_min_ms", "event_dist_max_ms",
            )},
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.settings_form.gui.native)
        layout.addWidget(self.manual_form.gui.native)

    def state(self, *, output_suffix: str) -> TrainPredictionSectionState:
        return TrainPredictionSectionState(
            output_suffix=output_suffix,
            syllable_postprocessor=str(self.settings_form.syllable_postprocessor),
            fill_gap_ms=float(self.manual_form.fill_gap_ms),
            min_syllable_ms=float(self.manual_form.min_syllable_ms),
            segment_threshold_low=float(self.manual_form.segment_threshold_low),
            segment_threshold_high=float(self.manual_form.segment_threshold_high),
            event_threshold=float(self.manual_form.event_threshold),
            event_dist_min_ms=float(self.manual_form.event_dist_min_ms),
            event_dist_max_ms=str(self.manual_form.event_dist_max_ms),
        )

    def load_state(self, state: TrainPredictionSectionState) -> None:
        self.settings_form.syllable_postprocessor = state.syllable_postprocessor
        self.manual_form.fill_gap_ms = state.fill_gap_ms
        self.manual_form.min_syllable_ms = state.min_syllable_ms
        self.manual_form.segment_threshold_low = state.segment_threshold_low
        self.manual_form.segment_threshold_high = state.segment_threshold_high
        self.manual_form.event_threshold = state.event_threshold
        self.manual_form.event_dist_min_ms = state.event_dist_min_ms
        self.manual_form.event_dist_max_ms = state.event_dist_max_ms


class ModelSectionWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.model_class = QComboBox()
        self.model_class.addItems(MODEL_CLASS_LABELS.values())
        _set_tooltip(self.model_class, "Top-level model class.")
        self.initial_model_field = InitialModelField()
        self.chunking_form = ChunkingSectionForm()
        self.frontend_type = QComboBox()
        self.frontend_type.addItems(DAS_FRONTEND_OPTIONS)
        self.encoder_type = QComboBox()
        self.encoder_type.addItems(DAS_ENCODER_OPTIONS)
        self.decoder_type = QComboBox()
        self.decoder_type.addItems(DAS_DECODER_OPTIONS)
        self.encoder_settings_form = EncoderSettingsForm()
        self.stft_frontend_form = STFTFrontendForm()
        self.mel_frontend_form = MelFrontendForm()
        self.conv_frontend_form = ConvFrontendForm()
        self.whisperseg_frontend_form = WhisperSegFrontendForm()
        self.conformer_encoder_form = ConformerEncoderForm()
        self.tcn_encoder_form = TCNEncoderForm()
        self.tweetynet_encoder_form = TweetynetEncoderForm()
        self.lstm_decoder_form = LSTMDecoderForm()
        self.conv_decoder_form = ConvDecoderForm()
        self.attention_decoder_form = AttentionDecoderForm()
        self.whisperseg_decoder_form = WhisperSegDecoderForm()
        self.hyperparameters_form = ModelHyperparametersForm()
        self.postprocessing_widget = TrainPostprocessingSectionWidget()

        _set_form_tooltips(
            self.chunking_form,
            {
                "num_time_steps": "num_time_steps",
                "chunk_stride": "chunk_stride",
                "split_within_files": "split_within_files",
            },
        )
        _set_config_tooltip(self.frontend_type, "frontend_type")
        _set_config_tooltip(self.encoder_type, "encoder_type")
        _set_config_tooltip(self.decoder_type, "decoder_type")
        _set_form_tooltips(self.encoder_settings_form, {"freeze_encoder": "freeze_encoder"})
        _set_form_tooltips(
            self.stft_frontend_form,
            {
                "num_channels": "frontend_num_channels",
                "kernel_size": "frontend_kernel_size",
                "hop_seconds": "frontend_hop_seconds",
                "fmin": "frontend_fmin",
                "fmax": "frontend_fmax",
                "trainable": "frontend_trainable",
            },
        )
        _set_form_tooltips(
            self.mel_frontend_form,
            {
                "num_channels": "frontend_num_channels",
                "kernel_size": "frontend_kernel_size",
                "hop_seconds": "frontend_hop_seconds",
                "fmin": "frontend_fmin",
                "fmax": "frontend_fmax",
                "trainable": "frontend_trainable",
            },
        )
        _set_form_tooltips(
            self.conv_frontend_form,
            {
                "num_channels": "frontend_num_channels",
                "kernel_size": "frontend_kernel_size",
                "hop_seconds": "frontend_hop_seconds",
                "pad_mode": "frontend_pad_mode",
            },
        )
        _set_form_tooltips(
            self.whisperseg_frontend_form,
            {
                "min_frequency": "min_frequency",
                "frequency_scale": "frequency_scale",
                "spec_time_step": "spec_time_step",
            },
        )
        _set_form_tooltips(
            self.conformer_encoder_form,
            {
                "num_heads": "encoder_num_heads",
                "hidden_size": "encoder_hidden_size",
                "num_layers": "encoder_num_layers",
                "kernel_size": "encoder_kernel_size",
            },
        )
        _set_form_tooltips(
            self.tcn_encoder_form,
            {
                "hidden_size": "encoder_hidden_size",
                "num_layers": "encoder_num_layers",
                "dilations": "encoder_dilations",
                "kernel_size": "encoder_kernel_size",
                "dropout": "encoder_dropout",
                "use_skip_connections": "encoder_use_skip_connections",
                "use_separable": "encoder_use_separable",
                "padding": "encoder_padding",
            },
        )
        _set_form_tooltips(
            self.tweetynet_encoder_form,
            {
                "hidden_size": "encoder_hidden_size",
                "num_layers": "encoder_num_layers",
                "kernel_size": "encoder_kernel_size",
                "dropout": "encoder_dropout",
            },
        )
        _set_form_tooltips(self.lstm_decoder_form, {"hidden_size": "decoder_hidden_size"})
        _set_form_tooltips(self.conv_decoder_form, {"kernel_size": "decoder_kernel_size"})
        _set_form_tooltips(
            self.attention_decoder_form,
            {
                "num_heads": "decoder_num_heads",
                "num_layers": "decoder_num_layers",
                "dropout": "decoder_dropout",
            },
        )
        _set_form_tooltips(
            self.whisperseg_decoder_form,
            {
                "decoder_dropout": "decoder_dropout",
                "max_length": "max_length",
                "generation_max_length": "generation_max_length",
                "num_trials": "num_trials",
                "num_beams": "num_beams",
                "top_k": "top_k",
                "top_p": "top_p",
                "length_penalty": "length_penalty",
            },
        )
        _set_form_tooltips(
            self.hyperparameters_form,
            {
                "cross_entropy_weight": "cross_entropy_weight",
                "learning_rate": "learning_rate",
                "early_stopping": "early_stopping",
                "early_stopping_patience": "early_stopping_patience",
                "reduce_lr": "reduce_lr",
                "reduce_lr_patience": "reduce_lr_patience",
                "reduce_lr_factor": "reduce_lr_factor",
                "reduce_lr_min": "reduce_lr_min",
                "linear_lr_schedule": "linear_lr_schedule",
                "weight_decay": "weight_decay",
                "warmup_steps": "warmup_steps",
            },
        )

        _set_placeholder(self.chunking_form.gui.chunk_stride, "default overlap behavior")
        _set_placeholder(self.stft_frontend_form.gui.fmax, "Nyquist frequency")
        _set_placeholder(self.mel_frontend_form.gui.fmax, "derived from sample rate")
        _set_placeholder(self.whisperseg_frontend_form.gui.min_frequency, "0")
        _set_placeholder(self.whisperseg_frontend_form.gui.frequency_scale, "1.0")
        _set_placeholder(self.whisperseg_frontend_form.gui.spec_time_step, "computed/model default")
        _set_placeholder(self.tcn_encoder_form.gui.dilations, "1, 2, 4, 8, 16")
        _set_placeholder(self.tcn_encoder_form.gui.use_separable, "false or true, false")
        _set_placeholder(self.hyperparameters_form.gui.weight_decay, "0.01")
        _hide_magicgui_field(self.hyperparameters_form.gui.reduce_lr_min)

        self.raw_frontend_placeholder = QWidget()
        raw_frontend_placeholder_layout = QVBoxLayout(self.raw_frontend_placeholder)
        raw_frontend_placeholder_layout.setContentsMargins(0, 0, 0, 0)
        raw_frontend_message = QLabel(
            "Raw waveform frontend uses the dataset channel count automatically."
        )
        raw_frontend_message.setWordWrap(True)
        raw_frontend_placeholder_layout.addWidget(raw_frontend_message)
        raw_frontend_placeholder_layout.addStretch(1)

        self.frontend_stack = QStackedWidget()
        self.frontend_stack.addWidget(self.raw_frontend_placeholder)
        self.frontend_stack.addWidget(self.stft_frontend_form.gui.native)
        self.frontend_stack.addWidget(self.mel_frontend_form.gui.native)
        self.frontend_stack.addWidget(self.conv_frontend_form.gui.native)

        self.encoder_stack = QStackedWidget()
        self.encoder_stack.addWidget(self.conformer_encoder_form.gui.native)
        self.encoder_stack.addWidget(self.tcn_encoder_form.gui.native)
        self.encoder_stack.addWidget(self.tweetynet_encoder_form.gui.native)

        self.decoder_stack = QStackedWidget()
        self.decoder_stack.addWidget(QWidget())
        self.decoder_stack.addWidget(self.lstm_decoder_form.gui.native)
        self.decoder_stack.addWidget(self.conv_decoder_form.gui.native)
        self.decoder_stack.addWidget(self.attention_decoder_form.gui.native)

        for form in (
            self.chunking_form,
            self.encoder_settings_form,
            self.stft_frontend_form,
            self.mel_frontend_form,
            self.conv_frontend_form,
            self.whisperseg_frontend_form,
            self.conformer_encoder_form,
            self.tcn_encoder_form,
            self.tweetynet_encoder_form,
            self.lstm_decoder_form,
            self.conv_decoder_form,
            self.attention_decoder_form,
            self.whisperseg_decoder_form,
            self.hyperparameters_form,
        ):
            _compact_magicgui_form(form)

        frontend_column = QWidget()
        frontend_layout = QVBoxLayout(frontend_column)
        frontend_layout.setSpacing(4)
        frontend_layout.addWidget(self.frontend_type)
        frontend_layout.addWidget(self.frontend_stack)

        encoder_column = QWidget()
        encoder_layout = QVBoxLayout(encoder_column)
        encoder_layout.setSpacing(4)
        encoder_layout.addWidget(self.encoder_type)
        encoder_layout.addWidget(self.encoder_stack)

        decoder_column = QWidget()
        decoder_layout = QVBoxLayout(decoder_column)
        decoder_layout.setSpacing(4)
        decoder_layout.addWidget(self.decoder_type)
        decoder_layout.addWidget(self.decoder_stack)

        self.das_frontend_panel = frontend_column
        self.das_encoder_panel = encoder_column
        self.das_decoder_panel = decoder_column
        self.whisperseg_frontend_panel = self.whisperseg_frontend_form.gui.native
        self.whisperseg_decoder_panel = self.whisperseg_decoder_form.gui.native

        self.frontend_section_stack = QStackedWidget()
        self.frontend_section_stack.addWidget(self.das_frontend_panel)
        self.frontend_section_stack.addWidget(self.whisperseg_frontend_panel)
        self.decoder_section_stack = QStackedWidget()
        self.decoder_section_stack.addWidget(self.das_decoder_panel)
        self.decoder_section_stack.addWidget(self.whisperseg_decoder_panel)

        self.frontend_section = _group_box("Frontend", self.frontend_section_stack)
        self.encoder_section = _group_box("Encoder", self.das_encoder_panel)
        self.decoder_section = _group_box("Decoder", self.decoder_section_stack)
        optimization_panel = QWidget()
        optimization_layout = QVBoxLayout(optimization_panel)
        optimization_layout.setContentsMargins(0, 0, 0, 0)
        optimization_layout.addWidget(self.hyperparameters_form.gui.native)
        optimization_layout.addWidget(self.encoder_settings_form.gui.native)
        self.chunking_section = _group_box("Chunking", self.chunking_form.gui.native)
        self.optimization_section = _group_box("Optimization", optimization_panel)
        self.postprocessing_section = _group_box("Postprocessing", self.postprocessing_widget)
        self.model_sections = (
            self.chunking_section,
            self.frontend_section,
            self.encoder_section,
            self.decoder_section,
            self.optimization_section,
            self.postprocessing_section,
        )

        self.model_columns_widget = QWidget()
        self.model_columns_layout = QHBoxLayout(self.model_columns_widget)
        self.model_columns = tuple(QWidget() for _ in range(3))
        self.model_column_layouts = []
        for column in self.model_columns:
            column_layout = QVBoxLayout(column)
            column_layout.setContentsMargins(0, 0, 0, 0)
            self.model_column_layouts.append(column_layout)
            self.model_columns_layout.addWidget(column, 1)

        model_class_panel = QWidget()
        model_class_layout = QHBoxLayout(model_class_panel)
        model_class_layout.setContentsMargins(0, 0, 0, 0)
        model_class_layout.addWidget(QLabel("Model class"))
        model_class_layout.addWidget(self.model_class)
        model_class_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(model_class_panel)
        layout.addWidget(self.initial_model_field)
        layout.addWidget(self.model_columns_widget)

        self.model_class.currentTextChanged.connect(self._sync_model_class)
        self.frontend_type.currentTextChanged.connect(self._sync_stacks)
        self.encoder_type.currentTextChanged.connect(self._sync_stacks)
        self.decoder_type.currentTextChanged.connect(self._sync_stacks)
        _set_combo_value(self.frontend_type, ModelSelectionState.frontend_type)
        _set_combo_value(self.encoder_type, ModelSelectionState.encoder_type)
        _set_combo_value(self.decoder_type, ModelSelectionState.decoder_type)
        self._sync_model_class()

    def _model_class_value(self) -> str:
        return MODEL_CLASS_VALUES.get(self.model_class.currentText(), "das")

    def _set_model_class_value(self, value: str) -> None:
        _set_combo_value(self.model_class, MODEL_CLASS_LABELS.get(value, "DAS"))

    def set_model_class(self, value: str) -> None:
        self._set_model_class_value(value)
        self._sync_model_class()

    def _set_model_section_columns(self, columns: tuple[tuple[QGroupBox, ...], ...]) -> None:
        for column_layout in self.model_column_layouts:
            while column_layout.count():
                item = column_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
        for section in self.model_sections:
            section.hide()
        for column_layout, sections in zip(self.model_column_layouts, columns):
            for section in sections:
                section.show()
                column_layout.addWidget(section)
            column_layout.addStretch(1)

    def _sync_model_class(self, *_args) -> None:
        self.initial_model_field.set_model_class(self._model_class_value())
        self.initial_model_field.setVisible(self._model_class_value() == "whisperseg")
        if self._model_class_value() == "whisperseg":
            self.frontend_section_stack.setCurrentWidget(self.whisperseg_frontend_panel)
            self.decoder_section_stack.setCurrentWidget(self.whisperseg_decoder_panel)
            self._set_model_section_columns(
                (
                    (self.frontend_section,),
                    (self.decoder_section,),
                    (self.optimization_section,),
                )
            )
        else:
            if self.frontend_type.currentText() not in DAS_FRONTEND_OPTIONS:
                _set_combo_value(self.frontend_type, "mel")
            if self.encoder_type.currentText() not in DAS_ENCODER_OPTIONS:
                _set_combo_value(self.encoder_type, "conformer")
            if self.decoder_type.currentText() not in DAS_DECODER_OPTIONS:
                _set_combo_value(self.decoder_type, "linear")
            self.frontend_section_stack.setCurrentWidget(self.das_frontend_panel)
            self.decoder_section_stack.setCurrentWidget(self.das_decoder_panel)
            self._set_model_section_columns(
                (
                    (self.chunking_section, self.frontend_section),
                    (self.encoder_section,),
                    (self.decoder_section, self.optimization_section, self.postprocessing_section),
                )
            )

        self._sync_stacks()

    def _sync_stacks(self, *_args) -> None:
        frontend_type = self.frontend_type.currentText()
        frontend_index = DAS_FRONTEND_OPTIONS.index(frontend_type)
        self.frontend_stack.setCurrentIndex(3 if frontend_type == "conv_resnet" else frontend_index)
        self.encoder_stack.setCurrentIndex(DAS_ENCODER_OPTIONS.index(self.encoder_type.currentText()))
        self.decoder_stack.setCurrentIndex(DAS_DECODER_OPTIONS.index(self.decoder_type.currentText()))
        whisperseg_training_enabled = self._model_class_value() == "whisperseg"
        _set_magicgui_field_visible(self.encoder_settings_form.gui.freeze_encoder, whisperseg_training_enabled)
        if (
            whisperseg_training_enabled
            and self.hyperparameters_form.reduce_lr
            and self.hyperparameters_form.linear_lr_schedule
        ):
            self.hyperparameters_form.reduce_lr = False
        _set_magicgui_field_visible(
            self.hyperparameters_form.gui.cross_entropy_weight,
            not whisperseg_training_enabled,
        )
        self.postprocessing_section.setVisible(not whisperseg_training_enabled)
        for field_name in ("linear_lr_schedule", "weight_decay", "warmup_steps"):
            widget = getattr(self.hyperparameters_form.gui, field_name, None)
            if widget is not None:
                _set_magicgui_field_visible(widget, whisperseg_training_enabled)
                _set_magicgui_field_enabled(widget, whisperseg_training_enabled)

    def load_state(self, state: TrainGuiState) -> None:
        model_class = state.model_selection.model_class
        if "whisperseg" in {
            state.model_selection.frontend_type,
            state.model_selection.encoder_type,
            state.model_selection.decoder_type,
        }:
            model_class = "whisperseg"
        self._set_model_class_value(model_class)
        if state.model_selection.frontend_type in DAS_FRONTEND_OPTIONS:
            _set_combo_value(self.frontend_type, state.model_selection.frontend_type)
        if state.model_selection.encoder_type in DAS_ENCODER_OPTIONS:
            _set_combo_value(self.encoder_type, state.model_selection.encoder_type)
        if state.model_selection.decoder_type in DAS_DECODER_OPTIONS:
            _set_combo_value(self.decoder_type, state.model_selection.decoder_type)
        _assign_to_form(self.encoder_settings_form, state.encoder_settings)
        _assign_to_form(self.stft_frontend_form, state.stft_frontend)
        _assign_to_form(self.mel_frontend_form, state.mel_frontend)
        _assign_to_form(self.conv_frontend_form, state.conv_frontend)
        _assign_to_form(self.whisperseg_frontend_form, state.whisperseg_frontend)
        _assign_to_form(self.conformer_encoder_form, state.conformer_encoder)
        _assign_to_form(self.tcn_encoder_form, state.tcn_encoder)
        _assign_to_form(self.tweetynet_encoder_form, state.tweetynet_encoder)
        _assign_to_form(self.lstm_decoder_form, state.lstm_decoder)
        _assign_to_form(self.conv_decoder_form, state.conv_decoder)
        _assign_to_form(self.attention_decoder_form, state.attention_decoder)
        _assign_to_form(self.whisperseg_decoder_form, state.whisperseg_decoder)
        _assign_to_form(self.hyperparameters_form, state.model_hyperparameters)
        _assign_to_form(self.chunking_form, state.data)
        self.postprocessing_widget.load_state(state.prediction)
        self._sync_model_class()
        self.initial_model = state.paths.initial_model

    @property
    def initial_model(self) -> str:
        return self.initial_model_field.value

    @initial_model.setter
    def initial_model(self, value: str) -> None:
        self.initial_model_field.value = value

    def selection_state(self) -> ModelSelectionState:
        if self._model_class_value() == "whisperseg":
            return ModelSelectionState(
                model_class="whisperseg",
                frontend_type="whisperseg",
                encoder_type="whisperseg",
                decoder_type="whisperseg",
            )
        return ModelSelectionState(
            model_class="das",
            frontend_type=self.frontend_type.currentText(),
            encoder_type=self.encoder_type.currentText(),
            decoder_type=self.decoder_type.currentText(),
        )

    def encoder_settings_state(self) -> EncoderSettingsState:
        return _copy_from_form(EncoderSettingsState, self.encoder_settings_form)

    def stft_frontend_state(self) -> STFTFrontendState:
        return STFTFrontendState(
            num_channels=self.stft_frontend_form.num_channels,
            kernel_size=self.stft_frontend_form.kernel_size,
            hop_seconds=float(self.stft_frontend_form.hop_seconds),
            fmin=self.stft_frontend_form.fmin,
            fmax=self.stft_frontend_form.fmax,
            trainable=self.stft_frontend_form.trainable,
        )

    def mel_frontend_state(self) -> MelFrontendState:
        return MelFrontendState(
            num_channels=self.mel_frontend_form.num_channels,
            kernel_size=self.mel_frontend_form.kernel_size,
            hop_seconds=float(self.mel_frontend_form.hop_seconds),
            fmin=self.mel_frontend_form.fmin,
            fmax=self.mel_frontend_form.fmax,
            trainable=self.mel_frontend_form.trainable,
        )

    def conv_frontend_state(self) -> ConvFrontendState:
        return ConvFrontendState(
            num_channels=self.conv_frontend_form.num_channels,
            kernel_size=self.conv_frontend_form.kernel_size,
            hop_seconds=float(self.conv_frontend_form.hop_seconds),
            pad_mode=self.conv_frontend_form.pad_mode,
        )

    def whisperseg_frontend_state(self) -> WhisperSegFrontendState:
        return _copy_from_form(WhisperSegFrontendState, self.whisperseg_frontend_form)

    def conformer_encoder_state(self) -> ConformerEncoderState:
        return _copy_from_form(ConformerEncoderState, self.conformer_encoder_form)

    def tcn_encoder_state(self) -> TCNEncoderState:
        return _copy_from_form(TCNEncoderState, self.tcn_encoder_form)

    def tweetynet_encoder_state(self) -> TweetynetEncoderState:
        return _copy_from_form(TweetynetEncoderState, self.tweetynet_encoder_form)

    def lstm_decoder_state(self) -> LSTMDecoderState:
        return _copy_from_form(LSTMDecoderState, self.lstm_decoder_form)

    def conv_decoder_state(self) -> ConvDecoderState:
        return _copy_from_form(ConvDecoderState, self.conv_decoder_form)

    def attention_decoder_state(self) -> AttentionDecoderState:
        return _copy_from_form(AttentionDecoderState, self.attention_decoder_form)

    def whisperseg_decoder_state(self) -> WhisperSegDecoderState:
        return _copy_from_form(WhisperSegDecoderState, self.whisperseg_decoder_form)

    def hyperparameter_state(self) -> ModelHyperparametersState:
        return ModelHyperparametersState(
            cross_entropy_weight=self.hyperparameters_form.cross_entropy_weight,
            learning_rate=float(self.hyperparameters_form.learning_rate),
            early_stopping=self.hyperparameters_form.early_stopping,
            early_stopping_patience=self.hyperparameters_form.early_stopping_patience,
            reduce_lr=self.hyperparameters_form.reduce_lr,
            reduce_lr_patience=self.hyperparameters_form.reduce_lr_patience,
            reduce_lr_factor=float(self.hyperparameters_form.reduce_lr_factor),
            reduce_lr_min=float(self.hyperparameters_form.reduce_lr_min),
            linear_lr_schedule=self.hyperparameters_form.linear_lr_schedule,
            weight_decay=float(self.hyperparameters_form.weight_decay),
            warmup_steps=self.hyperparameters_form.warmup_steps,
        )


class TrainPredictionSectionWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.output_form = TrainPredictionOutputForm()
        self.predict_after_train_combo = QComboBox()
        self.predict_after_train_combo.addItems(PREDICT_AFTER_TRAIN_OPTIONS)
        self.predict_after_train_label = QLabel("Predict after training")
        self.time_range_widget = CurrentAudioWidget()
        self.file_or_folder_field = PathField(
            label="Audio file or folder",
            selection_mode="file_or_directory",
            placeholder="/path/to/audio-file-or-folder",
            dialog_filter="Audio files (*.wav *.flac *.aiff *.aif *.h5 *.hdf5 *.mat *.zarr *.npz *.npy *.mmap);;All files (*)",
        )
        self.file_or_folder_field.label.hide()

        _set_form_tooltips(
            self.output_form,
            {"output_suffix": "output_suffix"},
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.output_form.gui.native)
        layout.addWidget(self._predict_after_train_panel())

        self.predict_after_train_combo.currentTextChanged.connect(self._sync_predict_after_train_options)
        self._sync_predict_after_train_options()

    def _predict_after_train_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.predict_after_train_label)
        layout.addWidget(self.predict_after_train_combo)
        layout.addWidget(self.time_range_widget)
        layout.addWidget(self.file_or_folder_field)
        return container

    def _sync_predict_after_train_options(self, *_args) -> None:
        mode = self.predict_after_train_mode()
        self.time_range_widget.setVisible(mode == PREDICT_AFTER_TRAIN_CURRENT_FILE)
        self.file_or_folder_field.setVisible(mode == PREDICT_AFTER_TRAIN_FILE_OR_FOLDER)

    def predict_after_train_mode(self) -> str:
        return self.predict_after_train_combo.currentText()

    def predict_after_train_time_range(self) -> tuple[float, float | None]:
        return self.time_range_widget.request()

    def predict_after_train_file_or_folder(self) -> str:
        return self.file_or_folder_field.value

    def set_time_range_default_end(self, duration_seconds: float | None) -> None:
        if duration_seconds is None:
            return
        if not self.time_range_widget.stop_seconds_field.text().strip():
            self.time_range_widget.stop_seconds_field.setText(f"{float(duration_seconds):g}")

    def set_file_or_folder_default(self, path: str) -> None:
        if path and not self.file_or_folder_field.value:
            self.file_or_folder_field.value = path

    @property
    def output_suffix(self) -> str:
        return str(self.output_form.output_suffix)

    def load_state(self, state: TrainPredictionSectionState) -> None:
        self.output_form.output_suffix = state.output_suffix


class TrainCommandWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.paths_form = TrainPathsWidget()
        self.data_form = DataSectionForm()
        self.include_labels_widget = IncludeLabelsWidget()
        self.model_section = ModelSectionWidget()
        self.model_class = self.model_section.model_class
        self.trainer_form = TrainerSectionForm()
        self.prediction_form = TrainPredictionSectionWidget()
        self.evaluation_form = EvaluationSectionForm()
        self._training_values_model_class = self._model_class_value()
        self._training_values_by_model_class = {
            self._training_values_model_class: self._training_values(),
        }
        self.model_class.currentTextChanged.connect(self._sync_model_class_training_values)

        _set_form_tooltips(
            self.data_form,
            {
                "batch_size": "batch_size",
                "num_workers": "num_workers",
                "validation_fraction": "validation_fraction",
                "test_fraction": "test_fraction",
                "ignore_class_names": "ignore_class_names",
            },
        )
        _set_form_tooltips(
            self.trainer_form,
            {
                "seed": "seed",
                "accelerator": "accelerator",
                "num_devices": "num_devices",
                "num_epochs": "num_epochs",
                "max_num_steps_per_epoch": "max_num_steps_per_epoch",
            },
        )
        _set_form_tooltips(self.evaluation_form, {"syllable_tolerance_ms": "syllable_tolerance_ms"})

        _hide_magicgui_field(self.trainer_form.gui.seed)
        _hide_magicgui_field(self.trainer_form.gui.num_devices)
        _set_placeholder(self.trainer_form.gui.max_num_steps_per_epoch, "full epoch")
        for signal in (
            self.paths_form.data_dir_field.line_edit.textChanged,
            self.paths_form.hf_dataset_combo.currentTextChanged,
            self.paths_form.data_source_combo.currentTextChanged,
        ):
            signal.connect(self._refresh_include_labels)
            signal.connect(self._refresh_predict_after_train_defaults)
        self._refresh_include_labels()
        self._refresh_predict_after_train_defaults()

        self.data_panel = QWidget()
        data_layout = QVBoxLayout(self.data_panel)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.addWidget(self.data_form.gui.native)
        data_layout.addWidget(self.include_labels_widget)

        prediction_evaluation_panel = QWidget()
        prediction_evaluation_layout = QVBoxLayout(prediction_evaluation_panel)
        prediction_evaluation_layout.setContentsMargins(0, 0, 0, 0)
        prediction_evaluation_layout.addWidget(self.prediction_form)
        prediction_evaluation_layout.addWidget(self.evaluation_form.gui.native)
        prediction_evaluation_layout.addStretch(1)

        self.setup_panel = _three_column_sections(
            (
                ("Paths", self.paths_form),
                ("Execution", self.trainer_form.gui.native),
            ),
            (("Data", self.data_panel),),
            (("Prediction and Evaluation", prediction_evaluation_panel),),
        )

        self.tabs = QTabWidget()
        self.tabs.addTab(_wrap_widget(self.setup_panel), "Setup")
        self.tabs.addTab(_wrap_widget(self.model_section), "Model")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

    def _model_class_value(self) -> str:
        return MODEL_CLASS_VALUES.get(self.model_class.currentText(), "das")

    def _set_model_class_value(self, value: str) -> None:
        _set_combo_value(self.model_class, MODEL_CLASS_LABELS.get(value, "DAS"))

    def _training_values(self) -> dict[str, int | float]:
        return {
            "batch_size": int(self.data_form.batch_size),
            "validation_fraction": float(self.data_form.validation_fraction),
            "test_fraction": float(self.data_form.test_fraction),
            "encoder_dropout": float(self.model_section.tcn_encoder_form.dropout),
            "decoder_dropout": float(self.model_section.whisperseg_decoder_form.decoder_dropout),
            "learning_rate": float(self.model_section.hyperparameters_form.learning_rate),
            "early_stopping_patience": int(self.model_section.hyperparameters_form.early_stopping_patience),
            "num_epochs": int(self.trainer_form.num_epochs),
        }

    def _sync_model_class_training_values(self, *_args) -> None:
        model_class = self._model_class_value()
        if model_class == self._training_values_model_class:
            return
        self._training_values_by_model_class[self._training_values_model_class] = self._training_values()
        values = self._training_values_by_model_class.get(model_class)
        if values is None:
            defaults = Config(encoder_type="whisperseg" if model_class == "whisperseg" else "conformer")
            values = {name: getattr(defaults, name) for name in self._training_values()}
        self.data_form.batch_size = values["batch_size"]
        self.data_form.validation_fraction = values["validation_fraction"]
        self.data_form.test_fraction = values["test_fraction"]
        self.model_section.tcn_encoder_form.dropout = values["encoder_dropout"]
        self.model_section.whisperseg_decoder_form.decoder_dropout = values["decoder_dropout"]
        self.model_section.hyperparameters_form.learning_rate = values["learning_rate"]
        self.model_section.hyperparameters_form.early_stopping_patience = values["early_stopping_patience"]
        self.trainer_form.num_epochs = values["num_epochs"]
        self._training_values_model_class = model_class

    def _refresh_include_labels(self, *_args, selected_labels: list[str] | None = None) -> None:
        self.include_labels_widget.load_from_data_dir(
            self.paths_form.data_dir,
            selected_labels=selected_labels,
        )

    def _refresh_predict_after_train_defaults(self, *_args) -> None:
        self.prediction_form.set_file_or_folder_default(self.paths_form.data_dir)

    def state(self) -> TrainGuiState:
        data_state = _copy_from_form(DataSectionState, self.data_form)
        data_state.num_time_steps = int(self.model_section.chunking_form.num_time_steps)
        data_state.chunk_stride = str(self.model_section.chunking_form.chunk_stride)
        data_state.split_within_files = bool(self.model_section.chunking_form.split_within_files)
        data_state.include_labels = self.include_labels_widget.include_labels()
        paths_state = _copy_from_form(TrainPathsState, self.paths_form)
        paths_state.initial_model = self.model_section.initial_model
        return TrainGuiState(
            paths=paths_state,
            data=data_state,
            model_selection=self.model_section.selection_state(),
            encoder_settings=self.model_section.encoder_settings_state(),
            stft_frontend=self.model_section.stft_frontend_state(),
            mel_frontend=self.model_section.mel_frontend_state(),
            conv_frontend=self.model_section.conv_frontend_state(),
            whisperseg_frontend=self.model_section.whisperseg_frontend_state(),
            conformer_encoder=self.model_section.conformer_encoder_state(),
            tcn_encoder=self.model_section.tcn_encoder_state(),
            tweetynet_encoder=self.model_section.tweetynet_encoder_state(),
            lstm_decoder=self.model_section.lstm_decoder_state(),
            conv_decoder=self.model_section.conv_decoder_state(),
            attention_decoder=self.model_section.attention_decoder_state(),
            whisperseg_decoder=self.model_section.whisperseg_decoder_state(),
            model_hyperparameters=self.model_section.hyperparameter_state(),
            trainer=_copy_from_form(TrainerSectionState, self.trainer_form),
            prediction=self.model_section.postprocessing_widget.state(
                output_suffix=self.prediction_form.output_suffix
            ),
            evaluation=_copy_from_form(EvaluationSectionState, self.evaluation_form),
        )

    def load_state(self, state: TrainGuiState) -> None:
        model_class = state.model_selection.model_class
        if "whisperseg" in {
            state.model_selection.frontend_type,
            state.model_selection.encoder_type,
            state.model_selection.decoder_type,
        }:
            model_class = "whisperseg"
        self._set_model_class_value(model_class)
        self.model_section.load_state(state)
        _assign_to_form(self.paths_form, state.paths)
        _assign_to_form(self.data_form, state.data)
        self._refresh_include_labels(selected_labels=state.data.include_labels)
        _assign_to_form(self.trainer_form, state.trainer)
        self.prediction_form.load_state(state.prediction)
        _assign_to_form(self.evaluation_form, state.evaluation)

    def build_config(self) -> Config:
        return train_gui_state_to_config(self.state())

    def default_save_path(self) -> str:
        config = self.build_config()
        return str(config.output_dir and f"{config.output_dir}/train-config.yaml" or "train-config.yaml")


class PredictCommandWidget(QWidget):
    def __init__(self, *, current_audio_available: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.current_audio_available = current_audio_available
        self.source_combo = QComboBox()
        if current_audio_available:
            self.source_combo.addItem("Current audio")
        self.source_combo.addItem("File or folder")
        self.current_audio_widget = CurrentAudioWidget()
        self.paths_form = PredictPathsWidget()
        self.data_form = PredictDataSectionForm()
        self.trainer_form = PredictTrainerSectionForm()
        self.runtime_form = PredictRuntimeSectionForm()
        self.prediction_form = PredictionSectionForm()
        self.postprocessing_form = PredictPostprocessingForm()
        self.evaluation_form = PredictEvaluationForm()
        self.whisperseg_decoder_form = WhisperSegDecoderForm()

        _set_form_tooltips(self.data_form, {"batch_size": "batch_size"})
        _set_form_tooltips(
            self.trainer_form,
            {
                "seed": "seed",
                "accelerator": "accelerator",
            },
        )
        _set_form_tooltips(
            self.runtime_form,
            {
                "num_workers": "num_workers",
                "num_devices": "num_devices",
            },
        )
        _set_form_tooltips(
            self.prediction_form,
            {
                "output_suffix": "output_suffix",
                "existing_annotations": "existing_annotations",
            },
        )
        _set_form_tooltips(
            self.postprocessing_form,
            {name: name for name in (
                "fill_gap_ms", "min_syllable_ms", "syllable_postprocessor", "segment_threshold_low",
                "segment_threshold_high", "event_threshold", "event_dist_min_ms", "event_dist_max_ms",
            )},
        )
        _set_form_tooltips(self.evaluation_form, {
            "evaluate": "evaluate", "split": "split", "syllable_tolerance_ms": "syllable_tolerance_ms",
        })
        _set_form_tooltips(
            self.whisperseg_decoder_form,
            {
                "generation_max_length": "generation_max_length",
                "num_trials": "num_trials",
                "num_beams": "num_beams",
                "top_k": "top_k",
                "top_p": "top_p",
                "length_penalty": "length_penalty",
            },
        )
        _hide_magicgui_field(self.whisperseg_decoder_form.gui.decoder_dropout)
        _hide_magicgui_field(self.whisperseg_decoder_form.gui.max_length)
        _compact_magicgui_form(self.whisperseg_decoder_form)
        _set_placeholder(self.trainer_form.gui.seed, "none")
        _set_placeholder(self.runtime_form.gui.num_devices, "auto")

        self.das_model_panel = _stacked_sections(
            ("Runtime", self.runtime_form.gui.native),
            ("Postprocessing", self.postprocessing_form.gui.native),
        )
        self.whisperseg_model_panel = _stacked_sections(
            ("Generation", self.whisperseg_decoder_form.gui.native),
        )
        self.model_stack = QStackedWidget()
        self.model_stack.addWidget(self.das_model_panel)
        self.model_stack.addWidget(self.whisperseg_model_panel)
        self.model_class_label = QLabel("DAS")
        model_header = QWidget()
        model_header_layout = QHBoxLayout(model_header)
        model_header_layout.setContentsMargins(0, 0, 0, 0)
        model_header_layout.addWidget(QLabel("Model class"))
        model_header_layout.addWidget(self.model_class_label)
        model_header_layout.addStretch(1)
        self.model_panel = QWidget()
        model_layout = QVBoxLayout(self.model_panel)
        model_layout.addWidget(model_header)
        model_layout.addWidget(self.model_stack)

        self.content_panel = _three_column_sections(
            (
                ("Source", self._source_panel()),
                ("Paths", self.paths_form),
                ("Execution", self.trainer_form.gui.native),
            ),
            (
                ("Data", self.data_form.gui.native),
                ("Model", self.model_panel),
            ),
            (("Output", self.prediction_form.gui.native), ("Evaluation", self.evaluation_form.gui.native)),
        )
        layout = QVBoxLayout(self)
        layout.addWidget(_wrap_widget(self.content_panel))

        self.source_combo.currentTextChanged.connect(self._sync_prediction_source)
        self.paths_form.checkpoint_field.changed.connect(self._sync_model_class)
        self.paths_form.checkpoint_field.changed.connect(self._apply_checkpoint_defaults)
        self._sync_prediction_source()
        self._sync_model_class()

    def _apply_checkpoint_defaults(self, *_args) -> None:
        try:
            predict_metadata = _checkpoint_predict_metadata(self.paths_form.checkpoint)
        except Exception:
            logging.debug("Failed to load predict defaults from checkpoint.", exc_info=True)
            return

        field_targets = (
            ("batch_size", self.data_form, int),
            ("num_trials", self.whisperseg_decoder_form, int),
            ("num_beams", self.whisperseg_decoder_form, int),
            ("fill_gap_ms", self.postprocessing_form, float),
            ("min_syllable_ms", self.postprocessing_form, float),
            ("event_dist_min_ms", self.postprocessing_form, float),
            ("segment_threshold_low", self.postprocessing_form, float),
            ("segment_threshold_high", self.postprocessing_form, float),
            ("event_threshold", self.postprocessing_form, float),
        )
        for metadata_key, form, cast in field_targets:
            value = predict_metadata.get(metadata_key)
            if value is not None:
                setattr(form, metadata_key, cast(value))

    def _source_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.source_combo)
        layout.addWidget(self.current_audio_widget)
        layout.addStretch(1)
        return container

    def _sync_prediction_source(self, *_args) -> None:
        current_audio = self.uses_current_audio()
        self.current_audio_widget.setVisible(current_audio)
        self.paths_form.set_folder_fields_enabled(not current_audio)

    def _sync_model_class(self, *_args) -> None:
        whisperseg = _is_whisperseg_predict_source(self.paths_form.checkpoint)
        self.model_class_label.setText("WhisperSeg" if whisperseg else "DAS")
        self.model_stack.setCurrentWidget(
            self.whisperseg_model_panel if whisperseg else self.das_model_panel
        )

    def uses_current_audio(self) -> bool:
        return self.source_combo.currentText() == "Current audio"

    def current_audio_request(self) -> tuple[float, float | None]:
        return self.current_audio_widget.request()

    def state(self) -> PredictGuiState:
        data_state = _copy_from_form(PredictDataSectionState, self.data_form)
        data_state.num_workers = int(self.runtime_form.num_workers)
        trainer_state = _copy_from_form(PredictTrainerSectionState, self.trainer_form)
        trainer_state.num_devices = str(self.runtime_form.num_devices)
        prediction_state = _copy_from_form(PredictionSectionState, self.prediction_form)
        prediction_state.fill_gap_ms = float(self.postprocessing_form.fill_gap_ms)
        prediction_state.min_syllable_ms = float(self.postprocessing_form.min_syllable_ms)
        prediction_state.syllable_postprocessor = str(self.postprocessing_form.syllable_postprocessor)
        prediction_state.segment_threshold_low = float(self.postprocessing_form.segment_threshold_low)
        prediction_state.segment_threshold_high = float(self.postprocessing_form.segment_threshold_high)
        prediction_state.event_threshold = float(self.postprocessing_form.event_threshold)
        prediction_state.event_dist_min_ms = float(self.postprocessing_form.event_dist_min_ms)
        prediction_state.event_dist_max_ms = str(self.postprocessing_form.event_dist_max_ms)
        prediction_state.evaluate = bool(self.evaluation_form.evaluate)
        prediction_state.split = str(self.evaluation_form.split)
        prediction_state.syllable_tolerance_ms = float(self.evaluation_form.syllable_tolerance_ms)
        return PredictGuiState(
            paths=_copy_from_form(PredictPathsState, self.paths_form),
            data=data_state,
            trainer=trainer_state,
            prediction=prediction_state,
            whisperseg_decoder=_copy_from_form(WhisperSegDecoderState, self.whisperseg_decoder_form),
        )

    def load_state(self, state: PredictGuiState) -> None:
        _assign_to_form(self.paths_form, state.paths)
        _assign_to_form(self.data_form, state.data)
        _assign_to_form(self.runtime_form, state.data)
        _assign_to_form(self.trainer_form, state.trainer)
        _assign_to_form(self.runtime_form, state.trainer)
        _assign_to_form(self.prediction_form, state.prediction)
        _assign_to_form(self.postprocessing_form, state.prediction)
        _assign_to_form(self.evaluation_form, state.prediction)
        _assign_to_form(self.whisperseg_decoder_form, state.whisperseg_decoder)
        self._sync_model_class()

    def build_config(self) -> Config:
        return predict_gui_state_to_config(self.state())

    def default_save_path(self) -> str:
        config = self.build_config()
        return str(config.output_dir and f"{config.output_dir}/predict-config.yaml" or "predict-config.yaml")


class _SignalWriter:
    def __init__(self, emit):
        self._emit = emit

    def write(self, text: str) -> int:
        if text:
            self._emit(text)
        return len(text)

    def flush(self) -> None:
        return None


class _TeeWriter:
    def __init__(self, *streams):
        self._streams = [stream for stream in streams if stream is not None]

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            flush = getattr(stream, "flush", None)
            if flush is not None:
                flush()


class _SignalLogHandler(logging.Handler):
    def __init__(self, emit):
        super().__init__(level=logging.INFO)
        self._emit = emit
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self._emit(f"{message}\n")


class _ProcessMessageWriter:
    def __init__(self, connection):
        self._connection = connection

    def write(self, text: str) -> int:
        if text:
            self._connection.send(("log", text))
        return len(text)

    def flush(self) -> None:
        pass


def _run_training_command(
    config: Config,
    *,
    predict_after_train: dict[str, object] | None,
    stop_event,
    log_emit,
) -> dict:
    result = train(config, verbose=True, stop_event=stop_event, emit_epoch_logs=True)
    prediction_after_train = None
    if predict_after_train is not None and not bool(stop_event is not None and stop_event.is_set()):
        log_emit("Starting prediction after training...\n")
        predict_config = predict_after_train["config"].copy(checkpoint=str(result))
        predict_audio = predict_after_train.get("audio")
        if predict_audio is not None:
            predict_result = predict(
                predict_config,
                audio=predict_audio,
                samplerate=predict_after_train.get("samplerate"),
                verbose=True,
                stop_event=stop_event,
            )
        else:
            predict_result = predict(predict_config, verbose=True, stop_event=stop_event)
        prediction_after_train = {
            "result": predict_result,
            "time_offset_seconds": float(predict_after_train.get("time_offset_seconds", 0.0)),
        }
    payload = {
        "ok": True,
        "command": "train",
        "result": result,
        "time_offset_seconds": 0.0,
        "cancelled": bool(stop_event is not None and stop_event.is_set()),
    }
    if prediction_after_train is not None:
        payload["prediction_after_train"] = prediction_after_train
    return payload


def _run_command_process(
    command_name: str,
    config: Config,
    predict_after_train: dict[str, object] | None,
    audio,
    samplerate: int | None,
    time_offset_seconds: float,
    stop_event,
    output_connection,
) -> None:
    writer = _ProcessMessageWriter(output_connection)
    stdout_tee = _TeeWriter(sys.__stdout__ or sys.stdout, writer)
    stderr_tee = _TeeWriter(sys.__stderr__ or sys.stderr, writer)
    root_logger = logging.getLogger()
    gui_log_handler = _SignalLogHandler(lambda text: output_connection.send(("log", text)))
    root_logger.addHandler(gui_log_handler)
    try:
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            if command_name == "train":
                payload = _run_training_command(
                    config,
                    predict_after_train=predict_after_train,
                    stop_event=stop_event,
                    log_emit=lambda text: output_connection.send(("log", text)),
                )
            else:
                result = predict(
                    config,
                    audio=audio,
                    samplerate=samplerate,
                    verbose=True,
                    stop_event=stop_event,
                )
                payload = {
                    "ok": True,
                    "command": "predict",
                    "result": result,
                    "time_offset_seconds": time_offset_seconds,
                    "cancelled": bool(stop_event is not None and stop_event.is_set()),
                }
    except Exception:
        output_connection.send(("finished", {"ok": False, "command": command_name, "error": traceback.format_exc()}))
    else:
        output_connection.send(("finished", payload))
    finally:
        root_logger.removeHandler(gui_log_handler)
        output_connection.close()


class JobWorker(QObject):
    log = Signal(str)
    finished = Signal(dict)

    def __init__(
        self,
        command_name: str,
        config: Config,
        *,
        audio: np.ndarray | None = None,
        samplerate: int | None = None,
        time_offset_seconds: float = 0.0,
        predict_after_train: dict[str, object] | None = None,
        stop_event: threading.Event | None = None,
        cancel_event: threading.Event | None = None,
        run_train_in_process: bool = False,
    ):
        super().__init__()
        self.command_name = command_name
        self.config = config
        self.audio = audio
        self.samplerate = samplerate
        self.time_offset_seconds = time_offset_seconds
        self.predict_after_train = predict_after_train
        self.stop_event = stop_event
        self.cancel_event = cancel_event or threading.Event()
        self.run_train_in_process = run_train_in_process
        self._train_process = None
        self._cancel_requested = False

    def cancel_training_process(self) -> None:
        self._cancel_requested = True
        self.cancel_event.set()

    def _kill_training_process(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        process = self._train_process
        if process is None or not process.is_alive():
            return
        kill = getattr(process, "kill", None)
        if kill is not None:
            kill()
        else:
            process.terminate()

    def _run_train_in_process(self) -> dict:
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        self._train_process = context.Process(
            target=_run_command_process,
            args=(
                self.command_name,
                self.config,
                self.predict_after_train,
                self.audio,
                self.samplerate,
                self.time_offset_seconds,
                self.stop_event,
                child_connection,
            ),
        )
        self._train_process.start()
        child_connection.close()
        if self._cancel_requested:
            self._kill_training_process()

        payload = None
        while True:
            if self.cancel_event.is_set():
                self._cancel_requested = True
                self._kill_training_process()
            if parent_connection.poll(0.1):
                try:
                    message_type, message = parent_connection.recv()
                except EOFError:
                    break
                if message_type == "log":
                    self.log.emit(message)
                elif message_type == "finished":
                    payload = message
                    break
            elif self._train_process is not None and not self._train_process.is_alive():
                break

        if self._train_process is not None:
            self._train_process.join(timeout=1.0)
            if self._train_process.is_alive():
                self._kill_training_process()
                self._train_process.join(timeout=1.0)
            exitcode = self._train_process.exitcode
            self._train_process = None
        else:
            exitcode = None

        while True:
            if not parent_connection.poll():
                break
            try:
                message_type, message = parent_connection.recv()
            except EOFError:
                break
            if message_type == "log":
                self.log.emit(message)
            elif message_type == "finished" and payload is None:
                payload = message
        parent_connection.close()

        if payload is not None:
            return payload
        if self._cancel_requested or self.cancel_event.is_set() or bool(self.stop_event is not None and self.stop_event.is_set()):
            return {
                "ok": True,
                "command": self.command_name,
                "result": "" if self.command_name == "train" else [],
                "time_offset_seconds": self.time_offset_seconds,
                "cancelled": True,
            }
        return {
            "ok": False,
            "command": self.command_name,
            "error": f"{'Training' if self.command_name == 'train' else 'Prediction'} process exited with code {exitcode}.",
        }

    def run(self) -> None:
        writer = _SignalWriter(self.log.emit)
        stdout_tee = _TeeWriter(sys.__stdout__ or sys.stdout, writer)
        stderr_tee = _TeeWriter(sys.__stderr__ or sys.stderr, writer)
        root_logger = logging.getLogger()
        gui_log_handler = _SignalLogHandler(self.log.emit)
        root_logger.addHandler(gui_log_handler)
        try:
            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                if self.run_train_in_process and self.command_name in {"train", "predict"}:
                    payload = self._run_train_in_process()
                elif self.command_name == "train":
                    payload = _run_training_command(
                        self.config,
                        predict_after_train=self.predict_after_train,
                        stop_event=self.stop_event,
                        log_emit=self.log.emit,
                    )
                elif self.audio is not None:
                    result = predict(self.config, audio=self.audio, samplerate=self.samplerate, verbose=True)
                    prediction_after_train = None
                else:
                    result = predict(self.config, verbose=True)
                    prediction_after_train = None
        except Exception:
            self.finished.emit({"ok": False, "command": self.command_name, "error": traceback.format_exc()})
        else:
            if self.command_name != "train" and not self.run_train_in_process:
                payload = {
                    "ok": True,
                    "command": self.command_name,
                    "result": result,
                    "time_offset_seconds": self.time_offset_seconds,
                    "cancelled": bool(self.stop_event is not None and self.stop_event.is_set()),
                }
                if prediction_after_train is not None:
                    payload["prediction_after_train"] = prediction_after_train
            self.finished.emit(payload)
        finally:
            root_logger.removeHandler(gui_log_handler)


class DASConformerWindow(QMainWindow):
    def __init__(
        self,
        *,
        startup_config: Config | None = None,
        initial_tab: Literal["train", "predict"] = "train",
        current_audio_provider: CurrentAudioProvider | None = None,
        current_duration_provider: CurrentDurationProvider | None = None,
        annotated_region_provider: AnnotatedRegionProvider | None = None,
        on_predictions: PredictionCallback | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.current_audio_provider = current_audio_provider
        self.current_duration_provider = current_duration_provider
        self.annotated_region_provider = annotated_region_provider
        self.on_predictions = on_predictions
        self.setWindowTitle("DAS")
        self.resize(1100, 850)

        self.command_tabs = QTabWidget()
        self.train_widget = TrainCommandWidget()
        self.predict_widget = PredictCommandWidget(current_audio_available=current_audio_provider is not None)
        self.command_tabs.addTab(self.train_widget, "Train")
        self.command_tabs.addTab(self.predict_widget, "Predict")
        self.command_tabs.currentChanged.connect(self._update_run_button)
        self.train_widget.prediction_form.predict_after_train_combo.currentTextChanged.connect(
            self._refresh_predict_after_train_duration_default
        )

        self.preset_dropdown = QComboBox()
        self.preset_dropdown.addItem("Apply Built-in Config...", None)
        for config_name, label in BUILTIN_CONFIG_OPTIONS:
            self.preset_dropdown.addItem(label, config_name)
        self.preset_dropdown.setToolTip(
            "Built-in configs loaded like YAML configs: "
            + ", ".join(config_name for config_name, _ in BUILTIN_CONFIG_OPTIONS)
        )
        self.load_button = QPushButton("Load Config")
        self.save_button = QPushButton("Save Config")
        self.stop_early_button = QPushButton("Stop early")
        self.cancel_button = QPushButton("Cancel")
        self.run_button = QPushButton()
        self._update_run_button()
        self.stop_early_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

        self.preset_dropdown.activated.connect(self.apply_selected_preset)
        self.load_button.clicked.connect(self.load_active_config)
        self.save_button.clicked.connect(self.save_active_config)
        self.stop_early_button.clicked.connect(self.stop_training_early)
        self.cancel_button.clicked.connect(self.cancel_training)
        self.run_button.clicked.connect(self.run_active_command)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.addWidget(self.preset_dropdown)
        footer_layout.addWidget(self.load_button)
        footer_layout.addWidget(self.save_button)
        footer_layout.addStretch(1)

        run_actions = QWidget()
        run_actions_layout = QHBoxLayout(run_actions)
        run_actions_layout.setContentsMargins(0, 0, 0, 0)
        run_actions_layout.addWidget(self.cancel_button)
        run_actions_layout.addWidget(self.stop_early_button)
        run_actions_layout.addWidget(self.run_button)
        footer_layout.addWidget(run_actions)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(2000)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        fixed_font.setStyleHint(fixed_font.StyleHint.Monospace)
        self.log_output.setFont(fixed_font)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.addWidget(self.command_tabs)
        central_layout.addWidget(footer)
        central_layout.addWidget(self.log_output)
        self.setCentralWidget(central)

        self._thread: QThread | None = None
        self._worker: JobWorker | None = None
        self._stop_event: threading.Event | None = None
        self._cancel_event: threading.Event | None = None
        self._running_command: str | None = None
        self._hard_cancel_requested = False

        if startup_config:
            self.load_config(startup_config)
        else:
            self.select_tab(initial_tab)
        self._refresh_predict_after_train_duration_default()

    def select_tab(self, tab: Literal["train", "predict"]) -> None:
        if tab == "train":
            self.command_tabs.setCurrentWidget(self.train_widget)
        elif tab == "predict":
            self.command_tabs.setCurrentWidget(self.predict_widget)
        else:
            raise ValueError(f"Unknown DAS tab: {tab!r}")
        self._update_run_button()

    def _append_log(self, text: str) -> None:
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        for index, part in enumerate(text.replace("\r\n", "\n").split("\r")):
            if index:
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
                cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
            cursor.insertText(part)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def _append_log_line(self, text: str) -> None:
        self._append_log(f"{text}\n")

    def _update_run_button(self, *_args) -> None:
        labels = {
            "train": "Start Training",
            "predict": "Start Prediction",
        }
        self.run_button.setText(labels[self.active_command_name()])

    def active_command_name(self) -> str:
        current = self.command_tabs.currentWidget()
        if current is self.predict_widget:
            return "predict"
        return "train"

    def active_widget(self) -> TrainCommandWidget | PredictCommandWidget:
        return self.command_tabs.currentWidget()

    def _config_for_mode(self, mode: Literal["train", "predict"]) -> Config:
        try:
            if mode == "predict":
                return self.predict_widget.build_config()
            return self.train_widget.build_config()
        except Exception:
            return Config(mode=mode)

    def _set_running(self, running: bool) -> None:
        self.command_tabs.setEnabled(not running)
        self.preset_dropdown.setEnabled(not running)
        self.load_button.setEnabled(not running)
        self.save_button.setEnabled(not running)
        self.run_button.setEnabled(not running)
        stoppable_command_running = bool(running and self._running_command in {"train", "predict"})
        self.stop_early_button.setEnabled(stoppable_command_running)
        self.cancel_button.setEnabled(stoppable_command_running)

    def _show_error(self, title: str, message: str) -> None:
        dialog = QDialog(self)
        dialog.setObjectName("error_dialog")
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        summary = QLabel(message.rstrip().splitlines()[-1])
        summary.setObjectName("error_message")
        summary.setWordWrap(True)
        trace = QPlainTextEdit(message)
        trace.setObjectName("error_trace")
        trace.setReadOnly(True)
        trace.setFixedHeight(280)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(summary)
        layout.addWidget(trace)
        layout.addWidget(buttons)
        dialog.resize(720, 400)
        dialog.exec()

    def _refresh_predict_after_train_duration_default(self, *_args) -> None:
        if self.current_duration_provider is None:
            return
        try:
            duration_seconds = float(self.current_duration_provider())
        except Exception:
            return
        self.train_widget.prediction_form.set_time_range_default_end(duration_seconds)

    def _current_duration_seconds(self) -> float:
        if self.current_duration_provider is None:
            raise ValueError("No current file duration provider is available.")
        duration_seconds = float(self.current_duration_provider())
        if duration_seconds <= 0:
            raise ValueError("Current file duration must be greater than zero.")
        return duration_seconds

    def _annotated_region_bounds(self) -> tuple[float, float]:
        if self.annotated_region_provider is None:
            raise ValueError("No annotated region provider is available.")
        regions = self.annotated_region_provider()
        if (
            isinstance(regions, tuple)
            and len(regions) == 2
            and all(isinstance(value, (int, float)) for value in regions)
        ):
            intervals = [regions]
        else:
            intervals = list(regions)
        if not intervals:
            raise ValueError("No annotated region is available.")
        starts = [float(start) for start, _stop in intervals]
        stops = [float(stop) for _start, stop in intervals]
        start_seconds = min(starts)
        stop_seconds = max(stops)
        if stop_seconds <= start_seconds:
            raise ValueError("Annotated region end must be after its start.")
        return start_seconds, stop_seconds

    def _predict_config_after_training(self, train_config: Config, *, data_dir: str, output_dir: str) -> Config:
        return train_config.copy(
            mode="predict",
            data_dir=data_dir,
            checkpoint="",
            output_dir=output_dir,
            evaluate=False,
            split=None,
        )

    def _build_predict_after_train_request(self, train_config: Config) -> dict[str, object] | None:
        prediction_widget = self.train_widget.prediction_form
        mode = prediction_widget.predict_after_train_mode()
        if mode == PREDICT_AFTER_TRAIN_NONE:
            return None

        if mode == PREDICT_AFTER_TRAIN_FILE_OR_FOLDER:
            data_dir = prediction_widget.predict_after_train_file_or_folder() or str(train_config.data_dir)
            if not data_dir:
                raise ValueError("Choose an audio file or folder for prediction after training.")
            output_dir = str(Path(train_config.output_dir).expanduser() / "predictions")
            return {"config": self._predict_config_after_training(train_config, data_dir=data_dir, output_dir=output_dir)}

        if self.current_audio_provider is None:
            raise ValueError("No current audio provider is available.")

        if mode == PREDICT_AFTER_TRAIN_CURRENT_FILE:
            start_seconds, stop_seconds = prediction_widget.predict_after_train_time_range()
        elif mode == PREDICT_AFTER_TRAIN_ANNOTATED_REGION:
            start_seconds, stop_seconds = self._annotated_region_bounds()
        else:
            raise ValueError(f"Unknown predict-after-training mode: {mode!r}")

        audio, samplerate, time_offset_seconds = self.current_audio_provider(start_seconds, stop_seconds)
        return {
            "config": self._predict_config_after_training(train_config, data_dir="", output_dir=""),
            "audio": audio,
            "samplerate": samplerate,
            "time_offset_seconds": time_offset_seconds,
        }

    def stop_training_early(self) -> None:
        if self._running_command not in {"train", "predict"} or self._stop_event is None or self._stop_event.is_set():
            return
        self._stop_event.set()
        self.stop_early_button.setEnabled(False)
        if self._running_command == "predict":
            self._append_log_line("Stop requested. Waiting for the current file to finish...")
        else:
            self._append_log_line("Stop requested. Waiting for the current training step to finish...")

    def cancel_training(self) -> None:
        if self._running_command not in {"train", "predict"} or self._thread is None or not self._thread.isRunning():
            return
        self._hard_cancel_requested = True
        self.stop_early_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        activity = "training" if self._running_command == "train" else "prediction"
        self._append_log_line(f"Cancel requested. Terminating {activity} immediately...")
        if self._cancel_event is not None:
            self._cancel_event.set()

    def _detect_config_command(self, path: str) -> str:
        payload = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
        if payload is None:
            raise ValueError(f"Config file '{path}' is empty.")
        if not isinstance(payload, dict):
            raise ValueError(f"Expected '{path}' to contain a YAML mapping.")
        mode = payload.get("mode")
        if mode in {"train", "predict"}:
            return str(mode)
        if payload.get("checkpoint"):
            return "predict"
        return self.active_command_name()

    def load_config_path(self, path: str) -> None:
        command_name = self._detect_config_command(path)
        try:
            if command_name == "train":
                self.command_tabs.setCurrentWidget(self.train_widget)
                self.train_widget.load_state(load_train_gui_state_from_yaml(path))
            else:
                self.command_tabs.setCurrentWidget(self.predict_widget)
                self.predict_widget.load_state(load_predict_gui_state_from_yaml(path))
        except Exception as exc:
            raise ValueError(str(exc)) from exc
        self._append_log_line(f"Loaded config from {path}")

    def load_config(self, config: Config) -> None:
        if config.mode == "predict":
            self.command_tabs.setCurrentWidget(self.predict_widget)
            self.predict_widget.load_state(predict_config_to_gui_state(config))
        else:
            self.command_tabs.setCurrentWidget(self.train_widget)
            self.train_widget.load_state(train_config_to_gui_state(config))

    def apply_selected_preset(self, index: int) -> None:
        preset_name = self.preset_dropdown.itemData(index)
        if preset_name is None:
            return

        mode = self.active_command_name()
        try:
            config = Config.from_config_sources([str(preset_name)], mode=mode, base=self._config_for_mode(mode))
            self.load_config(config)
        except Exception:
            self._show_error("Apply Config Failed", traceback.format_exc())
        else:
            self._append_log_line(f"Applied {self.preset_dropdown.itemText(index)} {mode} config.")
        finally:
            self.preset_dropdown.blockSignals(True)
            self.preset_dropdown.setCurrentIndex(0)
            self.preset_dropdown.blockSignals(False)

    def load_active_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Config",
            "",
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return

        try:
            self.load_config_path(path)
        except Exception:
            self._show_error("Load Config Failed", traceback.format_exc())
            return

    def save_active_config(self) -> None:
        widget = self.active_widget()
        default_path = widget.default_save_path()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config",
            default_path,
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return

        try:
            if self.active_command_name() == "train":
                config = self.train_widget.build_config()
                save_train_config(config, path)
                yaml_text = format_train_config_yaml(config)
            else:
                config = self.predict_widget.build_config()
                save_predict_config(config, path)
                yaml_text = format_predict_config_yaml(config)
        except Exception:
            self._show_error("Save Config Failed", traceback.format_exc())
            return
        self._append_log_line(f"Saved config to {path}")
        self._append_log_line(yaml_text)

    def run_active_command(self) -> None:
        audio = None
        samplerate = None
        time_offset_seconds = 0.0
        predict_after_train = None
        try:
            command_name = self.active_command_name()
            config = self.active_widget().build_config()
            if command_name == "predict" and self.predict_widget.uses_current_audio():
                if self.current_audio_provider is None:
                    raise ValueError("No current audio provider is available.")
                start_seconds, stop_seconds = self.predict_widget.current_audio_request()
                audio, samplerate, time_offset_seconds = self.current_audio_provider(start_seconds, stop_seconds)
            elif command_name == "train":
                predict_after_train = self._build_predict_after_train_request(config)
        except Exception:
            self._show_error("Invalid Configuration", traceback.format_exc())
            return

        self._append_log_line(f"Starting {command_name}...")
        self._running_command = command_name
        self._stop_event = multiprocessing.get_context("spawn").Event() if command_name in {"train", "predict"} else None
        self._cancel_event = threading.Event() if command_name in {"train", "predict"} else None
        self._hard_cancel_requested = False
        self._set_running(True)

        self._thread = QThread(self)
        self._worker = JobWorker(
            command_name,
            config,
            audio=audio,
            samplerate=samplerate,
            time_offset_seconds=time_offset_seconds,
            predict_after_train=predict_after_train,
            stop_event=self._stop_event,
            cancel_event=self._cancel_event,
            run_train_in_process=command_name in {"train", "predict"},
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._handle_job_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    def _handle_job_finished(self, payload: dict) -> None:
        if self._hard_cancel_requested:
            activity = "Training" if self._running_command == "train" else "Prediction"
            self._append_log_line(f"{activity} cancelled.")
            self._hard_cancel_requested = False
            self._set_running(False)
            return
        if payload["ok"]:
            if payload["command"] == "train":
                if payload.get("cancelled"):
                    self._append_log_line(f"Training stopped early. Checkpoint: {payload['result']}")
                else:
                    self._append_log_line(f"Training finished. Checkpoint: {payload['result']}")
                prediction_after_train = payload.get("prediction_after_train")
                if prediction_after_train is not None:
                    self._append_log_line("Prediction after training finished.")
                    self._handle_prediction_outputs(
                        prediction_after_train["result"],
                        float(prediction_after_train.get("time_offset_seconds", 0.0)),
                    )
            else:
                self._append_log_line("Prediction stopped early." if payload.get("cancelled") else "Prediction finished.")
                self._handle_prediction_outputs(
                    payload["result"],
                    float(payload.get("time_offset_seconds", 0.0)),
                )
        else:
            self._append_log_line(payload["error"])
            self._show_error("Command Failed", payload["error"])
        self._set_running(False)

    def _handle_prediction_outputs(self, outputs, time_offset_seconds: float) -> None:
        if isinstance(outputs, pd.DataFrame):
            callback_failed = False
            if self.on_predictions is not None:
                try:
                    self.on_predictions(outputs, time_offset_seconds)
                except Exception:
                    error = traceback.format_exc()
                    self._append_log_line(error)
                    self._show_error("Prediction Callback Failed", error)
                    callback_failed = True
            if not callback_failed:
                self._append_log_line(f"Returned {len(outputs)} predictions.")
        else:
            for output_path in outputs:
                self._append_log_line(str(output_path))

    def _cleanup_worker(self) -> None:
        hard_cancelled = self._hard_cancel_requested
        cancelled_command = self._running_command
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._stop_event = None
        self._cancel_event = None
        self._running_command = None
        self._hard_cancel_requested = False
        if hard_cancelled:
            activity = "Training" if cancelled_command == "train" else "Prediction"
            self._append_log_line(f"{activity} cancelled.")
            self._set_running(False)


def main(argv: list[str] | None = None, *, startup_config: Config | None = None) -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv if argv is None else argv)
    window = DASConformerWindow(startup_config=startup_config)
    window.show()
    if not owns_app:
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
