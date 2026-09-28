from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.metrics import classification_report, confusion_matrix
import torch

SYLLABLE_NONE_LABEL = "<none>"


@dataclass
class PredictionResultContext:
    class_names: list[str]
    class_types: list[str] | None
    frame_rate_hz: float
    fill_gap_seconds: float
    min_syllable_seconds: float
    tolerance_seconds: float
    event_dist_min_seconds: float = 0.0
    event_dist_max_seconds: float | None = None
    postprocessor: str = "label_aware_dense"
    segment_threshold_low: float = 0.5
    segment_threshold_high: float = 0.5
    event_threshold: float = 0.5
    legacy_data_padding: int = 0

    def __post_init__(self) -> None:
        self.class_names = [str(name) for name in self.class_names]
        self.class_types = normalize_class_types(class_names=self.class_names, class_types=self.class_types)
        self.frame_rate_hz = float(self.frame_rate_hz)
        self.fill_gap_seconds = float(self.fill_gap_seconds)
        self.min_syllable_seconds = float(self.min_syllable_seconds)
        self.tolerance_seconds = float(self.tolerance_seconds)
        self.event_dist_min_seconds = float(self.event_dist_min_seconds)
        self.event_dist_max_seconds = (
            None if self.event_dist_max_seconds is None else float(self.event_dist_max_seconds)
        )
        self.postprocessor = str(self.postprocessor)
        self.segment_threshold_low = float(self.segment_threshold_low)
        self.segment_threshold_high = float(self.segment_threshold_high)
        self.event_threshold = float(self.event_threshold)
        self.legacy_data_padding = int(self.legacy_data_padding)

@dataclass
class PredictionEvaluation:
    summary: dict[str, int]
    class_names: list[str]
    dense_matrix: np.ndarray
    dense_report: str
    syllable_matrix: np.ndarray
    syllable_report: str
    syllable_wer: float


def normalize_class_types(*, class_names: list[str], class_types) -> list[str]:
    if class_types is None:
        return ["segment"] * len(class_names)
    normalized = [str(class_type) for class_type in class_types]
    return normalized if len(normalized) == len(class_names) else ["segment"] * len(class_names)


def evaluate_file_predictions(
    *,
    dataset,
    predictions=None,
    chunk_probabilities: list[np.ndarray] | None = None,
    context: PredictionResultContext,
) -> PredictionEvaluation:
    if chunk_probabilities is None:
        chunk_probabilities = _prediction_batches_to_chunk_probabilities(predictions)
    dense_matrix, dense_report, syllable_matrix, syllable_report, syllable_wer, summary = (
        _classification_summaries_from_file_predictions(
            dataset=dataset,
            chunk_probabilities=chunk_probabilities,
            class_names=context.class_names,
            class_types=context.class_types,
            frame_rate_hz=context.frame_rate_hz,
            fill_gap_seconds=context.fill_gap_seconds,
            min_syllable_seconds=context.min_syllable_seconds,
            tolerance_seconds=context.tolerance_seconds,
            event_dist_min_seconds=context.event_dist_min_seconds,
            event_dist_max_seconds=context.event_dist_max_seconds,
            postprocessor=context.postprocessor,
            segment_threshold_low=context.segment_threshold_low,
            segment_threshold_high=context.segment_threshold_high,
            event_threshold=context.event_threshold,
        )
    )
    return PredictionEvaluation(
        summary=summary,
        class_names=context.class_names,
        dense_matrix=dense_matrix,
        dense_report=dense_report,
        syllable_matrix=syllable_matrix,
        syllable_report=syllable_report,
        syllable_wer=syllable_wer,
    )


def evaluate_supervised_predictions(
    *,
    model,
    dataloader,
    device,
    context: PredictionResultContext,
) -> PredictionEvaluation:

    dense_matrix, dense_report = _classification_summary(
        model=model,
        dataloader=dataloader,
        class_names=context.class_names,
        device=device,
    )
    syllable_matrix, syllable_report, syllable_wer = _syllable_classification_summary(
        model=model,
        dataloader=dataloader,
        class_names=context.class_names,
        device=device,
        frame_rate_hz=context.frame_rate_hz,
        fill_gap_seconds=context.fill_gap_seconds,
        min_syllable_seconds=context.min_syllable_seconds,
        tolerance_seconds=context.tolerance_seconds,
        postprocessor=context.postprocessor,
        segment_threshold_low=context.segment_threshold_low,
        segment_threshold_high=context.segment_threshold_high,
    )
    dataset = getattr(dataloader, "dataset", None)
    return PredictionEvaluation(
        summary={
            "evaluated_file_count": len(getattr(dataset, "audio_files", [])),
            "skipped_file_count": 0,
        },
        class_names=context.class_names,
        dense_matrix=dense_matrix,
        dense_report=dense_report,
        syllable_matrix=syllable_matrix,
        syllable_report=syllable_report,
        syllable_wer=syllable_wer,
    )


def write_prediction_outputs(
    *,
    output_dir: Path | None,
    dataset,
    predictions,
    context: PredictionResultContext,
    output_suffix: str,
    existing_annotations: str = "overwrite",
    annotations: list[pd.DataFrame] | None = None,
) -> list[str]:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if annotations is None:
        annotations = prediction_annotations(
            dataset=dataset,
            predictions=predictions,
            context=context,
        )

    written_files = []
    for audio_file, annotation in zip(dataset.audio_files, annotations, strict=True):
        output_path = prediction_output_path(audio_file, output_dir=output_dir, output_suffix=output_suffix)
        if existing_annotations == "skip" and output_path.exists():
            continue
        write_annotation_file(output_path, annotation, merge=existing_annotations == "merge")
        written_files.append(str(output_path))
    return written_files


def prediction_output_path(audio_file, *, output_dir: Path | None, output_suffix: str) -> Path:
    audio_path = Path(audio_file)
    return (output_dir or audio_path.parent) / f"{audio_path.stem}{output_suffix}"


def prediction_annotations(*, dataset, predictions, context: PredictionResultContext) -> list[pd.DataFrame]:
    chunk_probabilities = _prediction_batches_to_chunk_probabilities(predictions)
    annotations = [
        _probabilities_to_syllable_annotations(
            probabilities=probabilities,
            class_names=context.class_names,
            class_types=context.class_types,
            frame_rate_hz=context.frame_rate_hz,
            fill_gap_seconds=context.fill_gap_seconds,
            min_syllable_seconds=context.min_syllable_seconds,
            event_dist_min_seconds=context.event_dist_min_seconds,
            event_dist_max_seconds=context.event_dist_max_seconds,
            postprocessor=context.postprocessor,
            segment_threshold_low=context.segment_threshold_low,
            segment_threshold_high=context.segment_threshold_high,
            event_threshold=context.event_threshold,
        )
        for probabilities in _stitch_chunk_probabilities_by_file(
            dataset,
            chunk_probabilities,
            trim_chunk_frames=context.legacy_data_padding,
        )
    ]
    for index, annotation in enumerate(annotations):
        duration = dataset.nb_samples_in_file[index] / dataset.samplerate_per_file[index]
        annotation = annotation.loc[annotation.start_seconds <= duration].copy()
        annotation[["start_seconds", "stop_seconds"]] = annotation[["start_seconds", "stop_seconds"]].clip(0, duration)
        annotations[index] = annotation.reset_index(drop=True)
    return annotations


def print_evaluation_report(report: PredictionEvaluation) -> None:
    syllable_labels = [str(name) for name in report.class_names[1:]] + [SYLLABLE_NONE_LABEL]
    print(json.dumps(report.summary, indent=2, sort_keys=True))
    print("Dense confusion matrix:")
    print(_format_confusion_matrix_table(matrix=report.dense_matrix, labels=report.class_names))
    print("Dense classification report:")
    print(report.dense_report)
    print("Syllable confusion matrix:")
    print(_format_confusion_matrix_table(matrix=report.syllable_matrix, labels=syllable_labels))
    print("Syllable classification report:")
    print(report.syllable_report)
    print(f"Syllable WER: {report.syllable_wer:.4f}")


def _classification_summary(*, model, dataloader, class_names: list[str], device) -> tuple[np.ndarray, str]:
    chunk_probabilities, chunk_targets = _collect_supervised_chunk_outputs(
        model=model,
        dataloader=dataloader,
        device=device,
    )
    y_pred = np.concatenate([np.argmax(probabilities, axis=1) for probabilities in chunk_probabilities])
    y_true = np.concatenate(chunk_targets)
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, normalize="true", labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    return matrix, report


def _normalize_confusion_matrix_by_support(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(matrix, dtype=np.float64)
    np.divide(matrix, support, out=normalized, where=support != 0)
    return normalized


def _format_confusion_matrix_table(*, matrix: np.ndarray, labels: list[str]) -> str:
    expected_shape = (len(labels), len(labels))
    if matrix.shape != expected_shape:
        raise ValueError(f"Confusion matrix shape {matrix.shape} does not match labels {expected_shape}.")

    table = Table(box=box.ASCII, show_header=True)
    table.add_column("true/data \\ predicted", no_wrap=True)
    for label in labels:
        table.add_column(label, justify="right", no_wrap=True)
    for row_label, row_values in zip(labels, np.asarray(matrix, dtype=np.float64), strict=True):
        table.add_row(row_label, *[f"{value:.2f}" for value in row_values])

    buffer = StringIO()
    Console(file=buffer, force_terminal=False, color_system=None, width=512).print(table)
    return buffer.getvalue().rstrip()


def _frame_targets_from_annotation(
    *,
    annotation_file,
    class_names: list[str],
    frame_count: int,
    frame_rate_hz: float,
) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros((0,), dtype=np.int64)

    labels = np.zeros((frame_count,), dtype=np.int64)
    annotations = annotation_file if isinstance(annotation_file, pd.DataFrame) else pd.read_csv(annotation_file)
    name_to_index = {str(name): index for index, name in enumerate(class_names)}

    for row in annotations.itertuples(index=False):
        label_index = name_to_index.get(str(row.name))
        if label_index is None:
            continue
        start_frame = max(int(np.floor(float(row.start_seconds) * frame_rate_hz)), 0)
        stop_frame = min(int(np.ceil(float(row.stop_seconds) * frame_rate_hz)), frame_count)
        if stop_frame > start_frame:
            labels[start_frame:stop_frame] = label_index
        elif 0 <= start_frame < frame_count and float(row.stop_seconds) == float(row.start_seconds):
            labels[start_frame] = label_index
    return labels


def classification_summary_from_files(
    *,
    annotated_entries: list[tuple[Path, Path | pd.DataFrame, np.ndarray]],
    class_names: list[str],
    frame_rate_hz: float,
) -> tuple[np.ndarray, str]:
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    for _audio_file, annotation_file, probabilities in annotated_entries:
        predicted_labels = np.argmax(probabilities, axis=1) if probabilities.size else np.zeros((0,), dtype=np.int64)
        target_labels = _frame_targets_from_annotation(
            annotation_file=annotation_file,
            class_names=class_names,
            frame_count=probabilities.shape[0],
            frame_rate_hz=frame_rate_hz,
        )
        time_steps = min(len(predicted_labels), len(target_labels))
        y_true.append(target_labels[:time_steps])
        y_pred.append(predicted_labels[:time_steps])

    labels = list(range(len(class_names)))
    flat_true = np.concatenate(y_true) if y_true else np.zeros((0,), dtype=np.int64)
    flat_pred = np.concatenate(y_pred) if y_pred else np.zeros((0,), dtype=np.int64)
    matrix = confusion_matrix(flat_true, flat_pred, normalize="true", labels=labels)
    report = classification_report(
        flat_true,
        flat_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    return matrix, report


def _syllable_classification_summary(
    *,
    model,
    dataloader,
    class_names: list[str],
    device,
    frame_rate_hz: float,
    fill_gap_seconds: float,
    min_syllable_seconds: float,
    tolerance_seconds: float,
    postprocessor: str = "label_aware_dense",
    segment_threshold_low: float = 0.5,
    segment_threshold_high: float = 0.5,
) -> tuple[np.ndarray, str, float]:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None or not hasattr(dataset, "annotation_files"):
        raise ValueError("Syllable evaluation requires a dataset with annotation files.")

    non_noise_labels = [str(name) for name in class_names[1:]]
    matrix_labels = non_noise_labels + [SYLLABLE_NONE_LABEL]

    chunk_probabilities, _ = _collect_supervised_chunk_outputs(model=model, dataloader=dataloader, device=device)
    file_probabilities = _stitch_chunk_probabilities_by_file(dataset, chunk_probabilities)

    y_true: list[str] = []
    y_pred: list[str] = []
    total_edits = 0
    total_reference_tokens = 0

    annotation_sources = getattr(dataset, "annotations", None) or dataset.annotation_files
    for annotation_file, probabilities in zip(annotation_sources, file_probabilities, strict=True):
        reference_syllables = load_reference_syllables(annotation_file=annotation_file, class_names=class_names)
        predicted_syllables = _probabilities_to_syllable_segments(
            probabilities=probabilities,
            class_names=class_names,
            frame_rate_hz=frame_rate_hz,
            fill_gap_seconds=fill_gap_seconds,
            min_syllable_seconds=min_syllable_seconds,
            postprocessor=postprocessor,
            segment_threshold_low=segment_threshold_low,
            segment_threshold_high=segment_threshold_high,
        )
        reference_sequence = [syllable["name"] for syllable in reference_syllables]
        predicted_sequence = [syllable["name"] for syllable in predicted_syllables]
        total_edits += _levenshtein_distance(reference_sequence, predicted_sequence)
        total_reference_tokens += len(reference_sequence)

        matched_pairs = _match_syllables(
            reference_syllables=reference_syllables,
            predicted_syllables=predicted_syllables,
            tolerance_seconds=tolerance_seconds,
        )
        matched_reference = {reference_idx for reference_idx, _ in matched_pairs}
        matched_predictions = {prediction_idx for _, prediction_idx in matched_pairs}

        for reference_idx, prediction_idx in matched_pairs:
            y_true.append(reference_syllables[reference_idx]["name"])
            y_pred.append(predicted_syllables[prediction_idx]["name"])
        for reference_idx, syllable in enumerate(reference_syllables):
            if reference_idx not in matched_reference:
                y_true.append(syllable["name"])
                y_pred.append(SYLLABLE_NONE_LABEL)
        for prediction_idx, syllable in enumerate(predicted_syllables):
            if prediction_idx not in matched_predictions:
                y_true.append(SYLLABLE_NONE_LABEL)
                y_pred.append(syllable["name"])

    matrix = (
        confusion_matrix(y_true, y_pred, labels=matrix_labels)
        if y_true
        else np.zeros((len(matrix_labels), len(matrix_labels)), dtype=np.int64)
    )
    matrix = _normalize_confusion_matrix_by_support(matrix)
    report = (
        classification_report(
            y_true,
            y_pred,
            labels=non_noise_labels,
            target_names=non_noise_labels,
            zero_division=0,
        )
        if y_true
        else ""
    )
    wer = 0.0 if total_reference_tokens == 0 and total_edits == 0 else float(total_edits) / float(total_reference_tokens or 1)
    return matrix, report, wer


def _syllable_classification_summary_from_files(
    *,
    annotated_entries: list[tuple[Path, Path | pd.DataFrame, np.ndarray]],
    class_names: list[str],
    class_types: list[str],
    frame_rate_hz: float,
    fill_gap_seconds: float,
    min_syllable_seconds: float,
    tolerance_seconds: float,
    event_dist_min_seconds: float = 0.0,
    event_dist_max_seconds: float | None = None,
    postprocessor: str = "label_aware_dense",
    segment_threshold_low: float = 0.5,
    segment_threshold_high: float = 0.5,
    event_threshold: float = 0.5,
) -> tuple[np.ndarray, str, float]:
    syllable_pairs = []
    for _audio_file, annotation_file, probabilities in annotated_entries:
        syllable_pairs.append(
            (
                load_reference_syllables(annotation_file=annotation_file, class_names=class_names),
                _probabilities_to_annotation_rows(
                    probabilities=probabilities,
                    class_names=class_names,
                    class_types=class_types,
                    frame_rate_hz=frame_rate_hz,
                    fill_gap_seconds=fill_gap_seconds,
                    min_syllable_seconds=min_syllable_seconds,
                    event_dist_min_seconds=event_dist_min_seconds,
                    event_dist_max_seconds=event_dist_max_seconds,
                    postprocessor=postprocessor,
                    segment_threshold_low=segment_threshold_low,
                    segment_threshold_high=segment_threshold_high,
                    event_threshold=event_threshold,
                ),
            )
        )
    return syllable_classification_summary_from_syllable_pairs(
        syllable_pairs=syllable_pairs,
        class_names=class_names,
        tolerance_seconds=tolerance_seconds,
    )


def syllable_classification_summary_from_syllable_pairs(
    *,
    syllable_pairs: list[tuple[list[dict[str, float | str]], list[dict[str, float | str]]]],
    class_names: list[str],
    tolerance_seconds: float,
) -> tuple[np.ndarray, str, float]:
    non_noise_labels = [str(name) for name in class_names[1:]]
    matrix_labels = non_noise_labels + [SYLLABLE_NONE_LABEL]

    y_true: list[str] = []
    y_pred: list[str] = []
    total_edits = 0
    total_reference_tokens = 0

    for reference_syllables, predicted_syllables in syllable_pairs:
        reference_sequence = [syllable["name"] for syllable in reference_syllables]
        predicted_sequence = [syllable["name"] for syllable in predicted_syllables]
        total_edits += _levenshtein_distance(reference_sequence, predicted_sequence)
        total_reference_tokens += len(reference_sequence)

        matched_pairs = _match_syllables(
            reference_syllables=reference_syllables,
            predicted_syllables=predicted_syllables,
            tolerance_seconds=tolerance_seconds,
        )
        matched_reference = {reference_idx for reference_idx, _prediction_idx in matched_pairs}
        matched_predictions = {prediction_idx for _reference_idx, prediction_idx in matched_pairs}

        for reference_idx, prediction_idx in matched_pairs:
            y_true.append(reference_syllables[reference_idx]["name"])
            y_pred.append(predicted_syllables[prediction_idx]["name"])
        for reference_idx, syllable in enumerate(reference_syllables):
            if reference_idx not in matched_reference:
                y_true.append(syllable["name"])
                y_pred.append(SYLLABLE_NONE_LABEL)
        for prediction_idx, syllable in enumerate(predicted_syllables):
            if prediction_idx not in matched_predictions:
                y_true.append(SYLLABLE_NONE_LABEL)
                y_pred.append(syllable["name"])

    matrix = (
        confusion_matrix(y_true, y_pred, labels=matrix_labels)
        if y_true
        else np.zeros((len(matrix_labels), len(matrix_labels)), dtype=np.int64)
    )
    matrix = _normalize_confusion_matrix_by_support(matrix)
    report = (
        classification_report(
            y_true,
            y_pred,
            labels=non_noise_labels,
            target_names=non_noise_labels,
            zero_division=0,
        )
        if y_true
        else ""
    )
    wer = 0.0 if total_reference_tokens == 0 and total_edits == 0 else float(total_edits) / float(total_reference_tokens or 1)
    return matrix, report, wer


def _prediction_batches_to_chunk_probabilities(predictions) -> list[np.ndarray]:
    chunk_probabilities: list[np.ndarray] = []
    for batch_logits, batch_lengths in predictions:
        probabilities = torch.softmax(batch_logits, dim=-1).detach().cpu().numpy()
        lengths = batch_lengths.detach().cpu().numpy()
        for sample_probabilities, sample_length in zip(probabilities, lengths, strict=True):
            chunk_probabilities.append(sample_probabilities[: int(sample_length)])
    return chunk_probabilities


def _classification_summaries_from_file_predictions(
    *,
    dataset,
    chunk_probabilities: list[np.ndarray],
    class_names: list[str],
    class_types: list[str],
    frame_rate_hz: float,
    fill_gap_seconds: float,
    min_syllable_seconds: float,
    tolerance_seconds: float,
    event_dist_min_seconds: float = 0.0,
    event_dist_max_seconds: float | None = None,
    postprocessor: str = "label_aware_dense",
    segment_threshold_low: float = 0.5,
    segment_threshold_high: float = 0.5,
    event_threshold: float = 0.5,
) -> tuple[np.ndarray, str, np.ndarray, str, float, dict[str, int]]:
    file_probabilities = _stitch_chunk_probabilities_by_file(dataset, chunk_probabilities)
    annotated_entries, summary = _annotated_probability_entries(
        dataset=dataset,
        file_probabilities=file_probabilities,
    )
    if not annotated_entries:
        raise ValueError("Evaluation requires at least one matching '*_annotations.csv' file.")

    dense_matrix, dense_report = classification_summary_from_files(
        annotated_entries=annotated_entries,
        class_names=class_names,
        frame_rate_hz=frame_rate_hz,
    )
    syllable_matrix, syllable_report, syllable_wer = _syllable_classification_summary_from_files(
        annotated_entries=annotated_entries,
        class_names=class_names,
        class_types=class_types,
        frame_rate_hz=frame_rate_hz,
        fill_gap_seconds=fill_gap_seconds,
        min_syllable_seconds=min_syllable_seconds,
        tolerance_seconds=tolerance_seconds,
        event_dist_min_seconds=event_dist_min_seconds,
        event_dist_max_seconds=event_dist_max_seconds,
        postprocessor=postprocessor,
        segment_threshold_low=segment_threshold_low,
        segment_threshold_high=segment_threshold_high,
        event_threshold=event_threshold,
    )
    return dense_matrix, dense_report, syllable_matrix, syllable_report, syllable_wer, summary


def _annotated_probability_entries(
    *,
    dataset,
    file_probabilities: list[np.ndarray],
) -> tuple[list[tuple[Path, Path | pd.DataFrame, np.ndarray]], dict[str, int]]:
    annotated_entries = []
    skipped_file_count = 0
    annotations_by_audio = getattr(dataset, "annotations_by_audio", {})
    annotation_files_by_audio = getattr(dataset, "annotation_files_by_audio", {})
    for audio_file, probabilities in zip(dataset.audio_files, file_probabilities, strict=True):
        audio_file_key = Path(audio_file).expanduser().resolve()
        annotation_file = annotations_by_audio.get(audio_file_key)
        if annotation_file is None:
            annotation_file = annotation_files_by_audio.get(audio_file_key)
        if annotation_file is None:
            annotation_file = Path(audio_file).with_name(f"{Path(audio_file).stem}_annotations.csv")
        if not isinstance(annotation_file, pd.DataFrame) and not annotation_file.exists():
            skipped_file_count += 1
            continue
        annotated_entries.append((Path(audio_file), annotation_file, probabilities))

    return annotated_entries, {
        "evaluated_file_count": len(annotated_entries),
        "skipped_file_count": skipped_file_count,
    }


def write_annotation_file(output_path: Path, annotation: pd.DataFrame, *, merge: bool = False) -> None:
    output_path = Path(output_path)
    if merge and output_path.exists():
        annotation = pd.concat([pd.read_csv(output_path), annotation], ignore_index=True, sort=False)
        if "start_seconds" in annotation.columns:
            annotation = annotation.sort_values("start_seconds", kind="stable", ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotation.to_csv(output_path, index=False)


def _collect_supervised_chunk_outputs(*, model, dataloader, device) -> tuple[list[np.ndarray], list[np.ndarray]]:
    chunk_probabilities: list[np.ndarray] = []
    chunk_targets: list[np.ndarray] = []

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    with torch.inference_mode():
        for batch in dataloader:
            if len(batch) != 4:
                raise ValueError(f"Expected batch with 4 items, got {len(batch)}.")

            inputs, input_lengths, batch_targets, batch_target_lengths = batch
            inputs = torch.as_tensor(inputs, device=device)
            input_lengths = torch.as_tensor(input_lengths, device=device, dtype=torch.long)
            batch_targets = torch.as_tensor(batch_targets, device=device)

            logits, output_lengths = model(inputs, input_lengths)
            probabilities = torch.softmax(logits, dim=-1)
            target_indices = torch.argmax(batch_targets, dim=-1) if batch_targets.ndim == logits.ndim else batch_targets.long()

            normalized_lengths = _normalize_output_lengths(
                output_lengths=output_lengths,
                batch_size=probabilities.shape[0],
                max_length=probabilities.shape[1],
            )
            normalized_target_lengths = _normalize_output_lengths(
                output_lengths=batch_target_lengths,
                batch_size=batch_targets.shape[0],
                max_length=batch_targets.shape[1],
            )
            for sample_probabilities, sample_targets, sample_length, sample_target_length in zip(
                probabilities,
                target_indices,
                normalized_lengths,
                normalized_target_lengths,
                strict=True,
            ):
                time_steps = min(
                    sample_probabilities.shape[0], sample_targets.shape[0], int(sample_length), int(sample_target_length)
                )
                chunk_probabilities.append(sample_probabilities[:time_steps].detach().cpu().numpy())
                chunk_targets.append(sample_targets[:time_steps].detach().cpu().numpy())

    return chunk_probabilities, chunk_targets


def _normalize_output_lengths(*, output_lengths, batch_size: int, max_length: int) -> np.ndarray:
    if output_lengths is None:
        return np.full(batch_size, max_length, dtype=np.int64)
    lengths = torch.as_tensor(output_lengths, dtype=torch.long).reshape(-1).detach().cpu().numpy()
    if lengths.size != batch_size:
        return np.full(batch_size, max_length, dtype=np.int64)
    return np.clip(lengths, 0, max_length)


def _sample_to_frame_index(sample_index: int, hop_samples: int) -> int:
    return int(sample_index) if hop_samples <= 1 else int(round(sample_index / hop_samples))


def _sample_count_to_frame_count(num_samples: int, hop_samples: int) -> int:
    if num_samples <= 0:
        return 0
    return int(num_samples) if hop_samples <= 1 else int(num_samples // hop_samples + 1)


def _stitch_chunk_probabilities_by_file(
    dataset,
    chunk_probabilities: list[np.ndarray],
    *,
    trim_chunk_frames: int = 0,
) -> list[np.ndarray]:
    file_probabilities: list[np.ndarray] = []
    num_classes = chunk_probabilities[0].shape[1] if chunk_probabilities else len(getattr(dataset, "class_names", []))
    trim_chunk_frames = max(int(trim_chunk_frames), 0)

    for file_idx, _audio_file in enumerate(dataset.audio_files):
        chunk_start = int(dataset.chunk_borders[file_idx])
        chunk_stop = int(dataset.chunk_borders[file_idx + 1])
        samplerate = int(dataset.samplerate_per_file[file_idx])
        hop_samples = max(int(round(samplerate * dataset.hop_s)), 1)
        file_frame_count = _sample_count_to_frame_count(int(dataset.nb_samples_in_file[file_idx]), hop_samples)
        if chunk_start == chunk_stop:
            file_probabilities.append(np.zeros((0, num_classes), dtype=np.float32))
            continue

        summed = np.zeros((file_frame_count, num_classes), dtype=np.float32)
        counts = np.zeros(file_frame_count, dtype=np.float32)
        for global_chunk_idx, probabilities in zip(
            range(chunk_start, chunk_stop),
            chunk_probabilities[chunk_start:chunk_stop],
            strict=True,
        ):
            start_sample = int(dataset.chunk_start_samples[global_chunk_idx])
            if trim_chunk_frames > 0 and start_sample % int(dataset.chunk_stride) != 0:
                continue
            start_frame = _sample_to_frame_index(start_sample, hop_samples)
            if trim_chunk_frames > 0:
                if probabilities.shape[0] <= 2 * trim_chunk_frames:
                    continue
                probabilities = probabilities[trim_chunk_frames:-trim_chunk_frames]
                start_frame += trim_chunk_frames
            end_frame = min(start_frame + probabilities.shape[0], file_frame_count)
            if end_frame <= start_frame:
                continue
            trimmed = probabilities[: end_frame - start_frame].astype(np.float32, copy=False)
            summed[start_frame:end_frame] += trimmed
            counts[start_frame:end_frame] += 1.0

        if file_frame_count == 0:
            file_probabilities.append(np.zeros((0, num_classes), dtype=np.float32))
            continue
        counts = np.maximum(counts, 1.0)
        file_probabilities.append(summed / counts[:, np.newaxis])

    return file_probabilities


def _probabilities_to_syllable_annotations(
    *,
    probabilities: np.ndarray,
    class_names: list[str],
    class_types: list[str] | None = None,
    frame_rate_hz: float,
    fill_gap_seconds: float,
    min_syllable_seconds: float,
    event_dist_min_seconds: float = 0.0,
    event_dist_max_seconds: float | None = None,
    postprocessor: str = "label_aware_dense",
    segment_threshold_low: float = 0.5,
    segment_threshold_high: float = 0.5,
    event_threshold: float = 0.5,
) -> pd.DataFrame:
    rows = _probabilities_to_annotation_rows(
        probabilities=probabilities,
        class_names=class_names,
        class_types=class_types,
        frame_rate_hz=frame_rate_hz,
        fill_gap_seconds=fill_gap_seconds,
        min_syllable_seconds=min_syllable_seconds,
        event_dist_min_seconds=event_dist_min_seconds,
        event_dist_max_seconds=event_dist_max_seconds,
        postprocessor=postprocessor,
        segment_threshold_low=segment_threshold_low,
        segment_threshold_high=segment_threshold_high,
        event_threshold=event_threshold,
    )
    return pd.DataFrame(rows, columns=["name", "start_seconds", "stop_seconds"])


def _probabilities_to_annotation_rows(
    *,
    probabilities: np.ndarray,
    class_names: list[str],
    class_types: list[str] | None,
    frame_rate_hz: float,
    fill_gap_seconds: float,
    min_syllable_seconds: float,
    event_dist_min_seconds: float = 0.0,
    event_dist_max_seconds: float | None = None,
    postprocessor: str = "label_aware_dense",
    segment_threshold_low: float = 0.5,
    segment_threshold_high: float = 0.5,
    event_threshold: float = 0.5,
) -> list[dict[str, float | str]]:
    normalized_class_types = normalize_class_types(class_names=class_names, class_types=class_types)
    rows: list[dict[str, float | str]] = []

    segment_dims = [index for index, class_type in enumerate(normalized_class_types) if class_type == "segment"]
    if len(segment_dims) >= 2:
        rows.extend(
            _probabilities_to_syllable_segments(
                probabilities=probabilities[:, segment_dims],
                class_names=[str(class_names[index]) for index in segment_dims],
                frame_rate_hz=frame_rate_hz,
                fill_gap_seconds=fill_gap_seconds,
                min_syllable_seconds=min_syllable_seconds,
                postprocessor=postprocessor,
                segment_threshold_low=segment_threshold_low,
                segment_threshold_high=segment_threshold_high,
            )
        )
    event_dims = [index for index, class_type in enumerate(normalized_class_types) if class_type == "event"]
    if event_dims:
        rows.extend(
            _probabilities_to_event_segments(
                probabilities=probabilities,
                class_names=class_names,
                event_dims=event_dims,
                frame_rate_hz=frame_rate_hz,
                event_dist_min_seconds=event_dist_min_seconds,
                event_dist_max_seconds=event_dist_max_seconds,
                event_threshold=event_threshold,
            )
        )
    rows.sort(key=lambda row: (float(row["start_seconds"]), float(row["stop_seconds"]), str(row["name"])))
    return rows


def _probabilities_to_syllable_segments(
    *,
    probabilities: np.ndarray,
    class_names: list[str],
    frame_rate_hz: float,
    fill_gap_seconds: float,
    min_syllable_seconds: float,
    postprocessor: str = "label_aware_dense",
    segment_threshold_low: float = 0.5,
    segment_threshold_high: float = 0.5,
) -> list[dict[str, float | str]]:
    if probabilities.size == 0 or len(class_names) <= 1:
        return []
    if probabilities.ndim != 2:
        raise ValueError(f"Expected probabilities with shape [time, classes], got {probabilities.shape}.")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("Number of probability columns must match the number of class names.")
    if frame_rate_hz <= 0:
        raise ValueError("frame_rate_hz must be positive.")

    labels = np.argmax(probabilities, axis=1).astype(np.int64, copy=False)
    if len(class_names) == 2:
        candidate = probabilities[:, 1] > segment_threshold_low
        seeds = probabilities[:, 1] > segment_threshold_high
        labels = np.zeros(len(candidate), dtype=np.int64)
        for start_frame, stop_frame in _true_runs(candidate):
            if seeds[start_frame:stop_frame].any():
                labels[start_frame:stop_frame] = 1
    max_gap_frames = int(fill_gap_seconds * frame_rate_hz)
    min_run_frames = int(min_syllable_seconds * frame_rate_hz)
    if postprocessor == "label_aware_dense":
        processed_labels = _postprocess_dense_labels(
            labels=labels,
            probabilities=probabilities,
            noise_label=0,
            max_gap_frames=max_gap_frames,
            min_run_frames=min_run_frames,
        )
        return _label_runs_to_syllable_segments(
            labels=processed_labels,
            class_names=class_names,
            frame_rate_hz=frame_rate_hz,
        )
    if postprocessor != "binary_mask":
        raise ValueError(f"Unknown syllable postprocessor '{postprocessor}'.")

    song_labels = labels > 0
    song_labels = _fill_short_gaps(song_labels, max_gap_frames)
    song_labels = _remove_short_runs(song_labels, min_run_frames)

    segments: list[dict[str, float | str]] = []
    for start_frame, stop_frame in _true_runs(song_labels):
        if stop_frame <= start_frame:
            continue
        if len(class_names) == 2:
            label_index = 1
        else:
            values, counts = np.unique(labels[start_frame:stop_frame], return_counts=True)
            non_noise = values != 0
            if not np.any(non_noise):
                continue
            values = values[non_noise]
            counts = counts[non_noise]
            label_index = int(values[int(np.argmax(counts))])
        start_seconds, stop_seconds = _frame_run_to_times(
            start_frame=start_frame,
            stop_frame=stop_frame,
            frame_rate_hz=frame_rate_hz,
        )
        segments.append(
            {
                "name": str(class_names[label_index]),
                "start_seconds": start_seconds,
                "stop_seconds": stop_seconds,
            }
        )
    return segments


def _label_runs_to_syllable_segments(
    *,
    labels: np.ndarray,
    class_names: list[str],
    frame_rate_hz: float,
) -> list[dict[str, float | str]]:
    segments: list[dict[str, float | str]] = []
    for start_frame, stop_frame, label_index in _label_runs(labels):
        if label_index == 0 or stop_frame <= start_frame:
            continue
        start_seconds, stop_seconds = _frame_run_to_times(
            start_frame=start_frame,
            stop_frame=stop_frame,
            frame_rate_hz=frame_rate_hz,
        )
        segments.append(
            {
                "name": str(class_names[label_index]),
                "start_seconds": start_seconds,
                "stop_seconds": stop_seconds,
            }
        )
    return segments


def _probabilities_to_event_segments(
    *,
    probabilities: np.ndarray,
    class_names: list[str],
    event_dims: list[int],
    frame_rate_hz: float,
    event_dist_min_seconds: float = 0.0,
    event_dist_max_seconds: float | None = None,
    event_threshold: float = 0.5,
) -> list[dict[str, float | str]]:
    if probabilities.size == 0 or frame_rate_hz <= 0:
        return []

    rows: list[dict[str, float | str]] = []
    peak_distance = max(int(round(0.01 * frame_rate_hz)), 1)
    for class_index in event_dims:
        if class_index <= 0 or class_index >= probabilities.shape[1]:
            continue
        peak_indices = _peak_indexes(probabilities[:, class_index], threshold=event_threshold, min_distance=peak_distance)
        peak_indices = np.asarray(peak_indices, dtype=np.int64)
        event_seconds = peak_indices.astype(np.float64) / float(frame_rate_hz)
        good_events = _event_interval_filter(
            event_seconds,
            event_dist_min_seconds=event_dist_min_seconds,
            event_dist_max_seconds=event_dist_max_seconds,
        )
        for peak_index in peak_indices[good_events].tolist():
            start_seconds = float(peak_index) / float(frame_rate_hz)
            rows.append(
                {
                    "name": str(class_names[class_index]),
                    "start_seconds": start_seconds,
                    "stop_seconds": start_seconds,
                }
            )
    return rows


def _event_interval_filter(
    events: np.ndarray,
    *,
    event_dist_min_seconds: float = 0.0,
    event_dist_max_seconds: float | None = None,
) -> np.ndarray:
    events = np.asarray(events, dtype=np.float64)
    if events.size == 0:
        return np.zeros((0,), dtype=bool)

    event_dist_max = np.inf if event_dist_max_seconds is None else float(event_dist_max_seconds)
    ipi_pre = np.diff(events, prepend=np.inf)
    ipi_post = np.diff(events, append=np.inf)

    ipi_too_long = np.logical_and(ipi_pre > event_dist_max, ipi_post > event_dist_max)
    ipi_too_short = np.logical_or(ipi_pre < float(event_dist_min_seconds), ipi_post < float(event_dist_min_seconds))
    ipi_too_short[0] = False
    return ~np.logical_or(ipi_too_long, ipi_too_short)


def _peak_indexes(y: np.ndarray, *, threshold: float, min_distance: int) -> np.ndarray:
    y = np.asarray(y)
    if y.size < 3:
        return np.zeros((0,), dtype=np.int64)

    dy = np.diff(y)
    (zeros,) = np.where(dy == 0)
    if len(zeros) == len(y) - 1:
        return np.zeros((0,), dtype=np.int64)

    if len(zeros):
        (split_at,) = np.add(np.where(np.diff(zeros) != 1), 1)
        zero_plateaus = list(np.split(zeros, split_at))
        if zero_plateaus and zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)
        if zero_plateaus and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)
        for plateau in zero_plateaus:
            median = np.median(plateau)
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    peaks = np.where((np.hstack([dy, 0.0]) < 0.0) & (np.hstack([0.0, dy]) > 0.0) & (y > threshold))[0]
    min_distance = int(min_distance)
    if peaks.size > 1 and min_distance > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        removed = np.ones(y.size, dtype=bool)
        removed[peaks] = False
        for peak in highest:
            if not removed[peak]:
                window = slice(max(0, peak - min_distance), peak + min_distance + 1)
                removed[window] = True
                removed[peak] = False
        peaks = np.arange(y.size)[~removed]
    return peaks.astype(np.int64, copy=False)


def _frame_run_to_times(*, start_frame: int, stop_frame: int, frame_rate_hz: float) -> tuple[float, float]:
    frame_width = 1.0 / float(frame_rate_hz)
    start_seconds = (float(start_frame) + 0.5) * frame_width
    stop_seconds = (float(stop_frame) - 0.5) * frame_width
    if stop_seconds <= start_seconds:
        start_seconds = float(start_frame) * frame_width
        stop_seconds = float(stop_frame) * frame_width
    return start_seconds, stop_seconds


def _postprocess_dense_labels(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    noise_label: int,
    max_gap_frames: int,
    min_run_frames: int,
) -> np.ndarray:
    processed = np.asarray(labels, dtype=np.int64).copy()
    if processed.size == 0:
        return processed
    if min_run_frames > 1:
        processed = _remove_short_label_runs(processed, noise_label=noise_label, min_run_frames=min_run_frames)
    if max_gap_frames > 0:
        processed = _fill_short_noise_gaps(
            processed,
            probabilities=probabilities,
            noise_label=noise_label,
            max_gap_frames=max_gap_frames,
        )
    if min_run_frames > 1:
        processed = _remove_short_label_runs(processed, noise_label=noise_label, min_run_frames=min_run_frames)
    return processed


def _fill_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0:
        return mask.astype(bool, copy=True)
    filled = mask.astype(bool, copy=True)
    runs = _true_runs(filled)
    for (_, previous_stop), (next_start, _next_stop) in zip(runs, runs[1:], strict=False):
        if next_start - previous_stop <= max_gap_frames:
            filled[previous_stop:next_start] = True
    return filled


def _remove_short_runs(mask: np.ndarray, min_run_frames: int) -> np.ndarray:
    if min_run_frames <= 1:
        return mask.astype(bool, copy=True)
    filtered = mask.astype(bool, copy=True)
    for start_frame, stop_frame in _true_runs(filtered):
        if stop_frame - start_frame < min_run_frames:
            filtered[start_frame:stop_frame] = False
    return filtered


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.int8)
    transitions = np.diff(values, prepend=0, append=0)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _label_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    values = np.asarray(labels, dtype=np.int64)
    if values.size == 0:
        return []
    change_points = np.flatnonzero(np.diff(values) != 0) + 1
    starts = np.concatenate(([0], change_points))
    stops = np.concatenate((change_points, [values.size]))
    return [(int(start), int(stop), int(values[start])) for start, stop in zip(starts, stops, strict=True)]


def _remove_short_label_runs(labels: np.ndarray, *, noise_label: int, min_run_frames: int) -> np.ndarray:
    filtered = np.asarray(labels, dtype=np.int64).copy()
    for start_frame, stop_frame, label_index in _label_runs(filtered):
        if label_index == noise_label:
            continue
        if stop_frame - start_frame < min_run_frames:
            filtered[start_frame:stop_frame] = noise_label
    return filtered


def _fill_short_noise_gaps(
    labels: np.ndarray,
    *,
    probabilities: np.ndarray,
    noise_label: int,
    max_gap_frames: int,
) -> np.ndarray:
    filled = np.asarray(labels, dtype=np.int64).copy()
    runs = _label_runs(filled)
    for run_index, (start_frame, stop_frame, label_index) in enumerate(runs):
        if label_index != noise_label or stop_frame - start_frame > max_gap_frames:
            continue
        if run_index == 0 or run_index == len(runs) - 1:
            continue

        previous_label = runs[run_index - 1][2]
        next_label = runs[run_index + 1][2]
        if previous_label == noise_label or next_label == noise_label:
            continue
        if previous_label == next_label:
            filled[start_frame:stop_frame] = previous_label
            continue

        gap_probabilities = probabilities[start_frame:stop_frame][:, [previous_label, next_label]]
        split_points = np.argmax(gap_probabilities, axis=1)
        filled[start_frame:stop_frame] = np.where(split_points == 0, previous_label, next_label)

        first_next = np.flatnonzero(filled[start_frame:stop_frame] == next_label)
        if first_next.size == 0:
            filled[start_frame:stop_frame] = previous_label
        elif first_next.size == stop_frame - start_frame:
            filled[start_frame:stop_frame] = next_label
        else:
            boundary = start_frame + int(first_next[0])
            filled[start_frame:boundary] = previous_label
            filled[boundary:stop_frame] = next_label
    return filled


def load_reference_syllables(*, annotation_file, class_names: list[str]) -> list[dict[str, float | str]]:
    annotations = annotation_file if isinstance(annotation_file, pd.DataFrame) else pd.read_csv(annotation_file)
    valid_names = {str(name) for name in class_names[1:]}
    syllables: list[dict[str, float | str]] = []

    for row in annotations.itertuples(index=False):
        name = str(row.name)
        start_seconds = float(row.start_seconds)
        stop_seconds = float(row.stop_seconds)
        if name not in valid_names:
            continue
        if not np.isfinite(start_seconds) or not np.isfinite(stop_seconds):
            continue
        syllables.append(
            {
                "name": name,
                "start_seconds": start_seconds,
                "stop_seconds": stop_seconds,
            }
        )

    syllables.sort(key=lambda syllable: (syllable["start_seconds"], syllable["stop_seconds"], syllable["name"]))
    return syllables


def _match_syllables(
    *,
    reference_syllables: list[dict[str, float | str]],
    predicted_syllables: list[dict[str, float | str]],
    tolerance_seconds: float,
) -> list[tuple[int, int]]:
    matched_predictions: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []

    for reference_idx, reference in enumerate(reference_syllables):
        best_prediction_idx: int | None = None
        best_distance: float | None = None
        for prediction_idx, prediction in enumerate(predicted_syllables):
            if prediction_idx in matched_predictions:
                continue
            onset_delta = abs(float(reference["start_seconds"]) - float(prediction["start_seconds"]))
            offset_delta = abs(float(reference["stop_seconds"]) - float(prediction["stop_seconds"]))
            if onset_delta > tolerance_seconds or offset_delta > tolerance_seconds:
                continue

            distance = onset_delta + offset_delta
            if best_distance is None or distance < best_distance:
                best_prediction_idx = prediction_idx
                best_distance = distance
        if best_prediction_idx is not None:
            matched_predictions.add(best_prediction_idx)
            matched_pairs.append((reference_idx, best_prediction_idx))
    return matched_pairs


def _levenshtein_distance(reference_tokens: list[str], predicted_tokens: list[str]) -> int:
    previous_row = list(range(len(predicted_tokens) + 1))
    for reference_idx, reference_token in enumerate(reference_tokens, start=1):
        current_row = [reference_idx]
        for prediction_idx, prediction_token in enumerate(predicted_tokens, start=1):
            substitution_cost = 0 if reference_token == prediction_token else 1
            current_row.append(
                min(
                    previous_row[prediction_idx] + 1,
                    current_row[prediction_idx - 1] + 1,
                    previous_row[prediction_idx - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]
