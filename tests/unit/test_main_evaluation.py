from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import das.api as api
import das.prediction_results as main
from das.config import Config


def test_classification_summary_builds_confusion_matrix_and_report():
    class FakeModel(torch.nn.Module):
        def forward(self, inputs, input_lengths):
            del inputs
            logits = torch.tensor(
                [[[5.0, 1.0], [1.0, 4.0], [4.0, 2.0]]],
                device=input_lengths.device,
            )
            return logits, input_lengths

    batch = (
        torch.zeros((1, 3), dtype=torch.float32),
        torch.tensor([3], dtype=torch.long),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        torch.tensor([3], dtype=torch.long),
    )

    matrix, report = main._classification_summary(
        model=FakeModel(),
        dataloader=[batch],
        class_names=["noise", "song"],
        device=torch.device("cpu"),
    )

    assert matrix.tolist() == [[1.0, 0.0], [0.5, 0.5]]
    assert "noise" in report
    assert "song" in report


def test_classification_summary_from_files_builds_confusion_matrix_and_report(tmp_path: Path):
    annotation_path = tmp_path / "clip_annotations.csv"
    pd.DataFrame(
        [
            {"name": "song", "start_seconds": 0.01, "stop_seconds": 0.03},
        ]
    ).to_csv(annotation_path, index=False)
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )

    matrix, report = main.classification_summary_from_files(
        annotated_entries=[(tmp_path / "clip.wav", annotation_path, probabilities)],
        class_names=["noise", "song"],
        frame_rate_hz=100.0,
    )

    assert matrix.shape == (2, 2)
    assert "noise" in report
    assert "song" in report


def test_file_prediction_evaluation_uses_dataset_annotation_tables(tmp_path: Path):
    audio_path = tmp_path / "clip.wav"
    annotations = pd.DataFrame([{"filepath": "clip.wav", "name": "song", "start_seconds": 0.01, "stop_seconds": 0.03}])
    dataset = SimpleNamespace(
        audio_files=[audio_path],
        chunk_borders=np.array([0, 1]),
        chunk_start_samples=np.array([0]),
        nb_samples_in_file=np.array([4]),
        samplerate_per_file=np.array([100]),
        hop_s=0.01,
        num_time_steps=4,
        class_names=["noise", "song"],
        annotations_by_audio={audio_path.resolve(): annotations},
        annotation_files_by_audio={},
    )
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )

    *_summaries, summary = main._classification_summaries_from_file_predictions(
        dataset=dataset,
        chunk_probabilities=[probabilities],
        class_names=["noise", "song"],
        class_types=["segment", "segment"],
        frame_rate_hz=100.0,
        fill_gap_seconds=0.0,
        min_syllable_seconds=0.0,
        tolerance_seconds=0.0,
    )

    assert summary == {"evaluated_file_count": 1, "skipped_file_count": 0}


def test_write_annotation_file_overwrites_by_default(tmp_path: Path):
    output_path = tmp_path / "clip_annotations.csv"
    pd.DataFrame([{"name": "old", "start_seconds": 0.0, "stop_seconds": 0.1}]).to_csv(output_path, index=False)

    main.write_annotation_file(
        output_path,
        pd.DataFrame([{"name": "new", "start_seconds": 0.2, "stop_seconds": 0.3}]),
    )

    assert pd.read_csv(output_path).to_dict("records") == [
        {"name": "new", "start_seconds": 0.2, "stop_seconds": 0.3},
    ]


def test_write_annotation_file_merges_existing_rows_and_sorts(tmp_path: Path):
    output_path = tmp_path / "clip_annotations.csv"
    pd.DataFrame(
        [
            {"name": "late", "start_seconds": 0.4, "stop_seconds": 0.5},
            {"name": "duplicate", "start_seconds": 0.2, "stop_seconds": 0.3},
        ]
    ).to_csv(output_path, index=False)

    main.write_annotation_file(
        output_path,
        pd.DataFrame(
            [
                {"name": "early", "start_seconds": 0.1, "stop_seconds": 0.2},
                {"name": "duplicate", "start_seconds": 0.2, "stop_seconds": 0.3},
            ]
        ),
        merge=True,
    )

    assert pd.read_csv(output_path).to_dict("records") == [
        {"name": "early", "start_seconds": 0.1, "stop_seconds": 0.2},
        {"name": "duplicate", "start_seconds": 0.2, "stop_seconds": 0.3},
        {"name": "duplicate", "start_seconds": 0.2, "stop_seconds": 0.3},
        {"name": "late", "start_seconds": 0.4, "stop_seconds": 0.5},
    ]


def test_write_prediction_outputs_skips_existing_files(tmp_path: Path):
    existing_path = tmp_path / "existing_annotations.csv"
    pd.DataFrame([{"name": "old"}]).to_csv(existing_path, index=False)

    written = main.write_prediction_outputs(
        output_dir=tmp_path,
        dataset=SimpleNamespace(audio_files=["existing.wav", "new.wav"]),
        predictions=None,
        context=None,
        output_suffix="_annotations.csv",
        existing_annotations="skip",
        annotations=[pd.DataFrame([{"name": "replacement"}]), pd.DataFrame([{"name": "new"}])],
    )

    assert written == [str(tmp_path / "new_annotations.csv")]
    assert pd.read_csv(existing_path).to_dict("records") == [{"name": "old"}]


def test_write_prediction_outputs_next_to_each_audio_by_default(tmp_path: Path):
    first_audio = tmp_path / "first" / "one.wav"
    second_audio = tmp_path / "second" / "two.wav"
    first_audio.parent.mkdir()
    second_audio.parent.mkdir()

    written = main.write_prediction_outputs(
        output_dir=None,
        dataset=SimpleNamespace(audio_files=[first_audio, second_audio]),
        predictions=None,
        context=None,
        output_suffix="_annotations.csv",
        annotations=[pd.DataFrame([{"name": "one"}]), pd.DataFrame([{"name": "two"}])],
    )

    assert written == [
        str(first_audio.with_name("one_annotations.csv")),
        str(second_audio.with_name("two_annotations.csv")),
    ]


def test_evaluate_prints_confusion_matrix_and_classification_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    class FakeDataModule:
        def __init__(self, **kwargs):
            del kwargs

        def predict_dataloader(self):
            return SimpleNamespace(dataset=SimpleNamespace(audio_files=["clip.wav"], chunk_borders=[0, 0]))

    class FakeTrainer:
        def __init__(self, **kwargs):
            del kwargs
            self.strategy = SimpleNamespace(root_device=torch.device("cpu"))

        def predict(self, model, dataloaders):
            del model, dataloaders
            return []

    monkeypatch.setattr(api, "AudioDirDataModule", FakeDataModule)
    monkeypatch.setattr(api, "_build_inference_trainer", lambda config, **kwargs: FakeTrainer())
    monkeypatch.setattr(
        api,
        "_load_predict_model",
        lambda checkpoint, config: (
            object(),
            api.PredictRuntime(
                sr=32_000,
                hop_seconds=1 / 32_000,
                num_time_steps=128,
                chunk_stride=64,
                class_names=["noise", "song"],
                class_types=["segment", "segment"],
            ),
        ),
    )
    monkeypatch.setattr(
        api.prediction_results,
        "evaluate_file_predictions",
        lambda **kwargs: main.PredictionEvaluation(
            summary={"evaluated_file_count": 1, "skipped_file_count": 0},
            class_names=kwargs["context"].class_names,
            dense_matrix=np.array([[2, 1], [0, 3]]),
            dense_report="report text",
            syllable_matrix=np.array([[1, 0], [2, 3]]),
            syllable_report="syllable report",
            syllable_wer=0.125,
        ),
    )

    summary = api.evaluate(
        Config(
            mode="predict",
            data_dir="data",
            checkpoint="model.ckpt",
            output_dir="predictions",
            evaluate=True,
        ),
        verbose=True,
    )

    assert summary == {"evaluated_file_count": 1, "skipped_file_count": 0}
    output = capsys.readouterr().out
    assert '"evaluated_file_count": 1' in output
    assert "Dense confusion matrix:" in output
    assert "Dense classification report:" in output
    assert "report text" in output
    assert "Syllable classification report:" in output
    assert "syllable report" in output
    assert "Syllable WER: 0.1250" in output
    assert re.search(r"\|\s*noise\s*\|\s*2\.00\s*\|\s*1\.00\s*\|", output)


def test_probabilities_to_syllable_annotations_fill_gaps_and_remove_short_runs():
    probabilities = np.array(
        [
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.98, 0.01, 0.01],
            [0.98, 0.01, 0.01],
            [0.98, 0.01, 0.01],
            [0.01, 0.01, 0.98],
            [0.98, 0.01, 0.01],
        ],
        dtype=np.float32,
    )

    annotations = main._probabilities_to_syllable_annotations(
        probabilities=probabilities,
        class_names=["noise", "a", "b"],
        frame_rate_hz=100.0,
        fill_gap_seconds=0.01,
        min_syllable_seconds=0.02,
    )

    assert annotations.to_dict("records") == [
        {"name": "a", "start_seconds": 0.005, "stop_seconds": 0.045},
    ]


def test_label_aware_syllable_postprocessor_keeps_neighbor_labels():
    probabilities = np.array(
        [
            [0.01, 0.98, 0.01],
            [0.01, 0.98, 0.01],
            [0.60, 0.30, 0.10],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
        ],
        dtype=np.float32,
    )

    binary_segments = main._probabilities_to_syllable_segments(
        probabilities=probabilities,
        class_names=["noise", "a", "b"],
        frame_rate_hz=100.0,
        fill_gap_seconds=0.01,
        min_syllable_seconds=0.0,
        postprocessor="binary_mask",
    )
    label_aware_segments = main._probabilities_to_syllable_segments(
        probabilities=probabilities,
        class_names=["noise", "a", "b"],
        frame_rate_hz=100.0,
        fill_gap_seconds=0.01,
        min_syllable_seconds=0.0,
        postprocessor="label_aware_dense",
    )

    assert [segment["name"] for segment in binary_segments] == ["a"]
    assert [segment["name"] for segment in label_aware_segments] == ["a", "b"]


def test_binary_segment_thresholds_require_high_confidence_seed():
    sine_probabilities = np.array([0.1, 0.6, 0.7, 0.1, 0.3, 0.95, 0.3, 0.1], dtype=np.float32)
    probabilities = np.column_stack((1.0 - sine_probabilities, sine_probabilities))

    segments = main._probabilities_to_syllable_segments(
        probabilities=probabilities,
        class_names=["noise", "sine"],
        frame_rate_hz=100.0,
        fill_gap_seconds=0.0,
        min_syllable_seconds=0.0,
        segment_threshold_low=0.2,
        segment_threshold_high=0.9,
    )

    assert segments == [{"name": "sine", "start_seconds": 0.045, "stop_seconds": 0.065}]


def test_probabilities_to_syllable_annotations_support_mixed_segment_and_event_classes():
    probabilities = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.85, 0.05],
            [0.90, 0.05, 0.05],
            [0.05, 0.05, 0.90],
            [0.05, 0.05, 0.90],
            [0.90, 0.05, 0.05],
            [0.10, 0.85, 0.05],
            [0.90, 0.05, 0.05],
        ],
        dtype=np.float32,
    )

    annotations = main._probabilities_to_syllable_annotations(
        probabilities=probabilities,
        class_names=["noise", "pulse", "sine"],
        class_types=["segment", "event", "segment"],
        frame_rate_hz=100.0,
        fill_gap_seconds=0.0,
        min_syllable_seconds=0.01,
    )

    assert annotations.to_dict("records") == [
        {"name": "pulse", "start_seconds": 0.01, "stop_seconds": 0.01},
        {"name": "sine", "start_seconds": 0.035, "stop_seconds": 0.045},
        {"name": "pulse", "start_seconds": 0.06, "stop_seconds": 0.06},
    ]


def test_event_postprocessing_removes_isolated_events():
    probabilities = np.full((24, 2), [0.90, 0.05], dtype=np.float32)
    probabilities[[1, 3, 20]] = [0.10, 0.85]

    annotations = main._probabilities_to_syllable_annotations(
        probabilities=probabilities,
        class_names=["noise", "pulse"],
        class_types=["segment", "event"],
        frame_rate_hz=100.0,
        fill_gap_seconds=0.0,
        min_syllable_seconds=0.0,
        event_dist_max_seconds=0.05,
    )

    assert annotations.to_dict("records") == [
        {"name": "pulse", "start_seconds": 0.01, "stop_seconds": 0.01},
        {"name": "pulse", "start_seconds": 0.03, "stop_seconds": 0.03},
    ]


def test_event_postprocessing_uses_configured_threshold():
    probabilities = np.full((20, 2), [0.9, 0.05], dtype=np.float32)
    probabilities[10] = [0.35, 0.65]

    annotations = main._probabilities_to_syllable_annotations(
        probabilities=probabilities,
        class_names=["noise", "pulse"],
        class_types=["segment", "event"],
        frame_rate_hz=1000.0,
        fill_gap_seconds=0.0,
        min_syllable_seconds=0.0,
        event_threshold=0.7,
    )

    assert annotations.empty


def test_event_postprocessing_removes_close_events_with_first_event_exception():
    probabilities = np.full((120, 2), [0.90, 0.05], dtype=np.float32)
    probabilities[[10, 40, 100]] = [0.10, 0.85]

    annotations = main._probabilities_to_syllable_annotations(
        probabilities=probabilities,
        class_names=["noise", "pulse"],
        class_types=["segment", "event"],
        frame_rate_hz=1000.0,
        fill_gap_seconds=0.0,
        min_syllable_seconds=0.0,
        event_dist_min_seconds=0.05,
    )

    assert annotations.to_dict("records") == [
        {"name": "pulse", "start_seconds": 0.01, "stop_seconds": 0.01},
        {"name": "pulse", "start_seconds": 0.1, "stop_seconds": 0.1},
    ]


def test_event_postprocessing_filters_each_event_class_independently():
    probabilities = np.full((60, 3), [0.90, 0.05, 0.05], dtype=np.float32)
    probabilities[10] = [0.10, 0.85, 0.05]
    probabilities[30] = [0.10, 0.05, 0.85]

    annotations = main._probabilities_to_syllable_annotations(
        probabilities=probabilities,
        class_names=["noise", "pulse", "click"],
        class_types=["segment", "event", "event"],
        frame_rate_hz=1000.0,
        fill_gap_seconds=0.0,
        min_syllable_seconds=0.0,
        event_dist_min_seconds=0.05,
    )

    assert annotations.to_dict("records") == [
        {"name": "pulse", "start_seconds": 0.01, "stop_seconds": 0.01},
        {"name": "click", "start_seconds": 0.03, "stop_seconds": 0.03},
    ]


def test_syllable_classification_summary_uses_tolerance_for_matching(tmp_path: Path):
    annotation_path = tmp_path / "audio_annotations.csv"
    pd.DataFrame(
        [
            {"name": "a", "start_seconds": 0.010, "stop_seconds": 0.020},
            {"name": "b", "start_seconds": 0.040, "stop_seconds": 0.050},
        ]
    ).to_csv(annotation_path, index=False)

    logits = torch.full((1, 100, 3), fill_value=-8.0, dtype=torch.float32)
    logits[:, :, 0] = 8.0
    logits[:, 15:25, 0] = -8.0
    logits[:, 15:25, 1] = 8.0
    logits[:, 70:80, 0] = -8.0
    logits[:, 70:80, 1] = 8.0

    class FakeModel(torch.nn.Module):
        def forward(self, inputs, input_lengths):
            del inputs
            return logits.to(input_lengths.device), input_lengths

    batch = (
        torch.zeros((1, 100), dtype=torch.float32),
        torch.tensor([100], dtype=torch.long),
        torch.zeros((1, 100, 3), dtype=torch.float32),
        torch.tensor([100], dtype=torch.long),
    )

    dataset = SimpleNamespace(
        audio_files=[tmp_path / "audio.wav"],
        annotation_files=[annotation_path],
        chunk_borders=np.array([0, 1]),
        chunk_start_samples=np.array([0]),
        nb_samples_in_file=np.array([100]),
        samplerate_per_file=np.array([1000]),
        hop_s=0.001,
        num_time_steps=100,
        class_names=["noise", "a", "b"],
    )

    class FakeLoader:
        def __init__(self, dataset, batch):
            self.dataset = dataset
            self._batch = batch

        def __iter__(self):
            yield self._batch

    matrix, report, wer = main._syllable_classification_summary(
        model=FakeModel(),
        dataloader=FakeLoader(dataset, batch),
        class_names=["noise", "a", "b"],
        device=torch.device("cpu"),
        frame_rate_hz=1000.0,
        fill_gap_seconds=0.0,
        min_syllable_seconds=0.001,
        tolerance_seconds=0.010,
    )

    assert matrix.shape == (3, 3)
    assert "a" in report
    assert wer >= 0.0
