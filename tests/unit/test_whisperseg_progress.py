import io

import numpy as np
import pytest
from tqdm import tqdm

import das.whisperseg.model as whisper_model


@pytest.mark.parametrize("segmenter_type", [whisper_model.WhisperSegmenter, whisper_model.WhisperSegmenterForEval])
@pytest.mark.parametrize("num_samples", [0, 20, 41])
@pytest.mark.parametrize("num_trials", [1, 3])
@pytest.mark.parametrize("show_progress", [False, True])
def test_chunk_progress_tracks_completed_batches(monkeypatch, segmenter_type, num_samples, num_trials, show_progress):
    segmenter = object.__new__(segmenter_type)
    segmenter.total_spec_columns = 80
    output = io.StringIO()
    bars = []
    completed = 0

    def progress_bar(**kwargs):
        bar = tqdm(**kwargs, file=output, mininterval=0)
        bars.append(bar)
        return bar

    def generate_text(batch, *_args):
        nonlocal completed
        assert bars[0].n == (completed if show_progress else 0)
        completed += len(batch)
        return ["chunk"] * len(batch)

    monkeypatch.setattr(whisper_model, "tqdm", progress_bar)
    monkeypatch.setattr(
        whisper_model, "WhisperSegFeatureExtractor",
        lambda *args, **kwargs: lambda *args, **kwargs: {"input_features": [np.ones((2, 80))]},
    )
    monkeypatch.setattr(segmenter, "generate_segment_text", generate_text)

    texts, metadata = segmenter.generate_segment_records(
        np.zeros(num_samples), 100, 0, 0.0025, num_trials, 2, 128, 1, 1, 1.0, 1.0, None,
        show_progress=show_progress,
    )

    assert bars[0].total == len(texts) == len(metadata) == completed
    if show_progress:
        assert bars[0].n == completed
        assert "Predicting chunks" in output.getvalue()
        assert "100%" in output.getvalue()
    else:
        assert output.getvalue() == ""
    assert bars[0].disable  # tqdm closes the bar on exit.
