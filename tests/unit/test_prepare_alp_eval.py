import runpy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import soundfile as sf


prepare = runpy.run_path(str(Path(__file__).parents[2] / "scripts/prepare_alp_eval.py"))


def test_candidates_reject_cut_calls_and_nested_overlaps():
    events = pd.DataFrame({
        "start": [1.0, 29.0, 65.0, 66.0, 68.0, 95.0, 96.0],
        "end": [2.0, 31.0, 75.0, 67.0, 69.0, 96.0, 97.0],
        "label": ["call"] * 7,
    })
    clips = prepare["candidates"](events, 150)
    assert [c["start"] for c in clips] == [90, 120]
    assert len(clips[0]["events"]) == 2  # Touching events are not overlaps.
    assert clips[1]["events"] == []


def test_select_covers_labels_keeps_negative_and_limits_recordings():
    clips = [{"source": "a", "events": []},
             {"source": "a", "events": [{"label": "common"}]},
             {"source": "b", "events": [{"label": "common"}, {"label": "rare"}]},
             {"source": "c", "events": [{"label": "common"}]}]
    selected = prepare["select"](clips, 3, random.Random(42), max_per_source=1)
    assert {c["source"] for c in selected} == {"a", "b", "c"}
    assert selected[0]["events"] == []
    assert {e["label"] for c in selected for e in c["events"]} == {"common", "rare"}
    assert selected == prepare["select"](clips, 3, random.Random(42), max_per_source=1)
    with pytest.raises(ValueError, match="eligible clips"):
        prepare["select"](clips, 4, random.Random(42), max_per_source=1)


def test_candidates_exclude_window_with_zero_duration_source_label():
    events = pd.DataFrame({"start": [12.0], "end": [12.0], "label": ["call"]})
    assert [c["start"] for c in prepare["candidates"](events, 60)] == [30]


def test_select_never_reuses_overlapping_source_audio():
    clips = [{"source": "a", "start": start, "end": start + 30, "events": []}
             for start in range(0, 61, 5)]
    selected = prepare["select"](clips, 2, random.Random(42))
    assert abs(selected[0]["start"] - selected[1]["start"]) >= 30


def test_extract_preserves_samples_and_rebases_labels(tmp_path):
    source = tmp_path / "source.wav"
    samples = np.linspace(-0.5, 0.5, 6000, dtype=np.float32)
    sf.write(source, samples, 100, subtype="FLOAT")
    plan = {"clips": [{"source": str(source), "file": "train/clip.wav", "start": 30,
                       "events": [{"start": 31.1, "end": 31.4, "label": "Phee"}]}]}
    output = tmp_path / "output"
    prepare["extract"](plan, output)
    audio, rate = sf.read(output / "train/clip.wav", dtype="float32")
    assert rate == 100
    np.testing.assert_array_equal(audio, samples[3000:])
    labels = pd.read_csv(output / "train/clip.csv")
    assert labels.name.tolist() == ["Phee"]
    np.testing.assert_allclose(labels.start_seconds, [1.1])
    np.testing.assert_allclose(labels.stop_seconds, [1.4])
