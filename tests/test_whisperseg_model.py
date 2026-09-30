from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from das.whisperseg.datautils import (
    VocalSegDataset,
    build_vocalseg_clip_records,
    build_vocalseg_interval_records,
    get_audio_and_label_paths,
    get_cluster_codebook,
    read_label,
)
from das.whisperseg.model import (
    WHISPERSEG_CHECKPOINT_FORMAT,
    WHISPERSEG_CHECKPOINT_FORMAT_VERSION,
    SegmenterBase,
    WhisperSegmenter,
    WhisperSegmenterForEval,
    _make_weights_only_safe,
    _torch_load,
    is_model_checkpoint_path,
    load_model_checkpoint,
)


def test_generation_casts_features_to_model_dtype():
    model = torch.nn.Linear(2, 2).half()

    def generate(**kwargs):
        assert kwargs["input_features"].dtype == torch.float16
        assert kwargs["force_unique_generate_call"] is True
        return torch.tensor([[1]])

    model.generate = generate
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda _: [1, 2, 3], pad_token_id=0, eos_token_id=1,
        batch_decode=lambda *args, **kwargs: ["segment"],
    )
    features = [(0, 0.0, np.ones((2, 2), dtype=np.float32), 1.0)]
    evaluator = object.__new__(WhisperSegmenterForEval)
    evaluator.model, evaluator.tokenizer, evaluator.device = model, tokenizer, torch.device("cpu")
    assert evaluator.generate_segment_text(features, 1, 10, 1) == ["segment"]
    predictor = object.__new__(WhisperSegmenter)
    predictor.model_list, predictor.tokenizer_list, predictor.device_list = [model], [tokenizer], [torch.device("cpu")]
    generated = {}
    predictor.generate_segment_text_core(features, 1, 10, 1, 1, 1.0, 1.0, generated, 0)
    assert generated == {0: ["segment"]}


def test_clustering_consolidation_preserves_overlapping_classes():
    segmenter = object.__new__(SegmenterBase)
    trials = [
        {"onset": [1.00, 1.01], "offset": [1.20, 1.21], "cluster": ["A", "B"]},
        {"onset": [1.01, 1.00], "offset": [1.21, 1.20], "cluster": ["A", "B"]},
        {"onset": [0.99, 1.02], "offset": [1.19, 1.22], "cluster": ["A", "B"]},
    ]

    prediction = segmenter.consolidate_trials_by_clustering(trials, eps=0.05, min_samples=2)

    assert prediction["cluster"] == ["A", "B"]


def test_eval_segmenter_streams_features_in_batches(monkeypatch):
    segmenter = object.__new__(WhisperSegmenterForEval)
    pending = 0
    max_pending = 0

    def iter_features(*_args, **kwargs):
        nonlocal pending, max_pending
        assert kwargs["max_frequency"] == 123
        for index in range(5):
            pending += 1
            max_pending = max(max_pending, pending)
            yield 0, float(index), np.full((2, 2), index, dtype=np.float32), 1.0

    def generate_text(batch, *_args):
        nonlocal pending
        pending -= len(batch)
        return [f"chunk-{int(item[1])}" for item in batch]

    monkeypatch.setattr(segmenter, "iter_sliced_audio_features", iter_features)
    monkeypatch.setattr(segmenter, "generate_segment_text", generate_text)

    texts, metadata = segmenter.generate_segment_records(
        np.zeros(1), 1, 0, 0.0025, 1, 2, 448, 1, 1, 1.0, 1.0, None, 123
    )

    assert texts == ["chunk-0", "chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    assert metadata == [(0, float(index), None, 1.0) for index in range(5)]
    assert pending == 0
    assert max_pending == 2


def test_torch_load_accepts_checkpoint_with_numpy_scalars(tmp_path: Path):
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save({"value": np.float64(1.5), "dtype": np.dtype("float64")}, checkpoint_path)

    loaded = _torch_load(checkpoint_path)

    assert loaded["value"] == np.float64(1.5)
    assert loaded["dtype"] == np.dtype("float64")


def test_make_weights_only_safe_converts_numpy_metadata():
    safe = _make_weights_only_safe(
        {
            "scalar": np.float64(1.5),
            "array": np.array([1, 2]),
            "dtype": np.dtype("float32"),
        }
    )

    assert safe == {"scalar": 1.5, "array": [1, 2], "dtype": "float32"}


def test_old_whisperseg_model_pt_bundle_is_not_checkpoint(tmp_path: Path):
    bundle_path = tmp_path / "model.pt"
    torch.save({"format": "das_whisper.model_bundle", "format_version": 1}, bundle_path)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "model.pt").write_bytes(b"bundle")

    assert is_model_checkpoint_path(bundle_path) is False
    assert is_model_checkpoint_path(bundle_dir) is False


def test_load_model_checkpoint_reconstructs_embedded_model(tmp_path: Path, monkeypatch):
    from das.whisperseg import model as whisper_model

    checkpoint_path = tmp_path / "model.ckpt"
    state_dict = {"weight": torch.tensor([1.0])}
    torch.save(
        {
            "das": {"backend": "whisperseg"},
            "whisperseg": {
                "format": WHISPERSEG_CHECKPOINT_FORMAT,
                "format_version": WHISPERSEG_CHECKPOINT_FORMAT_VERSION,
                "model_config": {"hidden_size": 4},
                "generation_config": {"max_length": 12},
                "model_state_dict": state_dict,
                "tokenizer_files": {"vocab.json": b"{}"},
                "metadata": {"current_step": 3},
            },
        },
        checkpoint_path,
    )

    class FakeWhisper:
        def __init__(self, config):
            self.config = config
            self.loaded_state_dict = None

        def load_state_dict(self, state, *, assign=False):
            assert assign
            self.loaded_state_dict = state

        def tie_weights(self):
            pass

    monkeypatch.setattr(whisper_model.WhisperConfig, "from_dict", lambda payload: ("model_config", payload))
    monkeypatch.setattr(whisper_model.GenerationConfig, "from_dict", lambda payload: ("generation_config", payload))
    monkeypatch.setattr(whisper_model, "WhisperForConditionalGeneration", FakeWhisper)
    monkeypatch.setattr(whisper_model, "_restore_tokenizer", lambda files: ("tokenizer", files))

    assert is_model_checkpoint_path(checkpoint_path) is True
    model, tokenizer = load_model_checkpoint(checkpoint_path)

    assert model.config == ("model_config", {"hidden_size": 4})
    assert torch.equal(model.loaded_state_dict["weight"], state_dict["weight"])
    assert model.generation_config == ("generation_config", {"max_length": 12})
    assert tokenizer == ("tokenizer", {"vocab.json": b"{}"})


def test_load_model_checkpoint_preserves_embedded_dtype(tmp_path: Path, monkeypatch):
    from das.whisperseg import model as whisper_model

    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "whisperseg": {
                "format": WHISPERSEG_CHECKPOINT_FORMAT,
                "format_version": WHISPERSEG_CHECKPOINT_FORMAT_VERSION,
                "model_config": {"dtype": "float16"},
                "generation_config": None,
                "model_state_dict": {},
                "tokenizer_files": {},
            },
        },
        checkpoint_path,
    )

    class FakeWhisper:
        def __init__(self, config):
            self.config = config
            self.dtype = None

        def load_state_dict(self, state, *, assign=False):
            assert assign
            assert state == {}
            self.dtype = self.config.dtype

        def tie_weights(self):
            pass

    monkeypatch.setattr(whisper_model, "WhisperForConditionalGeneration", FakeWhisper)
    monkeypatch.setattr(whisper_model, "_restore_tokenizer", lambda _files: object())

    model, _ = load_model_checkpoint(checkpoint_path)

    assert model.dtype == torch.float16


def test_load_model_applies_separate_encoder_and_decoder_dropout(monkeypatch):
    from das.whisperseg import model as whisper_model

    class FakeLayer:
        def __init__(self):
            self.dropout = None

    class FakeEmbedPositions:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.arange(20, dtype=torch.float32).reshape(10, 2))
            self.num_embeddings = 10

    class FakeEncoder:
        def __init__(self):
            self.dropout = None
            self.layers = [FakeLayer(), FakeLayer()]
            self.embed_positions = FakeEmbedPositions()

    class FakeDecoder:
        def __init__(self):
            self.dropout = None
            self.layers = [FakeLayer(), FakeLayer()]

    class FakeConfig:
        pass

    class FakeModel:
        def __init__(self):
            self.config = FakeConfig()
            self.model = type("FakeWhisper", (), {"encoder": FakeEncoder(), "decoder": FakeDecoder()})()

    class FakeTokenizer:
        def __init__(self):
            self.added_tokens = []

        def add_tokens(self, tokens, special_tokens=False):
            self.added_tokens.append((tokens, special_tokens))

    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr(whisper_model, "load_model_checkpoint", lambda _path: (fake_model, fake_tokenizer))

    model, tokenizer = whisper_model.load_model(
        "model.ckpt",
        total_spec_columns=8,
        encoder_dropout=0.2,
        decoder_dropout=0.3,
    )

    assert model is fake_model
    assert tokenizer is fake_tokenizer
    assert model.config.total_spec_columns == 8
    assert model.config.encoder_dropout == 0.2
    assert model.config.decoder_dropout == 0.3
    assert model.model.encoder.dropout == 0.2
    assert [layer.dropout for layer in model.model.encoder.layers] == [0.2, 0.2]
    assert model.model.decoder.dropout == 0.3
    assert [layer.dropout for layer in model.model.decoder.layers] == [0.3, 0.3]
    assert model.model.encoder.embed_positions.weight.shape[0] == 4
    assert model.model.encoder.embed_positions.num_embeddings == 4
    assert len(fake_tokenizer.added_tokens[0][0]) == 9


def test_whisperseg_read_label_excludes_annotation_markers(tmp_path: Path):
    label_path = tmp_path / "clip.csv"
    pd.DataFrame(
        [
            {"name": "annotation_start", "start_seconds": 0.1, "stop_seconds": 0.1},
            {"name": "song", "start_seconds": 0.2, "stop_seconds": 0.3},
            {"name": "annotation_end", "start_seconds": 0.4, "stop_seconds": 0.4},
        ]
    ).to_csv(label_path, index=False)

    label = read_label(label_path)

    assert label["cluster"] == ["song"]
    assert label["onset"] == [0.2]
    assert label["offset"] == [0.3]
    assert label["_annotation_markers"] == [("annotation_start", 0.1), ("annotation_end", 0.4)]


def test_whisperseg_include_clusters_filters_label_rows_and_codebook(tmp_path: Path):
    label_path = tmp_path / "clip.csv"
    pd.DataFrame(
        [
            {"name": "pulse", "start_seconds": 0.1, "stop_seconds": 0.2},
            {"name": "sine", "start_seconds": 0.3, "stop_seconds": 0.4},
        ]
    ).to_csv(label_path, index=False)

    label = read_label(label_path, include_clusters=["pulse"])
    codebook = get_cluster_codebook([label_path], {}, ignore_cluster=False, include_clusters=["pulse"])

    assert label["cluster"] == ["pulse"]
    assert codebook == {"pulse": 0}


def test_whisperseg_interval_records_crop_to_annotation_markers(tmp_path: Path):
    audio_path = tmp_path / "clip.wav"
    label_path = tmp_path / "clip.csv"
    audio = np.linspace(-0.5, 0.5, 1_000, dtype=np.float32)
    sf.write(audio_path, audio, 1_000, subtype="FLOAT")
    pd.DataFrame(
        [
            {"name": "annotation_start", "start_seconds": 0.2, "stop_seconds": 0.2},
            {"name": "song", "start_seconds": 0.25, "stop_seconds": 0.3},
            {"name": "annotation_end", "start_seconds": 0.5, "stop_seconds": 0.5},
            {"name": "outside", "start_seconds": 0.7, "stop_seconds": 0.8},
        ]
    ).to_csv(label_path, index=False)

    records = build_vocalseg_interval_records(
        [audio_path],
        [label_path],
        cluster_codebook={"song": 0, "outside": 1},
        default_config={
            "sr": 1_000,
            "input_sr": 1_000,
            "frequency_scale": 1.0,
            "spec_time_step": 0.001,
            "min_frequency": 0,
        },
    )

    assert len(records) == 1
    assert records[0].num_samples == 300
    assert records[0].label["cluster"] == ["song"]
    assert all(0 <= onset <= 0.3 for onset in records[0].label["onset"])
    assert "_annotation_markers" in records[0].label


def test_whisperseg_finds_and_indexes_non_wav_soundfile_audio(tmp_path: Path):
    audio_path = tmp_path / "clip.flac"
    label_path = tmp_path / "clip.csv"
    audio = np.linspace(-0.5, 0.5, 1_000, dtype=np.float32)
    sf.write(audio_path, audio, 1_000)
    pd.DataFrame(
        [{"name": "song", "start_seconds": 0.25, "stop_seconds": 0.3}]
    ).to_csv(label_path, index=False)

    audio_paths, label_paths = get_audio_and_label_paths(tmp_path)
    records = build_vocalseg_interval_records(
        audio_paths,
        label_paths,
        cluster_codebook={"song": 0},
        default_config={
            "sr": 1_000,
            "input_sr": 1_000,
            "frequency_scale": 1.0,
            "spec_time_step": 0.001,
            "min_frequency": 0,
        },
    )

    assert audio_paths == [str(audio_path)]
    assert label_paths == [str(label_path)]
    assert len(records) == 1
    assert records[0].label["cluster"] == ["song"]


def test_whisperseg_include_clusters_keeps_audio_when_all_labels_are_filtered(tmp_path: Path):
    audio_path = tmp_path / "clip.wav"
    label_path = tmp_path / "clip.csv"
    audio = np.linspace(-0.5, 0.5, 1_000, dtype=np.float32)
    sf.write(audio_path, audio, 1_000, subtype="FLOAT")
    pd.DataFrame(
        [{"name": "sine", "start_seconds": 0.25, "stop_seconds": 0.3}]
    ).to_csv(label_path, index=False)

    records = build_vocalseg_interval_records(
        [audio_path],
        [label_path],
        cluster_codebook={},
        default_config={
            "sr": 1_000,
            "input_sr": 1_000,
            "frequency_scale": 1.0,
            "spec_time_step": 0.001,
            "min_frequency": 0,
        },
        include_clusters=["pulse"],
    )

    assert len(records) == 1
    assert records[0].label["cluster"] == []


def test_vocalseg_dataset_reads_bounded_audio_chunk_on_getitem(tmp_path: Path, monkeypatch):
    from das.whisperseg import datautils

    audio_path = tmp_path / "clip.wav"
    label_path = tmp_path / "clip.csv"
    sf.write(audio_path, np.linspace(-0.5, 0.5, 1_000, dtype=np.float32), 1_000, subtype="FLOAT")
    pd.DataFrame(
        [{"name": "song", "start_seconds": 0.25, "stop_seconds": 0.3}]
    ).to_csv(label_path, index=False)

    train_records, _, _ = build_vocalseg_clip_records(
        [audio_path],
        [label_path],
        cluster_codebook={"song": 0},
        default_config={
            "sr": 1_000,
            "input_sr": 1_000,
            "frequency_scale": 1.0,
            "spec_time_step": 0.001,
            "min_frequency": 0,
            "max_frequency": 400,
        },
        total_spec_columns=100,
    )

    class FakeFeatureExtractor:
        n_fft = 4
        created_with = None

        def __init__(self, *args, **kwargs):
            self.__class__.created_with = (args, kwargs)

        def __call__(self, audio_clip, **kwargs):
            del kwargs
            return {"input_features": [np.ones((3, max(1, len(audio_clip) // 10)), dtype=np.float32)]}

    class FakeTokenizer:
        pad_token_id = 0

        def encode(self, text, **kwargs):
            del text, kwargs
            return [1, 2, 3]

    calls = []

    def fake_read_chunk(audio_file, start, frames, target_samplerate):
        del audio_file
        calls.append((start, frames, target_samplerate))
        return np.ones(frames, dtype=np.float32)

    monkeypatch.setattr(datautils, "WhisperSegFeatureExtractor", FakeFeatureExtractor)
    monkeypatch.setattr(datautils, "_read_resampled_audio_chunk", fake_read_chunk)

    dataset = VocalSegDataset(train_records, FakeTokenizer(), 4, 100, {"unknown": "<|unknown|>"})
    assert FakeFeatureExtractor.created_with[0] == (1_000, 0.001, 0, 400)
    assert calls == []

    item = dataset[0]

    assert set(item) == {"input_features", "decoder_input_ids", "labels"}
    assert calls
    assert max(frames for _, frames, _ in calls) < 1_000
