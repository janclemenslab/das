from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import transformers.pipelines as pipelines

from das import api, cli
from das.config import Config
from das.whisper_asr import WhisperTranscriber


@pytest.fixture
def speech_pipeline(monkeypatch):
    calls = {}

    class FakePipeline:
        feature_extractor = SimpleNamespace(sampling_rate=16000)

        def __call__(self, audio, **kwargs):
            calls["audio"] = audio
            calls["generation"] = kwargs
            return {
                "text": " Hello world!",
                "chunks": [
                    {"text": " Hello", "timestamp": (0.1, 0.4)},
                    {"text": " world!", "timestamp": (0.5, None)},
                ],
            }

    def factory(task, **kwargs):
        calls["task"] = task
        calls["load"] = kwargs
        return FakePipeline()

    monkeypatch.setattr(pipelines, "pipeline", factory)
    return calls


def test_speech_cli_is_prediction_only():
    config = cli.parse_command_config(["predict", "--data-dir", "data", "--checkpoint", "whisper:large"])
    assert config.checkpoint == "whisper:large"
    with pytest.raises(ValueError, match="Unknown Whisper speech model"):
        Config(mode="predict", data_dir="data", checkpoint="whisper:unknown").validate()
    with pytest.raises(ValueError, match="initial_model"):
        Config.from_mapping({"encoder_type": "whisperseg", "initial_model": "whisper:large"})


def test_transcription_loads_original_fp16_model_and_preserves_words(speech_pipeline):
    transcriber = WhisperTranscriber("openai/whisper-large", device="cpu")
    frame = transcriber.transcribe(np.zeros(16000, dtype=np.float32), num_beams=2, length_penalty=1.0)
    assert speech_pipeline["task"] == "automatic-speech-recognition"
    assert speech_pipeline["load"] == {
        "model": "openai/whisper-large",
        "device": "cpu",
        "dtype": torch.float16,
        "model_kwargs": {"attn_implementation": "eager"},
    }
    assert speech_pipeline["generation"] == {
        "return_timestamps": "word",
        "generate_kwargs": {"task": "transcribe", "num_beams": 2, "length_penalty": 1.0},
    }
    assert frame.to_dict("records") == [
        {"name": "Hello", "start_seconds": 0.1, "stop_seconds": 0.4},
        {"name": "world!", "start_seconds": 0.5, "stop_seconds": 1.0},
    ]


def test_speech_raw_prediction_resamples_stereo_without_segmentation_scaling(speech_pipeline):
    frame = api.predict(
        checkpoint="whisper:large",
        accelerator="cpu",
        audio=np.zeros((44100, 2), dtype=np.float32),
        samplerate=44100,
        frequency_scale=2.0,
        generation_max_length=1,
    )
    assert speech_pipeline["audio"].shape == (16000,)
    assert speech_pipeline["audio"].dtype == np.float32
    assert frame["name"].tolist() == ["Hello", "world!"]
    assert frame["stop_seconds"].tolist() == [0.4, 1.0]


def test_speech_file_prediction_writes_timed_das_annotations(tmp_path, speech_pipeline):
    audio_file = tmp_path / "speech.wav"
    sf.write(audio_file, np.zeros(48000, dtype=np.float32), 48000)
    result = api.predict(checkpoint="whisper:large", accelerator="cpu", data_dir=str(audio_file))
    assert len(result) == 1
    annotations = pd.read_csv(result[0])
    assert annotations.columns.tolist() == ["name", "start_seconds", "stop_seconds"]
    assert annotations["name"].tolist() == ["Hello", "world!"]
    assert annotations["start_seconds"].tolist() == [0.1, 0.5]
    assert annotations["stop_seconds"].tolist() == [0.4, 1.0]


def test_speech_silence_and_long_recording_timestamps(speech_pipeline):
    transcriber = WhisperTranscriber("openai/whisper-large", device="cpu")
    transcriber.pipeline = lambda *args, **kwargs: {"chunks": [{"text": " late", "timestamp": (35.0, 35.8)}]}
    frame = transcriber.transcribe(np.zeros(40 * 16000), num_beams=1, length_penalty=1.0, filename="long.wav")
    assert frame.to_dict("records") == [{"filename": "long.wav", "name": "late", "start_seconds": 35.0, "stop_seconds": 35.8}]
    transcriber.pipeline = lambda *args, **kwargs: {"chunks": []}
    frame = transcriber.transcribe(np.zeros(16000), num_beams=1, length_penalty=1.0)
    assert frame.empty
    assert frame.columns.tolist() == ["name", "start_seconds", "stop_seconds"]


def test_speech_rejects_segmentation_evaluation_before_model_load(monkeypatch):
    monkeypatch.setattr(api, "_whisper_segmenter", lambda config: pytest.fail("must reject before loading"))
    with pytest.raises(ValueError, match="Segmentation evaluation"):
        api.predict(checkpoint="whisper:large", evaluate=True, audio=np.zeros(16000), samplerate=16000)
