from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import transformers.pipelines as pipelines

from das import api, cli
from das.config import Config
from das.whisper_asr import WhisperTranscriber, _WhisperWordTimestampPipeline


@pytest.mark.parametrize("timestamp_format", ["tensor", "longform_list"])
def test_word_decoder_uses_current_and_next_token_boundaries(monkeypatch, speech_pipeline, timestamp_format):
    from transformers.models.whisper.tokenization_whisper import _decode_asr

    # A tiny vocabulary keeps the real Transformers word decoder offline.
    class Tokenizer:
        eos_token_id = 9
        all_special_ids = list(range(9, 15))
        language = "english"
        vocabulary = {
            0: " This",
            1: " is",
            2: " a",
            3: " speech",
            4: " record",
            5: "ing",
            6: ".",
            9: "<|endoftext|>",
            10: "<|startoftranscript|>",
            11: "<|en|>",
            12: "<|transcribe|>",
            13: "<|startofprev|>",
            14: "<|notimestamps|>",
        }

        def convert_tokens_to_ids(self, token):
            return {value: key for key, value in self.vocabulary.items()}[token]

        def decode(self, ids, **kwargs):
            return "".join(self.vocabulary[i] for i in ids)

        def _strip_prompt(self, ids, *args):
            return ids

    ids = [10, 11, 12, 15, 0, 1, 2, 3, 4, 5, 6, 75, 1765, 0, 1, 2, 3, 4, 5, 6, 1825, 9]
    onsets = torch.tensor(
        [[0, 0, 0, 0, 0, 0.2, 0.4, 0.5, 0.8, 0.9, 1.1, 1.2, 35, 35, 35.2, 35.4, 35.5, 35.8, 35.9, 36.1, 36.2, 36.2]]
    )
    output = {"tokens": torch.tensor([ids]), "token_timestamps": onsets if timestamp_format == "tensor" else list(onsets)}
    monkeypatch.setattr(pipelines.AutomaticSpeechRecognitionPipeline, "_forward", lambda *args, **kwargs: output)
    WhisperTranscriber("openai/whisper-large", device="cpu")
    pipeline_class = speech_pipeline["load"]["pipeline_class"]
    pipe = pipeline_class.__new__(pipeline_class)
    forwarded = pipe._forward({}, return_timestamps="word")
    text, decoded = _decode_asr(Tokenizer(), [forwarded], return_timestamps="word", return_language=False, time_precision=0.02)
    assert text == " This is a speech recording. This is a speech recording."
    assert decoded["chunks"] == [
        {"text": " This", "timestamp": (0.0, 0.2)},
        {"text": " is", "timestamp": (0.2, 0.4)},
        {"text": " a", "timestamp": (0.4, 0.5)},
        {"text": " speech", "timestamp": (0.5, 0.8)},
        {"text": " recording.", "timestamp": (0.8, 1.2)},
        {"text": " This", "timestamp": (35.0, 35.2)},
        {"text": " is", "timestamp": (35.2, 35.4)},
        {"text": " a", "timestamp": (35.4, 35.5)},
        {"text": " speech", "timestamp": (35.5, 35.8)},
        {"text": " recording.", "timestamp": (35.8, 36.2)},
    ]


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
        "pipeline_class": _WhisperWordTimestampPipeline,
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
