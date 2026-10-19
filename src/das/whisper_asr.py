"""Full-vocabulary Whisper speech transcription with word timestamps."""

from functools import partial

import pandas as pd
import torch
from transformers.pipelines import AutomaticSpeechRecognitionPipeline


def _retrieve_word_segments(retrieve_segment, tokenizer, **kwargs):
    segments, offset = retrieve_segment(**kwargs)
    window_frames = int(kwargs["seek_num_frames"][kwargs["prev_idx"]])
    if kwargs["return_token_timestamps"] and offset < window_frames:
        segment = segments[-1]
        text_indices = (segment["tokens"] < tokenizer.eos_token_id).nonzero().flatten().tolist()
        # Punctuation has no speech duration and can align deep into the following pause.
        while text_indices and all(
            char in "\"'.。,，!！?？:：”)]}、" for char in tokenizer.decode([int(segment["tokens"][text_indices[-1]])]).strip()
        ):
            text_indices.pop()
        if text_indices:
            end = float(segment["token_timestamps"][text_indices[-1] + 1])
            start = float(kwargs["time_offset"][kwargs["prev_idx"]])
            offset = min(window_frames, max(1, round((end - start) / kwargs["time_precision_features"])))
    return segments, offset


class _WhisperWordTimestampPipeline(AutomaticSpeechRecognitionPipeline):
    def _forward(self, model_inputs, return_timestamps=False, **generate_kwargs):
        output = super()._forward(model_inputs, return_timestamps=return_timestamps, **generate_kwargs)
        if return_timestamps == "word":
            # Transformers 4.57's word decoder expects token ends; Whisper generates token onsets.
            timestamps = output["token_timestamps"]
            if isinstance(timestamps, torch.Tensor):
                output["token_timestamps"] = torch.cat((timestamps[..., 1:], timestamps[..., -1:]), dim=-1)
            else:
                output["token_timestamps"] = [torch.cat((row[1:], row[-1:])) for row in timestamps]
        return output


class WhisperTranscriber:
    def __init__(self, model: str, *, device: str):
        from transformers.pipelines import pipeline

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=device,
            dtype=torch.float16,
            model_kwargs={"attn_implementation": "eager"},
            pipeline_class=_WhisperWordTimestampPipeline,
        )
        # Match OpenAI Whisper's word-aligned seek instead of decoding already transcribed speech again.
        self.pipeline.model._retrieve_segment = partial(
            _retrieve_word_segments, self.pipeline.model._retrieve_segment, self.pipeline.tokenizer
        )
        self.samplerate = self.pipeline.feature_extractor.sampling_rate

    def transcribe(self, audio, *, num_beams: int, length_penalty: float, filename: str | None = None) -> pd.DataFrame:
        # Whisper handles long recordings itself; do not apply WhisperSeg's short windows/token cap.
        result = self.pipeline(
            audio,
            return_timestamps="word",
            generate_kwargs={"task": "transcribe", "num_beams": num_beams, "length_penalty": length_penalty},
        )
        duration = len(audio) / self.samplerate
        rows = []
        for chunk in result["chunks"]:
            word = chunk["text"].strip()
            if not word:
                continue
            start, stop = chunk["timestamp"]
            # An unfinished word at the recording boundary can have an open timestamp.
            start = (rows[-1]["stop_seconds"] if rows else 0.0) if start is None else float(start)
            stop = duration if stop is None else float(stop)
            start = min(max(start, 0.0), duration)
            stop = min(max(stop, start), duration)
            if rows and rows[-1]["stop_seconds"] > start:
                rows[-1]["stop_seconds"] = start
            rows.append({"name": word, "start_seconds": start, "stop_seconds": stop})
        frame = pd.DataFrame(rows, columns=["name", "start_seconds", "stop_seconds"])
        if filename is not None:
            frame.insert(0, "filename", filename)
        return frame
