"""Full-vocabulary Whisper speech transcription with word timestamps."""

import pandas as pd
import torch
from transformers.pipelines import AutomaticSpeechRecognitionPipeline


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
            rows.append({"name": word, "start_seconds": start, "stop_seconds": stop})
        frame = pd.DataFrame(rows, columns=["name", "start_seconds", "stop_seconds"])
        if filename is not None:
            frame.insert(0, "filename", filename)
        return frame
