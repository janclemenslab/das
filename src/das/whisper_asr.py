"""Full-vocabulary Whisper speech transcription with word timestamps."""

from bisect import bisect_right
from functools import partial

import pandas as pd
import torch
from transformers.pipelines import AutomaticSpeechRecognitionPipeline


def _is_punctuation(tokenizer, token):
    return all(char in "\"'.。,，!！?？:：”)]}、" for char in tokenizer.decode([int(token)]).strip())


def _retrieve_word_segments(retrieve_segment, tokenizer, **kwargs):
    segments, offset = retrieve_segment(**kwargs)
    window_frames = int(kwargs["seek_num_frames"][kwargs["prev_idx"]])
    if kwargs["return_token_timestamps"] and offset < window_frames:
        segment = segments[-1]
        text_indices = (segment["tokens"] < tokenizer.eos_token_id).nonzero().flatten().tolist()
        # Punctuation has no speech duration and can align deep into the following pause.
        while text_indices and _is_punctuation(tokenizer, segment["tokens"][text_indices[-1]]):
            text_indices.pop()
        if text_indices:
            end = float(segment["token_timestamps"][text_indices[-1] + 1])
            start = float(kwargs["time_offset"][kwargs["prev_idx"]])
            offset = min(window_frames, max(1, round((end - start) / kwargs["time_precision_features"])))
    return segments, offset


def _refine_word_timestamps(rows, speech_regions):
    ends = [stop for _, stop in speech_regions]
    for row in rows:
        start, stop = row["start_seconds"], row["stop_seconds"]
        anchor_stop = min(stop, row.pop("_lexical_end", stop))
        row["stop_seconds"] = anchor_stop
        best_start, best_stop = start, start
        index = bisect_right(ends, start)
        while index < len(speech_regions) and speech_regions[index][0] < anchor_stop:
            speech_start, speech_stop = speech_regions[index]
            left, right = max(start, speech_start), min(anchor_stop, speech_stop)
            if right - left > best_stop - best_start:
                best_start, best_stop = left, right
                # Retain the final sound when punctuation aligns before the acoustic speech end.
                refined_stop = speech_stop if speech_stop <= stop else anchor_stop
            index += 1
        # Keep uncertain/undetected words; Whisper's alignment resolution is 20 ms.
        if best_stop - best_start >= 0.02:
            row["start_seconds"], row["stop_seconds"] = best_start, refined_stop


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

    def postprocess(self, model_outputs, **kwargs):
        result = super().postprocess([dict(output) for output in model_outputs], **kwargs)
        if kwargs.get("return_timestamps") == "word":
            end_outputs = []
            for output in model_outputs:
                rows = []
                for tokens, timestamps in zip(output["tokens"], output["token_timestamps"]):
                    ends = timestamps.clone()
                    for index, token in enumerate(tokens.tolist()):
                        if index and token < self.tokenizer.eos_token_id and _is_punctuation(self.tokenizer, token):
                            ends[index] = timestamps[index - 1]
                    rows.append(ends)
                end_outputs.append({**output, "token_timestamps": rows})
            # Punctuation must not add silence to a word or move the following word's onset.
            endings = super().postprocess(end_outputs, **kwargs)
            for word, ending in zip(result["chunks"], endings["chunks"]):
                start = word["timestamp"][0]
                stop = ending["timestamp"][1]
                if stop is not None and (start is None or stop > start):
                    word["lexical_end"] = stop
        return result


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
        self._vad_model = None

    def _speech_regions(self, audio):
        threads = torch.get_num_threads()
        try:
            # Silero changes this global setting on import; restore it for Whisper and other jobs.
            from silero_vad import get_speech_timestamps, load_silero_vad

            torch.set_num_threads(1)
            if self._vad_model is None:
                self._vad_model = load_silero_vad()
            regions = get_speech_timestamps(
                torch.as_tensor(audio, dtype=torch.float32, device="cpu"),
                self._vad_model,
                sampling_rate=self.samplerate,
                min_silence_duration_ms=200,
            )
        finally:
            torch.set_num_threads(threads)
        return [(region["start"] / self.samplerate, region["end"] / self.samplerate) for region in regions]

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
            if "lexical_end" in chunk:
                rows[-1]["_lexical_end"] = max(start, min(stop, chunk["lexical_end"]))
        if rows:
            _refine_word_timestamps(rows, self._speech_regions(audio))
        frame = pd.DataFrame(rows, columns=["name", "start_seconds", "stop_seconds"])
        if filename is not None:
            frame.insert(0, "filename", filename)
        return frame
