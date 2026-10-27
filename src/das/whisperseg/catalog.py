"""Published WhisperSeg AER checkpoints and Whisper speech models available in DAS."""

MODEL_REPO = "DAS-DeepAudioSegmenter/whisperseg-aer"

# Original speech models retain their full vocabulary when trained on DAS annotations.
ASR_MODEL_OPTIONS = {
    "whisper:large": ("Whisper Large · speech transcription · FP16", "openai/whisper-large"),
}
MODEL_OPTIONS = {
    "whisperseg-aer:tiny-ms": ("WhisperSeg-AER · Tiny", "whisperseg-tiny-ms-cmp-fp16.ckpt"),
    "whisperseg-aer:small-ms": ("WhisperSeg-AER · Small", "whisperseg-small-ms-cmp-fp16.ckpt"),
    "whisperseg-aer:large-ms": ("WhisperSeg-AER · Large", "whisperseg-large-ms-cmp-fp16.ckpt"),
}


def resolve_model_checkpoint(source: str) -> str:
    """Download a named release checkpoint on first use, using the Hub cache."""
    if source not in MODEL_OPTIONS:
        return source
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_OPTIONS[source][1], token=False)
