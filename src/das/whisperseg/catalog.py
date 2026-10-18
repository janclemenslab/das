"""Published WhisperSeg AER checkpoints available by name in DAS."""

MODEL_REPO = "DAS-DeepAudioSegmenter/whisperseg-aer"
MODEL_OPTIONS = {
    "whisperseg-aer:tiny-ms": ("Tiny multispecies · compact FP16", "whisperseg-tiny-ms-cmp-fp16.ckpt"),
    "whisperseg-aer:base-animal-vad": ("Animal VAD Base · compact FP16", "whisperseg-base-animal-vad-cmp-fp16.ckpt"),
    "whisperseg-aer:small-ms": ("Small multispecies · compact FP16", "whisperseg-small-ms-cmp-fp16.ckpt"),
    "whisperseg-aer:animal-vad": ("Animal VAD Large · compact FP16", "whisperseg-animal-vad-cmp-fp16.ckpt"),
    "whisperseg-aer:large-ms": ("Large multispecies · compact FP16", "whisperseg-large-ms-cmp-fp16.ckpt"),
}


def resolve_model_checkpoint(source: str) -> str:
    """Download a named release checkpoint on first use, using the Hub cache."""
    if source not in MODEL_OPTIONS:
        return source
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_OPTIONS[source][1])
