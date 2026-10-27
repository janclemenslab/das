"""Frozen, window-level bioacoustic embeddings with an optional reusable disk cache."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Literal
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from torch import nn


PRETRAINED_ENCODERS = ("aves2", "perch2", "birdcode")
DEFAULT_PRETRAINED_MODEL_IDS = {
    "aves2": "esp_aves2_sl_beats_all",
    "perch2": "DAS-DeepAudioSegmenter/perch2-torch",
    "birdcode": "EarthSpeciesProject/sed-birdcode",
}


def resolve_encoder_checkpoint(source: str) -> str:
    """Resolve a local export or Hugging Face file through the standard Hub cache."""
    if not source.startswith(("hf://", "https://huggingface.co/")):
        return str(Path(source).expanduser().resolve())
    from huggingface_hub import hf_hub_download

    if source.startswith("hf://"):
        parts = source[5:].split("/", 2)
        if len(parts) != 3 or not all(parts):
            raise ValueError("Use hf://owner/repository/model.pt2 for a Hugging Face encoder file.")
        owner, repo, filename = parts
        repo, _, revision = repo.partition("@")
    else:
        parts = urlparse(source).path.strip("/").split("/", 4)
        if len(parts) != 5 or parts[2] not in {"blob", "resolve"}:
            raise ValueError("Use a Hugging Face model file link, including its filename.")
        owner, repo, _, revision, filename = parts
    return hf_hub_download(f"{owner}/{repo}", unquote(filename), revision=unquote(revision) or None)


@dataclass
class PretrainedEncoderConfig:
    type: Literal["aves2", "perch2", "birdcode"] = "perch2"
    model_id: str | None = None
    checkpoint: str | None = None
    sample_rate: int | None = None
    embedding_dim: int | None = None
    window_seconds: float | None = None
    hop_seconds: float | None = None
    cache_dir: str | None = None

    def __post_init__(self):
        rate, dim, window = {
            "aves2": (16000, 768, 10.0),
            "perch2": (32000, 1536, 5.0),
            "birdcode": (32000, 1024, 5.0),
        }[self.type]
        self.model_id = self.model_id or DEFAULT_PRETRAINED_MODEL_IDS[self.type]
        self.sample_rate = rate if self.sample_rate is None else int(self.sample_rate)
        self.embedding_dim = dim if self.embedding_dim is None else int(self.embedding_dim)
        self.window_seconds = window if self.window_seconds is None else float(self.window_seconds)
        self.hop_seconds = self.window_seconds if self.hop_seconds is None else float(self.hop_seconds)
        if self.checkpoint and not self.checkpoint.startswith(("hf://", "https://huggingface.co/")):
            self.checkpoint = str(Path(self.checkpoint).expanduser().resolve())
        if self.cache_dir:
            self.cache_dir = str(Path(self.cache_dir).expanduser().resolve())
        if self.sample_rate <= 0 or self.embedding_dim <= 0 or not 0 < self.hop_seconds <= self.window_seconds:
            raise ValueError(
                "Pretrained encoders require positive sample rate/dimension and 0 < hop_seconds <= window_seconds."
            )
        if self.type == "perch2" and (self.sample_rate != 32000 or self.window_seconds != 5 or self.embedding_dim != 1536):
            raise ValueError("Perch2 requires 32 kHz, five-second windows, and 1536-dimensional embeddings.")
        if self.type == "birdcode" and (self.sample_rate, self.embedding_dim, self.window_seconds, self.hop_seconds) != (
            32000,
            1024,
            5,
            5,
        ):
            raise ValueError(
                "BirdCODE requires 32 kHz, 1024 dimensions, and five-second windows/hops; output frames are 5/38 seconds apart."
            )

    @property
    def frames_per_window(self) -> int:
        return 38 if self.type == "birdcode" else 1


class PretrainedAudioEncoder(nn.Module):
    """Frozen features: one frame per window, or 38 temporal frames for BirdCODE.

    Inputs follow the native encoder contract [batch, samples, channels]. Windows
    start at frame * hop, are downmixed to mono, resampled, and right-zero-padded.
    Cached .npy files contain each window's embeddings; cache hits need no model runtime.
    """

    def __init__(self, config: PretrainedEncoderConfig, *, sr: float):
        super().__init__()
        self.config = config
        self.sr = float(sr)
        self.output_dim = config.embedding_dim
        self.window_hop_samples = max(1, round(config.hop_seconds * sr))
        self.hop_samples = self.window_hop_samples / config.frames_per_window
        self.window_samples = max(1, round(config.window_seconds * sr))
        self.cache_dir = Path(config.cache_dir).expanduser() if config.cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        identity = asdict(config)
        identity.pop("cache_dir")
        identity.update(format_version=1, input_sample_rate=self.sr)
        self.checkpoint_path = resolve_encoder_checkpoint(config.checkpoint) if config.checkpoint else None
        if self.checkpoint_path:
            with Path(self.checkpoint_path).open("rb") as handle:
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            identity["checkpoint_sha256"] = digest.hexdigest()
        self.cache_identity = json.dumps(identity, sort_keys=True).encode()
        # External weights are deliberately absent from DAS decoder checkpoints.
        object.__setattr__(self, "_backbone", None)
        self._backbone_device = None

    def _load_backbone(self, device):
        if self._backbone is None or (self.config.checkpoint and self._backbone_device != device):
            if self.config.checkpoint:
                from torch.export.passes import move_to_device_pass

                metadata = {"das_embedding.json": ""}
                exported = torch.export.load(self.checkpoint_path, extra_files=metadata)
                expected = {
                    "sample_rate": self.config.sample_rate,
                    "window_seconds": self.config.window_seconds,
                    "embedding_dim": self.output_dim,
                }
                if not metadata["das_embedding.json"] or json.loads(metadata["das_embedding.json"]) != expected:
                    raise ValueError("Export metadata does not match the encoder sample rate, window, or dimension.")
                backbone = move_to_device_pass(exported, device).module()
            elif self.config.type == "perch2":
                backbone = torch.hub.load("janclemenslab/perch2_torch", "perch_v2", trust_repo=True)
            elif self.config.type == "birdcode":
                from .birdcode import BirdCODEEmbeddings

                backbone = BirdCODEEmbeddings(self.config.model_id)
            else:
                try:
                    from avex import load_model
                except ImportError as exc:
                    raise ImportError(
                        "Native AVEX requires avex on Python 3.11-3.13. On Python 3.14, set encoder_checkpoint to a .pt2 export. "
                        "See docs/pretrained-encoders.md."
                    ) from exc
                backbone = load_model(self.config.model_id, return_features_only=True, device=str(device))
            if not self.config.checkpoint:
                backbone.eval().requires_grad_(False)
            object.__setattr__(self, "_backbone", backbone)
            self._backbone_device = device
        return self._backbone if self.config.checkpoint else self._backbone.to(device).eval()

    @torch.no_grad()
    def _embed_window(self, waveform: np.ndarray, device) -> np.ndarray:
        from scipy.signal import resample_poly
        from fractions import Fraction

        ratio = Fraction(self.config.sample_rate / self.sr).limit_denominator(1000000)
        if ratio != 1:
            waveform = resample_poly(waveform, ratio.numerator, ratio.denominator)
        size = round(self.config.window_seconds * self.config.sample_rate)
        waveform = np.pad(waveform[:size], (0, max(0, size - len(waveform))))
        tensor = torch.as_tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)
        backbone = self._load_backbone(device)
        features = backbone.embed(tensor) if self.config.type == "perch2" and not self.config.checkpoint else backbone(tensor)
        if features.ndim == 3 and self.config.frames_per_window == 1:
            features = features.mean(dim=1)
        expected_shape = (
            (1, self.output_dim) if self.config.frames_per_window == 1 else (1, self.config.frames_per_window, self.output_dim)
        )
        if features.shape != expected_shape:
            raise ValueError(f"Expected embeddings {expected_shape}, got {tuple(features.shape)}; check encoder_embedding_dim.")
        return features[0].detach().cpu().float().numpy()

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio = features.detach().float().mean(dim=-1).cpu().numpy()
        frames_per_window = self.config.frames_per_window
        frame_count = (features.shape[1] * frames_per_window + self.window_hop_samples - 1) // self.window_hop_samples
        output = np.zeros((len(audio), frame_count, self.output_dim), dtype=np.float32)
        output_lengths = (lengths * frames_per_window + self.window_hop_samples - 1) // self.window_hop_samples
        for batch, length in enumerate(lengths.tolist()):
            for window, start in enumerate(range(0, length, self.window_hop_samples)):
                frame = window * frames_per_window
                waveform = np.zeros(self.window_samples, dtype=np.float32)
                chunk = audio[batch, start : min(start + self.window_samples, length)]
                waveform[: len(chunk)] = chunk
                cache_path = None
                if self.cache_dir is not None:
                    key = hashlib.sha256(self.cache_identity + waveform.tobytes()).hexdigest()
                    cache_path = self.cache_dir / f"{key}.npy"
                if cache_path is not None and cache_path.exists():
                    embedding = np.load(cache_path, allow_pickle=False)
                else:
                    embedding = self._embed_window(waveform, features.device)
                    if cache_path is not None:
                        with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npy", delete=False) as handle:
                            temporary = Path(handle.name)
                            np.save(handle, embedding, allow_pickle=False)
                        temporary.replace(cache_path)
                count = min(frames_per_window, int(output_lengths[batch]) - frame)
                output[batch, frame : frame + count] = embedding.reshape(-1, self.output_dim)[:count]
        return torch.from_numpy(output).to(features.device), output_lengths
