from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn


@dataclass
class DecoderConfig:
    """Decoder options.

    Args:
        type: Prediction head architecture to use. WhisperSeg is a backend sentinel, not a native decoder module.
        hidden_size: Hidden size for the LSTM decoder.
        kernel_size: Kernel size for the convolutional decoder.
        num_heads: Number of attention heads for the attention decoder.
        num_layers: Number of attention layers in the attention decoder.
        dropout: Dropout rate for the attention or WhisperSeg decoder.
        max_length: Maximum WhisperSeg decoder token length during training.
        generation_max_length: Maximum WhisperSeg decoder token length during prediction.
        num_trials: Number of WhisperSeg prediction trials.
        num_beams: Number of WhisperSeg generation beams.
        top_k: WhisperSeg generation top-k value.
        top_p: WhisperSeg generation top-p value.
        length_penalty: WhisperSeg generation length penalty.
    """

    type: Literal["linear", "lstm", "conv", "attention", "legacy_linear_upsample", "whisperseg"] = "linear"
    hidden_size: int = 64
    kernel_size: int = 8
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    max_length: int = 100
    generation_max_length: int = 448
    num_trials: int = 1
    num_beams: int = 4
    top_k: int = 1
    top_p: float = 1.0
    length_penalty: float = 1.0


@dataclass
class LinearDecoderConfig:
    type: Literal["linear"] = "linear"


@dataclass
class LSTMDecoderConfig:
    type: Literal["lstm"] = "lstm"
    hidden_size: int = 64


@dataclass
class ConvDecoderConfig:
    type: Literal["conv"] = "conv"
    kernel_size: int = 8


@dataclass
class AttentionDecoderConfig:
    type: Literal["attention"] = "attention"
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class LegacyLinearUpsampleDecoderConfig:
    type: Literal["legacy_linear_upsample"] = "legacy_linear_upsample"
    upsample_factor: int = 1


@dataclass
class WhisperSegDecoderConfig:
    type: Literal["whisperseg"] = "whisperseg"
    dropout: float = 0.1
    max_length: int = 100
    generation_max_length: int = 448
    num_trials: int = 1
    num_beams: int = 4
    top_k: int = 1
    top_p: float = 1.0
    length_penalty: float = 1.0


ResolvedDecoderConfig = (
    LinearDecoderConfig
    | LSTMDecoderConfig
    | ConvDecoderConfig
    | AttentionDecoderConfig
    | LegacyLinearUpsampleDecoderConfig
    | WhisperSegDecoderConfig
)


class LinearDecoder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.decoder = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded)


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, decoder_lstm_hidden_size: int = 64):
        super().__init__()
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=decoder_lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Linear(2 * decoder_lstm_hidden_size, num_classes, bias=False)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        decoded, _ = self.decoder(encoded)
        return self.projection(decoded)


class ConvDecoder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, decoder_conv_kernel_size: int = 8):
        super().__init__()
        self.decoder = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_classes,
            kernel_size=decoder_conv_kernel_size,
            padding="same",
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded.transpose(1, 2)).transpose(1, 2)


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        decoder_num_heads: int = 4,
        decoder_num_layers: int = 2,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim % decoder_num_heads != 0:
            raise ValueError(
                f"Attention decoder input_dim={input_dim} must be divisible by decoder_num_heads={decoder_num_heads}."
            )

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=decoder_num_heads,
                    dropout=decoder_dropout,
                    batch_first=True,
                )
                for _ in range(decoder_num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(decoder_num_layers)])
        self.dropout = nn.Dropout(decoder_dropout) if decoder_dropout > 0 else None
        self.projection = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        decoded = encoded
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms, strict=True):
            attention_output, _ = attention(decoded, decoded, decoded)
            decoded = layer_norm(decoded + attention_output)
            if self.dropout is not None:
                decoded = self.dropout(decoded)
        return self.projection(decoded)


class LegacyLinearUpsampleDecoder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, upsample_factor: int = 1):
        super().__init__()
        self.upsample_factor = int(upsample_factor)
        self.decoder = nn.Linear(input_dim, num_classes, bias=True)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(encoded)
        if self.upsample_factor > 1:
            logits = logits.repeat_interleave(self.upsample_factor, dim=1)
        return logits


def normalize_decoder_config(config: DecoderConfig | ResolvedDecoderConfig | Mapping[str, object]) -> ResolvedDecoderConfig:
    if isinstance(
        config,
        (
            LinearDecoderConfig,
            LSTMDecoderConfig,
            ConvDecoderConfig,
            AttentionDecoderConfig,
            LegacyLinearUpsampleDecoderConfig,
            WhisperSegDecoderConfig,
        ),
    ):
        return config
    if isinstance(config, Mapping) and str(config.get("type")) == "whisperseg":
        return WhisperSegDecoderConfig(
            dropout=float(config.get("dropout", 0.1)),
            max_length=int(config.get("max_length", 100)),
            generation_max_length=int(config.get("generation_max_length", 448)),
            num_trials=int(config.get("num_trials", 1)),
            num_beams=int(config.get("num_beams", 4)),
            top_k=int(config.get("top_k", 1)),
            top_p=float(config.get("top_p", 1.0)),
            length_penalty=float(config.get("length_penalty", 1.0)),
        )
    if isinstance(config, Mapping) and str(config.get("type")) == "legacy_linear_upsample":
        return LegacyLinearUpsampleDecoderConfig(upsample_factor=int(config.get("upsample_factor", 1)))
    if not isinstance(config, DecoderConfig):
        config = DecoderConfig(**dict(config))

    if config.type == "linear":
        return LinearDecoderConfig()
    if config.type == "lstm":
        return LSTMDecoderConfig(hidden_size=int(config.hidden_size))
    if config.type == "conv":
        return ConvDecoderConfig(kernel_size=int(config.kernel_size))
    if config.type == "attention":
        return AttentionDecoderConfig(
            num_heads=int(config.num_heads),
            num_layers=int(config.num_layers),
            dropout=float(config.dropout),
        )
    if config.type == "whisperseg":
        return WhisperSegDecoderConfig(
            dropout=float(config.dropout),
            max_length=int(config.max_length),
            generation_max_length=int(config.generation_max_length),
            num_trials=int(config.num_trials),
            num_beams=int(config.num_beams),
            top_k=int(config.top_k),
            top_p=float(config.top_p),
            length_penalty=float(config.length_penalty),
        )
    raise ValueError(f"Unknown decoder config '{config.type}'.")


def serialize_decoder_config(config: DecoderConfig | ResolvedDecoderConfig | Mapping[str, object]) -> dict[str, object]:
    return asdict(normalize_decoder_config(config))


def build_decoder(
    config: DecoderConfig | ResolvedDecoderConfig | Mapping[str, object],
    *,
    input_dim: int,
    num_classes: int,
) -> nn.Module:
    decoder_config = normalize_decoder_config(config)
    if isinstance(decoder_config, LinearDecoderConfig):
        return LinearDecoder(input_dim=input_dim, num_classes=num_classes)
    if isinstance(decoder_config, LSTMDecoderConfig):
        return LSTMDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            decoder_lstm_hidden_size=decoder_config.hidden_size,
        )
    if isinstance(decoder_config, ConvDecoderConfig):
        return ConvDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            decoder_conv_kernel_size=decoder_config.kernel_size,
        )
    if isinstance(decoder_config, AttentionDecoderConfig):
        return AttentionDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            decoder_num_heads=decoder_config.num_heads,
            decoder_num_layers=decoder_config.num_layers,
            decoder_dropout=decoder_config.dropout,
        )
    if isinstance(decoder_config, LegacyLinearUpsampleDecoderConfig):
        return LegacyLinearUpsampleDecoder(
            input_dim=input_dim,
            num_classes=num_classes,
            upsample_factor=decoder_config.upsample_factor,
        )
    if isinstance(decoder_config, WhisperSegDecoderConfig):
        raise ValueError("decoder_type=whisperseg uses the WhisperSeg backend and cannot be built as a native decoder.")
    raise TypeError(f"Unsupported decoder config '{type(decoder_config).__name__}'.")
