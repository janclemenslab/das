from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

import torch
from torch import nn

from .tcn import TemporalConvNet

TcnPadding = Literal["same", "causal"]


@dataclass
class EncoderConfig:
    """Encoder options.

    Args:
        type: Sequence encoder architecture to use. WhisperSeg is a backend sentinel, not a native encoder module.
        num_heads: Number of attention heads for the conformer encoder.
        hidden_size: Hidden size or filter count used by the encoder.
        num_layers: Number of encoder layers or stacks.
        kernel_size: Conformer depthwise kernel size or TCN kernel size.
        dilations: Dilation schedule for the TCN encoder.
        dropout: Dropout rate for TCN blocks or Tweetynet recurrent layers.
        use_skip_connections: Whether to sum skip connections in the TCN encoder.
        use_separable: Whether TCN blocks use separable convolutions.
        padding: Padding mode for the TCN encoder.
    """

    type: Literal["conformer", "tcn", "tweetynet", "whisperseg"] = "conformer"
    num_heads: int = 4
    hidden_size: int = 128
    num_layers: int = 3
    kernel_size: int | None = None
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    dropout: float = 0.1
    use_skip_connections: bool = True
    use_separable: bool | list[bool] = False
    padding: TcnPadding = "same"

    def __post_init__(self):
        if self.type == "conformer":
            if self.kernel_size is None:
                self.kernel_size = 31
            return
        if self.type == "tcn":
            if self.kernel_size is None:
                self.kernel_size = 16
            return
        if self.type == "tweetynet":
            if self.kernel_size is None:
                self.kernel_size = 5
            return
        if self.type == "whisperseg":
            return
        raise ValueError(f"Unknown encoder type '{self.type}'.")


@dataclass
class ConformerEncoderConfig:
    type: Literal["conformer"] = "conformer"
    num_heads: int = 4
    hidden_size: int = 128
    num_layers: int = 3
    kernel_size: int = 31


@dataclass
class TCNEncoderConfig:
    type: Literal["tcn"] = "tcn"
    hidden_size: int = 128
    num_layers: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    kernel_size: int = 16
    dropout: float = 0.1
    use_skip_connections: bool = True
    use_separable: bool | list[bool] = False
    padding: TcnPadding = "same"


@dataclass
class TweetynetEncoderConfig:
    type: Literal["tweetynet"] = "tweetynet"
    hidden_size: int = 512
    num_layers: int = 1
    kernel_size: int = 5
    dropout: float = 0.1


@dataclass
class WhisperSegEncoderConfig:
    type: Literal["whisperseg"] = "whisperseg"


ResolvedEncoderConfig = ConformerEncoderConfig | TCNEncoderConfig | TweetynetEncoderConfig | WhisperSegEncoderConfig


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_layers: int = 2,
        depthwise_conv_kernel_size: int = 31,
    ):
        super().__init__()
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("ConformerEncoder requires torchaudio to be installed.") from exc

        self.output_dim = input_dim
        self.encoder = torchaudio.models.Conformer(
            input_dim=input_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(features, lengths)


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        tcn_num_filters: int = 128,
        tcn_num_conv: int = 3,
        tcn_dilations: Sequence[int] = (1, 2, 4, 8, 16),
        tcn_kernel_size: int = 16,
        tcn_dropout: float = 0.1,
        tcn_use_skip_connections: bool = True,
        tcn_use_separable: bool | Sequence[bool] = False,
        tcn_padding: str = "same",
    ):
        super().__init__()
        self.output_dim = int(tcn_num_filters)
        self.encoder = TemporalConvNet(
            in_channels=input_dim,
            nb_filters=tcn_num_filters,
            kernel_size=tcn_kernel_size,
            nb_stacks=tcn_num_conv,
            dilations=tcn_dilations,
            dropout_rate=tcn_dropout,
            use_skip_connections=tcn_use_skip_connections,
            return_sequences=True,
            use_separable=tcn_use_separable,
            padding=tcn_padding,
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        return encoded, lengths


class TweetynetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 512,
        num_layers: int = 1,
        conv_kernel_size: int = 5,
        rnn_dropout: float = 0.0,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(conv_kernel_size, conv_kernel_size), padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1)),
            nn.Conv2d(32, 64, kernel_size=(conv_kernel_size, conv_kernel_size), padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1)),
        )
        with torch.no_grad():
            try:
                cnn_output = self.cnn(torch.zeros(1, 1, int(input_dim), 1))
            except RuntimeError as exc:
                raise ValueError("TweetynetEncoder requires at least 64 frontend channels.") from exc
        self.rnn_input_size = int(cnn_output.shape[1] * cnn_output.shape[2])
        self.output_dim = int(hidden_size) * 2
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(rnn_dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = features.transpose(1, 2).unsqueeze(1)
        features = self.cnn(features)
        features = features.reshape(features.shape[0], self.rnn_input_size, features.shape[-1]).transpose(1, 2)
        encoded, _ = self.rnn(features)
        return encoded, lengths


def normalize_encoder_config(config: EncoderConfig | ResolvedEncoderConfig | Mapping[str, object]) -> ResolvedEncoderConfig:
    if isinstance(config, (ConformerEncoderConfig, TCNEncoderConfig, TweetynetEncoderConfig, WhisperSegEncoderConfig)):
        return config
    if not isinstance(config, EncoderConfig):
        config = EncoderConfig(**dict(config))

    if config.type == "conformer":
        return ConformerEncoderConfig(
            num_heads=int(config.num_heads),
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_layers),
            kernel_size=int(config.kernel_size),
        )
    if config.type == "tcn":
        kernel_size = int(config.kernel_size)
        if kernel_size == 31 and config.num_heads == 4:
            kernel_size = 16
        return TCNEncoderConfig(
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_layers),
            dilations=[int(dilation) for dilation in config.dilations],
            kernel_size=kernel_size,
            dropout=float(config.dropout),
            use_skip_connections=bool(config.use_skip_connections),
            use_separable=config.use_separable,
            padding=str(config.padding),
        )
    if config.type == "tweetynet":
        return TweetynetEncoderConfig(
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_layers),
            kernel_size=int(config.kernel_size),
            dropout=float(config.dropout),
        )
    if config.type == "whisperseg":
        return WhisperSegEncoderConfig()
    raise ValueError(f"Unknown encoder config '{config.type}'.")


def serialize_encoder_config(config: EncoderConfig | ResolvedEncoderConfig | Mapping[str, object]) -> dict[str, object]:
    return asdict(normalize_encoder_config(config))


def model_hop_seconds(frontend, encoder, *, sr: float) -> float:
    from .frontends import frontend_hop_seconds

    encoder = normalize_encoder_config(encoder)
    return frontend_hop_seconds(frontend, sr=sr)


def build_encoder(config: EncoderConfig | ResolvedEncoderConfig | Mapping[str, object], *, input_dim: int, sr: float | None = None) -> nn.Module:
    encoder_config = normalize_encoder_config(config)
    if isinstance(encoder_config, ConformerEncoderConfig):
        return ConformerEncoder(
            input_dim=input_dim,
            num_heads=encoder_config.num_heads,
            ffn_dim=encoder_config.hidden_size,
            num_layers=encoder_config.num_layers,
            depthwise_conv_kernel_size=encoder_config.kernel_size,
        )
    if isinstance(encoder_config, TCNEncoderConfig):
        return TCNEncoder(
            input_dim=input_dim,
            tcn_num_filters=encoder_config.hidden_size,
            tcn_num_conv=encoder_config.num_layers,
            tcn_dilations=tuple(encoder_config.dilations),
            tcn_kernel_size=encoder_config.kernel_size,
            tcn_dropout=encoder_config.dropout,
            tcn_use_skip_connections=encoder_config.use_skip_connections,
            tcn_use_separable=encoder_config.use_separable,
            tcn_padding=encoder_config.padding,
        )
    if isinstance(encoder_config, TweetynetEncoderConfig):
        return TweetynetEncoder(
            input_dim=input_dim,
            hidden_size=encoder_config.hidden_size,
            num_layers=encoder_config.num_layers,
            conv_kernel_size=encoder_config.kernel_size,
            rnn_dropout=encoder_config.dropout,
        )
    if isinstance(encoder_config, WhisperSegEncoderConfig):
        raise ValueError("encoder_type=whisperseg uses the WhisperSeg backend and cannot be built as a native encoder.")
    raise TypeError(f"Unsupported encoder config '{type(encoder_config).__name__}'.")
