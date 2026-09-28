from collections.abc import Mapping
import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from scipy.signal import get_window

PadMode = Literal["constant", "reflect", "replicate", "circular"]


@dataclass
class FrontendConfig:
    """Frontend options.

    Args:
        type: Feature extractor to use: raw waveform, STFT, mel spectrogram, convolutional, convolutional ResNet or Sinc frontend, or WhisperSeg sentinel.
        num_channels: Output feature dimension or number of filters.
        kernel_size: STFT window size or convolution kernel size, depending on frontend type.
        hop_seconds: Time step between successive frames in seconds.
        pad_mode: Padding mode used by the convolutional frontend.
        fmin: Minimum frequency for STFT, mel, or Sinc features.
        fmax: Maximum frequency for STFT, mel, or Sinc features. Defaults to a value derived from the sample rate.
        trainable: Whether STFT or mel filter parameters are trainable.
    """

    type: Literal["raw", "stft", "mel", "conv", "conv_resnet", "sinc", "legacy_stft", "whisperseg"] = "mel"
    num_channels: int | None = None
    kernel_size: int | None = None
    hop_seconds: float | None = None
    pad_mode: PadMode = "constant"
    fmin: float = 100.0
    fmax: float | None = None
    trainable: bool = True
    min_frequency: int | None = 0
    frequency_scale: float | None = 1.0
    spec_time_step: float | None = None

    def __post_init__(self):
        if self.type == "raw":
            if self.num_channels is None:
                self.num_channels = 1
            return
        if self.type in {"stft", "mel", "legacy_stft"}:
            if self.num_channels is None:
                self.num_channels = 128
            if self.kernel_size is None:
                self.kernel_size = 1024
            if self.hop_seconds is None:
                self.hop_seconds = 0.004
            return
        if self.type in {"conv", "conv_resnet"}:
            if self.num_channels is None:
                self.num_channels = 128
            if self.kernel_size is None:
                self.kernel_size = 9
            if self.hop_seconds is None:
                self.hop_seconds = 0.004
            return
        if self.type == "sinc":
            if self.num_channels is None:
                self.num_channels = 128
            if self.kernel_size is None:
                self.kernel_size = 63
            if self.hop_seconds is None:
                self.hop_seconds = 0.005
            return
        if self.type == "whisperseg":
            return
        raise ValueError(f"Unknown frontend type '{self.type}'.")


@dataclass
class RawFrontendConfig:
    type: Literal["raw"] = "raw"
    num_channels: int = 1


@dataclass
class STFTFrontendConfig:
    type: Literal["stft"] = "stft"
    num_channels: int = 128
    kernel_size: int = 1024
    hop_seconds: float = 0.004
    fmin: float = 100.0
    fmax: float | None = None
    trainable: bool = True


@dataclass
class MelFrontendConfig:
    type: Literal["mel"] = "mel"
    num_channels: int = 128
    kernel_size: int = 1024
    hop_seconds: float = 0.004
    fmin: float = 100.0
    fmax: float | None = None
    trainable: bool = True


@dataclass
class ConvFrontendConfig:
    type: Literal["conv"] = "conv"
    num_channels: int = 128
    kernel_size: int = 9
    hop_seconds: float = 0.004
    pad_mode: PadMode = "constant"


@dataclass
class ConvResNetFrontendConfig:
    type: Literal["conv_resnet"] = "conv_resnet"
    num_channels: int = 128
    kernel_size: int = 9
    hop_seconds: float = 0.004
    pad_mode: PadMode = "reflect"


@dataclass
class SincFrontendConfig:
    type: Literal["sinc"] = "sinc"
    num_channels: int = 128
    kernel_size: int = 63
    hop_seconds: float = 0.005
    pad_mode: PadMode = "reflect"
    fmin: float = 50.0
    fmax: float | None = None


@dataclass
class LegacySTFTFrontendConfig:
    type: Literal["legacy_stft"] = "legacy_stft"
    num_channels: int = 1
    kernel_size: int = 64
    hop_seconds: float = 0.004


@dataclass
class WhisperSegFrontendConfig:
    type: Literal["whisperseg"] = "whisperseg"
    min_frequency: int | None = 0
    frequency_scale: float | None = 1.0
    spec_time_step: float | None = None


ResolvedFrontendConfig = (
    RawFrontendConfig
    | STFTFrontendConfig
    | MelFrontendConfig
    | ConvFrontendConfig
    | ConvResNetFrontendConfig
    | SincFrontendConfig
    | LegacySTFTFrontendConfig
    | WhisperSegFrontendConfig
)


class RawFrontend(nn.Module):
    def __init__(self, num_freq: int = 1):
        super().__init__()
        self.output_dim = num_freq

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim == 2:
            if self.output_dim != 1:
                raise ValueError(
                    "RawFrontend received waveform input with shape [B, T] but was configured "
                    f"for {self.output_dim} features. Use num_freq=1 for waveform input."
                )
            return inputs.unsqueeze(-1), lengths

        if inputs.ndim == 3:
            if inputs.shape[1] != self.output_dim:
                raise ValueError(f"RawFrontend expected {self.output_dim} input features, got {inputs.shape[1]}.")
            return inputs.transpose(1, 2), lengths

        raise ValueError(f"RawFrontend expects inputs shaped [B, T] or [B, F, T], got {tuple(inputs.shape)}.")


class STFTFrontend(nn.Module):
    def __init__(
        self,
        sr: float,
        num_fft: int = 1024,
        hop_length: int = 128,
        num_channels: int | None = None,
        fmin: float = 100.0,
        fmax: float | None = None,
        trainable: bool = False,
    ):
        super().__init__()
        try:
            from nnAudio.features.stft import STFT
        except ImportError as exc:
            raise ImportError("STFTFrontend requires nnAudio to be installed.") from exc

        self.hop_length = hop_length
        max_output_dim = num_fft // 2 + 1
        if fmax is None:
            fmax = sr / 2
        start_bin = max(0, int(math.ceil(float(fmin) * num_fft / float(sr))))
        stop_bin = min(max_output_dim, int(math.floor(float(fmax) * num_fft / float(sr))) + 1)
        available_bins = max(0, stop_bin - start_bin)
        if available_bins == 0:
            raise ValueError(f"STFT frequency range [{fmin}, {fmax}] Hz does not include any FFT bins.")
        self.start_bin = start_bin
        self.output_dim = available_bins if num_channels is None else min(int(num_channels), available_bins)
        self.spec_layer = STFT(
            n_fft=num_fft,
            hop_length=hop_length,
            center=True,
            pad_mode="constant",
            trainable=trainable,
            output_format="Magnitude",
            verbose=False,
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        specs = self.spec_layer(inputs)[:, self.start_bin : self.start_bin + self.output_dim, :].transpose(1, 2)
        output_lengths = _frame_lengths(lengths, hop_length=self.hop_length, max_length=specs.shape[1])
        return specs, output_lengths


class MelFrontend(nn.Module):
    def __init__(
        self,
        sr: float,
        num_freq: int = 128,
        num_fft: int = 1024,
        hop_length: int = 128,
        fmin: float = 100.0,
        fmax: float | None = None,
        trainable: bool = False,
    ):
        super().__init__()
        try:
            from nnAudio.features.mel import MelSpectrogram
        except ImportError as exc:
            raise ImportError("MelFrontend requires nnAudio to be installed.") from exc

        if fmax is None:
            fmax = (sr * 3) // 4

        self.hop_length = hop_length
        self.output_dim = num_freq
        self.spec_layer = MelSpectrogram(
            n_fft=num_fft,
            n_mels=num_freq,
            hop_length=hop_length,
            verbose=0,
            window="hann",
            center=True,
            pad_mode="constant",
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            trainable_mel=trainable,
            trainable_STFT=trainable,
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        specs = self.spec_layer(inputs).transpose(1, 2)
        output_lengths = _frame_lengths(lengths, hop_length=self.hop_length, max_length=specs.shape[1])
        return specs, output_lengths


class ConvFrontend(nn.Module):
    def __init__(
        self,
        hop_length: int = 4,
        num_channels: int = 128,
        kernel_size: int = 9,
        pad_mode: str = "constant",
    ):
        super().__init__()

        self.hop_length = int(hop_length)
        self.output_dim = int(num_channels)
        self.kernel_size = int(kernel_size)
        self.pad_mode = str(pad_mode)
        self.padding = self.kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.output_dim,
            kernel_size=self.kernel_size,
            stride=self.hop_length,
            padding=0,
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs.unsqueeze(1)
        if self.padding > 0:
            inputs = F.pad(inputs, pad=(self.padding, self.padding), mode=self.pad_mode)
        features = self.conv(inputs).transpose(1, 2)
        output_lengths = _conv_output_lengths(
            lengths,
            hop_length=self.hop_length,
            kernel_size=self.kernel_size,
            padding=self.padding,
            max_length=features.shape[1],
        )
        return features, output_lengths


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, pad_mode: str):
        super().__init__()
        self.padding = dilation * (kernel_size // 2)
        self.pad_mode = pad_mode
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0, bias=False),
                nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0, bias=False),
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(channels), nn.LayerNorm(channels)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs
        for conv, norm in zip(self.convs, self.norms, strict=True):
            features = F.pad(features, (self.padding, self.padding), mode=self.pad_mode)
            features = conv(features)
            features = F.gelu(norm(features.transpose(1, 2))).transpose(1, 2)
        return features + inputs


class ConvResNetFrontend(nn.Module):
    """SongExplorer-style local raw-waveform ResNet followed by learned downsampling."""

    def __init__(
        self,
        hop_length: int = 4,
        num_channels: int = 128,
        kernel_size: int = 9,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ConvResNet frontend kernel_size must be odd.")

        self.hop_length = int(hop_length)
        self.output_dim = int(num_channels)
        self.pad_mode = str(pad_mode)
        hidden_channels = max(8, self.output_dim // 2)
        self.stem_padding = kernel_size // 2
        self.stem = nn.Conv1d(1, hidden_channels, kernel_size, padding=0, bias=False)
        self.stem_norm = nn.LayerNorm(hidden_channels)
        self.blocks = nn.ModuleList(
            [_ResidualConvBlock(hidden_channels, kernel_size, dilation, self.pad_mode) for dilation in (1, 2, 4)]
        )
        downsample_kernel = max(3, 2 * self.hop_length - 1)
        self.downsample_padding = downsample_kernel // 2
        self.downsample_kernel = downsample_kernel
        self.downsample = nn.Conv1d(
            hidden_channels,
            self.output_dim,
            downsample_kernel,
            stride=self.hop_length,
            padding=0,
            bias=False,
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channel_shape = None
        if inputs.ndim == 3:
            batch_size, num_channels, num_samples = inputs.shape
            inputs = inputs.reshape(batch_size * num_channels, num_samples)
            channel_shape = (batch_size, num_channels)
        features = F.pad(inputs.unsqueeze(1), (self.stem_padding, self.stem_padding), mode=self.pad_mode)
        features = self.stem(features)
        features = F.gelu(self.stem_norm(features.transpose(1, 2))).transpose(1, 2)
        if channel_shape is not None:
            batch_size, num_channels = channel_shape
            features = features.reshape(batch_size, num_channels, *features.shape[1:]).amax(dim=1)
        for block in self.blocks:
            features = block(features)
        features = F.pad(
            features,
            (self.downsample_padding, self.downsample_padding),
            mode=self.pad_mode,
        )
        features = self.downsample(features).transpose(1, 2)
        features = F.gelu(self.output_norm(features))
        output_lengths = _conv_output_lengths(
            lengths,
            hop_length=self.hop_length,
            kernel_size=self.downsample_kernel,
            padding=self.downsample_padding,
            max_length=features.shape[1],
        )
        return features, output_lengths


class SincConv1d(nn.Module):
    def __init__(
        self,
        sr: float,
        out_channels: int,
        kernel_size: int,
        fmin: float = 50.0,
        fmax: float | None = None,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Sinc filter kernel_size must be odd.")

        max_frequency = float(sr) / 2 if fmax is None else min(float(fmax), float(sr) / 2)
        min_bandwidth = math.ceil(float(sr) / kernel_size)
        if max_frequency <= fmin + min_bandwidth:
            raise ValueError("Sinc frontend frequency range is too small for its kernel_size.")

        mel = torch.linspace(
            2595 * math.log10(1 + fmin / 700),
            2595 * math.log10(1 + max_frequency / 700),
            out_channels + 1,
        )
        edges = 700 * (10 ** (mel / 2595) - 1)
        self.low_hz = nn.Parameter(edges[:-1].unsqueeze(1) - fmin)
        self.band_hz = nn.Parameter((edges[1:] - edges[:-1]).unsqueeze(1) - min_bandwidth)
        self.register_buffer("times", torch.arange(-(kernel_size // 2), kernel_size // 2 + 1) / float(sr))
        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False))
        self.fmin = float(fmin)
        self.fmax = max_frequency
        self.min_bandwidth = float(min_bandwidth)
        self.padding = kernel_size // 2
        self.pad_mode = str(pad_mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        low = (self.fmin + self.low_hz.abs()).clamp(max=self.fmax - self.min_bandwidth)
        high = (low + self.min_bandwidth + self.band_hz.abs()).clamp(max=self.fmax)
        bandwidth = high - low
        filters = (
            2 * high * torch.sinc(2 * high * self.times)
            - 2 * low * torch.sinc(2 * low * self.times)
        )
        filters = filters * self.window / (2 * bandwidth)
        padded = F.pad(inputs, (self.padding, self.padding), mode=self.pad_mode)
        return F.conv1d(padded, filters.unsqueeze(1))


class SincFrontend(nn.Module):
    def __init__(
        self,
        sr: float,
        hop_length: int,
        num_channels: int,
        kernel_size: int,
        fmin: float = 50.0,
        fmax: float | None = None,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.hop_length = int(hop_length)
        self.output_dim = int(num_channels)
        if self.hop_length < 2:
            raise ValueError("Sinc frontend hop_length must be at least 2 so the Conv1d stack can project its filters.")
        sinc_channels = max(num_channels, int(round(float(sr) / kernel_size)))
        self.sinc = SincConv1d(sr, sinc_channels, kernel_size, fmin=fmin, fmax=fmax, pad_mode=pad_mode)

        remaining_stride = self.hop_length
        strides = []
        while remaining_stride > 1:
            stride = 5 if remaining_stride % 5 == 0 else 2 if remaining_stride % 2 == 0 else remaining_stride
            strides.append(stride)
            remaining_stride //= stride
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList([nn.LayerNorm(sinc_channels)])
        self.conv_specs = []
        in_channels = sinc_channels
        for stride in strides:
            conv_kernel = 10 if stride == 5 else 3 if stride == 2 else 2 * stride
            padding = conv_kernel // 2
            conv = nn.Conv1d(
                in_channels,
                num_channels,
                kernel_size=conv_kernel,
                stride=stride,
                padding=padding,
                bias=False,
            )
            nn.init.kaiming_normal_(conv.weight)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(num_channels))
            self.conv_specs.append((conv_kernel, stride, padding))
            in_channels = num_channels

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.sinc(inputs.unsqueeze(1))
        features = F.gelu(self.norms[0](features.transpose(1, 2))).transpose(1, 2)
        output_lengths = lengths
        for conv, norm, (kernel_size, stride, padding) in zip(
            self.convs,
            self.norms[1:],
            self.conv_specs,
            strict=True,
        ):
            features = conv(features)
            features = F.gelu(norm(features.transpose(1, 2))).transpose(1, 2)
            output_lengths = torch.div(
                torch.clamp(output_lengths + 2 * padding - kernel_size, min=0),
                stride,
                rounding_mode="floor",
            ) + 1
        return features.transpose(1, 2), torch.clamp(output_lengths, max=features.shape[-1])


def _legacy_stft_kernels(n_dft: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_dft <= 1 or (n_dft & (n_dft - 1)) != 0:
        raise ValueError(f"n_dft must be > 1 and a power of two, got {n_dft}.")

    nb_filter = int(n_dft // 2 + 1)
    timesteps = torch.arange(n_dft, dtype=torch.float32)
    w_ks = torch.arange(nb_filter, dtype=torch.float32) * 2 * math.pi / float(n_dft)
    real_kernels = torch.cos(w_ks.reshape(-1, 1) * timesteps.reshape(1, -1))
    imag_kernels = -torch.sin(w_ks.reshape(-1, 1) * timesteps.reshape(1, -1))
    window = torch.as_tensor(get_window("hann", n_dft, fftbins=True), dtype=torch.float32).reshape(1, -1)
    real_kernels = torch.multiply(real_kernels, window).transpose(0, 1)
    imag_kernels = torch.multiply(imag_kernels, window).transpose(0, 1)
    return real_kernels, imag_kernels


class LegacySTFTFrontend(nn.Module):
    def __init__(self, num_fft: int, hop_length: int, num_channels: int = 1):
        super().__init__()
        self.num_fft = int(num_fft)
        self.hop_length = int(hop_length)
        self.num_channels = int(num_channels)
        self.num_filter = self.num_fft // 2 + 1
        self.output_dim = self.num_channels * self.num_filter

        real_kernels, imag_kernels = _legacy_stft_kernels(self.num_fft)
        self.real_conv = nn.Conv1d(1, self.num_filter, kernel_size=self.num_fft, stride=self.hop_length, bias=False)
        self.imag_conv = nn.Conv1d(1, self.num_filter, kernel_size=self.num_fft, stride=self.hop_length, bias=False)
        with torch.no_grad():
            self.real_conv.weight.copy_(real_kernels.T.unsqueeze(1))
            self.imag_conv.weight.copy_(imag_kernels.T.unsqueeze(1))
        self.requires_grad_(False)

    def _pad_same(self, x: torch.Tensor) -> torch.Tensor:
        frames = int(math.ceil(x.shape[-1] / self.hop_length))
        total_length = max(self.num_fft, (frames - 1) * self.hop_length + self.num_fft)
        total_pad = max(total_length - x.shape[-1], 0)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        if total_pad == 0:
            return x
        return F.pad(x, (pad_left, pad_right))

    def _amplitude_to_decibel(self, x: torch.Tensor, amin: float = 1e-10, dynamic_range: float = 80.0) -> torch.Tensor:
        log_spec = 10.0 * torch.log10(torch.clamp(x, min=amin))
        reduce_dims = tuple(range(1, x.ndim))
        log_spec = log_spec - torch.amax(log_spec, dim=reduce_dims, keepdim=True)
        return torch.clamp(log_spec, min=-dynamic_range)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(-1)

        batch, _, channels = inputs.shape
        x = inputs.transpose(1, 2)
        outputs: list[torch.Tensor] = []
        for channel_idx in range(channels):
            channel_x = self._pad_same(x[:, channel_idx : channel_idx + 1, :])
            real = self.real_conv(channel_x)
            imag = self.imag_conv(channel_x)
            power = real.square() + imag.square()
            outputs.append(self._amplitude_to_decibel(torch.sqrt(power)))

        stacked = torch.stack(outputs, dim=1)
        features = stacked.permute(0, 3, 1, 2).reshape(batch, stacked.shape[-1], channels * self.num_filter)
        return features, lengths


def normalize_frontend_config(config: FrontendConfig | ResolvedFrontendConfig | Mapping[str, object]) -> ResolvedFrontendConfig:
    if isinstance(
        config,
        (
            RawFrontendConfig,
            STFTFrontendConfig,
            MelFrontendConfig,
            ConvFrontendConfig,
            ConvResNetFrontendConfig,
            SincFrontendConfig,
            LegacySTFTFrontendConfig,
            WhisperSegFrontendConfig,
        ),
    ):
        return config
    if not isinstance(config, FrontendConfig):
        config = dict(config)
        if config.get("type") == "stft":
            # Old serialized STFT configs predate frequency cropping and include DC.
            config.setdefault("fmin", 0.0)
        config = FrontendConfig(**config)

    if config.type == "raw":
        num_channels = int(config.num_channels)
        if num_channels == 128 and config.kernel_size == 1024 and config.hop_seconds == 0.004:
            num_channels = 1
        return RawFrontendConfig(num_channels=num_channels)
    if config.type == "stft":
        return STFTFrontendConfig(
            num_channels=int(config.num_channels),
            kernel_size=int(config.kernel_size),
            hop_seconds=float(config.hop_seconds),
            fmin=float(config.fmin),
            fmax=None if config.fmax is None else float(config.fmax),
            trainable=bool(config.trainable),
        )
    if config.type == "mel":
        return MelFrontendConfig(
            num_channels=int(config.num_channels),
            kernel_size=int(config.kernel_size),
            hop_seconds=float(config.hop_seconds),
            fmin=float(config.fmin),
            fmax=None if config.fmax is None else float(config.fmax),
            trainable=bool(config.trainable),
        )
    if config.type == "conv":
        kernel_size = int(config.kernel_size)
        if kernel_size == 1024 and config.num_channels == 128 and config.hop_seconds == 0.004:
            kernel_size = 9
        return ConvFrontendConfig(
            num_channels=int(config.num_channels),
            kernel_size=kernel_size,
            hop_seconds=float(config.hop_seconds),
            pad_mode=str(config.pad_mode),
        )
    if config.type == "conv_resnet":
        return ConvResNetFrontendConfig(
            num_channels=int(config.num_channels),
            kernel_size=int(config.kernel_size),
            hop_seconds=float(config.hop_seconds),
            pad_mode=str(config.pad_mode),
        )
    if config.type == "sinc":
        return SincFrontendConfig(
            num_channels=int(config.num_channels),
            kernel_size=int(config.kernel_size),
            hop_seconds=float(config.hop_seconds),
            pad_mode=str(config.pad_mode),
            fmin=float(config.fmin),
            fmax=None if config.fmax is None else float(config.fmax),
        )
    if config.type == "legacy_stft":
        return LegacySTFTFrontendConfig(
            num_channels=int(config.num_channels),
            kernel_size=int(config.kernel_size),
            hop_seconds=float(config.hop_seconds),
        )
    if config.type == "whisperseg":
        return WhisperSegFrontendConfig(
            min_frequency=None if config.min_frequency is None else int(config.min_frequency),
            frequency_scale=None if config.frequency_scale is None else float(config.frequency_scale),
            spec_time_step=None if config.spec_time_step is None else float(config.spec_time_step),
        )
    raise ValueError(f"Unknown frontend config '{config.type}'.")


def serialize_frontend_config(
    config: FrontendConfig | ResolvedFrontendConfig | Mapping[str, object],
    *,
    include_raw_num_channels: bool = True,
) -> dict[str, object]:
    payload = asdict(normalize_frontend_config(config))
    if payload["type"] == "raw" and not include_raw_num_channels:
        payload.pop("num_channels", None)
    return payload


def frontend_output_dim(config: FrontendConfig | ResolvedFrontendConfig | Mapping[str, object]) -> int:
    frontend_config = normalize_frontend_config(config)
    if isinstance(frontend_config, WhisperSegFrontendConfig):
        raise ValueError("frontend_type=whisperseg uses the WhisperSeg backend and cannot be built as a native frontend.")
    if isinstance(frontend_config, STFTFrontendConfig):
        return min(frontend_config.num_channels, frontend_config.kernel_size // 2 + 1)
    if isinstance(frontend_config, LegacySTFTFrontendConfig):
        return frontend_config.num_channels * (frontend_config.kernel_size // 2 + 1)
    return frontend_config.num_channels


def frontend_hop_length_samples(config: FrontendConfig | ResolvedFrontendConfig | Mapping[str, object], *, sr: float) -> int:
    frontend_config = normalize_frontend_config(config)
    if isinstance(frontend_config, WhisperSegFrontendConfig):
        raise ValueError("frontend_type=whisperseg uses the WhisperSeg backend and has no native hop length.")
    if isinstance(frontend_config, RawFrontendConfig):
        return 1
    return max(1, int(round(frontend_config.hop_seconds * sr)))


def frontend_hop_seconds(config: FrontendConfig | ResolvedFrontendConfig | Mapping[str, object], *, sr: float) -> float:
    frontend_config = normalize_frontend_config(config)
    if isinstance(frontend_config, WhisperSegFrontendConfig):
        raise ValueError("frontend_type=whisperseg uses the WhisperSeg backend and has no native hop duration.")
    if isinstance(frontend_config, RawFrontendConfig):
        return 1.0 / float(sr)
    if isinstance(frontend_config, LegacySTFTFrontendConfig):
        return 1.0 / sr
    hop_length = frontend_hop_length_samples(frontend_config, sr=sr)
    return hop_length / sr


def build_frontend(config: FrontendConfig | ResolvedFrontendConfig | Mapping[str, object], *, sr: float) -> nn.Module:
    frontend_config = normalize_frontend_config(config)
    if isinstance(frontend_config, WhisperSegFrontendConfig):
        raise ValueError("frontend_type=whisperseg uses the WhisperSeg backend and cannot be built as a native frontend.")
    if isinstance(frontend_config, RawFrontendConfig):
        return RawFrontend(num_freq=frontend_config.num_channels)
    if isinstance(frontend_config, STFTFrontendConfig):
        return STFTFrontend(
            sr=sr,
            num_fft=frontend_config.kernel_size,
            hop_length=frontend_hop_length_samples(frontend_config, sr=sr),
            num_channels=frontend_output_dim(frontend_config),
            fmin=frontend_config.fmin,
            fmax=frontend_config.fmax,
            trainable=frontend_config.trainable,
        )
    if isinstance(frontend_config, MelFrontendConfig):
        return MelFrontend(
            sr=sr,
            num_freq=frontend_config.num_channels,
            num_fft=frontend_config.kernel_size,
            hop_length=frontend_hop_length_samples(frontend_config, sr=sr),
            fmin=frontend_config.fmin,
            fmax=frontend_config.fmax,
            trainable=frontend_config.trainable,
        )
    if isinstance(frontend_config, ConvFrontendConfig):
        return ConvFrontend(
            hop_length=frontend_hop_length_samples(frontend_config, sr=sr),
            num_channels=frontend_config.num_channels,
            kernel_size=frontend_config.kernel_size,
            pad_mode=frontend_config.pad_mode,
        )
    if isinstance(frontend_config, ConvResNetFrontendConfig):
        return ConvResNetFrontend(
            hop_length=frontend_hop_length_samples(frontend_config, sr=sr),
            num_channels=frontend_config.num_channels,
            kernel_size=frontend_config.kernel_size,
            pad_mode=frontend_config.pad_mode,
        )
    if isinstance(frontend_config, SincFrontendConfig):
        return SincFrontend(
            sr=sr,
            hop_length=frontend_hop_length_samples(frontend_config, sr=sr),
            num_channels=frontend_config.num_channels,
            kernel_size=frontend_config.kernel_size,
            fmin=frontend_config.fmin,
            fmax=frontend_config.fmax,
            pad_mode=frontend_config.pad_mode,
        )
    if isinstance(frontend_config, LegacySTFTFrontendConfig):
        return LegacySTFTFrontend(
            num_fft=frontend_config.kernel_size,
            hop_length=frontend_hop_length_samples(frontend_config, sr=sr),
            num_channels=frontend_config.num_channels,
        )
    raise TypeError(f"Unsupported frontend config '{type(frontend_config).__name__}'.")


def _frame_lengths(lengths: torch.Tensor, hop_length: int, max_length: int) -> torch.Tensor:
    frame_lengths = torch.div(lengths, hop_length, rounding_mode="floor") + 1
    return torch.clamp(frame_lengths, max=max_length)


def _conv_output_lengths(
    lengths: torch.Tensor,
    hop_length: int,
    kernel_size: int,
    padding: int,
    max_length: int,
) -> torch.Tensor:
    padded = lengths + (2 * padding)
    effective = torch.clamp(padded - kernel_size, min=0)
    output_lengths = torch.div(effective, hop_length, rounding_mode="floor") + 1
    return torch.clamp(output_lengths, max=max_length)
