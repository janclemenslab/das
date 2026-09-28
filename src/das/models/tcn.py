from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _normalize_use_separable(use_separable: bool | Sequence[bool], nb_stacks: int) -> list[bool]:
    if isinstance(use_separable, bool):
        return [use_separable] * nb_stacks

    values = [bool(value) for value in use_separable]
    if len(values) == 0:
        return [False] * nb_stacks
    if len(values) == 1:
        return values * nb_stacks
    if len(values) != nb_stacks:
        raise ValueError(f"use_separable must have length 1 or {nb_stacks}, got {len(values)}.")
    return values


class ChannelNormalization(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().amax(dim=1, keepdim=True) + 1e-5
        return x / scale


class Conv1dWithPadding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        padding_mode: str = "same",
        groups: int = 1,
    ):
        super().__init__()
        if padding_mode == "same":
            self.left_padding = 0
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
                groups=groups,
            )
        elif padding_mode == "causal":
            self.left_padding = int((kernel_size - 1) * dilation)
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=0,
                groups=groups,
            )
        else:
            raise ValueError(f"Unsupported padding mode {padding_mode!r}. Expected 'same' or 'causal'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class SeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        *,
        padding_mode: str = "same",
        depth_multiplier: int = 4,
    ):
        super().__init__()
        depthwise_channels = int(in_channels * depth_multiplier)
        self.depthwise = Conv1dWithPadding(
            in_channels,
            depthwise_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding_mode=padding_mode,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(depthwise_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int,
        *,
        dropout_rate: float = 0.0,
        use_separable: bool = False,
        padding: str = "same",
    ):
        super().__init__()
        if use_separable:
            self.conv = SeparableConv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding_mode=padding,
            )
        else:
            self.conv = Conv1dWithPadding(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding_mode=padding,
            )
        self.activation = nn.ReLU()
        self.channel_norm = ChannelNormalization()
        self.dropout = nn.Dropout1d(dropout_rate)
        self.residual_projection = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.shape_match = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.conv(x)
        out = self.activation(out)
        out = self.channel_norm(out)
        out = self.dropout(out)
        skip = self.residual_projection(out)
        residual = self.shape_match(x)
        return residual + skip, skip


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nb_filters: int,
        kernel_size: int,
        nb_stacks: int,
        dilations: Iterable[int],
        *,
        dropout_rate: float = 0.0,
        use_skip_connections: bool = True,
        return_sequences: bool = True,
        use_separable: bool | Sequence[bool] = False,
        padding: str = "same",
    ):
        super().__init__()
        if nb_filters <= 0:
            raise ValueError("nb_filters must be > 0.")
        if nb_stacks <= 0:
            raise ValueError("nb_stacks must be > 0.")

        dilation_values = [int(dilation) for dilation in dilations]
        if not dilation_values:
            raise ValueError("dilations must contain at least one value.")
        if any(dilation <= 0 for dilation in dilation_values):
            raise ValueError("dilations must all be > 0.")

        self.return_sequences = bool(return_sequences)
        self.use_skip_connections = bool(use_skip_connections)
        self.input_projection = Conv1dWithPadding(in_channels, nb_filters, kernel_size=1, padding_mode=padding)

        separable_by_stack = _normalize_use_separable(use_separable, nb_stacks)
        blocks: list[ResidualBlock] = []
        current_channels = int(nb_filters)
        for stack_idx in range(nb_stacks):
            for dilation in dilation_values:
                blocks.append(
                    ResidualBlock(
                        in_channels=current_channels,
                        out_channels=nb_filters,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        dropout_rate=dropout_rate,
                        use_separable=separable_by_stack[stack_idx],
                        padding=padding,
                    )
                )
                current_channels = nb_filters

        self.blocks = nn.ModuleList(blocks)
        self.out_channels = current_channels
        self.apply(_initialize_tcn_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_projection(x)

        skips = []
        for block in self.blocks:
            x, skip = block(x)
            skips.append(skip)

        if self.use_skip_connections and skips:
            x = torch.stack(skips, dim=0).sum(dim=0)

        x = F.relu(x)
        if self.return_sequences:
            return x.transpose(1, 2)
        return x[:, :, x.shape[-1] // 2]


def _initialize_tcn_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


__all__ = [
    "ChannelNormalization",
    "Conv1dWithPadding",
    "ResidualBlock",
    "SeparableConv1d",
    "TemporalConvNet",
]
