from __future__ import annotations

import math
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.signal import get_window
from torch import nn

from ..models.tcn import TemporalConvNet


def is_legacy_model_source(source: str) -> bool:
    path = Path(source).expanduser()
    if path.suffix == ".ckpt":
        return False

    if path.name.endswith("_model.h5"):
        trunk = path.with_name(path.name[: -len("_model.h5")])
        return trunk.with_name(f"{trunk.name}_params.yaml").exists()

    return path.with_name(f"{path.name}_model.h5").exists() and path.with_name(f"{path.name}_params.yaml").exists()


def resolve_legacy_trunk(source: str) -> Path:
    path = Path(source).expanduser()
    if path.name.endswith("_model.h5"):
        trunk = path.with_name(path.name[: -len("_model.h5")])
    else:
        trunk = path

    model_path = trunk.with_name(f"{trunk.name}_model.h5")
    params_path = trunk.with_name(f"{trunk.name}_params.yaml")
    if not model_path.exists() or not params_path.exists():
        raise ValueError(
            f"Legacy DAS source '{source}' must resolve to adjacent '{model_path.name}' and '{params_path.name}' files."
        )
    return trunk


def load_legacy_params(trunk: str | Path) -> dict:
    trunk = Path(trunk)
    params_path = trunk.with_name(f"{trunk.name}_params.yaml")
    with params_path.open("r") as handle:
        try:
            params = yaml.unsafe_load(handle)
        except AttributeError:
            params = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(params, dict):
        raise ValueError(f"Expected legacy params in '{params_path}' to deserialize to a dictionary.")
    return params


def _normalize_class_types(class_names: list[str], class_types) -> list[str]:
    if class_types is None:
        return ["segment"] * len(class_names)

    normalized = [str(class_type) for class_type in class_types]
    if len(normalized) != len(class_names):
        return ["segment"] * len(class_names)
    return normalized


def normalize_legacy_params(params: dict) -> dict:
    normalized = dict(params)
    normalized["model_name"] = str(normalized.get("model_name", normalized.get("model", "")))
    normalized["nb_hist"] = int(normalized["nb_hist"])
    normalized["kernel_size"] = int(normalized["kernel_size"])
    normalized["nb_conv"] = int(normalized["nb_conv"])
    normalized["nb_filters"] = int(normalized["nb_filters"])
    normalized["nb_freq"] = int(normalized.get("nb_freq", 1))
    normalized["nb_classes"] = int(normalized.get("nb_classes", len(normalized.get("class_names", []))))
    normalized["pre_nb_dft"] = int(normalized.get("pre_nb_dft", 64))
    normalized["nb_pre_conv"] = int(normalized.get("nb_pre_conv", normalized.get("pre_nb_conv", 0)) or 0)
    normalized["nb_lstm_units"] = int(normalized.get("nb_lstm_units", 0) or 0)
    normalized["morph_nb_kernels"] = int(normalized.get("morph_nb_kernels", 0) or 0)
    normalized["samplerate_x_Hz"] = float(normalized.get("samplerate_x_Hz", normalized.get("sample_rate_hz")))
    normalized["upsample"] = bool(normalized.get("upsample", True))
    normalized["return_sequences"] = bool(normalized.get("return_sequences", True))
    normalized["use_skip_connections"] = bool(normalized.get("use_skip_connections", True))
    normalized["use_separable"] = bool(normalized.get("use_separable", False))
    normalized["dropout_rate"] = float(normalized.get("dropout_rate", 0.0))
    normalized["padding"] = str(normalized.get("padding", "same"))
    normalized["dilations"] = [int(dilation) for dilation in normalized.get("dilations", [1, 2, 4, 8, 16])]
    normalized["class_names"] = [str(name) for name in normalized.get("class_names", [])]
    normalized["class_types"] = _normalize_class_types(normalized["class_names"], normalized.get("class_types"))
    normalized["stride"] = int(normalized.get("stride", normalized["nb_hist"]))
    return normalized


def _validate_legacy_params(params: dict) -> None:
    if params["model_name"] not in {"tcn", "tcn_stft"}:
        raise ValueError(f"Unsupported legacy DAS model '{params['model_name']}'.")
    if not params["upsample"]:
        raise ValueError("Legacy DAS predict support requires upsample=True.")
    if not params["return_sequences"]:
        raise ValueError("Legacy DAS predict support requires return_sequences=True.")
    if params["nb_lstm_units"] > 0:
        raise ValueError("Legacy DAS predict support does not support LSTM legacy models.")
    if params["use_separable"]:
        raise ValueError("Legacy DAS predict support does not support separable-conv legacy models.")
    if bool(params.get("use_resnet", False)):
        raise ValueError("Legacy DAS predict support does not support resnet legacy models.")
    if params["morph_nb_kernels"] > 0:
        raise ValueError("Legacy DAS predict support does not support morphological legacy models.")


def _legacy_stft_kernels(n_dft: int) -> tuple[np.ndarray, np.ndarray]:
    if n_dft <= 1 or (n_dft & (n_dft - 1)) != 0:
        raise ValueError(f"n_dft must be > 1 and a power of two, got {n_dft}.")

    nb_filter = int(n_dft // 2 + 1)
    timesteps = np.arange(n_dft)
    w_ks = np.arange(nb_filter) * 2 * np.pi / float(n_dft)
    real_kernels = np.cos(w_ks.reshape(-1, 1) * timesteps.reshape(1, -1))
    imag_kernels = -np.sin(w_ks.reshape(-1, 1) * timesteps.reshape(1, -1))
    window = get_window("hann", n_dft, fftbins=True).astype(np.float32).reshape((1, -1))
    real_kernels = np.multiply(real_kernels, window).transpose()
    imag_kernels = np.multiply(imag_kernels, window).transpose()
    return real_kernels.astype(np.float32), imag_kernels.astype(np.float32)


class LegacyTrainableSpectrogramFrontend(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, *, trainable: bool = False):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_filter = self.n_fft // 2 + 1

        real_kernels, imag_kernels = _legacy_stft_kernels(self.n_fft)
        self.real_conv = nn.Conv1d(1, self.n_filter, kernel_size=self.n_fft, stride=self.hop_length, bias=False)
        self.imag_conv = nn.Conv1d(1, self.n_filter, kernel_size=self.n_fft, stride=self.hop_length, bias=False)
        with torch.no_grad():
            self.real_conv.weight.copy_(torch.from_numpy(real_kernels.T).unsqueeze(1))
            self.imag_conv.weight.copy_(torch.from_numpy(imag_kernels.T).unsqueeze(1))
        self.requires_grad_(bool(trainable))

    def _pad_same(self, x: torch.Tensor) -> torch.Tensor:
        frames = int(math.ceil(x.shape[-1] / self.hop_length))
        total_length = max(self.n_fft, (frames - 1) * self.hop_length + self.n_fft)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2)

        outputs: list[torch.Tensor] = []
        for channel_idx in range(channels):
            channel_x = self._pad_same(x[:, channel_idx : channel_idx + 1, :])
            real = self.real_conv(channel_x)
            imag = self.imag_conv(channel_x)
            power = real.square() + imag.square()
            outputs.append(self._amplitude_to_decibel(torch.sqrt(power)))

        stacked = torch.stack(outputs, dim=1)
        return stacked.permute(0, 3, 1, 2).reshape(batch, stacked.shape[-1], channels * self.n_filter)


class LegacyTorchTCNModel(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = dict(params)
        self.output_stride = int(2 ** int(params["nb_pre_conv"])) if int(params["nb_pre_conv"]) > 0 else 1
        self.upsample = bool(params["upsample"])

        if params["nb_pre_conv"] > 0:
            self.frontend: nn.Module | None = LegacyTrainableSpectrogramFrontend(
                params["pre_nb_dft"],
                int(2 ** params["nb_pre_conv"]),
                trainable=False,
            )
            input_dim = (params["pre_nb_dft"] // 2 + 1) * params["nb_freq"]
        else:
            self.frontend = None
            input_dim = int(params["nb_freq"])

        self.tcn = TemporalConvNet(
            in_channels=input_dim,
            nb_filters=params["nb_filters"],
            kernel_size=params["kernel_size"],
            nb_stacks=params["nb_conv"],
            dilations=params["dilations"],
            dropout_rate=params["dropout_rate"],
            use_skip_connections=params["use_skip_connections"],
            return_sequences=True,
            use_separable=False,
            padding=params["padding"],
        )
        self.classifier = nn.Linear(self.tcn.out_channels, params["nb_classes"], bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_length = x.shape[1]
        if self.frontend is not None:
            x = self.frontend(x)
        x = self.tcn(x)
        logits = self.classifier(x)
        if self.frontend is not None and self.upsample:
            logits = logits.repeat_interleave(self.output_stride, dim=1)[:, :original_length, :]
        return logits


def _copy_conv1d(target: nn.Conv1d, kernel, bias) -> None:
    weight = np.transpose(np.asarray(kernel), (2, 1, 0))
    with torch.no_grad():
        target.weight.copy_(torch.from_numpy(weight))
        if target.bias is not None and bias is not None:
            target.bias.copy_(torch.from_numpy(np.asarray(bias)))


def _load_legacy_h5_weights(model: LegacyTorchTCNModel, model_path: Path) -> None:
    with h5py.File(model_path, "r") as handle:
        weights_root = handle["model_weights"]

        if model.frontend is not None and "trainable_stft" in weights_root:
            real = np.asarray(weights_root["trainable_stft"]["trainable_stft"]["real_kernels:0"])
            imag = np.asarray(weights_root["trainable_stft"]["trainable_stft"]["imag_kernels:0"])
            real = np.squeeze(real, axis=(1, 2)).T[:, np.newaxis, :]
            imag = np.squeeze(imag, axis=(1, 2)).T[:, np.newaxis, :]
            with torch.no_grad():
                model.frontend.real_conv.weight.copy_(torch.from_numpy(real))
                model.frontend.imag_conv.weight.copy_(torch.from_numpy(imag))

        conv_names = sorted(
            [name for name in weights_root.keys() if name.startswith("conv1d")],
            key=lambda name: 0 if name == "conv1d" else int(name.split("_")[1]),
        )
        if "tcn_initial_conv" in weights_root:
            # Older Keras exports name the input/dilated convolutions explicitly.
            conv_names = [
                name for name in weights_root.attrs["layer_names"].astype(str)
                if name.startswith(("tcn_initial_conv", "tcn_dilated_conv", "conv1d"))
            ]
        expected_conv_count = 1 + 2 * len(model.tcn.blocks)
        if len(conv_names) != expected_conv_count:
            raise ValueError(f"Expected {expected_conv_count} legacy conv groups, found {len(conv_names)}.")

        conv_iter = iter(conv_names)
        input_name = next(conv_iter)
        input_weights = weights_root[input_name][input_name]
        _copy_conv1d(model.tcn.input_projection.conv, input_weights["kernel:0"], input_weights["bias:0"])

        for block in model.tcn.blocks:
            block_conv_name = next(conv_iter)
            block_conv_weights = weights_root[block_conv_name][block_conv_name]
            _copy_conv1d(block.conv.conv, block_conv_weights["kernel:0"], block_conv_weights["bias:0"])

            projection_name = next(conv_iter)
            projection_weights = weights_root[projection_name][projection_name]
            _copy_conv1d(block.residual_projection, projection_weights["kernel:0"], projection_weights["bias:0"])

        dense_weights = weights_root["dense"]["dense"]
        with torch.no_grad():
            model.classifier.weight.copy_(torch.from_numpy(np.asarray(dense_weights["kernel:0"]).T))
            model.classifier.bias.copy_(torch.from_numpy(np.asarray(dense_weights["bias:0"])))


class LegacyDASPredictor(L.LightningModule):
    def __init__(self, model: LegacyTorchTCNModel, num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = int(num_classes)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(-1)
        elif inputs.ndim != 3:
            raise ValueError(f"Legacy DAS predictor expects [B, T] or [B, T, C], got {tuple(inputs.shape)}.")

        if input_lengths is None:
            input_lengths = torch.full((inputs.shape[0],), inputs.shape[1], device=inputs.device, dtype=torch.long)
        else:
            input_lengths = torch.as_tensor(input_lengths, device=inputs.device, dtype=torch.long)

        logits = self.model(inputs)
        output_lengths = torch.clamp(input_lengths, max=logits.shape[1])
        return logits, output_lengths

    def predict_step(self, batch, batch_idx: int):
        del batch_idx
        if len(batch) == 2:
            inputs, input_lengths = batch
        elif len(batch) == 4:
            inputs, input_lengths, _, _ = batch
        else:
            raise ValueError(f"Expected predict batch with 2 or 4 items, got {len(batch)}.")
        return self.forward(inputs, input_lengths)


def load_legacy_predictor(source: str) -> tuple[LegacyDASPredictor, dict[str, object]]:
    trunk = resolve_legacy_trunk(source)
    params = normalize_legacy_params(load_legacy_params(trunk))
    _validate_legacy_params(params)

    model = LegacyTorchTCNModel(params)
    _load_legacy_h5_weights(model, trunk.with_name(f"{trunk.name}_model.h5"))
    model.eval()

    runtime = {
        "sr": float(params["samplerate_x_Hz"]),
        "hop_seconds": 1.0 / float(params["samplerate_x_Hz"]),
        "num_freq": 1,
        "num_time_steps": int(params["nb_hist"]),
        "chunk_stride": int(params["stride"]),
        "data_padding": int(params.get("data_padding", 0) or 0),
        "class_names": list(params["class_names"]),
        "class_types": list(params["class_types"]),
    }
    predictor = LegacyDASPredictor(model=model, num_classes=params["nb_classes"])
    predictor.eval()
    return predictor, runtime


def legacy_to_das_model(source: str, *, checkpoint_metadata: dict[str, object] | None = None):
    from ..models import DASModel

    trunk = resolve_legacy_trunk(source)
    params = normalize_legacy_params(load_legacy_params(trunk))
    _validate_legacy_params(params)

    legacy_model = LegacyTorchTCNModel(params)
    _load_legacy_h5_weights(legacy_model, trunk.with_name(f"{trunk.name}_model.h5"))

    sr = float(params["samplerate_x_Hz"])
    upsample_factor = int(legacy_model.output_stride) if legacy_model.frontend is not None and legacy_model.upsample else 1
    if legacy_model.frontend is None:
        frontend = {
            "type": "raw",
            "num_channels": int(params["nb_freq"]),
        }
    else:
        frontend = {
            "type": "legacy_stft",
            "num_channels": int(params["nb_freq"]),
            "kernel_size": int(params["pre_nb_dft"]),
            "hop_seconds": float(upsample_factor) / sr,
        }

    model = DASModel(
        num_classes=int(params["nb_classes"]),
        sr=sr,
        class_names=list(params["class_names"]),
        class_types=list(params["class_types"]),
        frontend=frontend,
        encoder={
            "type": "tcn",
            "hidden_size": int(params["nb_filters"]),
            "num_layers": int(params["nb_conv"]),
            "kernel_size": int(params["kernel_size"]),
            "dilations": list(params["dilations"]),
            "dropout": float(params["dropout_rate"]),
            "use_skip_connections": bool(params["use_skip_connections"]),
            "use_separable": False,
            "padding": str(params["padding"]),
        },
        decoder={
            "type": "legacy_linear_upsample",
            "upsample_factor": upsample_factor,
        },
        cross_entropy_weight=0.9,
        learning_rate=0.0001,
        num_time_steps=int(params["nb_hist"]),
        chunk_stride=int(params["stride"]),
        checkpoint_metadata=checkpoint_metadata,
    )

    if legacy_model.frontend is not None:
        model.frontend.real_conv.load_state_dict(legacy_model.frontend.real_conv.state_dict())
        model.frontend.imag_conv.load_state_dict(legacy_model.frontend.imag_conv.state_dict())
    model.encoder.encoder.load_state_dict(legacy_model.tcn.state_dict())
    model.decoder.decoder.load_state_dict(legacy_model.classifier.state_dict())
    model.eval()
    return model, params
