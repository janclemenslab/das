from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from das.api import convert_legacy_checkpoint
from das.das_legacy import is_legacy_model_source, load_legacy_predictor, resolve_legacy_trunk
from das.models import DASModel


def _write_legacy_fixture(tmp_path: Path, *, name: str = "test", params: dict | None = None) -> Path:
    trunk = tmp_path / name
    trunk.with_name(f"{name}_model.h5").write_bytes(b"legacy")
    trunk.with_name(f"{name}_params.yaml").write_text(
        yaml.safe_dump(
            params
            or {
                "model_name": "tcn",
                "nb_hist": 64,
                "kernel_size": 3,
                "nb_conv": 1,
                "nb_filters": 4,
                "nb_freq": 1,
                "nb_classes": 2,
                "samplerate_x_Hz": 10_000,
                "class_names": ["noise", "song"],
            }
        )
    )
    return trunk


def test_legacy_source_detection_accepts_trunk_and_model_h5(tmp_path: Path):
    trunk = _write_legacy_fixture(tmp_path)

    assert is_legacy_model_source(str(trunk)) is True
    assert is_legacy_model_source(str(trunk.with_name("test_model.h5"))) is True
    assert is_legacy_model_source(str(tmp_path / "model.ckpt")) is False
    assert resolve_legacy_trunk(str(trunk.with_name("test_model.h5"))) == trunk


def test_load_legacy_predictor_rejects_unsupported_legacy_models(tmp_path: Path):
    trunk = _write_legacy_fixture(
        tmp_path,
        params={
            "model_name": "tcn",
            "nb_hist": 64,
            "kernel_size": 3,
            "nb_conv": 1,
            "nb_filters": 4,
            "nb_freq": 1,
            "nb_classes": 2,
            "samplerate_x_Hz": 10_000,
            "class_names": ["noise", "song"],
            "nb_lstm_units": 8,
        },
    )

    with pytest.raises(ValueError, match="LSTM"):
        load_legacy_predictor(str(trunk))


def _write_valid_legacy_tcn_fixture(tmp_path: Path) -> Path:
    params = {
        "model_name": "tcn",
        "nb_hist": 32,
        "kernel_size": 3,
        "nb_conv": 1,
        "dilations": [1],
        "nb_filters": 4,
        "nb_freq": 1,
        "nb_classes": 2,
        "samplerate_x_Hz": 10_000,
        "class_names": ["noise", "song"],
    }
    trunk = _write_legacy_fixture(tmp_path, params=params)
    rng = np.random.default_rng(1234)
    with h5py.File(trunk.with_name("test_model.h5"), "w") as handle:
        weights = handle.create_group("model_weights")
        for name, kernel_shape, bias_shape in [
            ("conv1d", (1, 1, 4), (4,)),
            ("conv1d_1", (3, 4, 4), (4,)),
            ("conv1d_2", (1, 4, 4), (4,)),
        ]:
            group = weights.create_group(name).create_group(name)
            group.create_dataset("kernel:0", data=rng.normal(size=kernel_shape).astype(np.float32))
            group.create_dataset("bias:0", data=rng.normal(size=bias_shape).astype(np.float32))
        dense = weights.create_group("dense").create_group("dense")
        dense.create_dataset("kernel:0", data=rng.normal(size=(4, 2)).astype(np.float32))
        dense.create_dataset("bias:0", data=rng.normal(size=(2,)).astype(np.float32))
    return trunk


def test_convert_legacy_checkpoint_loads_as_native_das_model(tmp_path: Path):
    trunk = _write_valid_legacy_tcn_fixture(tmp_path)
    output_path = tmp_path / "converted.ckpt"

    converted_path = convert_legacy_checkpoint(str(trunk), str(output_path))

    legacy_model, _ = load_legacy_predictor(str(trunk))
    converted_model = DASModel.load_from_checkpoint(converted_path)
    inputs = torch.randn(2, 32)
    input_lengths = torch.tensor([32, 28])

    with torch.inference_mode():
        legacy_logits, legacy_lengths = legacy_model(inputs, input_lengths)
        converted_logits, converted_lengths = converted_model(inputs, input_lengths)

    assert torch.equal(converted_lengths, legacy_lengths)
    assert torch.allclose(converted_logits, legacy_logits, atol=1e-5)
