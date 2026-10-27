from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
import torch

from das import api
from das.config import Config


pytestmark = pytest.mark.regression
FIXTURES = Path(__file__).parents[1] / "fixtures" / "checkpoint_compatibility"


@pytest.mark.parametrize("kind,named_layers", [("raw", False), ("stft", False), ("native", False), ("raw", True), ("stft", True)])
def test_pre_ssl_removal_checkpoints_preserve_predictions(kind, named_layers, tmp_path):
    source = FIXTURES / "native-before-ssl-removal.pt" if kind == "native" else FIXTURES / kind / "test"
    expected_path = FIXTURES / "native-expected.npz" if kind == "native" else FIXTURES / kind / "expected.npz"
    with np.load(expected_path) as expected:
        inputs = torch.from_numpy(expected["inputs"])
        lengths = torch.from_numpy(expected["lengths"])
        expected_logits = torch.from_numpy(expected["logits"])
        expected_lengths = torch.from_numpy(expected["output_lengths"])

    if named_layers:
        shutil.copytree(source.parent, tmp_path / "named")
        source = tmp_path / "named" / "test"
        with h5py.File(source.with_name("test_model.h5"), "r+") as handle:
            weights = handle["model_weights"]
            names = ["tcn_initial_conv", "tcn_dilated_conv_1_tanh_s0", "conv1d"]
            for old, new in zip(["conv1d", "conv1d_1", "conv1d_2"], names):
                weights.move(old, new)
                weights[new].move(old, new)
            weights.attrs["layer_names"] = np.asarray(names + ["dense"], dtype="S")

    sources = [source]
    if kind != "native":
        sources.append(Path(api.convert_legacy_checkpoint(str(source), str(tmp_path / "converted.ckpt"))))
    for checkpoint in sources:
        model, runtime = api._load_predict_model(str(checkpoint), config=Config())
        model.cpu().eval()
        with torch.inference_mode():
            logits, output_lengths = model(inputs, lengths)
        torch.testing.assert_close(logits, expected_logits, rtol=1e-5, atol=1e-6)
        assert torch.equal(output_lengths, expected_lengths)
        assert runtime.class_names == ["noise", "song"]
        assert runtime.sr == 10_000
        assert runtime.num_time_steps == 32
        assert runtime.chunk_stride == (16 if kind == "native" else 32)
        assert runtime.hop_seconds == 1 / 10_000


@pytest.mark.parametrize("fmin", [0.0, 100.0])
def test_stft_checkpoint_preserves_old_and_explicit_frequency_bounds(fmin, tmp_path):
    checkpoint = torch.load(FIXTURES / "native-before-ssl-removal.pt", map_location="cpu", weights_only=False)
    hparams = dict(checkpoint["hyper_parameters"])
    hparams.update(sr=32_000, frontend={
        "type": "stft", "num_channels": 33, "kernel_size": 64, "hop_seconds": 0.0005, "fmin": fmin,
    })
    model = api.DASModel(**hparams).eval()
    checkpoint["state_dict"] = model.state_dict()
    checkpoint["hyper_parameters"] = dict(model.hparams)
    if fmin == 0.0:
        # Old STFT checkpoints did not store frequency bounds and included the DC bin.
        checkpoint["hyper_parameters"]["frontend"].pop("fmin")
        checkpoint["hyper_parameters"]["frontend"].pop("fmax")
    path = tmp_path / "stft.ckpt"
    torch.save(checkpoint, path)

    loaded = api.DASModel.load_from_checkpoint(path, map_location="cpu").eval()
    inputs = torch.randn(2, 128)
    lengths = torch.tensor([128, 112])
    with torch.inference_mode():
        expected, expected_lengths = model(inputs, lengths)
        actual, actual_lengths = loaded(inputs, lengths)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_lengths, expected_lengths)
    assert loaded.hparams["frontend"]["fmin"] == fmin
