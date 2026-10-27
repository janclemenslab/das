"""Run with DAS_AVES2_EXPORT pointing to the converted .pt2 file."""

import os

import numpy as np
import pytest
import torch

from das.models.pretrained import PretrainedAudioEncoder, PretrainedEncoderConfig


pytestmark = [pytest.mark.regression, pytest.mark.slow]


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable"))],
)
def test_aves2_export_matches_upstream_embeddings(device):
    path = os.environ.get("DAS_AVES2_EXPORT")
    if not path:
        pytest.skip("Set DAS_AVES2_EXPORT to run checkpoint-backed verification.")
    torch.set_num_threads(4)
    with np.load(path + ".verification.npz") as reference:
        waveform = torch.from_numpy(reference["inputs"]).to(device)
        expected = torch.from_numpy(reference["outputs"])
    encoder = PretrainedAudioEncoder(PretrainedEncoderConfig(type="aves2", checkpoint=path), sr=16000)
    with torch.no_grad():
        features = encoder._load_backbone(torch.device(device))(waveform).cpu()
        pooled, lengths = encoder(waveform.unsqueeze(-1), torch.tensor([waveform.shape[1]], device=device))
    torch.testing.assert_close(features, expected, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(pooled[:, 0].cpu(), expected.mean(1), rtol=1e-3, atol=1e-4)
    assert lengths.tolist() == [1]
