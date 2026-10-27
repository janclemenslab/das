"""The fixed-width decoder can learn from a wider Whisper teacher."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration


def test_large_teacher_fixed_decoder_distillation():
    source = Path(__file__).resolve().parents[2] / "models/das-conformer-multispecies"
    if not (source / "train_evaluate_hybrid.py").exists():
        pytest.skip("standalone experiment source is not present")
    sys.path.insert(0, str(source))
    try:
        spec = importlib.util.spec_from_file_location("large_distillation_test", source / "train_evaluate_hybrid.py")
        hybrid = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hybrid)
    finally:
        sys.path.remove(str(source))
    def config(width):
        return WhisperConfig(vocab_size=32, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                             decoder_start_token_id=1, num_mel_bins=80, d_model=width,
                             encoder_layers=1, decoder_layers=1, encoder_attention_heads=4,
                             decoder_attention_heads=4, encoder_ffn_dim=128, decoder_ffn_dim=128,
                             max_source_positions=20, max_target_positions=20)
    torch.manual_seed(0)
    student = WhisperForConditionalGeneration(config(64))
    student.model.encoder = hybrid.DasWhisperEncoder(student.config, 32, 64, 1)
    teacher = WhisperForConditionalGeneration(config(128)).eval()
    batch = {"input_features": torch.randn(2, 80, 40),
             "decoder_input_ids": torch.ones(2, 4, dtype=torch.long),
             "labels": torch.tensor([[1, 2, -100, -100], [2, 1, 2, -100]])}
    loss, ce, kl, hidden = hybrid.distillation_loss(student, teacher, batch, torch.device("cpu"))
    assert torch.isfinite(loss) and kl > 0 and hidden == 0
    assert torch.allclose(loss, ce + 0.5 * kl)
    loss.backward()
    assert student.model.encoder.projection.weight.grad.abs().sum() > 0
    assert student.model.decoder.layers[0].fc1.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
