from types import SimpleNamespace

import pytest
import torch

from das.cli import parse_command_config
from das.whisperseg.train import WhisperSegLightningModule


@pytest.mark.parametrize("freeze,after", [(True, 0), (True, 1), (True, 2), (True, 3), (False, 2)])
@pytest.mark.parametrize("method", ["full", "lora"])
def test_encoder_updates_only_after_scheduled_epochs(tiny_whisper_speech_model, freeze, after, method):
    model, _ = tiny_whisper_speech_model
    module = WhisperSegLightningModule(
        model, learning_rate=0.01, weight_decay=0.01, linear_lr_schedule=False,
        warmup_steps=0, reduce_lr=False, reduce_lr_patience=2, reduce_lr_factor=0.5,
        reduce_lr_min=0, lr_monitor="train_loss_epoch", total_training_steps=4,
        training_method=method, freeze_encoder=freeze, unfreeze_encoder_after=after,
    )
    optimizer = module.configure_optimizers()
    module.trainer = SimpleNamespace(current_epoch=0)
    for epoch in range(4):
        module.trainer.current_epoch = epoch
        module.on_train_epoch_start()
        before = [p.detach().clone() for p in model.model.encoder.parameters()]
        decoder_before = [p.detach().clone() for p in model.model.decoder.parameters()]
        optimizer.zero_grad()
        sum(p.sum() for p in module.model.parameters() if p.requires_grad).backward()
        optimizer.step()
        should_update = not freeze or (after > 0 and epoch >= after)
        assert any(p.requires_grad for p in model.model.encoder.parameters()) == should_update
        assert any(not torch.equal(old, p) for old, p in zip(before, model.model.encoder.parameters())) == should_update
        assert any(not torch.equal(old, p) for old, p in zip(decoder_before, model.model.decoder.parameters()))
        if method == "lora":
            assert all(not p.requires_grad for name, p in model.named_parameters() if "lora_" not in name)


def test_unfreeze_epoch_cli_and_validation():
    config = parse_command_config([
        "train", "--data-dir", "audio", "--output-dir", "run", "--encoder", "whisperseg",
        "--initial-model", "whisper:large", "--whisperseg-training-method", "full",
        "--freeze-encoder", "--unfreeze-encoder-after", "3",
    ])
    assert config.unfreeze_encoder_after == 3
    with pytest.raises(ValueError, match="unfreeze_encoder_after must be non-negative"):
        config.copy(unfreeze_encoder_after=-1).validate()
