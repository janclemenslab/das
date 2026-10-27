from inspect import signature

from das.cli import parse_command_config
from das.config import Config
from das.gui_helpers import WhisperSegDecoderState
from das.models.decoders import DecoderConfig, WhisperSegDecoderConfig, normalize_decoder_config
from das.whisperseg.model import WhisperSegmenter
from das.gui_helpers import train_config_to_gui_state, train_gui_state_to_config


def test_whisperseg_defaults_to_three_passes_across_entry_points():
    assert Config(mode="predict", checkpoint="model.ckpt").num_trials == 3
    assert parse_command_config(["predict", "--data-dir", "audio", "--checkpoint", "model.ckpt"]).num_trials == 3
    assert WhisperSegDecoderState().num_trials == 3
    assert DecoderConfig(type="whisperseg").num_trials == 3
    assert WhisperSegDecoderConfig().num_trials == 3
    assert normalize_decoder_config({"type": "whisperseg"}).num_trials == 3
    assert signature(WhisperSegmenter.segment).parameters["num_trials"].default == 3


def test_whisperseg_preserves_an_explicit_single_pass():
    assert Config(mode="predict", checkpoint="model.ckpt", num_trials=1).num_trials == 1
    assert parse_command_config(["predict", "--data-dir", "audio", "--checkpoint", "model.ckpt", "--num-trials", "1"]).num_trials == 1
    assert normalize_decoder_config({"type": "whisperseg", "num_trials": 1}).num_trials == 1


def test_lora_defaults_and_full_training_option_round_trip():
    lora = Config(encoder_type="whisperseg")
    assert (lora.batch_size, lora.accumulate_grad_batches, lora.max_length, lora.seed) == (1, 4, 128, 0)
    assert (lora.learning_rate, lora.num_epochs, lora.early_stopping_patience) == (1e-4, 24, 5)
    assert (lora.reduce_lr_patience, lora.reduce_lr_factor) == (2, 0.5)
    assert lora.reduce_lr and not lora.linear_lr_schedule and lora.freeze_encoder
    assert lora.unfreeze_encoder_after == 3
    for config in (lora, lora.copy(whisperseg_training_method="full")):
        restored = train_gui_state_to_config(train_config_to_gui_state(config))
        assert restored.whisperseg_training_method == config.whisperseg_training_method
        assert restored.accumulate_grad_batches == config.accumulate_grad_batches
        assert restored.learning_rate == config.learning_rate
        assert restored.seed == config.seed
        assert restored.unfreeze_encoder_after == 3
    full = lora.copy(whisperseg_training_method="full")
    assert (full.batch_size, full.accumulate_grad_batches, full.learning_rate) == (4, 1, 3e-6)
    assert full.linear_lr_schedule and not full.freeze_encoder
    assert full.copy(whisperseg_training_method="lora").reduce_lr


def test_save_resumable_checkpoint_defaults_off_and_round_trips():
    args = ["train", "--data-dir", "audio", "--output-dir", "run", "--encoder", "whisperseg",
            "--initial-model", "whisperseg-aer:tiny-ms"]
    assert not parse_command_config(args).save_resumable_checkpoint
    enabled = parse_command_config(args + ["--save-resumable-checkpoint"])
    assert enabled.save_resumable_checkpoint
    restored = train_gui_state_to_config(train_config_to_gui_state(enabled))
    assert restored.save_resumable_checkpoint
    assert Config.from_mapping(enabled.to_mapping()).save_resumable_checkpoint
