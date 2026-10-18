from das.config import Config
from das.gui_helpers import train_config_to_gui_state, train_gui_state_to_config


def test_lora_defaults_and_full_training_option_round_trip():
    lora = Config(encoder_type="whisperseg")
    assert (lora.batch_size, lora.accumulate_grad_batches, lora.max_length, lora.seed) == (1, 4, 128, 0)
    assert (lora.learning_rate, lora.num_epochs, lora.early_stopping_patience) == (1e-4, 24, 5)
    assert (lora.reduce_lr_patience, lora.reduce_lr_factor) == (2, 0.5)
    assert lora.reduce_lr and not lora.linear_lr_schedule and lora.freeze_encoder
    for config in (lora, lora.copy(whisperseg_training_method="full")):
        restored = train_gui_state_to_config(train_config_to_gui_state(config))
        assert restored.whisperseg_training_method == config.whisperseg_training_method
        assert restored.accumulate_grad_batches == config.accumulate_grad_batches
        assert restored.learning_rate == config.learning_rate
        assert restored.seed == config.seed
    full = lora.copy(whisperseg_training_method="full")
    assert (full.batch_size, full.accumulate_grad_batches, full.learning_rate) == (4, 1, 3e-6)
    assert full.linear_lr_schedule and not full.freeze_encoder
    assert full.copy(whisperseg_training_method="lora").reduce_lr
