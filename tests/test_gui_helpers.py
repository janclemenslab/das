from pathlib import Path

import yaml

import das.cli as cli
from das.config import BUILTIN_CONFIG_OPTIONS, Config
from das.gui_helpers import (
    PredictGuiState,
    TrainGuiState,
    format_predict_config_yaml,
    format_train_config_yaml,
    load_predict_gui_state_from_yaml,
    load_train_gui_state_from_yaml,
    predict_config_to_gui_state,
    predict_gui_state_to_config,
    train_gui_state_to_config,
)


def test_train_gui_defaults_match_cli_defaults():
    state = TrainGuiState()
    state.paths.data_dir = "data"
    state.paths.output_dir = "run"

    assert train_gui_state_to_config(state).to_mapping() == cli.parse_command_config(
        ["train", "--data-dir", "data", "--output-dir", "run"]
    ).to_mapping()


def test_predict_gui_defaults_match_cli_defaults():
    state = PredictGuiState()
    state.paths.data_dir = "data"
    state.paths.checkpoint = "model.ckpt"
    state.paths.output_dir = "predictions"

    assert predict_gui_state_to_config(state).to_mapping() == cli.parse_command_config(
        ["predict", "--data-dir", "data", "--checkpoint", "model.ckpt", "--output-dir", "predictions"]
    ).to_mapping()


def test_predict_gui_defaults_to_output_next_to_audio():
    state = PredictGuiState()

    assert state.paths.output_dir == ""
    assert predict_gui_state_to_config(state).output_dir is None


def test_train_yaml_round_trip_preserves_flat_structure(tmp_path: Path):
    initial_model = tmp_path / "initial.ckpt"
    initial_model.write_bytes(b"checkpoint")
    state = TrainGuiState()
    state.paths.data_dir = "/tmp/audio"
    state.paths.output_dir = "/tmp/run"
    state.paths.initial_model = str(initial_model)
    state.model_selection.frontend_type = "raw"
    state.model_selection.encoder_type = "tcn"
    state.model_selection.decoder_type = "lstm"
    state.paths.checkpoint_prefix = "zf"
    state.tcn_encoder.hidden_size = 16
    state.tcn_encoder.num_layers = 2
    state.tcn_encoder.dilations = "1, 2, 4"
    state.tcn_encoder.use_separable = "true, false"
    state.lstm_decoder.hidden_size = 32
    state.data.validation_fraction = 0.15
    state.data.test_fraction = 0.05
    state.data.ignore_class_names = True
    state.data.include_labels = ["pulse", "sine"]

    config = train_gui_state_to_config(state)
    path = tmp_path / "train.yaml"
    path.write_text(format_train_config_yaml(config), encoding="utf-8")

    loaded = train_gui_state_to_config(load_train_gui_state_from_yaml(str(path)))

    assert yaml.safe_load(format_train_config_yaml(loaded)) == yaml.safe_load(format_train_config_yaml(config))
    assert loaded.initial_model == str(initial_model)
    assert loaded.checkpoint_prefix == "zf"
    assert loaded.include_labels == ["pulse", "sine"]


def test_stft_frontend_frequency_bounds_round_trip(tmp_path: Path):
    state = TrainGuiState()
    state.paths.data_dir = "/tmp/audio"
    state.paths.output_dir = "/tmp/run"
    state.model_selection.frontend_type = "stft"
    state.stft_frontend.fmin = 250.0
    state.stft_frontend.fmax = "9000"

    config = train_gui_state_to_config(state)
    path = tmp_path / "train.yaml"
    path.write_text(format_train_config_yaml(config), encoding="utf-8")
    loaded = load_train_gui_state_from_yaml(str(path))

    assert config.frontend_fmin == 250.0
    assert config.frontend_fmax == 9000.0
    assert loaded.stft_frontend.fmin == 250.0
    assert float(loaded.stft_frontend.fmax) == 9000.0


def test_gui_detection_thresholds_round_trip(tmp_path: Path):
    state = TrainGuiState()
    state.paths.data_dir = "/tmp/audio"
    state.paths.output_dir = "/tmp/run"
    state.prediction.syllable_postprocessor = "binary_mask"
    state.prediction.segment_threshold_low = 0.2
    state.prediction.segment_threshold_high = 0.8
    state.prediction.event_threshold = 0.7
    state.prediction.event_dist_min_ms = 3.0
    state.prediction.event_dist_max_ms = "40"

    config = train_gui_state_to_config(state)
    path = tmp_path / "train.yaml"
    path.write_text(format_train_config_yaml(config), encoding="utf-8")
    loaded = train_gui_state_to_config(load_train_gui_state_from_yaml(str(path)))

    assert loaded.syllable_postprocessor == "binary_mask"
    assert loaded.segment_threshold_low == 0.2
    assert loaded.segment_threshold_high == 0.8
    assert loaded.event_threshold == 0.7
    assert loaded.event_dist_min_ms == 3.0
    assert loaded.event_dist_max_ms == 40.0
    assert yaml.safe_load(format_train_config_yaml(loaded)) == yaml.safe_load(format_train_config_yaml(config))

    predict_state = PredictGuiState()
    predict_state.prediction.segment_threshold_low = 0.3
    predict_state.prediction.segment_threshold_high = 0.9
    predict_state.prediction.event_threshold = 0.6
    predict_state.prediction.event_dist_max_ms = "75"
    predicted = predict_config_to_gui_state(predict_gui_state_to_config(predict_state))
    assert predicted.prediction.segment_threshold_low == 0.3
    assert predicted.prediction.segment_threshold_high == 0.9
    assert predicted.prediction.event_threshold == 0.6
    assert predicted.prediction.event_dist_max_ms == "75.0"


def test_train_gui_native_training_controls_round_trip(tmp_path: Path):
    state = TrainGuiState()
    state.paths.data_dir = "/tmp/audio"
    state.paths.output_dir = "/tmp/run"
    state.model_hyperparameters.early_stopping = False
    state.model_hyperparameters.early_stopping_patience = 4
    state.model_hyperparameters.reduce_lr = False
    state.model_hyperparameters.reduce_lr_patience = 3
    state.model_hyperparameters.reduce_lr_factor = 0.25
    state.model_hyperparameters.reduce_lr_min = 1e-8
    state.model_hyperparameters.linear_lr_schedule = False
    state.model_hyperparameters.weight_decay = 0.02
    state.model_hyperparameters.warmup_steps = 12
    state.encoder_settings.freeze_encoder = True

    config = train_gui_state_to_config(state)
    path = tmp_path / "train.yaml"
    path.write_text(format_train_config_yaml(config), encoding="utf-8")
    loaded = train_gui_state_to_config(load_train_gui_state_from_yaml(str(path)))

    assert loaded.early_stopping is False
    assert loaded.early_stopping_patience == 4
    assert loaded.reduce_lr is False
    assert loaded.reduce_lr_patience == 3
    assert loaded.reduce_lr_factor == 0.25
    assert loaded.reduce_lr_min == 1e-8
    assert loaded.linear_lr_schedule is False
    assert loaded.weight_decay == 0.02
    assert loaded.warmup_steps == 12
    assert loaded.freeze_encoder is True
    assert yaml.safe_load(format_train_config_yaml(loaded)) == yaml.safe_load(format_train_config_yaml(config))


def test_attention_decoder_round_trip_preserves_decoder_settings(tmp_path: Path):
    state = TrainGuiState()
    state.paths.data_dir = "/tmp/audio"
    state.paths.output_dir = "/tmp/run"
    state.model_selection.decoder_type = "attention"
    state.attention_decoder.num_heads = 2
    state.attention_decoder.num_layers = 3
    state.attention_decoder.dropout = 0.25

    config = train_gui_state_to_config(state)
    path = tmp_path / "train.yaml"
    path.write_text(format_train_config_yaml(config), encoding="utf-8")

    loaded = load_train_gui_state_from_yaml(str(path))

    assert loaded.model_selection.decoder_type == "attention"
    assert loaded.attention_decoder.num_heads == 2
    assert loaded.attention_decoder.num_layers == 3
    assert loaded.attention_decoder.dropout == 0.25


def test_whisperseg_gui_round_trip_preserves_dummy_frontend_and_decoder(tmp_path: Path):
    state = TrainGuiState()
    state.paths.data_dir = "/tmp/audio"
    state.paths.output_dir = "/tmp/run"
    state.model_selection.frontend_type = "mel"
    state.model_selection.encoder_type = "whisperseg"
    state.model_selection.decoder_type = "linear"
    state.whisperseg_frontend.min_frequency = "250"
    state.whisperseg_frontend.frequency_scale = "0.5"
    state.whisperseg_frontend.spec_time_step = "0.001"
    state.whisperseg_decoder.decoder_dropout = 0.25
    state.whisperseg_decoder.max_length = 77
    state.whisperseg_decoder.generation_max_length = 222
    state.whisperseg_decoder.num_trials = 3
    state.whisperseg_decoder.num_beams = 5
    state.whisperseg_decoder.top_k = 2
    state.whisperseg_decoder.top_p = 0.9
    state.whisperseg_decoder.length_penalty = 0.8

    config = train_gui_state_to_config(state)
    path = tmp_path / "train.yaml"
    path.write_text(format_train_config_yaml(config), encoding="utf-8")
    loaded = load_train_gui_state_from_yaml(str(path))

    assert config.frontend_type == "whisperseg"
    assert config.encoder_type == "whisperseg"
    assert config.decoder_type == "whisperseg"
    assert config.min_frequency == 250
    assert config.frequency_scale == 0.5
    assert config.spec_time_step == 0.001
    assert config.decoder_dropout == 0.25
    assert config.max_length == 77
    assert config.generation_max_length == 222
    assert config.num_trials == 3
    assert loaded.model_selection.frontend_type == "whisperseg"
    assert loaded.model_selection.encoder_type == "whisperseg"
    assert loaded.model_selection.decoder_type == "whisperseg"
    assert loaded.whisperseg_frontend.min_frequency == "250"
    assert loaded.whisperseg_frontend.frequency_scale == "0.5"
    assert loaded.whisperseg_frontend.spec_time_step == "0.001"
    assert loaded.whisperseg_decoder.decoder_dropout == 0.25
    assert loaded.whisperseg_decoder.max_length == 77
    assert loaded.whisperseg_decoder.generation_max_length == 222
    assert loaded.whisperseg_decoder.num_trials == 3


def test_predict_yaml_round_trip_preserves_flat_structure(tmp_path: Path):
    state = PredictGuiState()
    state.paths.data_dir = "/tmp/audio"
    state.paths.checkpoint = "/tmp/model.ckpt"
    state.paths.output_dir = "/tmp/predictions"
    state.data.batch_size = 4
    state.data.num_workers = 2
    state.trainer.accelerator = "cpu"
    state.prediction.output_suffix = "_pred.csv"
    state.prediction.existing_annotations = "merge"
    state.whisperseg_decoder.generation_max_length = 333
    state.whisperseg_decoder.num_trials = 4
    state.whisperseg_decoder.num_beams = 6
    state.whisperseg_decoder.top_k = 3
    state.whisperseg_decoder.top_p = 0.8
    state.whisperseg_decoder.length_penalty = 0.7

    config = predict_gui_state_to_config(state)
    path = tmp_path / "predict.yaml"
    path.write_text(format_predict_config_yaml(config), encoding="utf-8")

    loaded = predict_gui_state_to_config(load_predict_gui_state_from_yaml(str(path)))

    assert yaml.safe_load(format_predict_config_yaml(loaded)) == yaml.safe_load(format_predict_config_yaml(config))
    assert loaded.existing_annotations == "merge"
    assert loaded.generation_max_length == 333
    assert loaded.num_trials == 4
    assert loaded.num_beams == 6
    assert loaded.top_k == 3
    assert loaded.top_p == 0.8
    assert loaded.length_penalty == 0.7


def test_flat_yaml_loader_uses_mode_specific_defaults(tmp_path: Path):
    path = tmp_path / "predict.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data_dir": "/new/audio",
                "checkpoint": "/new/model.ckpt",
                "output_dir": "/new/predictions",
                "output_suffix": "_frames.csv",
                "existing_annotations": "skip",
                "evaluate": True,
                "split": "test",
                "syllable_tolerance_ms": 20.0,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    state = load_predict_gui_state_from_yaml(str(path))

    assert state.paths.data_dir == "/new/audio"
    assert state.paths.checkpoint == "/new/model.ckpt"
    assert state.prediction.output_suffix == "_frames.csv"
    assert state.prediction.existing_annotations == "skip"
    config = predict_gui_state_to_config(state)
    assert config.evaluate is True
    assert config.split == "test"
    assert config.existing_annotations == "skip"
    assert config.syllable_tolerance_ms == 20.0


def test_builtin_config_options_include_fly_and_zebra_finch():
    assert BUILTIN_CONFIG_OPTIONS == [
        ("fly-pulse", "Fly pulse"),
        ("fly", "Fly (classic)"),
        ("zebra-finch", "Zebra Finch"),
        ("tweetynet", "TweetyNet"),
    ]


def test_builtin_fly_pulse_config_uses_conv_resnet_tcn():
    config = Config.from_config_sources(
        ["fly-pulse"],
        base=Config(mode="train", data_dir="/tmp/audio", output_dir="/tmp/run"),
    )

    assert config.include_labels == ["pulse"]
    assert config.frontend_type == "conv_resnet"
    assert config.frontend_num_channels == 64
    assert config.frontend_hop_seconds == 0.001
    assert config.frontend_pad_mode == "reflect"
    assert config.encoder_type == "tcn"
    assert config.encoder_hidden_size == 64
    assert config.encoder_num_layers == 1
    assert config.encoder_kernel_size == 3


def test_builtin_config_train_merges_species_defaults_with_base_config():
    config = Config.from_config_sources(
        ["fly"],
        base=Config(mode="train", data_dir="/tmp/audio", output_dir="/tmp/run"),
    )

    assert config.mode == "train"
    assert config.data_dir == "/tmp/audio"
    assert config.output_dir == "/tmp/run"
    assert config.batch_size == 32
    assert config.num_time_steps == 1024
    assert config.chunk_stride == 928
    assert config.frontend_type == "raw"
    assert config.encoder_type == "tcn"
    assert config.encoder_hidden_size == 16
    assert config.encoder_num_layers == 3
    assert config.encoder_kernel_size == 16


def test_builtin_config_predict_uses_mode_specific_values_and_preserves_paths():
    config = Config.from_config_sources(
        ["zebra-finch"],
        mode="predict",
        base=Config(mode="predict", data_dir="/tmp/audio", checkpoint="/tmp/model.ckpt", output_dir="/tmp/pred"),
    )

    assert config.mode == "predict"
    assert config.data_dir == "/tmp/audio"
    assert config.checkpoint == "/tmp/model.ckpt"
    assert config.output_dir == "/tmp/pred"
    assert config.batch_size == 32
    assert config.fill_gap_ms == 5.0
    assert config.min_syllable_ms == 20.0
    assert config.syllable_tolerance_ms == 20.0


def test_builtin_config_tweetynet_train_sets_architecture_defaults():
    config = Config.from_config_sources(
        ["tweetynet"],
        mode="train",
        base=Config(mode="train", data_dir="/tmp/audio", output_dir="/tmp/run"),
    )

    assert config.mode == "train"
    assert config.data_dir == "/tmp/audio"
    assert config.output_dir == "/tmp/run"
    assert config.batch_size == 16
    assert config.frontend_type == "stft"
    assert config.frontend_num_channels == 513
    assert config.frontend_kernel_size == 1024
    assert config.frontend_trainable is False
    assert config.encoder_type == "tweetynet"
    assert config.encoder_hidden_size == 512
    assert config.encoder_num_layers == 1
    assert config.encoder_kernel_size == 5
    assert config.encoder_dropout == 0.1
    assert config.decoder_type == "linear"
    assert config.cross_entropy_weight == 1.0
    assert config.learning_rate == 0.001
