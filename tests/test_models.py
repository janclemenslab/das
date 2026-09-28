import itertools

import lightning as L
import pytest
import torch

from das.models import DASModel
from das.models.decoders import (
    AttentionDecoderConfig,
    ConvDecoderConfig,
    DecoderConfig,
    LSTMDecoderConfig,
    LinearDecoderConfig,
    WhisperSegDecoderConfig,
    build_decoder,
    serialize_decoder_config,
)
from das.models.encoders import ConformerEncoderConfig, TCNEncoderConfig, TweetynetEncoderConfig
from das.models.frontends import (
    ConvFrontendConfig,
    ConvResNetFrontendConfig,
    FrontendConfig,
    MelFrontendConfig,
    RawFrontendConfig,
    SincFrontendConfig,
    STFTFrontendConfig,
    WhisperSegFrontendConfig,
    build_frontend,
    normalize_frontend_config,
    serialize_frontend_config,
)


def _require_combo(frontend: str, encoder: str) -> None:
    if frontend in {"mel", "stft"}:
        pytest.importorskip("nnAudio")
    if encoder == "conformer":
        pytest.importorskip("torchaudio")


def _frontend_config(frontend: str):
    if frontend == "raw":
        return RawFrontendConfig(num_channels=8)
    if frontend == "stft":
        return STFTFrontendConfig(num_channels=8, kernel_size=16, hop_seconds=4 / 32_000)
    if frontend == "mel":
        return MelFrontendConfig(num_channels=8, kernel_size=16, hop_seconds=4 / 32_000)
    if frontend == "conv":
        return ConvFrontendConfig(num_channels=8, kernel_size=9, hop_seconds=4 / 32_000)
    raise ValueError(frontend)


def _encoder_config(encoder: str):
    if encoder == "conformer":
        return ConformerEncoderConfig(num_heads=1, hidden_size=16, num_layers=2, kernel_size=31)
    if encoder == "tcn":
        return TCNEncoderConfig(hidden_size=16, num_layers=2, dilations=[1, 2], kernel_size=3, dropout=0.1)
    raise ValueError(encoder)


def _decoder_config(decoder: str):
    if decoder == "linear":
        return LinearDecoderConfig()
    if decoder == "lstm":
        return LSTMDecoderConfig(hidden_size=64)
    if decoder == "conv":
        return ConvDecoderConfig(kernel_size=8)
    if decoder == "attention":
        return AttentionDecoderConfig(num_heads=4, num_layers=2, dropout=0.1)
    raise ValueError(decoder)


def _model_kwargs(frontend: str, encoder: str, decoder: str) -> dict[str, object]:
    return {
        "frontend": _frontend_config(frontend),
        "encoder": _encoder_config(encoder),
        "decoder": _decoder_config(decoder),
        "cross_entropy_weight": 0.9,
        "learning_rate": 0.0001,
    }


def test_dense_positive_class_weight():
    model = DASModel(
        num_classes=3,
        sr=1_000,
        positive_class_weight=2.5,
        **_model_kwargs("raw", "tcn", "linear"),
    )

    torch.testing.assert_close(model.criterion.weight, torch.tensor([1.0, 2.5, 2.5]))


def test_dense_boundary_weight_emphasizes_transition_errors():
    model = DASModel(
        num_classes=2,
        sr=1_000,
        boundary_weight=4.0,
        boundary_width_ms=1.0,
        **_model_kwargs("raw", "tcn", "linear"),
    )
    targets = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
    logits = torch.full((15, 2), -5.0)
    logits[torch.arange(15), targets[0]] = 5.0
    boundary_logits = logits.clone()
    boundary_logits[5] = 0.0
    interior_logits = logits.clone()
    interior_logits[8] = 0.0

    boundary_loss = model._dense_cross_entropy(boundary_logits, targets.reshape(-1), targets)
    interior_loss = model._dense_cross_entropy(interior_logits, targets.reshape(-1), targets)

    assert boundary_loss > interior_loss


def test_stft_frontend_config_preserves_frequency_bounds():
    frontend = normalize_frontend_config(
        FrontendConfig(type="stft", num_channels=16, kernel_size=64, hop_seconds=0.001, fmin=250.0, fmax=9000.0)
    )

    assert isinstance(frontend, STFTFrontendConfig)
    assert frontend.fmin == 250.0
    assert frontend.fmax == 9000.0
    assert serialize_frontend_config(frontend)["fmin"] == 250.0
    assert serialize_frontend_config(frontend)["fmax"] == 9000.0


def test_sinc_frontend_preserves_transient_resolution_and_learns_cutoffs():
    config = SincFrontendConfig(
        num_channels=8,
        kernel_size=31,
        hop_seconds=0.01,
        pad_mode="constant",
        fmin=20.0,
        fmax=480.0,
    )
    frontend = build_frontend(config, sr=1_000)

    features, lengths = frontend(torch.randn(2, 1_000), torch.tensor([1_000, 731]))
    features.square().mean().backward()

    assert features.shape == (2, 101, 8)
    assert lengths.tolist() == [101, 74]
    assert frontend.sinc.low_hz.grad is not None
    assert serialize_frontend_config(config)["type"] == "sinc"


def test_conv_resnet_frontend_preserves_one_millisecond_resolution():
    config = ConvResNetFrontendConfig(
        num_channels=8,
        kernel_size=9,
        hop_seconds=0.001,
        pad_mode="constant",
    )
    frontend = build_frontend(config, sr=10_000)

    features, lengths = frontend(torch.randn(2, 1_000), torch.tensor([1_000, 731]))
    features.square().mean().backward()

    assert features.shape == (2, 100, 8)
    assert lengths.tolist() == [100, 74]
    assert frontend.blocks[0].convs[0].weight.grad is not None
    assert serialize_frontend_config(config)["type"] == "conv_resnet"


def test_conv_resnet_compact_tcn_checkpoint_reconstruction(tmp_path):
    model = DASModel(
        num_classes=2,
        sr=10_000,
        class_names=["noise", "pulse"],
        class_types=["segment", "event"],
        frontend=ConvResNetFrontendConfig(
            num_channels=64,
            kernel_size=9,
            hop_seconds=0.001,
            pad_mode="reflect",
        ),
        encoder=TCNEncoderConfig(
            hidden_size=64,
            num_layers=1,
            dilations=[1, 2, 4, 8, 16],
            kernel_size=3,
            dropout=0.1,
        ),
        decoder=LinearDecoderConfig(),
        cross_entropy_weight=1.0,
        learning_rate=0.0003,
        num_time_steps=4096,
        chunk_stride=2048,
    )
    logits, lengths = model(torch.randn(2, 1000), torch.tensor([1000, 731]))
    checkpoint = tmp_path / "pulse.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        checkpoint,
    )

    loaded = DASModel.load_from_checkpoint(checkpoint)

    assert logits.shape == (2, 100, 2)
    assert lengths.tolist() == [100, 74]
    assert isinstance(loaded.frontend_config, ConvResNetFrontendConfig)
    assert isinstance(loaded.encoder_config, TCNEncoderConfig)


@pytest.mark.parametrize("num_channels", [1, 9, 16])
def test_conv_resnet_model_max_pools_multichannel_features(num_channels):
    model = DASModel(
        num_classes=2,
        sr=10_000,
        frontend=ConvResNetFrontendConfig(num_channels=8, kernel_size=9, hop_seconds=0.001, pad_mode="constant"),
        encoder=TCNEncoderConfig(hidden_size=8, num_layers=1, dilations=[1], kernel_size=3, dropout=0.0),
        decoder=LinearDecoderConfig(),
        cross_entropy_weight=1.0,
        learning_rate=0.001,
    ).eval()
    inputs = torch.randn(2, num_channels, 1_000)
    lengths = torch.tensor([1_000, 731])

    with torch.no_grad():
        logits, output_lengths = model(inputs, lengths)
        features, feature_lengths = model.frontend(inputs, lengths)
        encoded, _ = model.encoder(features, feature_lengths)
        expected = model.decoder(encoded)

    assert logits.shape == (2, 100, 2)
    assert output_lengths.tolist() == [100, 74]
    torch.testing.assert_close(logits, expected)


def test_whisperseg_frontend_config_preserves_dummy_settings():
    frontend = normalize_frontend_config(
        FrontendConfig(
            type="whisperseg",
            min_frequency=250,
            frequency_scale=0.5,
            spec_time_step=0.001,
        )
    )

    assert isinstance(frontend, WhisperSegFrontendConfig)
    assert serialize_frontend_config(frontend) == {
        "type": "whisperseg",
        "min_frequency": 250,
        "frequency_scale": 0.5,
        "spec_time_step": 0.001,
    }
    with pytest.raises(ValueError, match="frontend_type=whisperseg"):
        build_frontend(frontend, sr=32_000)


def test_whisperseg_decoder_config_preserves_dummy_settings():
    decoder = serialize_decoder_config(
        DecoderConfig(
            type="whisperseg",
            dropout=0.25,
            max_length=77,
            generation_max_length=222,
            num_trials=3,
            num_beams=5,
            top_k=2,
            top_p=0.9,
            length_penalty=0.8,
        )
    )

    assert decoder == {
        "type": "whisperseg",
        "dropout": 0.25,
        "max_length": 77,
        "generation_max_length": 222,
        "num_trials": 3,
        "num_beams": 5,
        "top_k": 2,
        "top_p": 0.9,
        "length_penalty": 0.8,
    }
    with pytest.raises(ValueError, match="decoder_type=whisperseg"):
        build_decoder(WhisperSegDecoderConfig(), input_dim=8, num_classes=4)


def test_stft_frontend_frequency_bounds_limit_output_bins():
    pytest.importorskip("nnAudio")

    frontend = build_frontend(
        STFTFrontendConfig(num_channels=8, kernel_size=8, hop_seconds=0.001, fmin=1000.0, fmax=3000.0),
        sr=8000,
    )

    assert frontend.output_dim == 3


@pytest.mark.parametrize(
    ("frontend", "encoder", "decoder"),
    list(itertools.product(("raw", "stft", "mel", "conv"), ("conformer", "tcn"), ("linear", "lstm", "conv", "attention"))),
)
def test_model_construction(frontend: str, encoder: str, decoder: str):
    _require_combo(frontend, encoder)
    model = DASModel(
        num_classes=4,
        sr=32_000,
        **_model_kwargs(frontend, encoder, decoder),
    )
    assert isinstance(model, DASModel)


def test_forward_waveform_mel_conformer_linear():
    _require_combo("mel", "conformer")
    model = DASModel(
        num_classes=4,
        sr=32_000,
        **_model_kwargs("mel", "conformer", "linear"),
    )
    inputs = torch.randn(2, 64)
    logits, output_lengths = model(inputs)

    assert model.frontend.output_dim == 8
    assert logits.shape[0] == 2
    assert logits.shape[2] == 4
    assert output_lengths.shape == (2,)
    assert torch.all(output_lengths == logits.shape[1])


def test_forward_waveform_stft_tcn_conv():
    _require_combo("stft", "tcn")
    model = DASModel(
        num_classes=4,
        sr=32_000,
        **_model_kwargs("stft", "tcn", "conv"),
    )
    inputs = torch.randn(2, 64)
    logits, output_lengths = model(inputs)

    assert model.frontend.output_dim == 8
    assert logits.shape[0] == 2
    assert logits.shape[2] == 4
    assert output_lengths.shape == (2,)
    assert torch.all(output_lengths == logits.shape[1])


def test_forward_precomputed_raw_conformer_lstm():
    _require_combo("raw", "conformer")
    model = DASModel(
        num_classes=4,
        sr=32_000,
        **_model_kwargs("raw", "conformer", "lstm"),
    )
    inputs = torch.randn(2, 8, 32)
    logits, output_lengths = model(inputs)

    assert logits.shape == (2, 32, 4)
    assert torch.all(output_lengths == 32)


def test_forward_precomputed_raw_tcn_attention():
    model = DASModel(
        num_classes=4,
        sr=32_000,
        **_model_kwargs("raw", "tcn", "attention"),
    )
    inputs = torch.randn(2, 8, 32)
    logits, output_lengths = model(inputs)

    assert logits.shape == (2, 32, 4)
    assert torch.all(output_lengths == 32)


def test_forward_precomputed_raw_tweetynet_linear():
    model = DASModel(
        num_classes=4,
        sr=32_000,
        frontend=RawFrontendConfig(num_channels=64),
        encoder=TweetynetEncoderConfig(hidden_size=16, num_layers=1, kernel_size=5, dropout=0.0),
        decoder=LinearDecoderConfig(),
        cross_entropy_weight=1.0,
        learning_rate=0.001,
    )
    inputs = torch.randn(2, 64, 32)
    logits, output_lengths = model(inputs)

    assert logits.shape == (2, 32, 4)
    assert torch.all(output_lengths == 32)


def test_training_step_with_one_hot_targets_and_time_alignment():
    _require_combo("mel", "conformer")
    model = DASModel(
        num_classes=4,
        sr=32_000,
        **_model_kwargs("mel", "conformer", "linear"),
    )
    inputs = torch.randn(2, 64)
    input_lengths = torch.full((2,), 64, dtype=torch.long)
    logits, _ = model(inputs, input_lengths)
    target_time = logits.shape[1] + 3
    target_indices = torch.randint(0, 4, (2, target_time))
    targets = torch.nn.functional.one_hot(target_indices, num_classes=4).float()
    target_lengths = torch.full((2,), target_time, dtype=torch.long)

    loss = model.training_step((inputs, input_lengths, targets, target_lengths), batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_configure_optimizers_uses_reduce_lr_settings():
    model = DASModel(
        num_classes=4,
        sr=32_000,
        reduce_lr_patience=3,
        reduce_lr_factor=0.25,
        reduce_lr_min=1e-7,
        **_model_kwargs("raw", "tcn", "linear"),
    )

    optimizer_config = model.configure_optimizers()

    scheduler = optimizer_config["lr_scheduler"]["scheduler"]
    assert optimizer_config["lr_scheduler"]["monitor"] == "train_loss"
    assert scheduler.factor == pytest.approx(0.25)
    assert scheduler.patience == 3
    assert scheduler.min_lrs == pytest.approx([1e-7])


def test_configure_optimizers_can_disable_reduce_lr():
    model = DASModel(
        num_classes=4,
        sr=32_000,
        reduce_lr=False,
        **_model_kwargs("raw", "tcn", "linear"),
    )

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)


def test_freeze_encoder_leaves_frontend_and_decoder_trainable():
    model = DASModel(
        num_classes=2,
        sr=32_000,
        **_model_kwargs("conv", "tcn", "linear"),
    )

    model.freeze_encoder()
    model.train()

    assert any(parameter.requires_grad for parameter in model.frontend.parameters())
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.decoder.parameters())
    assert model.frontend.training
    assert not model.encoder.training
    assert model.decoder.training


def test_import_smoke():
    import das
    from das.cli import cli_main
    from das.models import DASModel as ImportedModel

    assert ImportedModel is DASModel
    assert callable(cli_main)
    assert callable(das.train)
    assert callable(das.predict)
