import torch

from das.models import ResidualBlock, TemporalConvNet
from das.models.encoders import TCNEncoder


def test_temporal_conv_net_matches_legacy_fly_parameter_count():
    model = TemporalConvNet(
        in_channels=1,
        nb_filters=16,
        kernel_size=16,
        nb_stacks=3,
        dilations=[1, 2, 4, 8, 16],
        return_sequences=True,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == 65_792


def test_temporal_conv_net_matches_legacy_zf_parameter_count_without_frontend():
    model = TemporalConvNet(
        in_channels=33,
        nb_filters=32,
        kernel_size=32,
        nb_stacks=4,
        dilations=[1, 2, 4, 8, 16],
        return_sequences=True,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == 678_208


def test_residual_block_matches_legacy_no_post_add_relu():
    block = ResidualBlock(
        in_channels=2,
        out_channels=2,
        dilation=1,
        kernel_size=1,
        dropout_rate=0.0,
        padding="same",
    )
    with torch.no_grad():
        block.conv.conv.weight.zero_()
        block.conv.conv.bias.zero_()
        block.residual_projection.weight.zero_()
        block.residual_projection.bias.zero_()

    x = torch.full((1, 2, 4), -1.0)
    merged, skip = block(x)

    assert torch.allclose(skip, torch.zeros_like(skip))
    assert torch.allclose(merged, x)


def test_tcn_encoder_returns_sequence_outputs_and_preserves_lengths():
    encoder = TCNEncoder(
        input_dim=8,
        tcn_num_filters=16,
        tcn_num_conv=2,
        tcn_dilations=[1, 2],
        tcn_kernel_size=3,
        tcn_padding="causal",
    )
    features = torch.randn(2, 32, 8)
    lengths = torch.tensor([32, 28], dtype=torch.long)

    encoded, output_lengths = encoder(features, lengths)

    assert encoded.shape == (2, 32, 16)
    assert torch.equal(output_lengths, lengths)
