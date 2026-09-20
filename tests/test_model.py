import pytest
import torch

from resflow.model import AdaptiveNorm, ResFlowUNet


def _small_figure7_model() -> ResFlowUNet:
    return ResFlowUNet(
        base_channels=32,
        channel_multipliers=(1, 1, 2, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        image_size=64,
    )


def test_adaptive_norm_uses_documented_combination_formula():
    module = AdaptiveNorm(8, 16, auxiliary_channels=4)
    x = torch.randn(2, 8, 8, 8)
    time = torch.randn(2, 16)
    auxiliary = torch.randn(2, 4, 8, 8)

    time_scale, time_shift = module.time_to_scale_shift(time).chunk(2, dim=1)
    auxiliary_scale, auxiliary_shift = module.auxiliary_to_scale_shift(
        auxiliary
    ).chunk(2, dim=1)
    expected = (
        1.0 + time_scale[:, :, None, None] + auxiliary_scale
    ) * module.norm(x) + time_shift[:, :, None, None] + auxiliary_shift

    assert torch.allclose(module(x, time, auxiliary), expected)


def test_figure7_shapes_four_encoder_injections_and_initialization(capsys):
    model = _small_figure7_model()
    batch = 2
    x = torch.randn(batch, 3, 64, 64)
    y = torch.randn_like(x)
    t = torch.rand(batch)

    with torch.no_grad():
        adapter_features = model.adapter(y)
        prediction = model(x, y, t)

    expected_adapter_shapes = [
        (batch, 32, 64, 64),
        (batch, 32, 32, 32),
        (batch, 64, 16, 16),
        (batch, 64, 8, 8),
    ]
    print(f"x_t:        {list(x.shape)}")
    for index, feature in enumerate(adapter_features):
        print(f"adapter a{index}: {list(feature.shape)}")
    print(f"pred:       {list(prediction.shape)}")

    assert tuple(x.shape) == (batch, 3, 64, 64)
    assert [tuple(feature.shape) for feature in adapter_features] == expected_adapter_shapes
    assert tuple(prediction.shape) == (batch, 6, 64, 64)

    # Every serial adapter stage exposes a zero-initialized output projection.
    # Thus auxiliary conditioning has exactly zero initial effect on the U-Net.
    assert all(torch.count_nonzero(feature) == 0 for feature in adapter_features)

    injection_points = []
    for level, modules in enumerate(model.down_levels):
        for block in modules["blocks"]:
            projection = block.adaptive_norm.auxiliary_to_scale_shift
            if projection is not None:
                injection_points.append((level, projection))

    assert [level for level, _ in injection_points] == [0, 1, 2, 3]
    assert len(injection_points) == 4
    assert all(
        block.adaptive_norm.auxiliary_to_scale_shift is None
        for modules in model.up_levels
        for block in modules["blocks"]
    )

    for level, projection in injection_points:
        feature = adapter_features[level]
        projected = projection(feature)
        target_channels = expected_adapter_shapes[level][1]
        height, width = expected_adapter_shapes[level][2:]
        assert projected.shape == (batch, target_channels * 2, height, width)
        scale, shift = projected.chunk(2, dim=1)
        assert scale.shape == shift.shape == (batch, target_channels, height, width)

    # The complete augmented-velocity head is zero-initialized, following DDPM.
    assert torch.count_nonzero(prediction) == 0

    printed = capsys.readouterr().out
    assert "x_t:        [2, 3, 64, 64]" in printed
    assert "adapter a0: [2, 32, 64, 64]" in printed
    assert "adapter a1: [2, 32, 32, 32]" in printed
    assert "adapter a2: [2, 64, 16, 16]" in printed
    assert "adapter a3: [2, 64, 8, 8]" in printed
    assert "pred:       [2, 6, 64, 64]" in printed


def test_adapter_shape_mismatch_is_not_silently_resized():
    module = AdaptiveNorm(8, 16, auxiliary_channels=4)
    x = torch.randn(1, 8, 8, 8)
    time = torch.randn(1, 16)
    wrong_size = torch.randn(1, 4, 4, 4)

    with pytest.raises(ValueError, match="spatial mismatch"):
        module(x, time, wrong_size)
