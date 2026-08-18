from __future__ import annotations

import numpy as np
import pytest
import torch

from oisst_fno.data import SSTSequenceDataset
from oisst_fno.experiment import (
    measure_cost,
    set_global_seed,
    width_for_parameter_budget,
)
from oisst_fno.metrics import (
    parameter_count,
    temporal_increment_correlation,
    temporal_variability_ratio,
)
from oisst_fno.model import FNO2d
from oisst_fno.model3d import FNO3d, OperatorBlock3d, TruncatedFourierMix3d

# ------------------------------------------------------------------ spectral layer


def test_spectral_layer_preserves_shape() -> None:
    layer = TruncatedFourierMix3d(2, 3, modes_t=2, modes_y=4, modes_x=4)
    field = torch.randn(2, 2, 8, 16, 20)

    out = layer(field)

    assert out.shape == (2, 3, 8, 16, 20)
    assert torch.isfinite(out).all()


def test_spectral_layer_output_is_real() -> None:
    layer = TruncatedFourierMix3d(1, 1, modes_t=2, modes_y=3, modes_x=3)

    out = layer(torch.randn(1, 1, 8, 12, 12))

    assert not out.is_complex(), "the inverse real FFT must return a real field"


def test_spectral_layer_retains_four_corner_bands() -> None:
    """Real input halves only the last axis, so time and latitude each need both signs."""
    layer = TruncatedFourierMix3d(2, 3, modes_t=2, modes_y=4, modes_x=5)

    assert layer.kernel_components.shape == (4, 3, 2, 2, 4, 5, 2)


def test_spectral_layer_rejects_overlapping_modes() -> None:
    layer = TruncatedFourierMix3d(1, 1, modes_t=6, modes_y=2, modes_x=2)

    with pytest.raises(ValueError, match="overlap or exceed"):
        layer(torch.randn(1, 1, 8, 16, 16))  # 2 * modes_t > 8 time steps


def test_spectral_layer_rejects_wrong_rank() -> None:
    layer = TruncatedFourierMix3d(1, 1, modes_t=2, modes_y=2, modes_x=2)

    with pytest.raises(ValueError, match=r"\[batch, channels, time, height, width\]"):
        layer(torch.randn(1, 1, 16, 16))


def test_spectral_layer_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TruncatedFourierMix3d(1, 1, modes_t=0, modes_y=2, modes_x=2)


def test_temporal_mixing_is_translation_equivariant() -> None:
    """The point of transforming time: a circular shift in equals the same shift out.

    This is what the channel-stacked layer cannot offer, since permuting its channels
    carries no notion of adjacency.
    """
    set_global_seed(0)
    layer = TruncatedFourierMix3d(1, 1, modes_t=3, modes_y=4, modes_x=4)
    field = torch.randn(1, 1, 12, 16, 16)

    shifted_then_mixed = layer(torch.roll(field, shifts=3, dims=2))
    mixed_then_shifted = torch.roll(layer(field), shifts=3, dims=2)

    assert torch.allclose(shifted_then_mixed, mixed_then_shifted, atol=1e-5)


# -------------------------------------------------------------------- operator block


def test_operator_block_preserves_shape() -> None:
    block = OperatorBlock3d(width=6, modes_t=2, modes_y=3, modes_x=3)
    state = torch.randn(2, 6, 8, 12, 12)

    assert block(state).shape == state.shape
    assert block(state, activate=False).shape == state.shape


# ----------------------------------------------------------------------- full model


def test_model_maps_history_to_a_forecast_sequence() -> None:
    model = FNO3d(steps_out=7, width=6, modes_t=2, modes_y=4, modes_x=4, depth=2)
    history = torch.randn(2, 1, 14, 24, 28)

    out = model(history)

    assert out.shape == (2, 1, 7, 24, 28)
    assert torch.isfinite(out).all()


def test_model_can_return_the_same_length_it_consumed() -> None:
    model = FNO3d(steps_out=14, width=6, modes_t=2, modes_y=4, modes_x=4, depth=2)

    assert model(torch.randn(1, 1, 14, 20, 24)).shape == (1, 1, 14, 20, 24)


def test_model_rejects_wrong_rank_and_bad_configuration() -> None:
    model = FNO3d(steps_out=3, width=4, modes_t=2, modes_y=2, modes_x=2, depth=1)
    with pytest.raises(ValueError, match=r"\[batch, channels, time, height, width\]"):
        model(torch.randn(1, 1, 16, 16))

    with pytest.raises(ValueError, match="must be positive"):
        FNO3d(steps_out=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        FNO3d(steps_out=3, padding=-1)


def test_model_is_differentiable() -> None:
    model = FNO3d(steps_out=3, width=4, modes_t=2, modes_y=2, modes_x=2, depth=1)

    model(torch.randn(1, 1, 8, 12, 12)).square().mean().backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_parameters_grow_with_temporal_modes() -> None:
    """The extra cost of representing time explicitly, made concrete."""

    def build(modes_t: int) -> FNO3d:
        return FNO3d(steps_out=7, width=8, modes_t=modes_t, modes_y=4, modes_x=4, depth=2)

    few = parameter_count(build(1))
    many = parameter_count(build(4))

    assert many > few
    # Spectral weights dominate and scale linearly in modes_t.
    assert 3.0 < many / few < 4.5


# ------------------------------------------------------------------ budget matching


def test_width_for_parameter_budget_lands_close_to_the_target() -> None:
    target = parameter_count(FNO2d(in_channels=15, out_channels=14, width=24, depth=2))

    match = width_for_parameter_budget(
        lambda width: FNO3d(steps_out=14, width=width, modes_t=2, modes_y=8, modes_x=8, depth=2),
        target,
        max_width=64,
    )

    assert match.width >= 2
    assert abs(match.relative_error) < 0.25, "must get within a quarter of the target"
    assert match.target == target


def test_width_for_parameter_budget_validates_input() -> None:
    with pytest.raises(ValueError, match="target_parameters"):
        width_for_parameter_budget(lambda w: FNO2d(in_channels=2, width=w, depth=1), 0)
    with pytest.raises(ValueError, match="width range"):
        width_for_parameter_budget(
            lambda w: FNO2d(in_channels=2, width=w, depth=1), 1000, min_width=8, max_width=4
        )


def test_relative_error_is_signed() -> None:
    from oisst_fno.experiment import BudgetMatch

    assert BudgetMatch(width=4, parameters=110, target=100).relative_error == pytest.approx(0.1)
    assert BudgetMatch(width=4, parameters=90, target=100).relative_error == pytest.approx(-0.1)


# ------------------------------------------------------------------ cost measurement


def test_measure_cost_reports_time_and_parameters() -> None:
    model = FNO3d(steps_out=3, width=4, modes_t=2, modes_y=2, modes_x=2, depth=1)
    sample = torch.randn(1, 1, 8, 12, 12)

    cost = measure_cost(model, sample, repeats=2, warmup=1)

    assert cost.seconds_per_pass > 0
    assert cost.parameters == parameter_count(model)
    assert cost.peak_gpu_mb is None, "CPU runs report no GPU memory"
    assert "seconds_per_pass" in cost.to_dict()


def test_measure_cost_can_include_the_backward_pass() -> None:
    model = FNO3d(steps_out=3, width=4, modes_t=2, modes_y=2, modes_x=2, depth=1)
    sample = torch.randn(1, 1, 8, 12, 12)

    cost = measure_cost(model, sample, repeats=1, warmup=0, backward=True)

    assert cost.seconds_per_pass > 0


def test_measure_cost_validates_repeats() -> None:
    model = FNO3d(steps_out=3, width=4, modes_t=2, modes_y=2, modes_x=2, depth=1)
    with pytest.raises(ValueError, match="repeats"):
        measure_cost(model, torch.randn(1, 1, 8, 12, 12), repeats=0)


# ---------------------------------------------------------------- sequence dataset


def _field(steps: int = 20) -> np.ndarray:
    values = np.arange(steps * 3 * 4, dtype=np.float32).reshape(steps, 3, 4)
    values[:, 0, 0] = np.nan
    return values


def test_sequence_dataset_returns_a_target_trajectory() -> None:
    dataset = SSTSequenceDataset(_field(), lookback_days=5, horizon_days=3)
    x, y, mask = dataset[0]

    assert x.shape == (5, 3, 4)
    assert y.shape == (3, 3, 4), "the target is a sequence, not a single field"
    assert mask.shape == (1, 3, 4)
    assert len(dataset) == 20 - 5 - 3 + 1


def test_sequence_targets_immediately_follow_the_history() -> None:
    values = _field()
    dataset = SSTSequenceDataset(values, lookback_days=5, horizon_days=3)

    x, y, _ = dataset[2]

    assert np.allclose(np.nan_to_num(values[2:7]), x.numpy())
    assert np.allclose(np.nan_to_num(values[7:10]), y.numpy())


def test_target_offsets_are_lead_times_in_days() -> None:
    dataset = SSTSequenceDataset(_field(), lookback_days=5, horizon_days=14)

    assert np.array_equal(dataset.target_offsets(), np.arange(1, 15))


def test_sequence_dataset_validates_input() -> None:
    with pytest.raises(ValueError, match=r"\[time, height, width\]"):
        SSTSequenceDataset(np.zeros((5, 5), dtype=np.float32), 2, 2)
    with pytest.raises(ValueError, match="must be positive"):
        SSTSequenceDataset(_field(), lookback_days=0, horizon_days=2)
    with pytest.raises(ValueError, match="at least"):
        SSTSequenceDataset(_field(steps=5), lookback_days=4, horizon_days=4)
    with pytest.raises(IndexError):
        SSTSequenceDataset(_field(), 5, 3)[999]


def test_both_architectures_consume_the_same_window() -> None:
    """The comparison is only fair if the two models see identical inputs and targets."""
    dataset = SSTSequenceDataset(_field(steps=30), lookback_days=14, horizon_days=14)
    x, y, mask = dataset[0]

    # 2-D baseline: time as channels, plus the mask channel, predicting 14 output channels.
    two_d = FNO2d(in_channels=15, out_channels=14, width=8, modes_y=1, modes_x=1, depth=1)
    stacked = torch.cat((x, mask), dim=0).unsqueeze(0)
    out_2d = two_d(stacked)

    # 3-D: the same window with time as a dimension.
    three_d = FNO3d(steps_out=14, width=4, modes_t=2, modes_y=1, modes_x=1, depth=1)
    out_3d = three_d(x.unsqueeze(0).unsqueeze(0))

    assert out_2d.shape == (1, 14, 3, 4)
    assert out_3d.shape == (1, 1, 14, 3, 4)
    assert out_3d.squeeze(1).shape == out_2d.shape, "comparable after removing the channel axis"
    assert y.shape == (14, 3, 4)


# ------------------------------------------------------------- temporal coherence


def test_increment_correlation_is_one_for_a_perfect_forecast() -> None:
    rng = np.random.default_rng(0)
    target = rng.normal(size=(14, 6, 8)).astype(np.float32)

    assert temporal_increment_correlation(target, target) == pytest.approx(1.0, abs=1e-6)


def test_increment_correlation_falls_for_a_temporally_flat_forecast() -> None:
    """A forecast can track the mean well and still not move like the truth."""
    rng = np.random.default_rng(1)
    target = np.cumsum(rng.normal(size=(14, 6, 8)), axis=0).astype(np.float32)
    frozen = np.repeat(target[:1], 14, axis=0)  # persistence: no temporal change at all

    assert temporal_increment_correlation(frozen, target) == pytest.approx(0.0)


def test_increment_correlation_respects_the_mask() -> None:
    rng = np.random.default_rng(2)
    target = rng.normal(size=(6, 4, 5)).astype(np.float32)
    prediction = target.copy()
    prediction[:, 0, 0] = 999.0  # a wild value on a land cell
    mask = np.ones((4, 5), dtype=np.float32)
    mask[0, 0] = 0.0

    assert temporal_increment_correlation(prediction, target, mask) == pytest.approx(1.0, abs=1e-6)


def test_variability_ratio_detects_temporal_over_smoothing() -> None:
    rng = np.random.default_rng(3)
    target = np.cumsum(rng.normal(size=(14, 6, 8)), axis=0).astype(np.float32)
    smoothed = target * 0.0 + target.mean(axis=0, keepdims=True)

    assert temporal_variability_ratio(target, target) == pytest.approx(1.0, abs=1e-6)
    assert temporal_variability_ratio(smoothed, target) == pytest.approx(0.0, abs=1e-6)


def test_coherence_metrics_validate_input() -> None:
    a = np.zeros((3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        temporal_increment_correlation(a, np.zeros((3, 4, 6), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[lead, height, width\]"):
        temporal_increment_correlation(
            np.zeros((4, 5), dtype=np.float32), np.zeros((4, 5), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="two lead times"):
        temporal_variability_ratio(a[:1], a[:1])
