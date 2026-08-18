from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from oisst_fno.experiment import (
    GPU_NONDETERMINISM_NOTES,
    CurveDiagnosis,
    EnvironmentInfo,
    EpochRecord,
    ExperimentConfig,
    TrainingHistory,
    amp_is_supported,
    collect_environment,
    diagnose_learning_curves,
    gradient_norm,
    set_global_seed,
)


def _draw() -> tuple[float, float, float]:
    """Draw one value from each seeded generator."""
    import random

    return (random.random(), float(np.random.rand()), float(torch.rand(1)))


def test_seeding_makes_all_three_generators_reproducible() -> None:
    set_global_seed(1234)
    first = _draw()
    set_global_seed(1234)
    second = _draw()

    assert first == second


def test_different_seeds_give_different_draws() -> None:
    set_global_seed(1)
    first = _draw()
    set_global_seed(2)
    second = _draw()

    assert first != second


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_global_seed(-1)


def test_gpu_nondeterminism_is_documented() -> None:
    assert len(GPU_NONDETERMINISM_NOTES) >= 5
    joined = " ".join(GPU_NONDETERMINISM_NOTES).lower()
    for topic in ("cudnn", "atomic", "fft", "mixed precision", "driver"):
        assert topic in joined


def test_collect_environment_records_versions_and_platform() -> None:
    env = collect_environment(packages=("numpy", "torch", "definitely-not-installed"))

    assert isinstance(env, EnvironmentInfo)
    assert env.python_version.count(".") == 2
    assert env.packages["numpy"] is not None
    assert env.packages["torch"] is not None
    assert env.packages["definitely-not-installed"] is None, "absent packages record None"
    assert env.torch_version == torch.__version__
    assert isinstance(env.cuda_available, bool)
    assert json.dumps(env.to_dict())  # must survive JSON serialization


def test_collect_environment_tolerates_missing_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("oisst_fno.experiment._git_output", lambda *args: None)

    env = collect_environment(packages=("numpy",))

    assert env.git_commit is None
    assert env.git_dirty is None


def make_config(**overrides: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "name": "fno-baseline",
        "seed": 42,
        "deterministic": False,
        "data_path": "data/raw/oisst.nc",
        "train_end": "2024-12-31",
        "validation_end": "2025-12-31",
        "region": {"lat_min": 30.125, "lat_max": 50.125, "lon_min": 330.125, "lon_max": 355.125},
        "lookback_days": 14,
        "horizon_days": 7,
        "scaler": {"mean": 17.5, "std": 3.2},
        "model": {"width": 48, "modes_y": 16, "modes_x": 16, "depth": 4, "padding": 8},
        "optimizer": "AdamW",
        "learning_rate": 2e-3,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "epochs_requested": 50,
        "early_stopping": {"monitor": "val_loss", "patience": 8, "mode": "min"},
        "scheduler": "CosineAnnealingLR",
        "mixed_precision": False,
        "gradient_clip_norm": 1.0,
    }
    base.update(overrides)
    return ExperimentConfig(**base)  # type: ignore[arg-type]


def test_config_records_every_field_the_prompt_requires() -> None:
    payload = make_config().to_dict()

    for key in (
        "train_end",
        "validation_end",
        "region",
        "lookback_days",
        "horizon_days",
        "scaler",
        "model",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs_requested",
        "early_stopping",
        "seed",
    ):
        assert key in payload, f"{key} must be recorded"

    for key in ("width", "modes_y", "modes_x", "depth", "padding"):
        assert key in payload["model"]


def test_config_roundtrips_and_can_carry_the_environment(tmp_path: Path) -> None:
    config = make_config()
    env = collect_environment(packages=("numpy",))
    path = config.save(tmp_path / "experiment.json", environment=env)

    assert ExperimentConfig.load(path) == config

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["environment"]["packages"]["numpy"] is not None


def test_config_saves_without_an_environment(tmp_path: Path) -> None:
    path = make_config().save(tmp_path / "experiment.json")

    assert "environment" not in json.loads(path.read_text(encoding="utf-8"))


def _history(train: list[float], val: list[float]) -> TrainingHistory:
    history = TrainingHistory()
    for index, (t, v) in enumerate(zip(train, val, strict=True)):
        history.append(
            EpochRecord(
                epoch=index + 1,
                train_loss=t,
                val_loss=v,
                learning_rate=1e-3,
                seconds=2.0,
                grad_norm=0.5,
                peak_gpu_mb=100.0 + index,
            )
        )
    return history


def test_history_tracks_best_time_and_memory() -> None:
    history = _history([1.0, 0.6, 0.4], [1.1, 0.5, 0.7])

    assert len(history) == 3
    assert history.best.epoch == 2
    assert history.best.val_loss == 0.5
    assert history.total_seconds == 6.0
    assert history.peak_gpu_mb == 102.0


def test_history_reports_no_gpu_memory_on_cpu_runs() -> None:
    history = TrainingHistory()
    history.append(
        EpochRecord(epoch=1, train_loss=1.0, val_loss=1.0, learning_rate=1e-3, seconds=1.0)
    )

    assert history.peak_gpu_mb is None


def test_history_roundtrips_through_json(tmp_path: Path) -> None:
    history = _history([1.0, 0.6], [1.1, 0.5])
    path = history.save(tmp_path / "history.json")

    restored = TrainingHistory.load(path)

    assert restored.records == history.records
    assert json.loads(path.read_text(encoding="utf-8"))[0]["epoch"] == 1


def test_best_of_empty_history_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        _ = TrainingHistory().best


def test_diagnosis_detects_overfitting() -> None:
    diagnosis = diagnose_learning_curves(
        _history([1.0, 0.7, 0.5, 0.3, 0.2], [1.0, 0.6, 0.5, 0.8, 1.2])
    )

    assert diagnosis.verdict == "overfit"
    assert "best epoch" in diagnosis.detail


def test_diagnosis_detects_underfitting() -> None:
    # Still improving when the run ended.
    diagnosis = diagnose_learning_curves(
        _history([1.0, 0.8, 0.6, 0.5, 0.4], [1.0, 0.8, 0.7, 0.6, 0.5])
    )

    assert diagnosis.verdict == "underfit"
    assert "still improving" in diagnosis.detail


def test_diagnosis_detects_instability() -> None:
    diagnosis = diagnose_learning_curves(
        _history([1.0, 0.8, 9.0, 0.7, 0.6], [1.0, 0.9, 8.0, 0.8, 0.7])
    )

    assert diagnosis.verdict == "unstable"
    assert "learning rate" in diagnosis.detail


def test_diagnosis_detects_non_finite_loss() -> None:
    diagnosis = diagnose_learning_curves(_history([1.0, float("nan")], [1.0, 0.9]))

    assert diagnosis.verdict == "unstable"
    assert "non-finite" in diagnosis.detail


def test_diagnosis_accepts_a_converged_run() -> None:
    diagnosis = diagnose_learning_curves(
        _history(
            [1.0, 0.7, 0.5, 0.4, 0.35, 0.33, 0.32, 0.31, 0.30, 0.29],
            [1.0, 0.7, 0.5, 0.42, 0.40, 0.40, 0.41, 0.41, 0.41, 0.41],
        )
    )

    assert diagnosis.verdict == "converged"
    assert isinstance(diagnosis, CurveDiagnosis)


def test_diagnosis_needs_at_least_two_epochs() -> None:
    with pytest.raises(ValueError, match="two epochs"):
        diagnose_learning_curves(_history([1.0], [1.0]))


def test_gradient_norm_matches_manual_computation() -> None:
    a = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    b = torch.nn.Parameter(torch.tensor([12.0]))
    a.grad = torch.tensor([3.0, 4.0])
    b.grad = torch.tensor([12.0])

    # sqrt(3^2 + 4^2 + 12^2) = 13
    assert gradient_norm([a, b]) == pytest.approx(13.0)


def test_gradient_norm_ignores_parameters_without_gradients() -> None:
    a = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    b = torch.nn.Parameter(torch.tensor([12.0]))
    a.grad = torch.tensor([3.0, 4.0])

    assert gradient_norm([a, b]) == pytest.approx(5.0)
    assert gradient_norm([b]) == 0.0


def test_amp_is_restricted_to_cuda() -> None:
    assert amp_is_supported(torch.device("cpu")) is False
    assert amp_is_supported(torch.device("cuda")) is torch.cuda.is_available()
