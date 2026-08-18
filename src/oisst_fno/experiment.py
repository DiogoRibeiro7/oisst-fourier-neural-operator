"""Reproducibility helpers shared by the model-training notebooks.

Notebooks ``05`` (local CNN baseline), ``07`` (FNO), and ``10`` (ablations) all train
models, so seeding, environment capture, configuration records, epoch history, and
learning-curve diagnosis live here rather than being copied three times. The training
loops themselves stay in the notebooks, where the scientific narrative belongs.

Nothing here touches the network or fits any preprocessing.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "xarray",
    "netCDF4",
)

# Remaining sources of run-to-run variation once seeds are fixed. Stated explicitly
# because "seeded" is often mistaken for "bitwise reproducible".
GPU_NONDETERMINISM_NOTES: tuple[str, ...] = (
    "cuDNN may select different algorithms per run unless deterministic mode is on; "
    "set deterministic=True to pin it, at some cost in speed.",
    "Atomic float accumulation on GPU reorders additions, so reductions and scatter "
    "operations can differ in the last bits even when seeded.",
    "cuFFT plan selection can vary with available memory, which matters here because "
    "the spectral layers are FFT-based.",
    "Mixed precision changes accumulation order and rounding, so AMP runs are not "
    "bitwise comparable with full-precision runs.",
    "Different GPU models, driver versions, or CUDA/cuDNN versions can change results "
    "regardless of seeding.",
    "DataLoader worker count changes batch composition unless the workers are seeded identically.",
)


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch, optionally forcing deterministic kernels.

    ``deterministic=True`` additionally pins cuDNN algorithm selection and asks PyTorch
    for deterministic implementations. That makes repeated GPU runs far more
    comparable, but it is slower and raises if an operation used has no deterministic
    kernel — which is why it is opt-in rather than the default.

    ``PYTHONHASHSEED`` cannot be changed after interpreter start; set it in the
    environment if hash-order reproducibility matters.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_output(*args: str) -> str | None:
    """Run a read-only git command, returning None when git or the repo is absent."""
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Software environment a run happened in.

    Recorded so that a result which cannot be reproduced later can at least be
    attributed to a specific set of versions.
    """

    recorded_at: str
    python_version: str
    platform: str
    packages: dict[str, str | None]
    git_commit: str | None
    git_dirty: bool | None
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    device_name: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def collect_environment(
    packages: tuple[str, ...] = DEFAULT_TRACKED_PACKAGES,
) -> EnvironmentInfo:
    """Capture interpreter, package, git, and accelerator details for the current run.

    Git information is best-effort: a source tree exported without ``.git``, or a
    machine without git installed, records ``None`` rather than failing the run.
    """
    commit = _git_output("rev-parse", "HEAD")
    dirty: bool | None = None
    if commit is not None:
        status = _git_output("status", "--porcelain")
        dirty = bool(status)

    cuda_available = torch.cuda.is_available()
    return EnvironmentInfo(
        recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        python_version=platform.python_version(),
        platform=platform.platform(),
        packages={name: _package_version(name) for name in packages},
        git_commit=commit,
        git_dirty=dirty,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda if cuda_available else None,
        device_name=torch.cuda.get_device_name(0) if cuda_available else None,
    )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Everything needed to reconstruct a training run.

    Saved alongside the checkpoint so a second run can be set up from the artifact
    rather than from memory of what the notebook contained at the time.
    """

    name: str
    seed: int
    deterministic: bool

    data_path: str
    train_end: str
    validation_end: str
    region: dict[str, float]
    lookback_days: int
    horizon_days: int
    scaler: dict[str, float]

    model: dict[str, int]

    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs_requested: int
    early_stopping: dict[str, Any]

    scheduler: str | None = None
    mixed_precision: bool = False
    gradient_clip_norm: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def save(self, path: Path, *, environment: EnvironmentInfo | None = None) -> Path:
        """Write the configuration, and optionally the environment, as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"config": self.to_dict()}
        if environment is not None:
            payload["environment"] = environment.to_dict()
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        """Read a configuration written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload["config"])


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """One epoch of training telemetry."""

    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    seconds: float
    grad_norm: float | None = None
    peak_gpu_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass
class TrainingHistory:
    """Epoch-level history with the bookkeeping every training loop repeats."""

    records: list[EpochRecord] = field(default_factory=list)

    def append(self, record: EpochRecord) -> None:
        """Add one epoch of telemetry."""
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def train_losses(self) -> list[float]:
        """Training loss per epoch, in order."""
        return [record.train_loss for record in self.records]

    @property
    def val_losses(self) -> list[float]:
        """Validation loss per epoch, in order."""
        return [record.val_loss for record in self.records]

    @property
    def best(self) -> EpochRecord:
        """The epoch with the lowest validation loss."""
        if not self.records:
            raise ValueError("History is empty.")
        return min(self.records, key=lambda record: record.val_loss)

    @property
    def total_seconds(self) -> float:
        """Wall-clock time across all recorded epochs."""
        return float(sum(record.seconds for record in self.records))

    @property
    def peak_gpu_mb(self) -> float | None:
        """Largest peak GPU memory seen, or None when running on CPU."""
        seen = [r.peak_gpu_mb for r in self.records if r.peak_gpu_mb is not None]
        return max(seen) if seen else None

    def save(self, path: Path) -> Path:
        """Write the history as a JSON list of epoch records."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [record.to_dict() for record in self.records]
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> TrainingHistory:
        """Read a history written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(records=[EpochRecord(**row) for row in payload])


@dataclass(frozen=True, slots=True)
class CurveDiagnosis:
    """Verdict on a pair of learning curves."""

    verdict: str
    detail: str

    def __str__(self) -> str:
        return f"{self.verdict}: {self.detail}"


def diagnose_learning_curves(
    history: TrainingHistory,
    *,
    overfit_ratio: float = 1.05,
    underfit_tail_fraction: float = 0.2,
    instability_ratio: float = 1.5,
) -> CurveDiagnosis:
    """Classify learning curves as unstable, overfitting, underfitting, or converged.

    The heuristics are deliberately blunt and are meant to prompt a human look at the
    curves, not to replace it:

    - **unstable** — a non-finite loss, or an epoch-to-epoch increase in training loss
      larger than ``instability_ratio``;
    - **overfit** — validation loss rose above ``overfit_ratio`` times its minimum after
      that minimum, while training loss kept falling;
    - **underfit** — the best validation epoch falls in the last
      ``underfit_tail_fraction`` of the run, so training stopped while still improving;
    - **converged** — none of the above.
    """
    if len(history) < 2:
        raise ValueError("At least two epochs are required to diagnose curves.")

    train = history.train_losses
    val = history.val_losses

    if not all(np.isfinite(train)) or not all(np.isfinite(val)):
        return CurveDiagnosis("unstable", "training produced a non-finite loss")

    for index in range(1, len(train)):
        previous = train[index - 1]
        if previous > 0 and train[index] > instability_ratio * previous:
            return CurveDiagnosis(
                "unstable",
                f"training loss jumped from {previous:.4g} to {train[index]:.4g} at "
                f"epoch {index + 1}; reduce the learning rate or clip gradients harder",
            )

    best_index = int(np.argmin(val))
    best_val = val[best_index]

    after_best = val[best_index + 1 :]
    if after_best and max(after_best) > overfit_ratio * best_val:
        worst = max(after_best)
        train_still_falling = train[-1] < train[best_index]
        qualifier = (
            " while training loss kept falling"
            if train_still_falling
            else " although training loss also stopped improving"
        )
        return CurveDiagnosis(
            "overfit",
            f"validation loss rose from {best_val:.4g} at epoch {best_index + 1} to "
            f"{worst:.4g}{qualifier}; the saved checkpoint is the best epoch, not the last",
        )

    tail_start = len(val) - max(1, int(round(underfit_tail_fraction * len(val))))
    if best_index >= tail_start:
        return CurveDiagnosis(
            "underfit",
            f"validation loss was still at its minimum ({best_val:.4g}) at epoch "
            f"{best_index + 1} of {len(val)}; the run stopped while still improving, so "
            "train longer or increase capacity before concluding anything",
        )

    return CurveDiagnosis(
        "converged",
        f"validation loss bottomed at {best_val:.4g} on epoch {best_index + 1} of "
        f"{len(val)} and did not deteriorate materially afterwards",
    )


def gradient_norm(parameters: Any) -> float:
    """Return the total L2 norm of the gradients currently attached to ``parameters``.

    Reported per epoch so that a vanishing or exploding gradient is visible in the
    saved history rather than only in a lost console log.
    """
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().norm(2).item() ** 2)
    return float(total**0.5)


@dataclass(frozen=True, slots=True)
class BudgetMatch:
    """Result of matching one architecture's parameter count to another's."""

    width: int
    parameters: int
    target: int

    @property
    def relative_error(self) -> float:
        """Signed fractional difference from the target parameter count."""
        return (self.parameters - self.target) / self.target


def width_for_parameter_budget(
    factory: Callable[[int], torch.nn.Module],
    target_parameters: int,
    *,
    min_width: int = 2,
    max_width: int = 512,
) -> BudgetMatch:
    """Find the width whose parameter count sits closest to ``target_parameters``.

    Comparing two architectures is only informative if capacity is held roughly fixed;
    otherwise "the elaborate model won" and "the bigger model won" are indistinguishable.
    Parameter count rises monotonically with width, so a scan over the range is enough and
    is more robust than solving the scaling relation analytically.

    The achieved count will rarely be exact. ``BudgetMatch.relative_error`` reports the
    residual mismatch so it can be stated rather than glossed over.
    """
    if target_parameters < 1:
        raise ValueError("target_parameters must be positive.")
    if min_width < 1 or max_width < min_width:
        raise ValueError("Invalid width range.")

    best: BudgetMatch | None = None
    for width in range(min_width, max_width + 1):
        count = sum(p.numel() for p in factory(width).parameters() if p.requires_grad)
        candidate = BudgetMatch(width=width, parameters=count, target=target_parameters)
        if best is None or abs(candidate.relative_error) < abs(best.relative_error):
            best = candidate
        if count > target_parameters:
            break

    assert best is not None
    return best


@dataclass(frozen=True, slots=True)
class CostMeasurement:
    """Wall-clock and memory cost of a forward or forward-backward pass."""

    seconds_per_pass: float
    peak_gpu_mb: float | None
    parameters: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def measure_cost(
    model: torch.nn.Module,
    sample: torch.Tensor,
    *,
    repeats: int = 5,
    warmup: int = 1,
    backward: bool = False,
) -> CostMeasurement:
    """Time a model on one sample and record peak GPU memory.

    A structurally richer operator can win on error and still be the wrong choice if it
    costs several times more per forecast, so cost is measured rather than assumed.
    Warm-up passes are excluded because the first call pays lazy-initialisation and, on
    CUDA, kernel autotuning.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive.")

    device = sample.device
    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    def one_pass() -> None:
        if backward:
            model.zero_grad(set_to_none=True)
            model(sample).square().mean().backward()
        else:
            with torch.no_grad():
                model(sample)

    for _ in range(warmup):
        one_pass()
    if on_cuda:
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(repeats):
        one_pass()
    if on_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return CostMeasurement(
        seconds_per_pass=elapsed / repeats,
        peak_gpu_mb=(torch.cuda.max_memory_allocated() / 1024**2) if on_cuda else None,
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


def amp_is_supported(device: torch.device) -> bool:
    """True when mixed precision is worth enabling on this device.

    Restricted to CUDA. CPU autocast exists but changes numerics for no practical
    speedup here, and the study's correctness matters more than a marginal gain.
    """
    return device.type == "cuda" and torch.cuda.is_available()
