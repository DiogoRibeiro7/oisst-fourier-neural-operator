"""Utilities for the notebook-first NOAA OISST FNO study."""

from .data import (
    DownloadError,
    ForecastSpec,
    Region,
    SSTWindowDataset,
    Standardizer,
)
from .experiment import (
    EpochRecord,
    ExperimentConfig,
    TrainingHistory,
    collect_environment,
    diagnose_learning_curves,
    set_global_seed,
)
from .model import FNO2d, TruncatedFourierMix2d
from .multivariate import (
    AblationSpec,
    MultivariateWindowDataset,
    PerVariableStandardizer,
    VariableSpec,
    standard_ablations,
)
from .provenance import DownloadManifest, ProvenanceError
from .validation import DataValidationError, ValidationReport

__all__ = [
    "AblationSpec",
    "DataValidationError",
    "DownloadError",
    "DownloadManifest",
    "EpochRecord",
    "ExperimentConfig",
    "FNO2d",
    "ForecastSpec",
    "MultivariateWindowDataset",
    "PerVariableStandardizer",
    "ProvenanceError",
    "Region",
    "SSTWindowDataset",
    "Standardizer",
    "TrainingHistory",
    "TruncatedFourierMix2d",
    "VariableSpec",
    "ValidationReport",
    "collect_environment",
    "diagnose_learning_curves",
    "set_global_seed",
    "standard_ablations",
]
