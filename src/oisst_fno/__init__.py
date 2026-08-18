"""Utilities for the notebook-first NOAA OISST FNO study."""

from .data import (
    DownloadError,
    ForecastSpec,
    Region,
    SSTSequenceDataset,
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
from .model3d import FNO3d, TruncatedFourierMix3d
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
    "FNO3d",
    "ForecastSpec",
    "MultivariateWindowDataset",
    "PerVariableStandardizer",
    "ProvenanceError",
    "Region",
    "SSTSequenceDataset",
    "SSTWindowDataset",
    "Standardizer",
    "TrainingHistory",
    "TruncatedFourierMix2d",
    "TruncatedFourierMix3d",
    "VariableSpec",
    "ValidationReport",
    "collect_environment",
    "diagnose_learning_curves",
    "set_global_seed",
    "standard_ablations",
]
