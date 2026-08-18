"""Provenance manifests for downloaded NOAA OISST subsets.

A processed dataset is only trustworthy if the exact source that produced it can be
identified later. Every download therefore writes a sidecar manifest recording the
request, the response, and a content hash.

Exploratory summaries and plots belong in the notebooks; this module only holds the
reusable, testable manifest logic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_SUFFIX = ".manifest.json"
_HASH_CHUNK_BYTES = 1024 * 1024


class ProvenanceError(RuntimeError):
    """Raised when a file does not match its recorded provenance."""


def sha256_file(path: Path, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """Return the SHA-256 hex digest of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(data_path: Path) -> Path:
    """Return the manifest path that accompanies a downloaded data file."""
    data_path = Path(data_path)
    return data_path.with_name(data_path.name + MANIFEST_SUFFIX)


@dataclass(frozen=True, slots=True)
class DownloadManifest:
    """Everything needed to identify the source data behind a processed dataset.

    Attributes mirror what a reader needs in order to reproduce or audit a download:
    what was asked for, what came back, and how to detect later corruption.
    """

    source_url: str
    dataset_id: str
    dataset_doi: str
    product_version: str
    downloaded_at: str
    start_date: str
    end_date: str
    variables: tuple[str, ...]
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    file_name: str
    file_bytes: int
    sha256: str
    smoke_test: bool = False

    def __post_init__(self) -> None:
        if self.file_bytes <= 0:
            raise ValueError("file_bytes must be positive.")
        if len(self.sha256) != 64:
            raise ValueError("sha256 must be a 64-character hex digest.")
        if not self.variables:
            raise ValueError("variables must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["variables"] = list(self.variables)
        return payload

    def write(self, path: Path) -> Path:
        """Write the manifest as indented JSON and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> DownloadManifest:
        """Load a manifest written by :meth:`write`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["variables"] = tuple(payload["variables"])
        return cls(**payload)

    @classmethod
    def build(
        cls,
        *,
        data_path: Path,
        source_url: str,
        dataset_id: str,
        dataset_doi: str,
        product_version: str,
        start_date: str,
        end_date: str,
        variables: tuple[str, ...],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        smoke_test: bool = False,
    ) -> DownloadManifest:
        """Build a manifest by hashing and measuring an already-downloaded file."""
        data_path = Path(data_path)
        return cls(
            source_url=source_url,
            dataset_id=dataset_id,
            dataset_doi=dataset_doi,
            product_version=product_version,
            downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            file_name=data_path.name,
            file_bytes=data_path.stat().st_size,
            sha256=sha256_file(data_path),
            smoke_test=smoke_test,
        )

    def verify(self, data_path: Path) -> None:
        """Raise :class:`ProvenanceError` if the file no longer matches the manifest.

        Checks size before hashing so an obviously truncated file fails fast.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise ProvenanceError(f"{data_path} is missing but a manifest exists for it.")

        actual_bytes = data_path.stat().st_size
        if actual_bytes != self.file_bytes:
            raise ProvenanceError(
                f"{data_path.name} is {actual_bytes} bytes but the manifest records "
                f"{self.file_bytes}. The file is truncated or was replaced."
            )

        actual_hash = sha256_file(data_path)
        if actual_hash != self.sha256:
            raise ProvenanceError(
                f"{data_path.name} has SHA-256 {actual_hash} but the manifest records "
                f"{self.sha256}. The file contents changed."
            )
