from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oisst_fno.provenance import (
    DownloadManifest,
    ProvenanceError,
    manifest_path_for,
    sha256_file,
)


def _manifest_for(path: Path, **overrides: object) -> DownloadManifest:
    manifest = DownloadManifest.build(
        data_path=path,
        source_url="https://example.invalid/erddap/griddap/dataset.nc?sst[...]",
        dataset_id="ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon",
        dataset_doi="10.25921/RE9P-PT57",
        product_version="Version v02r01",
        start_date="2024-01-01",
        end_date="2024-01-03",
        variables=("sst",),
        lat_min=30.125,
        lat_max=50.125,
        lon_min=330.125,
        lon_max=355.125,
    )
    if overrides:
        return DownloadManifest(**{**manifest.to_dict(), **overrides})  # type: ignore[arg-type]
    return manifest


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    payload = b"CDF\x01" + b"oisst" * 1000
    target.write_bytes(payload)

    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_manifest_path_sits_beside_the_data_file() -> None:
    assert manifest_path_for(Path("data/raw/oisst.nc")) == Path("data/raw/oisst.nc.manifest.json")


def test_manifest_records_request_and_content(tmp_path: Path) -> None:
    target = tmp_path / "oisst.nc"
    target.write_bytes(b"CDF\x01payload")

    manifest = _manifest_for(target)

    assert manifest.file_name == "oisst.nc"
    assert manifest.file_bytes == len(b"CDF\x01payload")
    assert manifest.sha256 == sha256_file(target)
    assert manifest.dataset_doi == "10.25921/RE9P-PT57"
    assert manifest.variables == ("sst",)
    assert manifest.downloaded_at.startswith("20")


def test_manifest_roundtrips_through_json(tmp_path: Path) -> None:
    target = tmp_path / "oisst.nc"
    target.write_bytes(b"CDF\x01payload")
    manifest = _manifest_for(target)

    path = manifest.write(manifest_path_for(target))
    restored = DownloadManifest.read(path)

    assert restored == manifest
    assert json.loads(path.read_text(encoding="utf-8"))["variables"] == ["sst"]


def test_verify_accepts_an_untouched_file(tmp_path: Path) -> None:
    target = tmp_path / "oisst.nc"
    target.write_bytes(b"CDF\x01payload")

    _manifest_for(target).verify(target)


def test_verify_rejects_a_truncated_file(tmp_path: Path) -> None:
    target = tmp_path / "oisst.nc"
    target.write_bytes(b"CDF\x01payload")
    manifest = _manifest_for(target)

    target.write_bytes(b"CDF\x01pay")

    with pytest.raises(ProvenanceError, match="truncated or was replaced"):
        manifest.verify(target)


def test_verify_rejects_modified_content_of_the_same_size(tmp_path: Path) -> None:
    target = tmp_path / "oisst.nc"
    target.write_bytes(b"CDF\x01payload")
    manifest = _manifest_for(target)

    target.write_bytes(b"CDF\x01paylOAD")

    with pytest.raises(ProvenanceError, match="contents changed"):
        manifest.verify(target)


def test_verify_rejects_a_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "oisst.nc"
    target.write_bytes(b"CDF\x01payload")
    manifest = _manifest_for(target)

    target.unlink()

    with pytest.raises(ProvenanceError, match="missing"):
        manifest.verify(target)


def test_manifest_rejects_implausible_fields(tmp_path: Path) -> None:
    target = tmp_path / "oisst.nc"
    target.write_bytes(b"CDF\x01payload")

    with pytest.raises(ValueError, match="file_bytes"):
        _manifest_for(target, file_bytes=0)
    with pytest.raises(ValueError, match="sha256"):
        _manifest_for(target, sha256="tooshort")
    with pytest.raises(ValueError, match="variables"):
        _manifest_for(target, variables=())
