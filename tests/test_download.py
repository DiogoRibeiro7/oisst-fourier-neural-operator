"""Download-path tests. Every network call is mocked; nothing here touches the internet."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import pytest
import requests

from oisst_fno import data as data_module
from oisst_fno.data import (
    DownloadError,
    Region,
    download_subset,
    smoke_test_end_date,
)
from oisst_fno.provenance import DownloadManifest, manifest_path_for

NETCDF_PAYLOAD = b"CDF\x01" + b"oisst-bytes" * 64


class FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(
        self,
        *,
        body: bytes = NETCDF_PAYLOAD,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: int = 3,
        raise_midstream: Exception | None = None,
    ) -> None:
        self.body = body
        self.status_code = status_code
        self.headers = headers if headers is not None else {"Content-Length": str(len(body))}
        self._chunks = chunks
        self._raise_midstream = raise_midstream

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self  # type: ignore[assignment]
            raise error

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        size = max(1, len(self.body) // self._chunks)
        for start in range(0, len(self.body), size):
            if self._raise_midstream is not None and start > 0:
                raise self._raise_midstream
            yield self.body[start : start + size]


class RecordingGet:
    """Callable that replaces ``requests.get`` and returns queued responses."""

    def __init__(self, *responses: FakeResponse | Exception) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.urls: list[str] = []

    def __call__(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls += 1
        self.urls.append(url)
        item = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep retry backoff from actually pausing the test suite."""
    monkeypatch.setattr(data_module.time, "sleep", lambda seconds: None)


def test_successful_download_writes_file_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    get = RecordingGet(FakeResponse())
    monkeypatch.setattr(data_module.requests, "get", get)
    target = tmp_path / "oisst.nc"

    result = download_subset(target, "2024-01-01", "2024-01-05", Region())

    assert result == target
    assert target.read_bytes() == NETCDF_PAYLOAD
    assert get.calls == 1

    manifest = DownloadManifest.read(manifest_path_for(target))
    manifest.verify(target)
    assert manifest.start_date == "2024-01-01"
    assert manifest.end_date == "2024-01-05"
    assert manifest.dataset_doi == "10.25921/RE9P-PT57"
    assert manifest.file_bytes == len(NETCDF_PAYLOAD)
    assert manifest.smoke_test is False


def test_manifest_can_be_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module.requests, "get", RecordingGet(FakeResponse()))
    target = tmp_path / "oisst.nc"

    download_subset(target, "2024-01-01", "2024-01-05", Region(), write_manifest=False)

    assert not manifest_path_for(target).exists()


def test_existing_file_is_not_redownloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    get = RecordingGet(FakeResponse())
    monkeypatch.setattr(data_module.requests, "get", get)
    target = tmp_path / "oisst.nc"
    target.write_bytes(b"CDF\x01existing")

    download_subset(target, "2024-01-01", "2024-01-05", Region())

    assert get.calls == 0
    assert target.read_bytes() == b"CDF\x01existing"


def test_transient_failures_are_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    get = RecordingGet(
        requests.ConnectionError("reset by peer"),
        FakeResponse(status_code=503),
        FakeResponse(),
    )
    monkeypatch.setattr(data_module.requests, "get", get)
    target = tmp_path / "oisst.nc"

    download_subset(target, "2024-01-01", "2024-01-05", Region())

    assert get.calls == 3
    assert target.read_bytes() == NETCDF_PAYLOAD


def test_retries_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    get = RecordingGet(requests.Timeout("too slow"))
    monkeypatch.setattr(data_module.requests, "get", get)
    target = tmp_path / "oisst.nc"

    with pytest.raises(DownloadError, match="failed after 3 attempt"):
        download_subset(target, "2024-01-01", "2024-01-05", Region(), max_attempts=3)

    assert get.calls == 3
    assert not target.exists()


def test_client_errors_fail_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    get = RecordingGet(FakeResponse(status_code=400))
    monkeypatch.setattr(data_module.requests, "get", get)
    target = tmp_path / "oisst.nc"

    with pytest.raises(DownloadError, match="will not"):
        download_subset(target, "2024-01-01", "2024-01-05", Region())

    assert get.calls == 1, "a 400 cannot be fixed by retrying"


def test_truncated_body_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    short = FakeResponse(body=NETCDF_PAYLOAD[:50], headers={"Content-Length": "999999"})
    monkeypatch.setattr(data_module.requests, "get", RecordingGet(short))
    target = tmp_path / "oisst.nc"

    with pytest.raises(DownloadError, match="failed after"):
        download_subset(target, "2024-01-01", "2024-01-05", Region(), max_attempts=1)

    assert not target.exists()
    assert not target.with_suffix(".nc.part").exists()


def test_html_error_page_is_not_saved_as_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = b"<html><body>Error: your query produced no matching results.</body></html>"
    monkeypatch.setattr(
        data_module.requests,
        "get",
        RecordingGet(FakeResponse(body=html, headers={"Content-Length": str(len(html))})),
    )
    target = tmp_path / "oisst.nc"

    with pytest.raises(DownloadError, match="failed after"):
        download_subset(target, "2024-01-01", "2024-01-05", Region(), max_attempts=1)

    assert not target.exists()


def test_empty_body_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        data_module.requests,
        "get",
        RecordingGet(FakeResponse(body=b"", headers={})),
    )
    target = tmp_path / "oisst.nc"

    with pytest.raises(DownloadError):
        download_subset(target, "2024-01-01", "2024-01-05", Region(), max_attempts=1)

    assert not target.exists()


def test_interrupted_stream_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interrupted = FakeResponse(raise_midstream=requests.ConnectionError("dropped"))
    monkeypatch.setattr(data_module.requests, "get", RecordingGet(interrupted))
    target = tmp_path / "oisst.nc"

    with pytest.raises(DownloadError):
        download_subset(target, "2024-01-01", "2024-01-05", Region(), max_attempts=2)

    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_smoke_test_mode_shortens_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    get = RecordingGet(FakeResponse())
    monkeypatch.setattr(data_module.requests, "get", get)
    target = tmp_path / "oisst_smoke.nc"

    download_subset(target, "2024-01-01", "2024-12-31", Region(), smoke_test=True)

    assert "2024-01-03" in get.urls[0]
    assert "2024-12-31" not in get.urls[0]

    manifest = DownloadManifest.read(manifest_path_for(target))
    assert manifest.smoke_test is True
    assert manifest.end_date == "2024-01-03"


def test_smoke_test_end_date_is_inclusive() -> None:
    assert smoke_test_end_date("2024-01-01", days=1) == "2024-01-01"
    assert smoke_test_end_date("2024-01-01", days=3) == "2024-01-03"
    assert smoke_test_end_date("2024-02-27", days=3) == "2024-02-29"

    with pytest.raises(ValueError, match="days must be positive"):
        smoke_test_end_date("2024-01-01", days=0)


def test_invalid_attempt_count_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        download_subset(tmp_path / "x.nc", "2024-01-01", "2024-01-05", Region(), max_attempts=0)
