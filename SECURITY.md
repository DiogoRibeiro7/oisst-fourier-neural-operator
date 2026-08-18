# Security policy

## Scope

This is a research repository: notebooks, a small Python package, and documentation. It
runs no service, stores no credentials, and processes only publicly available NOAA data.
The realistic risk surface is therefore limited to:

- vulnerabilities in pinned dependencies (`torch`, `xarray`, `netcdf4`, `requests`, and
  their transitive dependencies);
- unsafe deserialization when loading model checkpoints (`torch.load`) that did not come
  from you;
- the ERDDAP download path in `src/oisst_fno/data.py`, which builds a URL and writes a
  file to disk.

## Supported versions

The `main` branch is the only supported version. Archived Zenodo releases are immutable
snapshots and are not patched; fixes land on `main` and appear in the next release.

## Reporting a vulnerability

Please report privately rather than in a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/DiogoRibeiro7/oisst-fourier-neural-operator/security/advisories/new)
on this repository.

Include the affected file or dependency, the version, what an attacker could achieve,
and reproduction steps if you have them. Expect an initial response within 14 days; this
is a personal research project without a staffed on-call rotation.

## Handling model checkpoints

`torch.load` executes arbitrary code when unpickling. Load checkpoints only from
`artifacts/models/` that you produced yourself, or pass `weights_only=True`. Do not load
checkpoints supplied by third parties.

## Scientific integrity issues

Data leakage, an invalid baseline, or a broken statistical comparison is not a security
vulnerability, but it matters just as much here. Report those as a public issue using
the "Scientific issue" template so the discussion stays open.
