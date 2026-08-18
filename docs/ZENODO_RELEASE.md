# Zenodo archiving and citation

This repository carries machine-readable citation metadata so that an archived release
gets a DOI with correct authorship, licensing, and provenance instead of Zenodo's
auto-generated defaults.

| File | Consumed by | Purpose |
|---|---|---|
| [`.zenodo.json`](../.zenodo.json) | Zenodo | Deposition metadata for the archived release (title, creators, license, description, keywords, related identifiers). Overrides Zenodo's defaults and `CITATION.cff`. |
| [`CITATION.cff`](../CITATION.cff) | GitHub, `cffconvert`, reference managers | Powers GitHub's "Cite this repository" button and BibTeX/APA export. |

Both are kept in sync by hand; `.zenodo.json` wins on Zenodo if the two disagree.

## Before the first deposit

Author identity is recorded in both files and does not need changing:

```json
"creators": [
  {
    "name": "Ribeiro, Diogo",
    "affiliation": "ESMAD - Instituto Politécnico do Porto",
    "orcid": "0009-0001-2022-7072"
  }
]
```

`CITATION.cff` carries the same values, with the ORCID as a full `https://orcid.org/...`
URI because CFF 1.2.0 requires that form; `.zenodo.json` takes the bare identifier.

`CITATION.cff` also records `repository-code` and `url` pointing at
<https://github.com/DiogoRibeiro7/oisst-fourier-neural-operator>. `.zenodo.json` does
not need the repository URL: Zenodo's GitHub integration attaches the `isSupplementTo`
link to the release automatically.

One field remains open: **co-authors**, if any, listed in the same order in both files.

## Archiving a release

1. Sign in to [zenodo.org](https://zenodo.org) with the GitHub account that owns the
   repository.
2. Open **Account → GitHub** and toggle this repository **on**. Zenodo only archives
   releases created *after* the toggle is enabled.
3. Confirm `.zenodo.json` and `CITATION.cff` are committed and that versions match
   `pyproject.toml`.
4. Publish a GitHub release with a tag such as `v0.1.0`.
5. Zenodo mints two DOIs: a **concept DOI** that always resolves to the newest version,
   and a **version DOI** fixed to that release. Cite the concept DOI in the README badge
   and the version DOI when reproducing a specific result.

To archive without GitHub, create the deposit manually on Zenodo and upload a source
archive; `.zenodo.json` is not read on manual uploads, so the metadata has to be entered
in the web form.

## Version bump checklist

Update these in the same commit, then tag:

- `pyproject.toml` → `version`
- `.zenodo.json` → `version`
- `CITATION.cff` → `version` and `date-released`
- `CHANGELOG.md` → move `Unreleased` entries under the new version, with a **Results**
  note saying whether any reported metric, figure, or conclusion changed
- `README.md` → DOI badge, if pinned to a version DOI rather than the concept DOI

The `citation metadata` CI job fails the build if `pyproject.toml`, `.zenodo.json`, and
`CITATION.cff` disagree on the version, so a partial bump cannot reach `main`.

## What the archive does and does not contain

The deposit is a snapshot of the tagged source tree. Per [`.gitignore`](../.gitignore),
it excludes raw and processed OISST data, model checkpoints, metrics, predictions, and
figures. Reproduction requires re-downloading OISST v2.1 from NOAA NCEI ERDDAP
(dataset `ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon`) and re-running the notebooks in
order.

If a release should be reproducible without a re-download, attach the processed subset
and evaluation artifacts as a **separate** Zenodo deposit of type *dataset*, cite the
NOAA source DOI ([10.25921/RE9P-PT57](https://doi.org/10.25921/RE9P-PT57)) in it, and
link the two deposits with an `isSupplementTo` / `isSupplementedBy` relation. Do not
redistribute NOAA data without preserving its source terms and attribution
requirements.

## Metadata claims

The description in `.zenodo.json` deliberately repeats the repository's scope limits:
no novel architecture, no first application of FNOs to ocean or Atlantic SST
forecasting, OISST as an analyzed product rather than raw observations, and a negative
result as an acceptable outcome. Keep those statements in the deposit metadata — the
DOI record is often read without the README. See
[`RESEARCH_POSITIONING.md`](RESEARCH_POSITIONING.md),
[`THREATS_TO_VALIDITY.md`](THREATS_TO_VALIDITY.md), and
[`PROVENANCE.md`](PROVENANCE.md).

## Validation

```bash
python -c "import json; json.load(open('.zenodo.json'))"
pipx run cffconvert --validate -i CITATION.cff
```
