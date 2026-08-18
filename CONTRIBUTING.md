# Contributing

Thanks for your interest in this project. It is a small research repository, so the
contribution process is deliberately lightweight — but the scientific rules are not
negotiable, because they are what the results depend on.

## Development setup

```bash
poetry install
poetry run pre-commit install
poetry run python -m ipykernel install --user --name oisst-fno --display-name "OISST FNO"
```

## Before opening a pull request

```bash
make quality        # ruff, mypy, pytest
```

or individually:

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy src
poetry run pytest
```

CI runs the same commands on Python 3.11 and 3.12. `mypy` runs in `strict` mode; when a
third-party stub widens a type to `Any`, bind the value to an annotated local at the
boundary rather than adding a blanket `# type: ignore`.

## Scientific rules for code changes

These come from the README's guardrails and apply to every change that touches data
handling, baselines, or evaluation:

1. **Fit preprocessing on training data only.** Any normalization, EOF basis, or
   climatology must be estimated inside the training split.
2. **Never introduce a random train/test split.** Neighbouring daily SST fields are
   dependent; splits stay temporal.
3. **Do not weaken a baseline to favour the FNO.** Persistence is the primary null
   model. If a change makes the FNO look better by making a baseline worse, it will be
   rejected.
4. **Paired comparisons keep their pairing.** Models are compared on identical forecast
   target dates, with block-bootstrap uncertainty that respects serial dependence.
5. **A change that alters reported numbers must say so** in the pull-request
   description, with the before/after values.

Negative results are welcome. A pull request showing the FNO does *not* beat a baseline
is as valuable as one showing that it does.

## Scope

`src/oisst_fno/` holds only what is reused across notebooks or needs unit tests: data
acquisition and windowing, baselines, the FNO, and metrics. Scientific narrative,
plotting, and interpretation belong in the notebooks. Please do not migrate analysis
into the package to "tidy" it — that separation is intentional.

## Notebooks

Commit notebooks **without outputs**. Outputs bloat diffs and can leak local paths:

```bash
poetry run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

Keep the numeric ordering; a notebook must run correctly after the ones before it.

## Data

Never commit data, model checkpoints, figures, or predictions — `.gitignore` covers
`data/` and `artifacts/`. OISST is re-downloadable from NOAA NCEI ERDDAP through
`01_data_acquisition.ipynb`, and NOAA's terms and attribution requirements travel with
it.

## Commit messages

Write a short imperative subject line, then explain *why* in the body. Reference the
notebook or module affected.

## Reporting problems

Open an issue using one of the templates. For anything that looks like a **scientific**
error — leakage, a mis-specified baseline, an invalid statistical comparison — use the
"Scientific issue" template and say which reported result is affected. Those are treated
as higher priority than feature requests.
