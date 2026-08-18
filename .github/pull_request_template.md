## What this changes

<!-- One or two sentences. -->

## Why

<!-- The motivation. For method changes, state which research question it affects. -->

## Checks

- [ ] `make quality` passes (ruff, mypy strict, pytest)
- [ ] Notebooks committed without outputs
- [ ] No data, checkpoints, or figures added to the repository

## Scientific review

- [ ] Preprocessing is still fit on the training split only
- [ ] Splits remain temporal — no random train/test split introduced
- [ ] No baseline was weakened
- [ ] Paired comparisons still use identical forecast target dates

## Reported numbers

<!--
If this changes any metric, figure, or conclusion, give before/after values here.
If it changes none, write "no change to reported results".
-->
