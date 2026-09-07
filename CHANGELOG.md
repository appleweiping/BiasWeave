# Changelog

## Unreleased

### Added

- `biasweave quality`: hypervolume, spacing, and extent for the feasible front a ledger records,
  with `--curve` reporting hypervolume against evaluations spent and `--compare` measuring a second
  ledger of the same problem. The hypervolume is exact in every dimension, checked against
  inclusive-exclusive summation and independent Monte Carlo integration, and is refused rather than
  estimated above a front size that halves per objective past five, because a sampled volume carries
  more relative error than the difference between two runs worth comparing.
- `--curve` also reports the share of the final hypervolume reached at each cut and the evaluation
  by which the run came within one percent of it. A run can keep improving to its final evaluation
  while the improvement is immaterial, and a column of raw volumes hides that; on the bundled
  example at 240 evaluations, two thirds of the budget bought a tenth of a percent.
- Coverage and the additive epsilon indicator are reported beside the volume in a comparison,
  because hypervolume rewards points near the knee over points at the extremes and so ranks fronts
  that no dominance relation orders.
- `biasweave.quality` as a Python API: `measure_run`, `front_quality`, `attainment_curve`,
  `compare_fronts`, `hypervolume`, `coverage`, `epsilon_indicator`, `spacing`, and
  `derive_reference_point`.

### Changed

- `frontier.json` is schema version 2, adding a `quality` block, and `summary.md` reports the same
  numbers. Both derive the reference point from every feasible trial of the run, which is what
  `biasweave quality` derives for the same ledger, so a run summary and a later measurement agree.
  A front the indicator cannot measure is recorded as absent rather than raised, since every
  evaluation is already spent by the time a run is written out and a diagnostic must not discard it.

## 0.2.0 - 2026-08-31

- Add strict TopologyLantern analog benchmark ingestion and analytic evaluation.
- Add a same-budget deterministic uniform-random-search comparison.
- Add bounded strict JSON ingestion, scaling contracts, and synchronized runtime version metadata.

## 0.1.0 - 2026-08-31

- Added strict mixed-variable problem definitions.
- Added constraint-first Pareto assessment and an exact frontier archive.
- Added deterministic coverage, frontier, and repair proposal strands.
- Added callable and JSON subprocess evaluators.
- Added crash-tolerant JSONL checkpoints and reproducible resume.
- Added CLI validation, optimization, resume, and frontier inspection.
