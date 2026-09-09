# Changelog

## 0.4.0 - 2026-09-08

### Added

- A typed synchronous optimizer contract and seven explicit strategies: the existing `weave`, uniform `random`,
  simulated annealing (`sa`), particle swarm (`pso`), differential evolution (`de`), NSGA-II (`nsga2`), and MOEA/D
  (`moead`). Each has independent proposal and survival logic over the common mixed-variable decoder.
- `optimize_strategy` and `create_optimizer` Python APIs, with exact budgets, decoded-point uniqueness, stable
  feasibility-first ties, seeded worker-count-independent ordering, and finite-space exhaustion.
- `biasweave run --strategy` and `--population-size`, plus a same-budget `catalog-benchmark` command and explicit
  algorithm/benchmark methodology documentation.
- Independent oracle and property tests for non-dominated sorting, crowding distance, boundary projection, hard
  constraints, budgets, mixed types, finite spaces, and deterministic parallel scheduling.

### Changed

- Fresh non-weave catalog runs write `catalog-run.json` beside the common ledger and result artifacts. Exact
  checkpoint/resume remains scoped to `weave` until population state receives a versioned recovery schema.
- Harden ask/tell against forged assessments and evaluator mutation with immutable snapshots, canonical point replay,
  metric validation, and complete constraint/objective recomputation before state or archive updates.
- Remember told keys across direct API calls and add a resource-bounded exact fallback for small discrete and quantized
  domains. Stochastic proposal failure is now `proposal_stalled`; only proven coverage is `search_space_exhausted`.
- Implement physically reflected, damped PSO boundary motion, including multiple crossings in one update.
- Include a pinned-tool SPDX 2.3 SBOM whose target package and complete file inventory are bound to the installed
  wheel's `RECORD`, plus verified SHA-256 checksums and GitHub build provenance attestations with each release.
- Unify native and catalog weave finite-domain behavior: small decoded products are provably exhausted, while a
  bounded miss in a continuous or oversized domain is `proposal_stalled`; checkpoint replay includes the finite cursor.
- Preserve integers beyond IEEE-754's exact range with rational decoding and all-strategy coverage; explicit encoding
  now rejects a wide-interval integer when its normalized float coordinate would decode to a neighbor. Stabilize finite
  real defaults, interpolation, normalization, and finite-domain enumeration without overflowing intermediate spans;
  ULP-scale logarithmic intervals use `log1p`/`expm1` rather than subtracting equal rounded logs. Reject variable names
  that collide after trimming. The durable JSON envelope now includes the sign on every accepted 128-digit integer,
  and seeds use the same explicit 128-digit resource bound.
- Correct NSGA-II crowding so constant objectives contribute nothing and extreme finite scales cannot overflow.
- Bound evaluator-error formatting and drain command stdout/stderr concurrently, with timeout, overflow, and child-reap
  tests that remain stable on loaded Windows coverage runners.
- Coordinate every checkpoint transition with a run-level single-writer claim. Multi-file rollback now restores only
  transaction-owned identities, preserves concurrent replacements, and distinguishes pre-commit `BaseException`
  rollback from post-commit cleanup interruption without changing the original exception. Fresh weave runs atomically
  initialize all four artifacts as one `in_progress` generation. Restored trials must reproduce their complete
  canonical success or failure representation.
- Remove repeated full seen-set copying; add a content-identified deduplication scaling harness, packaged run schemas,
  complete effective-hyperparameter provenance, and seven-strategy seeded golden trajectories.
- Define seeded reproducibility at its actual numeric boundary: repeat and worker-count invariance remain bit-exact in
  one Python/OS/libm environment, while the cross-platform semantic golden fixture permits at most four binary64 ULPs
  for continuous coordinates and decoded values and keeps discrete values, types, shape, and order exact. Point keys
  continue to bind exact, unrounded decoded values, so cross-environment checkpoint replay fails closed when they differ.
- Move DCO checking to a trusted-base `pull_request_target` verifier bound before and after verification to the PR base,
  immutable head, commit count, and API commit list; base edits reset the head status to pending and final sign-offs must
  be valid trailers. Release gates authenticate the signed tag before source execution, require CI from the exact main
  push, audit every archive member, validate the SBOM document header and full SHA-1/SHA-256 file graph, inventory the
  exact four assets, and rerun the complete offline verifier after every download.
- Make extreme finite archive-cell and public quality arithmetic fail closed with clear diagnostics; finished run
  artifacts retain the existing `quality: null` fallback when a diagnostic is not representable.

## 0.3.0 - 2026-09-07

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
