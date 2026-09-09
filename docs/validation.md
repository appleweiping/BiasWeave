# Benchmark validation

BiasWeave rejects missing or unknown contract fields, wrong JSON types,
duplicate candidate IDs, inconsistent counts, unsupported metrics and models,
and any canonical SHA-256 mismatch. The analytic evaluator is deterministic and
finite over bounded inputs.

The comparison gives both algorithms the same evaluation budget and seed.
Uniform random search is a well-established black-box baseline; this
implementation samples normalized dimensions independently and rejects
duplicate decoded points. Repeat-run tests require identical reports.

The optional representative is selected by equal L1 contributions after each
non-constant objective is normalized to the observed feasible Pareto-front
range. This is a deterministic reporting convention, not evidence that an
engineer's preferences have equal weights. A sizing decision binds the full
comparison digest and is exported only after the budget/seed run, frontier,
representative, metrics, and point key replay exactly.

The proxy model is intentionally technology-neutral and is not a SPICE model.
Reported values must not be interpreted as predicted silicon performance.

Optimizer-catalog validation is deliberately independent of the analog proxy. Property tests compare non-dominated
sorting with a direct dominance-peeling oracle and check hand-calculated crowding distances. Pure variation tests force
PSO and DE beyond both unit-box boundaries. End-to-end tests exercise every algorithm on real, integer, categorical,
and linked variables; assert exact budgets and unique decoded points; compare serial and four-worker seeded sequences;
and exhaust integer, categorical, and linear/log-quantized products. Adversarial tests mutate evaluator inputs, forge
assessment fields, retry rejected tells, and force stochastic stalls to verify that point identity, pending state, and
exhaustion provenance remain sound. Constraint tests include feasible designs with deliberately worse objective values
to verify that objective quality cannot buy passage across a hard-constraint boundary. PSO tests cover damped reflection
through multiple boundary crossings. Every strategy also runs against real bounds spanning `-1e308` to `1e308` and a
wide integer interval. Encoder regressions cover endpoints, implicit defaults, exact round trips, and explicit rejection
when an interior integer cannot be represented by the normalized float coordinate. Quality indicators either return a
finite value or a clear configuration error at extreme finite scales; run-result serialization keeps its established
fallback of reporting an unmeasurable quality block as `null`.

Benchmark contracts are read through a 1 MiB bounded input and are limited to
64 nested container levels and 10,000 decoded values. Duplicate keys,
non-finite numeric spellings, parser recursion, and numeric overflow are
reported as normal input errors; these runtime checks are intentionally stricter
than the public JSON Schema.

Release validation is executable policy, not only workflow shell. The release helper rejects archive traversal,
backslashes and drive paths, control or non-NFC names, normalized-casefold duplicates, file/directory prefix
collisions, links, sparse or special tar entries, encrypted or symlink-like wheel entries, and bounded-size violations.
The release contains exactly one wheel, one source distribution, `SBOM.spdx.json` produced by pinned Syft from an
isolated wheel installation, and `SHA256SUMS`. The helper validates the SPDX document header, binds the canonical
BiasWeave PyPI package and every `CONTAINS` file to the installed distribution's complete `RECORD` inventory, requires
SHA-1 and SHA-256 for every bound file, and checks the package verification code. Both downstream jobs rerun that full
offline validation over the exact four-asset inventory after artifact download; provenance attests the checksum-listed
subjects rather than an unbounded glob.
