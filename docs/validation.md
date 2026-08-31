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

Benchmark contracts are read through a 1 MiB bounded input and are limited to
64 nested container levels and 10,000 decoded values. Duplicate keys,
non-finite numeric spellings, parser recursion, and numeric overflow are
reported as normal input errors; these runtime checks are intentionally stricter
than the public JSON Schema.
