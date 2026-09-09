# Architecture

## Design constraints

BiasWeave is organized around four guarantees: strict input validation, constraint-first ordering, deterministic
proposal state, and recoverable persistence. The runtime uses only the Python standard library. Simulator integration is
kept outside the search logic through a narrow evaluator protocol.

## Data path

1. `problem.py` parses strict TOML into immutable objects from `model.py`.
2. `encoding.py` maps a normalized hypercube to mixed variable values and resolves linked values.
3. `proposal.py` implements the resumable weave policy; `optimizers/` implements the common ask/tell contract and
   independent catalog algorithms.
4. `evaluator.py` adapts a Python callable or a no-shell JSON subprocess and validates finite required metrics. Command
   stdout and stderr are drained concurrently under independent byte limits; timeout or overflow kills and reaps the
   child instead of buffering an unbounded pipe.
5. `dominance.py` converts metrics to normalized violations and objective vectors.
6. `archive.py` recomputes the exact feasible Pareto front and supplies sparse or low-violation anchors.
7. `engine.py` assigns trial IDs and checkpoints weave state; `strategy.py` provides the budget-exact catalog runner.
8. `ledger.py` persists trials; `results.py` emits machine- and human-readable views.
9. `cli.py` exposes validation, fresh run, resume, and frontier inspection.

The evaluator is the only boundary expected to know circuit or simulator semantics. The optimizer treats variable and
metric names as opaque identifiers and numerical values as already expressed in a consistent unit system.

## Constraint-first ordering

Each constraint produces a non-negative normalized violation. Violations are squared and summed for a total score; the
largest violation is retained as a tie-breaker. This yields the ordering:

1. successful feasible over successful infeasible;
2. successful infeasible by total violation, then largest violation;
3. successful over evaluator failure;
4. feasible points by Pareto dominance in normalized minimization coordinates.

A maximized metric changes sign when converted to the objective vector. Reference and scale affect conditioning, not
the original metric stored in the ledger. The reported front is exact: epsilon cells affect only anchor selection.

## Proposal strands

The proposal generator advances deterministic counters and a seeded pseudo-random state.

### Coverage

A family of coprime strides maps the proposal index into each normalized dimension. It supplies broad, repeatable
coverage without allocating a grid whose size grows exponentially.

### Frontier refinement

A low-occupancy epsilon cell supplies a frontier anchor. One to three coordinates are perturbed inside an adaptive
radius. The radius grows after frontier change and shrinks after a non-improving batch.

### Feasibility repair

The lowest-violation infeasible point is the anchor. Median finite-difference slopes estimated from successful history
identify a coordinate and direction likely to reduce violation. When history is insufficient, local mutation supplies
a conservative fallback.

All decoded points receive a SHA-256 key derived from canonical JSON. Integer interpolation uses exact rational
arithmetic before the final normalized coordinate, including bounds beyond IEEE-754's exact-integer range. Explicit
integer encoding validates that the float coordinate decodes to the requested integer and fails closed when a very wide
interval cannot represent that interior value. Linear real interpolation and normalization avoid forming an overflowing
`high - low`; log-scaled defaults and coordinates stay in log space, with `log1p`/`expm1` retaining ULP-scale positive
intervals whose separately rounded logarithms are equal. A proposal already present in the ledger or current
batch is skipped. Products of at most 100,000 discrete decoded values are
walked by one shared deterministic enumerator, so both native and catalog weave terminate with
`search_space_exhausted` only after a proof; a bounded miss in any other domain is `proposal_stalled`.

## Ordered parallelism

One batch is proposed from one archive snapshot. Evaluation may use several worker threads, but results are consumed in
the same order as their submitted `(trial_id, point)` pairs. Archive updates and ledger writes happen only on the calling
thread. Changing worker count cannot reorder trials; changing batch size is considered a checkpoint incompatibility
because it changes when archive feedback reaches the proposal generator.

Evaluator `Exception` instances become explicit failed trials. Process-control exceptions such as `KeyboardInterrupt`
and `SystemExit` are not swallowed. A failed trial remains part of the reproducible sequence but never enters the Pareto
front or repair history.

## Optimizer catalog contract

Catalog optimizers operate only on normalized coordinates and therefore share the exact decoder for real, integer,
choice, quantized, and linked variables. `ask` cannot run while a batch is pending. `tell` must return the complete
batch in proposal order with matching point identities and non-negative, strictly increasing trial IDs. The runner
allocates contiguous trial IDs, evaluates at most the remaining budget, and tells results only after their deterministic
ordering is restored.

The archive remains external to strategy survival state. Consequently every strategy reports the same exact feasible
Pareto definition even though SA has one chain, PSO and DE have incumbent populations, NSGA-II uses ranks and crowding,
and MOEA/D uses decomposition neighborhoods. Stable trial ID and point-key tie-breaks make otherwise equal choices
reproducible. A decoded point is never evaluated twice, including when integer or categorical coordinates collapse a
large portion of the normalized cube.

The boundary retains canonical immutable point and metric snapshots. A successful report is accepted only when its
decoded values and key reproduce from the pending coordinates and its feasibility, violations, and minimized objective
vector reproduce from validated metrics. Rejected reports leave pending and seen-key state untouched. Optimizers
remember told keys internally and merge them with caller-supplied preexisting keys.

Small finite decoded products have a resource-bounded exact enumeration fallback. An empty ask means
`search_space_exhausted` only after that enumerator covers the product; stochastic failure in a continuous or larger
domain is reported separately as `proposal_stalled`.

## Checkpoint protocol

This recovery protocol applies to the native `weave` engine. `trials.jsonl` is append-only. Each completed batch is
encoded with strict JSON (`allow_nan=False`), flushed, and fsync'd.
`run.json` is transactionally staged and atomically replaced after the ledger append. Its v2 schema is packaged in the
wheel, and it records:

- checkpoint, package, and weave-strategy schema versions plus the effective hyperparameter object;
- problem SHA-256;
- evaluator identifier;
- seed and batch size;
- completed-trial and stagnation counters;
- coverage/proposal counters, finite-domain cursor, radius, and pseudo-random state.

On resume, ledger IDs must be contiguous. The proposal generator and stagnation counter are reconstructed from the
durable trial sequence and compared with `run.json`, so even structurally valid state tampering is rejected.
Compatibility fields must equal the new request before any new evaluation. A budget is a total target in the API; the
CLI turns `--additional-budget` into that target after reading the completed count.

The order is intentionally ledger then metadata. If interruption leaves the ledger ahead, the originally proposed
batch is regenerated from trusted prior state. Its complete prefix is verified and any missing suffix is evaluated
before normal search resumes, preserving the uninterrupted proposal sequence. A truncated last ledger line is ignored
only when the file lacks its final newline; corruption in any complete line is an error.
Once a pending batch is recorded, resume must use a total budget large enough to finish that entire committed batch.
This prevents silently dropping proposals that have already advanced the deterministic generator state.

One nonce-owned claim with the exact `run` scope covers resume reads, canonical trial validation, proposal-state reconstruction, ledger
append/fsync, and the following metadata transition. A second live writer fails before reading mutable state. Claim
directories are removed only while their owner marker still has the identity and bounded nonce written by the holder;
an uncertain or abandoned marker is left fail-closed for operator inspection instead of guessed stale and removed.

## Stop conditions

The hard evaluation budget is always present. Optional wall time is checked between batches. Optional stagnation counts
evaluations since the last frontier change. A finite enumerator that has visited its complete decoded product reports
`search_space_exhausted`. A continuous, oversized, or otherwise non-enumerable domain that cannot find a new point in
its bounded retry window reports `proposal_stalled`; stochastic failure is never presented as proof.

## Output transaction boundary

All JSON, Markdown, checkpoint metadata, benchmark, quality, and catalog artifacts use the same UTF-8/LF staging
layer. Destinations are no-clobber by default; explicit `--force` can replace outputs but can never replace an input
alias, including case-folded, symlink, and existing hard-link identities. Multi-file installs stage every payload first
and acquire deterministic per-target claims before the final preflight. A fresh persisted weave run initializes its
ledger, checkpoint, frontier, and summary as one `in_progress` generation before evaluation. Before the explicit commit
point, rollback removes or restores a path only if its filesystem identity is still the one installed by that
transaction; a concurrent replacement is preserved. Cancellation through `BaseException` runs the same
ownership-checked recovery and then re-raises the original exception. After every new identity is installed, the
transaction is committed: backup-cleanup cancellation preserves the new generation, annotates and re-raises the
original exception, and leaves uncertain state fail-closed rather than claiming rollback. Ledger append
validates line, file, record, node, metric, and text limits before opening the durable file, then rechecks the opened
descriptor against every protected input identity to close a preflight/open race. A rejected record cannot damage the
old ledger.

## Extension boundaries

New variable kinds belong in the strict parser and encode/decode layer together. New proposal policies should preserve
snapshot/restore completeness and never mutate the archive. Evaluator transports must return the same metric mapping
contract and avoid implicit shell interpretation. Approximate archive strategies, if added, should remain distinct from
the exact reported front so users can tell exploration heuristics from result semantics.
