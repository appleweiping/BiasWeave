# Architecture

## Design constraints

BiasWeave is organized around four guarantees: strict input validation, constraint-first ordering, deterministic
proposal state, and recoverable persistence. The runtime uses only the Python standard library. Simulator integration is
kept outside the search logic through a narrow evaluator protocol.

## Data path

1. `problem.py` parses strict TOML into immutable objects from `model.py`.
2. `encoding.py` maps a normalized hypercube to mixed variable values and resolves linked values.
3. `proposal.py` interleaves coverage, frontier mutation, and feasibility repair.
4. `evaluator.py` adapts a Python callable or a no-shell JSON subprocess and validates finite required metrics.
5. `dominance.py` converts metrics to normalized violations and objective vectors.
6. `archive.py` recomputes the exact feasible Pareto front and supplies sparse or low-violation anchors.
7. `engine.py` assigns trial IDs, evaluates ordered batches, updates the archive, and checkpoints state.
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

All decoded points receive a SHA-256 key derived from canonical JSON. A proposal already present in the ledger or the
current batch is skipped. Finite discrete spaces therefore terminate with `search_space_exhausted` instead of repeating
work indefinitely.

## Ordered parallelism

One batch is proposed from one archive snapshot. Evaluation may use several worker threads, but results are consumed in
the same order as their submitted `(trial_id, point)` pairs. Archive updates and ledger writes happen only on the calling
thread. Changing worker count cannot reorder trials; changing batch size is considered a checkpoint incompatibility
because it changes when archive feedback reaches the proposal generator.

Evaluator `Exception` instances become explicit failed trials. Process-control exceptions such as `KeyboardInterrupt`
and `SystemExit` are not swallowed. A failed trial remains part of the reproducible sequence but never enters the Pareto
front or repair history.

## Checkpoint protocol

`trials.jsonl` is append-only. Each completed batch is encoded with strict JSON (`allow_nan=False`), flushed, and fsync'd.
`run.json` is written to a sibling temporary file and atomically replaced after the ledger append. It records:

- schema version;
- problem SHA-256;
- evaluator identifier;
- seed and batch size;
- completed-trial and stagnation counters;
- coverage/proposal counters, radius, and pseudo-random state.

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

## Stop conditions

The hard evaluation budget is always present. Optional wall time is checked between batches. Optional stagnation counts
evaluations since the last frontier change. A proposal generator that cannot find a new point within its bounded retry
window reports search-space exhaustion. The resulting stop reason is persisted in the result views, while the checkpoint
remains resumable when a larger total budget is meaningful.

## Extension boundaries

New variable kinds belong in the strict parser and encode/decode layer together. New proposal policies should preserve
snapshot/restore completeness and never mutate the archive. Evaluator transports must return the same metric mapping
contract and avoid implicit shell interpretation. Approximate archive strategies, if added, should remain distinct from
the exact reported front so users can tell exploration heuristics from result semantics.
