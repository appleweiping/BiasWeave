# Optimizer catalog

BiasWeave 0.4 provides seven real search policies behind one strict synchronous ask/tell contract. They share problem
decoding, hard-constraint assessment, exact Pareto reporting, decoded-point deduplication, ordered parallel evaluation,
and seed handling. They do not share an implementation path or masquerade as aliases.

| Strategy | Search state | Proposal and survival behavior | Natural use |
|---|---|---|---|
| `weave` | exact archive plus adaptive radius | coprime-stride coverage, sparse-front mutation, violation-slope repair | default mixed-variable sizing and resumable runs |
| `random` | seeded random stream | independent uniform normalized samples | transparent baseline |
| `sa` | one current state and temperature | Gaussian neighborhood, Metropolis acceptance within the same feasibility class, geometric cooling | small budgets or local refinement |
| `pso` | particles, velocities, personal/global best | inertia plus cognitive/social motion, reflected and damped unit-box boundaries | smooth objectives with a reusable population |
| `de` | target population | DE/rand/1/bin mutation and crossover, one-to-one constraint-first replacement | continuous or lightly discrete black-box search |
| `nsga2` | parent and offspring populations | binary rank/crowding tournament, simulated-binary crossover, polynomial mutation, elitist environmental selection | broad multi-objective fronts |
| `moead` | weighted subproblems and neighborhoods | deterministic simplex weights, neighborhood DE variation, constraint-first weighted-Chebyshev replacement | decomposing fronts across objective trade-offs |

All algorithms search the normalized unit cube. The central decoder alone turns a coordinate into a real, log-scaled,
quantized, integer, or categorical value and resolves linked variables. Candidate identity is the SHA-256 of those
decoded values, so two coordinates that describe the same device sizing are one evaluation, not two.
Each optimizer also remembers successfully told keys itself; `seen_keys` only imports evaluations performed before it
was created. Returned points and accepted trial metrics are immutable snapshots.

Finite real bounds remain valid even when their direct difference would overflow: linear coordinates use scaled ratios
and convex interpolation, and logarithmic defaults are formed in log space. Narrow positive intervals use log-ratio
arithmetic so adjacent representable bounds and their interior floats remain encodable. Integer decoding is exact; the
public encoder rejects a supplied value if no normalized float coordinate round-trips to that exact integer in a very
wide interval. Optimizers use the same decoder at every entry point and never silently replace a requested encoded value.

## Hard constraints and ties

Feasibility is never folded into a weighted objective. A successful feasible trial outranks an infeasible trial; an
infeasible trial outranks an evaluator failure and is compared by total normalized squared violation and then maximum
violation. Pareto dominance orders feasible vectors. When dominance leaves a tie, objective vector, trial ID, and point
key provide a stable total order. SA applies probabilistic acceptance only within the same success/feasibility class,
so temperature cannot buy a move across a hard-constraint boundary.

## Budget and reproducibility

The runner asks for at most `min(batch_size, remaining_budget)` points. SA may return one because it is a single chain;
population algorithms may stop at a generation or initialization boundary and continue in the next ask. A final partial
batch is legal, and every returned point counts exactly once whether evaluation succeeds or fails. Integer, choice, and
quantized-real products of at most 100,000 decoded points have a deterministic lazy enumeration fallback. They stop
with `search_space_exhausted` only after complete coverage is proven. Larger or continuous domains use bounded random
immigrants when a native operator stalls and report `proposal_stalled` if none is found. Wall time and stagnation are
checked only between batches.

A seed fixes algorithm randomness. Results are told in proposal order after worker threads finish, so worker count does
not alter the sequence when the evaluator itself is deterministic and thread-safe. Batch size can alter adaptive
feedback timing and is therefore part of a reproducible run specification.
`catalog-run.json` records the package version, strategy-schema version, and complete effective hyperparameters. The
versioned `tests/data/optimizer-golden-v1.json` fixture pins all seven seeded point-key trajectories; changes require an
intentional fixture review rather than silently redefining reproducibility.

## CLI and API

```console
biasweave run --problem problem.toml --evaluator python:models:evaluate \
  --strategy de --population-size 20 --budget 100 --seed 7 --batch-size 8 \
  --out artifacts/de-7
```

Output destinations are no-clobber by default. `--force` replaces prior outputs transactionally, but input aliases are
permanently protected even with force.

`--population-size` is valid only for PSO, DE, NSGA-II, and MOEA/D and must be between four and 512. The upper bound
prevents hostile or accidental settings from allocating unbounded population and quadratic neighborhood state. The
default is 16. Choose a population no larger than the budget if the run should reach adaptive proposals rather than
spend the entire budget on initialization. MOEA/D requires at least one population slot per objective, accepts at most
128 objectives, and uses non-repeating Halton-simplex weights after the objective extremes.

```python
from biasweave import StrategyName, create_optimizer, optimize_strategy
from biasweave.model import RunConfig

result = optimize_strategy(
    problem,
    evaluate,
    evaluator_id="python:models:evaluate",
    config=RunConfig(budget=100, seed=7, workers=4, batch_size=8),
    strategy=StrategyName.NSGA2,
    population_size=20,
)

optimizer = create_optimizer(problem, StrategyName.SA, seed=7)
points = optimizer.ask(1, seen_keys=set())
# Evaluate and assess the returned point(s), preserving order, then:
optimizer.tell(trials)
```

Direct ask/tell users own evaluation and Trial construction, but `tell` reconstructs the pending point, validates
metrics, recomputes constraint and objective assessment, and rejects a mismatch without consuming the pending batch.
`optimize_strategy` gives evaluators a defensive value copy and revalidates trials before archiving. Native
weave runs use `run.json` for exact crash recovery and `biasweave resume`. Other catalog runs currently emit the common
`trials.jsonl`, `frontier.json`, and `summary.md` plus final `catalog-run.json`; they deliberately do not advertise
resume until all population and pending-generation state has a versioned checkpoint schema.
