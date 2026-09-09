# Bounded Gaussian-process surrogate

The development `biasweave.surrogate` module implements an original, small
Gaussian-process regression kernel for sequential optimization. It is not yet
an optimizer catalog entry: fitting a statistical model alone does not provide
a complete ask/tell search policy or circuit-validity evidence.

Inputs are immutable tuples in the normalized unit cube, with at most 256
observations and 64 coordinates per observation. Targets must already be in
`[-1, 1]`. The model uses their empirical mean, unit signal variance, a fixed
squared-exponential length scale, and explicitly supplied observation-noise
variance. It solves through a Cholesky factor and triangular substitutions,
never by an explicit inverse. Predictions return latent function variance,
excluding observation noise. These are standard regression equations described
in [Gaussian Processes for Machine Learning, chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf).

Kernel values, group statistics, factorization and prediction use 60-digit
decimal arithmetic in an explicit, isolated context. Inputs retain their exact
binary64 meaning on conversion; only public results are converted back to
binary64. This matters for tightly clustered samples at low observation noise:
forming the covariance in binary64 first can already lose information, and
opposite large coefficients can then produce errors larger than the reported
uncertainty. Increasing precision does not change the declared noise or inflate
the latent variance. Caller decimal precision, rounding and exception traps do
not affect the model. Negative zero is canonicalized to positive zero.

`expected_improvement` evaluates positive improvement for minimization, with an
explicit exploration offset, a deterministic zero-variance limit, and bounded
Gaussian-tail handling. Its interpretation follows the acquisition formulation
in [Frazier's Bayesian optimization tutorial](https://arxiv.org/abs/1807.02811).
Neither statistic proves that a physical circuit will meet a specification.

```python
from biasweave.surrogate import GaussianProcess, expected_improvement

model = GaussianProcess(((0.0,), (1.0,)), (-1.0, 1.0), noise_variance=1e-6)
prediction = model.predict((0.25,))
acquisition = expected_improvement(prediction, best=-1.0)
assert prediction.variance >= 0.0 and acquisition >= 0.0
```

Length scale is bounded to `[1e-4, 10]`; noise **variance**, not standard
deviation, is bounded to `[1e-10, 1]`. Repeated coordinates are allowed under
this explicit independent-noise model. Exactly equal coordinates are grouped
in canonical order using sufficient statistics: the group-average target has
noise variance divided by the replicate count. The likelihood retains the
within-group squared residual and determinant correction, so contradictory
replicates are not discarded or treated as exact measurements. `points` and
`targets` expose these unique coordinates and averages; `replicate_counts` and
`observation_count` preserve the number of contributing observations. The
empirical mean is computed over all observations, not over unique locations.
The implementation does not silently tune
hyperparameters, inject additional jitter, discard observations, repair
non-positive Cholesky pivots, or substitute a pseudoinverse. A negative latent
variance below `-1e-40` is a numerical error; only smaller internal decimal
roundoff is clamped to zero.

Training costs `O(n³ + n²d)`, prediction costs `O(n² + nd)`, and storage costs
`O(n² + nd)`. Bounds are checked before matrix construction. This dense kernel
is intended for expensive-evaluation budgets, not unbounded training tables.
On one Windows/CPython 3.14.5 run, a seeded 256-point, 64-dimensional fit took
6.39 seconds and 32 predictions took 2.02 seconds after reusing exact coordinate
conversions. These are observed timings, not a latency guarantee or an algorithm
comparison. Fit once per training set and reuse it for candidate predictions.
The independent tests include an 80-digit, pivoted full-covariance elimination
oracle for near-duplicate samples, distinct from the production Cholesky solve.
It has no learned hyperparameter fitting, multi-output covariance, categorical
kernel, serialized checkpoint, or claim of cross-platform bitwise identity.
The strategy layer must preserve hard-constraint assessment, decoded-point
deduplication, truthful evaluation budgets and evaluator failures separately.
