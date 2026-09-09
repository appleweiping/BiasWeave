"""Bounded normalized Gaussian-process regression for sequential search.

The model uses a fixed unit-amplitude squared-exponential covariance, an
empirical constant mean, and an explicit observation-noise variance. Returned
uncertainty is latent-function variance, not measurement-noise variance.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)

from biasweave.errors import ConfigurationError

MAX_TRAINING_POINTS = 256
MAX_DIMENSIONS = 64
_CONTEXT = Context(
    prec=60,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    traps=[InvalidOperation, DivisionByZero, Overflow],
)
_ZERO = Decimal(0)
_ONE = Decimal(1)
_HALF = Decimal("0.5")
_VARIANCE_ROUNDOFF = Decimal("1e-40")
_TWO_PI = Decimal("6.283185307179586476925286766559005768394338798750211641949889")


class SurrogateError(ConfigurationError):
    """Surrogate inputs or numerical evidence cannot satisfy the model contract."""


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SurrogateError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise SurrogateError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise SurrogateError(f"{name} must be a finite number")
    return 0.0 if result == 0.0 else result


def _coordinates(value: object, dimension: int) -> tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != dimension:
        raise SurrogateError("coordinates must be an immutable tuple of the fitted dimension")
    result = tuple(_number(item, "coordinate") for item in value)
    if any(not 0.0 <= item <= 1.0 for item in result):
        raise SurrogateError("coordinates must belong to the normalized unit cube")
    return result


def _covariance(left: tuple[Decimal, ...], right: tuple[Decimal, ...], scale: Decimal) -> Decimal:
    distance = sum(((a - b) / scale) ** 2 for a, b in zip(left, right, strict=True))
    return (-_HALF * distance).exp()


def _observation_sum(values: Iterable[Decimal]) -> Decimal:
    # Each validated binary64 target lies in [-1, 1]. Its exact decimal
    # expansion has at most 1074 fractional places. 1200 digits therefore
    # suffice to sum all 256 observations exactly, independent of order,
    # even when a tiny value lies between large cancelling observations.
    with localcontext(_CONTEXT) as context:
        context.prec = 1200
        return sum(values, _ZERO)


def _forward(
    lower: tuple[tuple[Decimal, ...], ...], values: tuple[Decimal, ...]
) -> tuple[Decimal, ...]:
    solved: list[Decimal] = []
    for row, value in zip(lower, values, strict=True):
        index = len(solved)
        product = sum((a * b for a, b in zip(row[:index], solved, strict=True)), _ZERO)
        solved.append((value - product) / row[index])
    return tuple(solved)


def _backward(
    lower: tuple[tuple[Decimal, ...], ...], values: tuple[Decimal, ...]
) -> tuple[Decimal, ...]:
    size = len(values)
    solved = [_ZERO] * size
    for index in range(size - 1, -1, -1):
        product = sum((lower[j][index] * solved[j] for j in range(index + 1, size)), _ZERO)
        solved[index] = (values[index] - product) / lower[index][index]
    return tuple(solved)


@dataclass(frozen=True, slots=True)
class GaussianPrediction:
    """Mean and non-negative latent variance in normalized target units."""

    mean: float
    variance: float


@dataclass(frozen=True, slots=True, init=False)
class GaussianProcess:
    """Fit at most 256 observations in at most 64 normalized dimensions.

    Targets must be in [-1, 1]; objective normalization belongs to the caller.
    Duplicate coordinates are supported through the declared positive noise.
    No hidden jitter, optimization, point truncation, or pseudoinverse is used.
    Training costs O(n^3 + n^2 d), storage O(n^2 + nd), prediction O(n^2 + nd).
    """

    points: tuple[tuple[float, ...], ...]
    targets: tuple[float, ...]
    replicate_counts: tuple[int, ...]
    observation_count: int
    length_scale: float
    noise_variance: float
    constant_mean: float
    log_marginal_likelihood: float
    _lower: tuple[tuple[Decimal, ...], ...]
    _alpha: tuple[Decimal, ...]
    _mean: Decimal
    _scale: Decimal
    _points: tuple[tuple[Decimal, ...], ...]

    def __init__(
        self,
        points: tuple[tuple[float, ...], ...],
        targets: tuple[float, ...],
        *,
        length_scale: float = 0.25,
        noise_variance: float = 1e-6,
    ) -> None:
        if not isinstance(points, tuple) or not 1 <= len(points) <= MAX_TRAINING_POINTS:
            raise SurrogateError("training points must be an immutable tuple of size 1 through 256")
        if not isinstance(points[0], tuple) or not 1 <= len(points[0]) <= MAX_DIMENSIONS:
            raise SurrogateError("training dimension must be between 1 and 64")
        if not isinstance(targets, tuple) or len(targets) != len(points):
            raise SurrogateError("targets must be an immutable tuple matching training points")
        scale = _number(length_scale, "length_scale")
        noise = _number(noise_variance, "noise_variance")
        if not 1e-4 <= scale <= 10.0:
            raise SurrogateError("length_scale must be between 1e-4 and 10")
        if not 1e-10 <= noise <= 1.0:
            raise SurrogateError("noise_variance must be between 1e-10 and 1")
        checked = tuple(_coordinates(point, len(points[0])) for point in points)
        observations = tuple(_number(value, "target") for value in targets)
        if any(not -1.0 <= value <= 1.0 for value in observations):
            raise SurrogateError("targets must be normalized to [-1, 1]")
        # Equal coordinates have exact Gaussian sufficient statistics. Collapse
        # each group to its mean with noise/count, retaining its within-group
        # residual and determinant correction in the full-data likelihood.
        # This avoids cancellation of O(1/noise) opposite alpha coefficients.
        groups: dict[tuple[float, ...], list[Decimal]] = {}
        for point, target in zip(checked, observations, strict=True):
            groups.setdefault(point, []).append(Decimal.from_float(target))
        unique = tuple(sorted(groups))
        counts = tuple(len(groups[point]) for point in unique)
        decimal_points = tuple(
            tuple(Decimal.from_float(value) for value in point) for point in unique
        )
        # Binary64 covariance rounding followed by subtraction of O(1/noise)
        # coefficients can dwarf the reported posterior uncertainty. Compute
        # both the kernel and solves in an isolated fixed-precision context;
        # converting an already rounded float covariance would not fix this.
        with localcontext(_CONTEXT):
            decimal_scale = Decimal.from_float(scale)
            decimal_noise = Decimal.from_float(noise)
            mean = _observation_sum(value for values in groups.values() for value in values) / len(
                checked
            )
            averages = tuple(
                _observation_sum(groups[point]) / len(groups[point]) for point in unique
            )
            within = sum(
                (
                    (value - average) ** 2
                    for point, average in zip(unique, averages, strict=True)
                    for value in groups[point]
                ),
                _ZERO,
            )
            centered = tuple(value - mean for value in averages)
            rows: list[tuple[Decimal, ...]] = []
            for index, decimal_point in enumerate(decimal_points):
                row: list[Decimal] = []
                for column in range(index + 1):
                    covariance = _covariance(decimal_point, decimal_points[column], decimal_scale)
                    if index == column:
                        residual = (
                            _ONE
                            + decimal_noise / counts[index]
                            - sum((value * value for value in row), _ZERO)
                        )
                        if not residual.is_finite() or residual <= 0:
                            raise SurrogateError(
                                "covariance Cholesky factor is not positive definite"
                            )
                        row.append(residual.sqrt())
                    else:
                        product = sum(
                            (a * b for a, b in zip(row, rows[column][:column], strict=True)), _ZERO
                        )
                        row.append((covariance - product) / rows[column][column])
                rows.append(tuple(row))
            lower = tuple(rows)
            alpha = _backward(lower, _forward(lower, centered))
            likelihood = (
                -_HALF * sum((a * b for a, b in zip(centered, alpha, strict=True)), _ZERO)
                - sum((row[index].ln() for index, row in enumerate(lower)), _ZERO)
                - _HALF * len(checked) * _TWO_PI.ln()
                - _HALF * within / decimal_noise
                - _HALF * (len(checked) - len(unique)) * decimal_noise.ln()
                - _HALF * sum((Decimal(count).ln() for count in counts), _ZERO)
            )
            if not likelihood.is_finite() or any(not value.is_finite() for value in alpha):
                raise SurrogateError("non-finite Gaussian-process factorization")
        for name, value in (
            ("points", unique),
            ("targets", tuple(float(value) for value in averages)),
            ("replicate_counts", counts),
            ("observation_count", len(checked)),
            ("length_scale", scale),
            ("noise_variance", noise),
            ("constant_mean", float(mean)),
            ("log_marginal_likelihood", float(likelihood)),
            ("_lower", lower),
            ("_alpha", alpha),
            ("_mean", mean),
            ("_scale", decimal_scale),
            ("_points", decimal_points),
        ):
            object.__setattr__(self, name, value)

    def predict(self, coordinates: tuple[float, ...]) -> GaussianPrediction:
        """Predict a latent value; numerical breakdown is never zero uncertainty."""
        point = _coordinates(coordinates, len(self.points[0]))
        with localcontext(_CONTEXT):
            decimal_point = tuple(Decimal.from_float(value) for value in point)
            covariances = tuple(
                _covariance(decimal_point, other, self._scale) for other in self._points
            )
            mean = self._mean + sum(
                (a * b for a, b in zip(covariances, self._alpha, strict=True)), _ZERO
            )
            projected = _forward(self._lower, covariances)
            variance = _ONE - sum((value * value for value in projected), _ZERO)
            if not mean.is_finite() or not variance.is_finite() or variance < -_VARIANCE_ROUNDOFF:
                raise SurrogateError("non-finite prediction or negative predictive variance")
            return GaussianPrediction(float(mean), float(max(_ZERO, variance)))


def expected_improvement(
    prediction: GaussianPrediction, best: float, *, exploration: float = 0.0
) -> float:
    """Expected positive improvement for minimization in normalized units."""
    if not isinstance(prediction, GaussianPrediction):
        raise SurrogateError("prediction must be a GaussianPrediction")
    mean = _number(prediction.mean, "predictive mean")
    variance = _number(prediction.variance, "predictive variance")
    incumbent = _number(best, "best target")
    offset = _number(exploration, "exploration")
    if variance < 0.0 or variance > 1.0 or not -1.0 <= incumbent <= 1.0:
        raise SurrogateError("variance or best target is outside normalized model bounds")
    if not 0.0 <= offset <= 1.0:
        raise SurrogateError("exploration must be between 0 and 1")
    improvement = incumbent - mean - offset
    if variance == 0.0:
        return max(0.0, improvement)
    sigma = math.sqrt(variance)
    # Tail branches avoid overflowing z or evaluating an underflowed density.
    if improvement > 38.0 * sigma:
        return improvement
    if improvement < -38.0 * sigma:
        return 0.0
    z = improvement / sigma
    cdf = 0.5 * math.erfc(-z / math.sqrt(2.0))
    density = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return max(0.0, improvement * cdf + sigma * density)
