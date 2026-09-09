from __future__ import annotations

import math
from decimal import Decimal, Inexact, Rounded, Underflow, localcontext

import pytest

from biasweave.surrogate import GaussianProcess


def _eliminate(
    matrix: tuple[tuple[float, ...], ...], right_hand_side: tuple[float, ...]
) -> tuple[tuple[float, ...], float]:
    """Solve and compute log(det) by pivoted elimination, independently of Cholesky."""
    augmented = [list(row) + [value] for row, value in zip(matrix, right_hand_side, strict=True)]
    log_determinant = 0.0
    determinant_sign = 1
    for column in range(len(augmented)):
        pivot = max(range(column, len(augmented)), key=lambda row: abs(augmented[row][column]))
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            determinant_sign *= -1
        diagonal = augmented[column][column]
        assert diagonal != 0.0
        determinant_sign *= 1 if diagonal > 0.0 else -1
        log_determinant += math.log(abs(diagonal))
        for row in range(column + 1, len(augmented)):
            ratio = augmented[row][column] / diagonal
            for index in range(column, len(augmented) + 1):
                augmented[row][index] -= ratio * augmented[column][index]
    assert determinant_sign == 1
    solved = [0.0] * len(augmented)
    for row in range(len(augmented) - 1, -1, -1):
        residual = augmented[row][-1] - math.fsum(
            augmented[row][column] * solved[column] for column in range(row + 1, len(augmented))
        )
        solved[row] = residual / augmented[row][row]
    return tuple(solved), log_determinant


def _decimal_solve(
    matrix: tuple[tuple[Decimal, ...], ...], right_hand_side: tuple[Decimal, ...]
) -> tuple[Decimal, ...]:
    augmented = [list(row) + [value] for row, value in zip(matrix, right_hand_side, strict=True)]
    for column in range(len(augmented)):
        pivot = max(range(column, len(augmented)), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        diagonal = augmented[column][column]
        assert diagonal != 0
        for row in range(column + 1, len(augmented)):
            ratio = augmented[row][column] / diagonal
            for index in range(column, len(augmented) + 1):
                augmented[row][index] -= ratio * augmented[column][index]
    solved = [Decimal(0)] * len(augmented)
    for row in range(len(augmented) - 1, -1, -1):
        residual = augmented[row][-1] - sum(
            (augmented[row][column] * solved[column] for column in range(row + 1, len(augmented))),
            start=Decimal(0),
        )
        solved[row] = residual / augmented[row][row]
    return tuple(solved)


def _decimal_prediction(
    points: tuple[tuple[float, ...], ...],
    targets: tuple[float, ...],
    query: tuple[float, ...],
    *,
    length_scale: float,
    noise_variance: float,
) -> tuple[float, float]:
    """Evaluate the full covariance solve at 80 decimal digits from exact float inputs."""
    with localcontext() as context:
        context.prec = 80
        decimal_points = tuple(
            tuple(Decimal.from_float(value) for value in point) for point in points
        )
        decimal_targets = tuple(Decimal.from_float(value) for value in targets)
        decimal_query = tuple(Decimal.from_float(value) for value in query)
        scale = Decimal.from_float(length_scale)
        noise = Decimal.from_float(noise_variance)
        half = Decimal("0.5")

        def covariance(left: tuple[Decimal, ...], right: tuple[Decimal, ...]) -> Decimal:
            squared_distance = sum(
                (((a - b) / scale) ** 2 for a, b in zip(left, right, strict=True)),
                start=Decimal(0),
            )
            return (-half * squared_distance).exp()

        matrix = tuple(
            tuple(
                covariance(left, right) + (noise if row == column else Decimal(0))
                for column, right in enumerate(decimal_points)
            )
            for row, left in enumerate(decimal_points)
        )
        constant = sum(decimal_targets, start=Decimal(0)) / Decimal(len(decimal_targets))
        centered = tuple(value - constant for value in decimal_targets)
        alpha = _decimal_solve(matrix, centered)
        cross_covariance = tuple(covariance(decimal_query, point) for point in decimal_points)
        prediction = constant + sum(
            (left * right for left, right in zip(cross_covariance, alpha, strict=True)),
            start=Decimal(0),
        )
        projected = _decimal_solve(matrix, cross_covariance)
        variance = Decimal(1) - sum(
            (left * right for left, right in zip(cross_covariance, projected, strict=True)),
            start=Decimal(0),
        )
        return float(prediction), float(variance)


@pytest.mark.parametrize("size", [128, 256])
@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_observations_preserve_the_analytical_mean(size: int, reverse: bool) -> None:
    """Identical inputs reduce to their sufficient statistic, independent of order."""
    points = ((0.5,),) * size
    targets = (-1.0,) * (size // 2) + (1.0,) * (size // 2)
    if reverse:
        targets = tuple(reversed(targets))

    model = GaussianProcess(points, targets, noise_variance=1e-10)
    prediction = model.predict((0.5,))

    assert prediction.mean == pytest.approx(0.0, abs=1e-10)
    assert prediction.variance == pytest.approx(1e-10 / (size + 1e-10), rel=1e-12, abs=0.0)


def test_mixed_replicates_match_the_full_observation_covariance() -> None:
    points = ((0.0,), (0.0,), (0.4,), (1.0,), (0.4,))
    targets = (-1.0, 0.5, 0.2, 0.75, -0.4)
    scale = 0.4
    noise = 0.03
    model = GaussianProcess(points, targets, length_scale=scale, noise_variance=noise)

    covariance = tuple(
        tuple(
            math.exp(-0.5 * ((left[0] - right[0]) / scale) ** 2) + (noise if row == column else 0.0)
            for column, right in enumerate(points)
        )
        for row, left in enumerate(points)
    )
    constant = math.fsum(targets) / len(targets)
    centered = tuple(target - constant for target in targets)
    alpha, log_determinant = _eliminate(covariance, centered)
    query = (0.25,)
    cross_covariance = tuple(
        math.exp(-0.5 * ((query[0] - point[0]) / scale) ** 2) for point in points
    )
    projected, _ = _eliminate(covariance, cross_covariance)
    expected_mean = constant + math.fsum(
        left * right for left, right in zip(cross_covariance, alpha, strict=True)
    )
    expected_variance = 1.0 - math.fsum(
        left * right for left, right in zip(cross_covariance, projected, strict=True)
    )
    expected_likelihood = (
        -0.5 * math.fsum(left * right for left, right in zip(centered, alpha, strict=True))
        - 0.5 * log_determinant
        - 0.5 * len(points) * math.log(2.0 * math.pi)
    )

    assert model.points == ((0.0,), (0.4,), (1.0,))
    assert model.targets == pytest.approx((-0.25, -0.1, 0.75))
    assert model.replicate_counts == (2, 2, 1)
    assert model.observation_count == 5
    prediction = model.predict(query)
    assert prediction.mean == pytest.approx(expected_mean, abs=1e-13)
    assert prediction.variance == pytest.approx(expected_variance, abs=1e-13)
    assert model.log_marginal_likelihood == pytest.approx(expected_likelihood, abs=1e-12)


@pytest.mark.parametrize("size", [8, 32])
def test_near_duplicate_prediction_matches_high_precision_full_covariance(size: int) -> None:
    points = tuple((0.5 + 1e-9 * (index - (size - 1) / 2) / (size - 1),) for index in range(size))
    targets = tuple(math.sin((index + 0.25) * 1.7) for index in range(size))
    query = (0.5 + 1e-9 * 0.123,)
    scale = 0.25
    noise = 1e-10
    expected_mean, expected_variance = _decimal_prediction(
        points,
        targets,
        query,
        length_scale=scale,
        noise_variance=noise,
    )

    prediction = GaussianProcess(points, targets, length_scale=scale, noise_variance=noise).predict(
        query
    )

    assert prediction.mean == pytest.approx(expected_mean, rel=0.0, abs=1e-13)
    assert prediction.variance == pytest.approx(expected_variance, rel=1e-12, abs=1e-24)


def test_canonical_duplicate_coordinate_normalizes_signed_zero() -> None:
    negative_first = GaussianProcess(((-0.0,), (0.0,)), (-1.0, 1.0))
    positive_first = GaussianProcess(((0.0,), (-0.0,)), (1.0, -1.0))

    assert math.copysign(1.0, negative_first.points[0][0]) == 1.0
    assert math.copysign(1.0, positive_first.points[0][0]) == 1.0


def test_fit_and_prediction_are_isolated_from_the_callers_decimal_context() -> None:
    arguments = (((0.0,), (0.4,), (1.0,)), (-1.0, 0.25, 1.0))
    baseline = GaussianProcess(*arguments, noise_variance=1e-10)
    baseline_prediction = baseline.predict((0.3,))

    with localcontext() as hostile:
        hostile.prec = 2
        hostile.Emin = -2
        hostile.Emax = 2
        hostile.traps[Inexact] = True
        hostile.traps[Rounded] = True
        hostile.traps[Underflow] = True
        isolated = GaussianProcess(*arguments, noise_variance=1e-10)
        isolated_prediction = isolated.predict((0.3,))

    assert isolated == baseline
    assert isolated_prediction == baseline_prediction
