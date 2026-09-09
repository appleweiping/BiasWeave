from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from biasweave import surrogate as module
from biasweave.surrogate import (
    GaussianPrediction,
    GaussianProcess,
    SurrogateError,
    expected_improvement,
)


def test_single_point_posterior_has_analytical_latent_variance():
    model = GaussianProcess(((0.25,),), (0.75,), length_scale=0.5, noise_variance=0.1)
    same = model.predict((0.25,))
    distant = model.predict((0.75,))
    assert same.mean == distant.mean == 0.75
    assert same.variance == pytest.approx(1 - 1 / 1.1)
    assert distant.variance == pytest.approx(1 - math.exp(-1.0) / 1.1)
    assert model.log_marginal_likelihood == pytest.approx(-0.5 * math.log(2 * math.pi * 1.1))
    with pytest.raises(FrozenInstanceError):
        model.noise_variance = 0


def test_two_point_predictions_match_an_explicit_two_by_two_inverse():
    scale, noise = 0.4, 0.03
    model = GaussianProcess(((0.0,), (1.0,)), (-1.0, 1.0), length_scale=scale, noise_variance=noise)
    cross = math.exp(-0.5 / scale**2)
    diagonal = 1 + noise
    determinant = diagonal**2 - cross**2
    for x in (0.0, 0.2, 0.5, 0.7, 1.0):
        left = math.exp(-0.5 * (x / scale) ** 2)
        right = math.exp(-0.5 * ((x - 1) / scale) ** 2)
        expected_mean = (right - left) / (diagonal - cross)
        expected_variance = (
            1 - (diagonal * (left**2 + right**2) - 2 * cross * left * right) / determinant
        )
        prediction = model.predict((x,))
        assert prediction.mean == pytest.approx(expected_mean, abs=1e-13)
        assert prediction.variance == pytest.approx(expected_variance, abs=1e-13)


def test_duplicate_points_have_explicit_noise_and_finite_posterior():
    model = GaussianProcess(((0.5,),) * 32, (-1.0, 1.0) * 16, noise_variance=1e-10)
    prediction = model.predict((0.5,))
    assert math.isfinite(prediction.mean)
    assert prediction.variance >= 0
    assert prediction.variance == pytest.approx(1e-10 / (32 + 1e-10), abs=2e-15)
    assert model.noise_variance == 1e-10


@pytest.mark.parametrize(
    "points,targets",
    [
        ((), ()),
        ([], []),
        (((0.0,),) * 257, (0.0,) * 257),
        (((),), (0.0,)),
        (((0.0,) * 65,), (0.0,)),
        (((0.0,),), []),
        (((0.0,),), ()),
        (((0.0,), (0.0, 1.0)), (0.0, 0.0)),
        (([0.0],), (0.0,)),
        (((True,),), (0.0,)),
        (((-0.1,),), (0.0,)),
        (((1.1,),), (0.0,)),
        (((math.nan,),), (0.0,)),
        (((math.inf,),), (0.0,)),
        (((10**1000,),), (0.0,)),
        (((0.0,),), (math.nan,)),
        (((0.0,),), (True,)),
        (((0.0,),), (2.0,)),
    ],
)
def test_training_input_contract_is_fail_closed(points, targets):
    with pytest.raises(SurrogateError):
        GaussianProcess(points, targets)


@pytest.mark.parametrize(
    "parameters",
    [
        {"length_scale": 0},
        {"length_scale": 11},
        {"length_scale": math.nan},
        {"noise_variance": 0},
        {"noise_variance": 1e-11},
        {"noise_variance": 2},
        {"noise_variance": True},
    ],
)
def test_hyperparameters_are_explicit_and_bounded(parameters):
    with pytest.raises(SurrogateError):
        GaussianProcess(((0.0,),), (0.0,), **parameters)


def test_predict_coordinates_are_checked():
    model = GaussianProcess(((0.0,),), (0.0,))
    for point in ((), (2.0,), (math.inf,), (False,), [0.0]):
        with pytest.raises(SurrogateError):
            model.predict(point)


def test_ei_matches_closed_form_and_degenerate_limits():
    assert expected_improvement(GaussianPrediction(0, 1), 0) == pytest.approx(
        1 / math.sqrt(2 * math.pi)
    )
    assert expected_improvement(GaussianPrediction(0.25, 0), 0.75) == 0.5
    assert expected_improvement(GaussianPrediction(0.75, 0), 0.25) == 0
    assert expected_improvement(GaussianPrediction(0.25, 0), 0.75, exploration=0.1) == 0.4
    assert expected_improvement(GaussianPrediction(-1e100, 1), 0) == 1e100
    assert expected_improvement(GaussianPrediction(1e100, 1), 0) == 0
    assert expected_improvement(GaussianPrediction(0, 5e-324), 0) > 0


@pytest.mark.parametrize(
    "prediction,best,exploration",
    [
        (None, 0, 0),
        (GaussianPrediction(math.nan, 1), 0, 0),
        (GaussianPrediction(0, -1), 0, 0),
        (GaussianPrediction(0, 2), 0, 0),
        (GaussianPrediction(0, 1), 2, 0),
        (GaussianPrediction(0, 1), 0, -1),
        (GaussianPrediction(0, 1), 0, 2),
    ],
)
def test_ei_rejects_invalid_evidence(prediction, best, exploration):
    with pytest.raises(SurrogateError):
        expected_improvement(prediction, best, exploration=exploration)


def test_numerical_breakdown_does_not_become_confident_prediction(monkeypatch):
    model = GaussianProcess(((0.0,),), (0.0,))
    monkeypatch.setattr(module, "_forward", lambda *_args: (Decimal(2),))
    with pytest.raises(SurrogateError, match="variance"):
        model.predict((0.0,))


def test_training_preflight_happens_before_covariance_allocation(monkeypatch):
    def forbidden(*_args):
        pytest.fail("invalid inputs must not allocate a covariance matrix")

    monkeypatch.setattr(module, "_covariance", forbidden)
    with pytest.raises(SurrogateError):
        GaussianProcess(((0.0,),) * 257, (0.0,) * 257)


def test_replicate_mean_retains_small_values_after_exact_cancellation():
    tiny = 1e-100
    first = GaussianProcess(((0.5,),) * 3, (1.0, tiny, -1.0))
    last = GaussianProcess(((0.5,),) * 3, (1.0, -1.0, tiny))
    assert first.constant_mean == last.constant_mean == tiny / 3
    assert first.predict((0.5,)).mean == last.predict((0.5,)).mean == tiny / 3
