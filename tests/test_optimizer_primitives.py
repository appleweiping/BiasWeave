from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from biasweave.dominance import assess
from biasweave.encoding import make_point
from biasweave.errors import ConfigurationError
from biasweave.optimizers import StrategyName, create_optimizer
from biasweave.optimizers.annealing import SimulatedAnnealingOptimizer
from biasweave.optimizers.base import clamp_coordinate, clamp_vector, finite_parameter
from biasweave.optimizers.differential_evolution import (
    DifferentialEvolutionOptimizer,
    differential_trial,
)
from biasweave.optimizers.moead import MOEADOptimizer, simplex_weights, tchebycheff
from biasweave.optimizers.nsga2 import (
    NSGA2Optimizer,
    polynomial_mutation_coordinate,
    simulated_binary_coordinate,
)
from biasweave.optimizers.pso import ParticleSwarmOptimizer, advance_particle
from biasweave.optimizers.random_search import RandomOptimizer
from biasweave.optimizers.weave import WeaveOptimizer
from tests.helpers import evaluator, make_problem


def _assess(point, trial_id=0):
    problem = make_problem()
    return assess(problem, trial_id, point, evaluator(point.values))


def test_unit_box_projection_rejects_nonfinite_values() -> None:
    assert clamp_coordinate(-2.0) == 0.0
    assert clamp_coordinate(2.0) == 1.0
    assert clamp_vector((0.25, -1.0, 4.0)) == (0.25, 0.0, 1.0)
    with pytest.raises(ConfigurationError, match="non-finite"):
        clamp_coordinate(math.nan)
    assert finite_parameter(3, "coefficient") == 3.0
    with pytest.raises(ConfigurationError, match="numeric"):
        finite_parameter("three", "coefficient")
    with pytest.raises(ConfigurationError, match="finite"):
        finite_parameter(10**10_000, "coefficient")


def test_ask_tell_contract_rejects_protocol_errors() -> None:
    with pytest.raises(ConfigurationError, match="seed"):
        RandomOptimizer(make_problem(), True)
    optimizer = RandomOptimizer(make_problem(), 4)
    assert optimizer.recommended_batch_size == 1
    with pytest.raises(ConfigurationError, match="positive integer"):
        optimizer.ask(0)
    with pytest.raises(ConfigurationError, match="pending"):
        optimizer.tell(())
    points = optimizer.ask(2)
    assert optimizer.pending == points
    with pytest.raises(ConfigurationError, match="before ask"):
        optimizer.ask(1)
    with pytest.raises(ConfigurationError, match="complete"):
        optimizer.tell((_assess(points[0]),))
    forged_point = replace(points[0], coordinates=(0.99,) * len(points[0].coordinates))
    forged_trial = replace(_assess(points[0]), point=forged_point)
    with pytest.raises(ConfigurationError, match="do not match"):
        optimizer.tell((forged_trial, _assess(points[1], 1)))
    optimizer.tell(tuple(_assess(point, index) for index, point in enumerate(points)))
    replacement = optimizer.ask(1, {point.key for point in points})
    wrong = _assess(make_point(make_problem(), (0.99, 0.99, 0.99)), 2)
    with pytest.raises(ConfigurationError, match="do not match"):
        optimizer.tell((wrong,))
    duplicate_id = _assess(replacement[0], 1)
    with pytest.raises(ConfigurationError, match="strictly increasing"):
        optimizer.tell((duplicate_id,))
    optimizer.tell((_assess(replacement[0], 2),))
    assert optimizer.ask(1, {point.key for point in (*points, *replacement)})


def test_tell_rejects_nonfinite_or_internally_inconsistent_assessments() -> None:
    optimizer = RandomOptimizer(make_problem(), 1)
    point = optimizer.ask(1)[0]
    bad = replace(_assess(point), objective_vector=(math.nan, 0.0))
    with pytest.raises(ConfigurationError, match="inconsistent assessment"):
        optimizer.tell((bad,))
    bad = replace(_assess(point), objective_vector=("not-a-number", 0.0))
    with pytest.raises(ConfigurationError, match="inconsistent assessment"):
        optimizer.tell((bad,))
    bad = replace(_assess(point), feasible=True, violation=0.1)
    with pytest.raises(ConfigurationError, match="inconsistent assessment"):
        optimizer.tell((bad,))
    optimizer.tell((_assess(point),))


def test_point_and_trial_mappings_are_immutable_but_serialize_as_objects() -> None:
    point = make_point(make_problem(), (0.2, 0.4, 0.6))
    trial = _assess(point)
    with pytest.raises(TypeError):
        point.values["x"] = 999.0  # type: ignore[index]
    with pytest.raises(TypeError):
        trial.metrics["loss"] = -999.0  # type: ignore[index]
    payload = trial.as_dict()
    payload["point"]["values"]["x"] = 999.0
    assert point.values["x"] != 999.0
    assert json.loads(json.dumps(trial.as_dict()))["point"]["values"]["x"] == point.values["x"]


def test_tell_rejects_forged_assessment_without_consuming_pending() -> None:
    optimizer = RandomOptimizer(make_problem(), 1)
    point = optimizer.ask(1)[0]
    valid = _assess(point)
    forged = replace(
        valid,
        metrics={"unrelated": 1.0},
        objective_vector=(-999.0, -999.0),
    )
    with pytest.raises(ConfigurationError, match="metrics are invalid"):
        optimizer.tell((forged,))
    assert optimizer.pending == (point,)
    optimizer.tell((valid,))


def test_rejected_failed_trial_does_not_consume_pending_or_seen_state() -> None:
    optimizer = RandomOptimizer(make_problem(), 2)
    point = optimizer.ask(1)[0]
    invalid = replace(_assess(point), status="failed", error="boom")
    with pytest.raises(ConfigurationError, match="status"):
        optimizer.tell((invalid,))
    assert optimizer.pending == (point,)
    optimizer.tell((_assess(point),))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("weave", WeaveOptimizer),
        ("random", RandomOptimizer),
        ("sa", SimulatedAnnealingOptimizer),
        ("pso", ParticleSwarmOptimizer),
        ("de", DifferentialEvolutionOptimizer),
        ("nsga2", NSGA2Optimizer),
        ("moead", MOEADOptimizer),
    ],
)
def test_catalog_constructs_distinct_concrete_strategies(name, expected) -> None:
    optimizer = create_optimizer(
        make_problem(),
        name,
        seed=11,
        population_size=4 if name in {"pso", "de", "nsga2", "moead"} else None,
    )
    assert type(optimizer) is expected
    assert optimizer.name == name


def test_catalog_rejects_unknown_and_inapplicable_options() -> None:
    with pytest.raises(ConfigurationError, match="strategy must be"):
        create_optimizer(make_problem(), "alias-for-de", seed=0)
    for name in (StrategyName.WEAVE, StrategyName.RANDOM, StrategyName.SA):
        with pytest.raises(ConfigurationError, match="does not apply"):
            create_optimizer(make_problem(), name, seed=0, population_size=8)


def test_particle_step_reflects_and_damps_both_boundaries() -> None:
    position, velocity = advance_particle(
        (0.9, 0.1),
        (0.5, -0.5),
        (0.9, 0.1),
        (0.9, 0.1),
        inertia=1.0,
        cognitive=0.0,
        social=0.0,
        personal_draws=(0.0, 0.0),
        social_draws=(0.0, 0.0),
    )
    assert position == pytest.approx((0.8, 0.2))
    assert velocity == (-0.25, 0.25)
    position, velocity = advance_particle(
        (0.5,),
        (4.0,),
        (0.5,),
        (0.5,),
        inertia=1.0,
        cognitive=0.0,
        social=0.0,
        personal_draws=(0.0,),
        social_draws=(0.0,),
    )
    assert position == pytest.approx((0.375,))
    assert velocity == pytest.approx((1.0,))
    with pytest.raises(ConfigurationError, match="matching"):
        advance_particle(
            (0.5,),
            (),
            (0.5,),
            (0.5,),
            inertia=1.0,
            cognitive=1.0,
            social=1.0,
            personal_draws=(0.5,),
            social_draws=(0.5,),
        )


def test_de_trial_forces_a_donor_coordinate_and_clamps() -> None:
    trial = differential_trial(
        (0.25, 0.75),
        (0.9, 0.1),
        (1.0, 0.0),
        (0.0, 1.0),
        differential_weight=1.0,
        crossover_rate=0.0,
        crossover_draws=(0.8, 0.8),
        forced_dimension=1,
    )
    assert trial == (0.25, 0.0)
    with pytest.raises(ConfigurationError, match="matching"):
        differential_trial(
            (0.5,),
            (),
            (0.5,),
            (0.5,),
            differential_weight=0.5,
            crossover_rate=0.5,
            crossover_draws=(0.5,),
            forced_dimension=0,
        )
    with pytest.raises(ConfigurationError, match="outside"):
        differential_trial(
            (0.5,),
            (0.5,),
            (0.5,),
            (0.5,),
            differential_weight=0.5,
            crossover_rate=0.5,
            crossover_draws=(0.5,),
            forced_dimension=1,
        )


def test_nsga_variation_operators_are_bounded_and_directional() -> None:
    low_draw = simulated_binary_coordinate(0.2, 0.8, 0.25, 15.0)
    high_draw = simulated_binary_coordinate(0.2, 0.8, 0.75, 15.0)
    assert 0.0 <= low_draw <= 1.0
    assert 0.0 <= high_draw <= 1.0
    assert low_draw > high_draw
    mutated_down = polynomial_mutation_coordinate(0.5, 0.25, 20.0)
    mutated_up = polynomial_mutation_coordinate(0.5, 0.75, 20.0)
    assert 0.0 <= mutated_down < 0.5 < mutated_up <= 1.0


def test_moead_weights_and_scalarization_are_well_formed() -> None:
    weights = simplex_weights(6, 3)
    assert weights[:3] == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert all(math.fsum(weight) == pytest.approx(1.0) for weight in weights)
    trial = _assess(make_point(make_problem(), (0.2, 0.4, 0.6)))
    ideal = tuple(value - 1.0 for value in trial.objective_vector)
    assert tchebycheff(trial, (0.25, 0.75), ideal) == pytest.approx(0.75)
    with pytest.raises(ConfigurationError, match="matching"):
        tchebycheff(trial, (1.0,), (0.0,))
    with pytest.raises(ConfigurationError, match="positive"):
        simplex_weights(0, 2)


def test_moead_weight_stream_stays_unique_beyond_the_old_short_period() -> None:
    weights = simplex_weights(140, 3)
    assert len(set(weights)) == len(weights) == 140


def test_population_and_objective_allocations_are_bounded() -> None:
    with pytest.raises(ConfigurationError, match="at most 512"):
        ParticleSwarmOptimizer(make_problem(), 0, population_size=513)
    with pytest.raises(ConfigurationError, match="at most 128"):
        simplex_weights(4, 129)
    many_objectives = replace(make_problem(), objectives=make_problem().objectives * 3)
    with pytest.raises(ConfigurationError, match="number of objectives"):
        MOEADOptimizer(many_objectives, 0, population_size=4)


@pytest.mark.parametrize(
    "optimizer",
    [
        ParticleSwarmOptimizer(make_problem(), 8, population_size=4),
        DifferentialEvolutionOptimizer(make_problem(), 8, population_size=4),
        NSGA2Optimizer(make_problem(), 8, population_size=4),
        MOEADOptimizer(make_problem(), 8, population_size=4),
    ],
    ids=["pso-personal-best", "de-survival", "nsga2-survival", "moead-replacement"],
)
def test_population_algorithms_never_trade_feasibility_for_objectives(optimizer) -> None:
    initial_points = optimizer.ask(4)
    initial = tuple(
        assess(
            make_problem(),
            index,
            point,
            {"loss": 100.0, "score": -200.0, "quality": 1.0, "window": 1.0},
        )
        for index, point in enumerate(initial_points)
    )
    optimizer.tell(initial)
    seen = {point.key for point in initial_points}
    candidates = optimizer.ask(4, seen)
    infeasible = tuple(
        assess(
            make_problem(),
            index + 4,
            point,
            {"loss": -100.0, "score": 200.0, "quality": 0.349, "window": 1.0},
        )
        for index, point in enumerate(candidates)
    )
    optimizer.tell(infeasible)
    if isinstance(optimizer, ParticleSwarmOptimizer):
        assert all(particle.best.feasible for particle in optimizer.particles)
    else:
        assert all(trial.feasible for trial in optimizer.population)


@pytest.mark.parametrize(
    "constructor,kwargs,message",
    [
        (ParticleSwarmOptimizer, {"population_size": 3}, "at least 4"),
        (ParticleSwarmOptimizer, {"inertia": -1.0}, "coefficients"),
        (DifferentialEvolutionOptimizer, {"population_size": 3}, "at least 4"),
        (DifferentialEvolutionOptimizer, {"differential_weight": 0.0}, "weight"),
        (DifferentialEvolutionOptimizer, {"crossover_rate": 2.0}, "rate"),
        (NSGA2Optimizer, {"population_size": 3}, "at least 4"),
        (NSGA2Optimizer, {"crossover_probability": 2.0}, "probability"),
        (NSGA2Optimizer, {"crossover_eta": 0.0}, "crossover_eta"),
        (NSGA2Optimizer, {"mutation_eta": math.inf}, "mutation_eta"),
        (MOEADOptimizer, {"population_size": 3}, "at least 4"),
        (MOEADOptimizer, {"neighborhood_size": 0}, "neighborhood"),
        (MOEADOptimizer, {"population_size": 5, "neighborhood_size": 3}, "neighborhood"),
        (MOEADOptimizer, {"differential_weight": 0.0}, "weight"),
        (MOEADOptimizer, {"crossover_rate": -1.0}, "rate"),
    ],
)
def test_strategy_parameters_are_strictly_validated(constructor, kwargs, message) -> None:
    with pytest.raises(ConfigurationError, match=message):
        constructor(make_problem(), 0, **kwargs)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"initial_temperature": 0.0}, "initial_temperature"),
        ({"cooling": 1.0}, "cooling"),
        ({"minimum_temperature": math.nan}, "minimum_temperature"),
    ],
)
def test_annealing_parameters_are_validated(kwargs, message) -> None:
    with pytest.raises(ConfigurationError, match=message):
        SimulatedAnnealingOptimizer(make_problem(), 0, **kwargs)
