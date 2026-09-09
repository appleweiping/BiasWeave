"""Typed construction for BiasWeave's independently implemented strategies."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from biasweave.errors import ConfigurationError
from biasweave.model import Problem
from biasweave.optimizers.annealing import SimulatedAnnealingOptimizer
from biasweave.optimizers.base import AskTellOptimizer
from biasweave.optimizers.differential_evolution import DifferentialEvolutionOptimizer
from biasweave.optimizers.moead import MOEADOptimizer
from biasweave.optimizers.nsga2 import NSGA2Optimizer
from biasweave.optimizers.pso import ParticleSwarmOptimizer
from biasweave.optimizers.random_search import RandomOptimizer
from biasweave.optimizers.weave import WeaveOptimizer


class StrategyName(StrEnum):
    """Stable strategy identifiers accepted by the API and CLI."""

    WEAVE = "weave"
    RANDOM = "random"
    SA = "sa"
    PSO = "pso"
    DE = "de"
    NSGA2 = "nsga2"
    MOEAD = "moead"


def parse_strategy(value: StrategyName | str) -> StrategyName:
    """Parse a strategy while returning a package-level configuration error."""
    try:
        return StrategyName(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(item.value for item in StrategyName)
        raise ConfigurationError(f"strategy must be one of: {choices}") from error


def create_optimizer(
    problem: Problem,
    strategy: StrategyName | str,
    *,
    seed: int,
    population_size: int | None = None,
) -> AskTellOptimizer:
    """Create one concrete strategy without silently substituting another."""
    selected = parse_strategy(strategy)
    if selected is StrategyName.WEAVE:
        if population_size is not None:
            raise ConfigurationError("population_size does not apply to weave")
        return WeaveOptimizer(problem, seed)
    if selected is StrategyName.RANDOM:
        if population_size is not None:
            raise ConfigurationError("population_size does not apply to random")
        return RandomOptimizer(problem, seed)
    if selected is StrategyName.SA:
        if population_size is not None:
            raise ConfigurationError("population_size does not apply to sa")
        return SimulatedAnnealingOptimizer(problem, seed)

    size = population_size if population_size is not None else 16
    if selected is StrategyName.PSO:
        return ParticleSwarmOptimizer(problem, seed, population_size=size)
    if selected is StrategyName.DE:
        return DifferentialEvolutionOptimizer(problem, seed, population_size=size)
    if selected is StrategyName.NSGA2:
        return NSGA2Optimizer(problem, seed, population_size=size)
    if selected is StrategyName.MOEAD:
        return MOEADOptimizer(problem, seed, population_size=size)
    raise AssertionError("unhandled optimizer strategy")


def optimizer_parameters(
    strategy: StrategyName | str, optimizer: AskTellOptimizer
) -> dict[str, Any]:
    """Return the complete effective public hyperparameter set before a run starts."""

    selected = parse_strategy(strategy)
    if selected in (StrategyName.WEAVE, StrategyName.RANDOM):
        return {}
    if selected is StrategyName.SA:
        candidate = optimizer
        if not isinstance(candidate, SimulatedAnnealingOptimizer):
            raise ConfigurationError("optimizer instance does not match sa")
        return {
            "initial_temperature": candidate.temperature,
            "cooling": candidate.cooling,
            "minimum_temperature": candidate.minimum_temperature,
        }
    if selected is StrategyName.PSO:
        candidate = optimizer
        if not isinstance(candidate, ParticleSwarmOptimizer):
            raise ConfigurationError("optimizer instance does not match pso")
        return {
            "population_size": candidate.population_size,
            "inertia": candidate.inertia,
            "cognitive": candidate.cognitive,
            "social": candidate.social,
        }
    if selected is StrategyName.DE:
        candidate = optimizer
        if not isinstance(candidate, DifferentialEvolutionOptimizer):
            raise ConfigurationError("optimizer instance does not match de")
        return {
            "population_size": candidate.population_size,
            "differential_weight": candidate.differential_weight,
            "crossover_rate": candidate.crossover_rate,
        }
    if selected is StrategyName.NSGA2:
        candidate = optimizer
        if not isinstance(candidate, NSGA2Optimizer):
            raise ConfigurationError("optimizer instance does not match nsga2")
        return {
            "population_size": candidate.population_size,
            "crossover_probability": candidate.crossover_probability,
            "crossover_eta": candidate.crossover_eta,
            "mutation_eta": candidate.mutation_eta,
        }
    candidate = optimizer
    if not isinstance(candidate, MOEADOptimizer):
        raise ConfigurationError("optimizer instance does not match moead")
    return {
        "population_size": candidate.population_size,
        "neighborhood_size": candidate.neighborhood_size,
        "differential_weight": candidate.differential_weight,
        "crossover_rate": candidate.crossover_rate,
    }
