"""BiasWeave public API."""

from biasweave._version import __version__
from biasweave.benchmark import (
    compare_optimizer_catalog,
    compare_with_random,
    load_analog_benchmark,
    sizing_decision,
)
from biasweave.engine import optimize
from biasweave.evaluator import CommandEvaluator, load_python_evaluator
from biasweave.optimizers import AskTellOptimizer, BayesianOptimizer, StrategyName, create_optimizer
from biasweave.problem import load_problem
from biasweave.strategy import optimize_strategy

__all__ = [
    "AskTellOptimizer",
    "BayesianOptimizer",
    "CommandEvaluator",
    "StrategyName",
    "compare_optimizer_catalog",
    "compare_with_random",
    "create_optimizer",
    "load_analog_benchmark",
    "load_problem",
    "load_python_evaluator",
    "optimize",
    "optimize_strategy",
    "sizing_decision",
    "__version__",
]
