"""Expected errors exposed by BiasWeave."""


class BiasWeaveError(Exception):
    """Base class for failures that callers may handle."""


class ProblemError(BiasWeaveError):
    """The optimization problem is invalid."""


class EvaluationError(BiasWeaveError):
    """An evaluator could not produce a valid metric mapping."""


class CheckpointError(BiasWeaveError):
    """A checkpoint is corrupt or incompatible with the requested run."""


class ConfigurationError(BiasWeaveError):
    """Run configuration is inconsistent or outside supported bounds."""
