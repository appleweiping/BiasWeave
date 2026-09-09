"""Ask/tell adapter for BiasWeave's original coverage-and-repair strategy."""

from __future__ import annotations

from biasweave.archive import Archive
from biasweave.model import Point, Problem, Trial
from biasweave.optimizers.base import AskTellOptimizer
from biasweave.proposal import ProposalGenerator


class WeaveOptimizer(AskTellOptimizer):
    """Expose the native weave proposal stream through the catalog contract."""

    name = "weave"

    def __init__(self, problem: Problem, seed: int):
        super().__init__(problem, seed)
        self.generator = ProposalGenerator(problem, seed)
        self.archive = Archive(problem)
        self.trials: list[Trial] = []
        self._signature: tuple[int, ...] = ()

    def _ask(self, count: int, seen_keys: set[str]) -> tuple[Point, ...]:
        self._signature = self.archive.signature
        points = list(
            self.generator.propose(
                count,
                self.archive,
                tuple(self.trials),
                seen_keys,
            )
        )
        if len(points) < count:
            self._empty_ask_reason = self.generator.empty_reason
        return tuple(points)

    def _tell(self, trials: tuple[Trial, ...]) -> None:
        self.trials.extend(trials)
        for trial in trials:
            self.archive.add(trial)
        self.generator.observe(self.archive.signature != self._signature)
