# Development guide

Contributions should improve a measurable optimizer property, input diagnostic,
or reproducibility guarantee. Start by describing the expected behavior and a
small synthetic problem. Keep evaluator fixtures independent of commercial
simulators and technology files.

Before submitting a change, run:

```bash
ruff check .
ruff format --check .
pytest --cov=biasweave --cov-report=term-missing
python -m build
```

Algorithm changes need deterministic tests with a fixed seed, a statement of
budget impact, and a changelog entry. Never replace a failed evaluation with a
fabricated objective value. Declare significant automated assistance and review
every generated change.
