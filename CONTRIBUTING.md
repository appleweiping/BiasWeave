# Development guide

Contributions should improve a measurable optimizer property, input diagnostic,
or reproducibility guarantee. Start by describing the expected behavior and a
small synthetic problem. Keep evaluator fixtures independent of commercial
simulators and technology files.

Before submitting a change, run:

```bash
ruff check .
ruff format --check .
mypy --python-version 3.11 src
bandit -q -r src
pytest --cov=biasweave --cov-report=term-missing
python -m build
twine check --strict dist/*
```

Run `actionlint` when a workflow changes; CI applies its bundled ShellCheck rules
to every workflow shell block as well as validating the YAML expressions.

Algorithm changes need deterministic tests with a fixed seed, a statement of
budget impact, and a changelog entry. Never replace a failed evaluation with a
fabricated objective value. Declare significant automated assistance and review
every generated change.

Every pull-request commit must carry an author-matching Developer Certificate of
Origin trailer. Use `git commit -s`. The `pull_request_target` DCO workflow runs
only the verifier from the trusted base revision, requires the author-matching
sign-off inside a syntactically valid final trailer block, and never executes
pull-request code. Its bounded JSON reader uses only the standard library;
the workflow runs `python -I -S` to disable site-packages, and tests repeat that
invocation so an editable install cannot mask a missing dependency. It marks the event head
pending, handles PR base edits, and
binds the REST commit list to the event and final base repository/ref/SHA, head,
and declared count before publishing success.
Protected `main` requires the `DCO / commits` status from GitHub Actions in
addition to its existing platform tests, lint and CodeQL checks. A passing
ordinary test workflow does not replace this trusted-base sign-off check.
Release tags are annotated SSH-signed tags and are verified against the signer
policy on `main` before tag source or dependencies are executed.
