# Synthetic two-stage OTA example

This example exercises mixed real, integer, choice, and linked variables. The bundled evaluator is a deterministic
analytic teaching surrogate; it does not represent a foundry process and must not be used for tape-out decisions.

Run it from the repository root:

```console
biasweave validate --problem examples/two_stage_ota/problem.toml
biasweave run --problem examples/two_stage_ota/problem.toml \
  --evaluator python:biasweave.demo:evaluate --budget 80 --workers 4 \
  --out artifacts/ota
```

All quantities in the problem and evaluator use SI units. The optimizer attaches no implicit units to names or values.
