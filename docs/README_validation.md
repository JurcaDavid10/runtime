# QARW — compiled-circuit functional correctness

Copy `validate_hardware_aware_adders.py` alongside these existing files:
`benchmark_hardware_aware.py`, `modular_arithmetic.py`, and
`modular_arithmetic_optimized.py`.

Use the same activated `.venv` as the hardware-aware benchmark.

## Step 1 — fast smoke test

```powershell
python validate_hardware_aware_adders.py --quick --seeds 42 --topologies unrestricted linear_16
```

## Step 2 — full selected-input test for all topologies

```powershell
python validate_hardware_aware_adders.py --seeds 42
```

## Step 3 — optional multiple-seed follow-up

```powershell
python validate_hardware_aware_adders.py --quick --seeds 42 101 202 303 404
```

The tool tests the compiled **u/cx** circuits, respects initial/final
transpilation layouts, and validates selected computational basis inputs,
clean ancillas, out-of-range `b` behavior, and one coherent superposition.
It saves `compiled_correctness.csv` and `compiled_correctness_summary.md`
under `benchmark_outputs/hardware_aware/validation/` and logs to an
MLflow experiment when MLflow is available. It is not an exhaustive
proof or a hardware execution.

**Git:** `benchmark_outputs/` is ignored. If you want selected figures,
CSVs and markdown artifacts in Git, use `git add -f` on those exact files.
Review staged files with `git diff --cached --stat` before committing.
