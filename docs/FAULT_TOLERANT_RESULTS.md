# QARW — Milestone 7: Fault-tolerant resource estimation

## Scope and measured logical results

Modulo-17 arithmetic benchmark, Qiskit 2.1.2. Exact recursive Clifford+T resource accounting, with explicit clean-ancilla MCX lowering. T-depth is dependency-aware, not optimized over equivalent circuits. One ideal logical T state per T/Tdg.

| Architecture | MCX policy | Source | Scratch | Logical | T-count | T-depth | Clifford |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | CCX chain | 10 | 3 | 13 | 2576 | 1328 | 3212 |
| Optimized RCCX | CCX chain | 13 | 0 | 13 | 1712 | 1057 | 2348 |
| Baseline | RCCX chain | 10 | 3 | 13 | 1712 | 1057 | 2348 |
| Optimized RCCX | RCCX chain | 13 | 0 | 13 | 1712 | 1057 | 2348 |

Relative to the baseline CCX chain, T-count decreases 33.54%, T-depth 20.41%, Clifford count 26.90%. Relative to the RCCX-synthesized baseline, manual optimization has no further measured logical-resource benefit. The CCX-chain policy preserves existing RCCX operations in the optimized source.

Phase-aware numerical validation: six configurations, 1024 five-bit input pairs each (6144 checks); 289 valid arithmetic pairs per configuration. Clean ancillas restored, reversible mapping and equivalence on tested clean-ancilla subspace up to global phase. Maximum relative phase error 3.09e-14. This does not claim an arbitrary dirty-ancilla guarantee. Input consistency checks are not circuit-version provenance proofs.

Final regression results: 58 tests and 28 subtests passed. The Qrisp-specific module was excluded due to a missing optional dependency.
All 36 surface-code scenarios satisfied the modeled failure budget. Both logical and physical resource benchmarks were successfully published to the local MLflow experiment `QARW-Fault-Tolerant-Resources`.

## Conditional physical model

`surface_code_resources.py` implements an explicitly illustrative, serial surface-code scenario. `benchmark_surface_code.py` sweeps four measured logical configurations, three physical error rates (1e-4, 1e-3, 3e-3) and three requested factory counts (1, 4, 8). Input phase-validation and resource records must pass the existing reporting consistency checks.

Default assumptions: 1 microsecond per syndrome cycle, failure budget 0.01, threshold 0.01, prefactor 0.1, odd code distances 3 through 99. The rotated patch uses `2*d*d-1` physical qubits including check qubits. The phenomenological patch-cycle error fit is `pL = 0.1*(p/0.01)**((d+1)/2)`. These are approximate modeling assumptions, not guaranteed device thresholds or error probabilities.

For logical width `Q`, T-count `T`, Clifford count `C`, requested factory count `F`, active count `f=min(F,T)` and distance `d`:

- Data patches: `2*Q` by default (one extra routing/work patch per logical qubit, not a verified layout).
- Factory patches: `11*f`; buffer patches: `f`. The default generic factory envelope uses 11 patches and 11 distance-scaled steps per accepted output. It is inspired by the size/time of a 15-to-1 example, not an implemented distillation circuit or fidelity/yield calculation.
- Batches: `ceil(T/f)` when T is nonzero; zero otherwise. Assume each active factory produces one accepted state per batch. Buffer capacity is reserved; a batch is consumed before the next batch is produced. All gate and factory times are serial with no overlap, irrespective of T-depth.
- Gate cycles: `(T+C)*d`; factory cycles: `ceil(T/f)*11*d`. Every primitive gate is assigned d cycles, an illustrative uniform cost assumption, not a lattice-surgery schedule.
- Runtime: total cycles times cycle duration. The idealized T-layer time `T_depth*d*cycle_duration` is also exported as a separate reference, not a total runtime estimate.
- All reserved patches accrue the patch-cycle error term. Combined union estimate: `patches*total_cycles*pL + T*p_magic`, with externally assumed residual error per accepted T state `p_magic=1e-7` (excluding modeled patch faults). This is a union bound only conditional on the assumed per-location bounds; the fit itself is not a rigorous bound. No independence is needed for the union sum. The smallest modeled odd distance meeting the budget is selected. If no such distance exists or the residual error exhausts the budget, the scenario is marked infeasible.

Physical estimates exclude validated routing layouts, correlated noise, decoder throughput, factory rejection/retries, raw-state injection overhead, transport details, logical gate-specific error rates, factory-specific code distances and hardware control. Extra work and buffer patches are schematic. No claim is made that the generic factory produces states with the assumed error at the given physical error rate. The runtime is not a rigorous upper or lower bound for real hardware. Physical estimates are conditional scenarios, not measurements, forecasts or an estimate of an actual cryptocurrency attack.

## Reproduction

Run from the runtime directory in the project's activated virtual environment:

```powershell
python benchmark_fault_tolerant.py
python validate_fault_tolerant.py
python report_fault_tolerant.py --log-mlflow
python benchmark_surface_code.py
python benchmark_surface_code.py --log-mlflow
python -m pytest -q --ignore=tests/test_qrisp_autologging.py
```

The physical benchmark exports CSV, JSON, Markdown and three plots under `benchmark_outputs/fault_tolerant/physical/`. The optional MLflow run is separate from the logical benchmark, with a parent run and child runs for feasible scenarios. The local physical benchmark, MLflow integration and final regression suite were successfully verified on November 27, 2026. All 36 modeled scenarios satisfied the failure budget, and 58 tests and 28 subtests passed. Milestone 7 is complete.

## References

- A. G. Fowler et al., *Surface codes: Towards practical large-scale quantum computation*, Phys. Rev. A 86, 032324 (2012), https://doi.org/10.1103/PhysRevA.86.032324.
- D. Litinski, *A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery*, Quantum 3, 128 (2019), https://doi.org/10.22331/q-2019-03-05-128.
- IBM Quantum Learning, Error correction II, https://quantum.cloud.ibm.com/learning/en/courses/tools-for-quantum-advantage/error-correction-ii.
- Qualtran, Architecture-dependent physical costs, https://qualtran.readthedocs.io/en/latest/surface_code/physical_cost_model.html.
