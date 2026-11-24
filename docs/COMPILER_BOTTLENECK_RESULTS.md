# QARW — Milestone 6: Compiler Bottleneck Profiling Results

## 1. Overview

Milestone 6 extends the existing Qubernetes runtime compiler profiler with detailed, per-pass resource tracking, compiler bottleneck analysis, performance visualizations, and automatic MLflow experiment reporting.

The objective is to identify which Qiskit compiler passes consume the most execution time and how different compilation stages affect quantum circuit resources.

The implementation builds upon the modular arithmetic benchmarks and hardware-aware compilation infrastructure developed in previous milestones.

## 2. Implemented Features

**Per-pass resource tracking**

The compiler profiler now records changes between consecutive compilation passes, including circuit depth, total operation count, two-qubit gate count, two-qubit depth, and individual gate counts (CX, CZ, ECR, etc.).

**Compiler bottleneck analysis**

The `compiler_analysis.py` module identifies the slowest recorded compiler passes, summarizes accumulated pass execution time by compiler stage, and ranks passes associated with the largest resource increases and reductions.

**Performance visualizations**

The `compiler_graphs.py` module generates four visualizations:

- `slowest_passes.png`
- `stage_runtime.png`
- `depth_changes.png`
- `two_qubit_changes.png`

**MLflow integration**

Compiler performance summaries and visualizations are automatically uploaded as experiment artifacts under the `compiler_bottlenecks/` directory. The structured report is saved as `summary.json`.

## 3. Experimental Methodology

The primary benchmark uses the baseline modulo-17 quantum adder.

| Parameter | Configuration |
|---|---|
| Quantum circuit | Baseline modulo-17 adder |
| Logical qubits | 10 |
| Optimization level | 3 |
| Gate bases | CX, CZ, ECR |
| Topologies | Unrestricted, Linear-16 |
| Seeds | 42, 101, 202, 303, 404 |
| Single-qubit basis | U |
| Routing and layout | SABRE for constrained topologies |
| Initial-qubit assumption | `qubits_initially_zero=False` |
| Total configurations | 6 |
| Total benchmark runs | 30 |

These hardware topologies are abstract compilation targets rather than calibrated physical quantum devices.

## 4. Multi-Seed Benchmark Results

The following table summarizes the arithmetic means across five transpiler seeds.

| Topology | Gate basis | Mean depth | Mean 2Q gates | Mean recorded pass time (s) |
|---|---|---:|---:|---:|
| Unrestricted | CX | 3375.0 | 1824.0 | 0.3052 |
| Unrestricted | CZ | 3375.0 | 1824.0 | 0.4033 |
| Unrestricted | ECR | 3519.0 | 1824.0 | 0.3020 |
| Linear-16 | CX | 6028.0 | 5100.2 | 1.8889 |
| Linear-16 | CZ | 7876.8 | 5100.2 | 2.7948 |
| Linear-16 | ECR | 8280.2 | 5100.2 | 3.1928 |

The complete individual-run results are available in `benchmark_outputs/compiler_bottlenecks/multiseed_baseline/compiler_comparison.csv` and the associated MLflow experiment.

## 5. Compiler Bottleneck Findings

**Unrestricted topology**

Across all 15 unrestricted compilation runs, `MinimumPoint` was identified as the slowest recorded compiler pass.

The final circuit depth and two-qubit gate count remained identical across all five seeds within each gate-basis configuration.

**Linear-16 topology**

Across all 15 linear-connectivity compilation runs, `SabreLayout` was identified as the slowest recorded compiler pass.

This demonstrates that connectivity restrictions can shift the dominant compilation workload toward hardware-aware layout selection.

**Resource overhead**

For the CX gate basis, mean circuit depth increased from 3375 to 6028 when switching from unrestricted connectivity to Linear-16, approximately a 78.6% increase.

Mean two-qubit gate count increased from 1824 to 5100.2, approximately a 179.6% increase.

For Linear-16, the CX basis produced the lowest mean circuit depth among the three tested bases.

The two-qubit counts were identical across CX, CZ, and ECR for matching seeds and topologies, although the resulting circuit depths differed.

## 6. Functional Correctness Validation

The real-circuit profiling experiment successfully validated sampled functional equivalence for the unrestricted baseline modulo-17 adder.

Cross-topology benchmarking also performed sampled computational-basis correctness checks and connectivity validation for mapped circuits.

A separate coherence-enabled smoke test passed for the baseline modulo-17 adder using Linear-16 connectivity, the CX basis, optimization level 3, and seed 42.

These checks provide evidence of correct operation for the tested inputs and states. They do not constitute exhaustive equivalence verification across all possible quantum states or hardware configurations.

## 7. Unit Testing and Integration Validation

The available test suite completed with:

**38 passed, 8 subtests passed.**

The suite was run with `tests/test_qrisp_autologging.py` excluded because the optional Qrisp dependency was unavailable in the current environment.

Additional validation confirmed that real compiler-pass records, compiler analysis summaries, and all five expected bottleneck reporting artifacts were successfully uploaded to MLflow.

## 8. Experimental Limitations

Pass durations represent recorded Qiskit pass execution times, not necessarily the complete transpilation wall-clock duration.

Instrumentation, callback processing, resource analysis, Python runtime behavior, and system load can influence measured performance.

Multiple transpiler seeds allow us to evaluate sensitivity to randomized compilation decisions, but do not replace repeated timing measurements under controlled conditions.

Resource changes are measured between consecutive recorded circuit snapshots and should not automatically be interpreted as direct causal effects of individual compiler passes.

The 30-run comparison covers the baseline architecture only. Broader comparisons involving the optimized RCCX architecture, additional topologies, and independent repeated trials remain possible extensions.

## 9. Reproducing the Experiments

From the `runtime` directory, activate the Python virtual environment and start the local MLflow server using the existing experiment store.

Run the baseline comparison with:

`python benchmark_compiler_bottlenecks.py --architectures baseline --topologies unrestricted linear_16 --bases CX CZ ECR --seeds 42 101 202 303 404 --output-dir benchmark_outputs/compiler_bottlenecks/multiseed_baseline`

Run the coherence validation with:

`python benchmark_compiler_bottlenecks.py --architectures baseline --topologies linear_16 --bases CX --seeds 42 --check-coherence --output-dir benchmark_outputs/compiler_bottlenecks/coherence_smoke`

Run the available regression tests with:

`python -m pytest -q --ignore=tests/test_qrisp_autologging.py`

The results are recorded in the MLflow experiment `QARW-Compiler-Bottleneck-Comparisons`.

## 10. Conclusion

Milestone 6 successfully extends the QARW compiler profiler from collecting compilation provenance to automatically analyzing resource changes and identifying compilation bottlenecks.

The experiments demonstrate measurable differences in compiler behavior across gate bases and connectivity constraints.

The implementation provides a reusable foundation for studying compiler optimization, hardware-aware quantum arithmetic, and future cryptographic quantum circuits.