# QARW — Modulo-17 quantum adder: optimization comparison

Compiled with basis gates `u,cx`, idealized all-to-all connectivity; the same Qiskit version and transpiler seed for all levels.

| Metric | Level 0 | Level 1 | Level 2 | Level 3 | Change L0 → L3 |
|---|---:|---:|---:|---:|---|
| Compiled circuit depth | 3,809 | 3,311 | 3,307 | 3,307 | 13.18% reduction |
| Compiled total operations | 4,865 | 3,986 | 3,898 | 3,898 | 19.88% reduction |
| CX (two-qubit) gate count | 1,794 | 1,794 | 1,794 | 1,794 | 0.0% |
| Two-qubit circuit depth | 1,725 | 1,725 | 1,725 | 1,725 | 0.0% |
| Single-qubit U gate count | 3,071 | 2,192 | 2,104 | 2,104 | 31.49% reduction |
| Compilation wall time (seconds) | 0.407 | 1.686 | 1.848 | 1.819 | 346.51% increase |

## Key observations

- Circuit depth: 13.18% reduction from level 0 to level 3.
- Total operations: 19.88% reduction.
- CX gate count: 0.0%.
- Two-qubit depth: 0.0%.
- Single-qubit U gates: 31.49% reduction.
- Levels 2 and 3 have identical measured structural resource metrics in these runs (an observed plateau).

## Interpretation and limitations

- The source circuit contains high-level multi-controlled gates; its depth/operation count should not be compared directly with the compiled `u,cx` circuit as if the gate primitives were equivalent.
- Each level's compilation time is one measured run, not a mean. Repeat runs before drawing statistical timing conclusions.
- These are compiler resource estimates, not hardware performance or hardware-aware routing results.
- The benchmark itself transpiles the circuit; functional correctness was tested separately.

## Experimental settings

- Workload: `modular_add_17`
- Qiskit version: `2.1.2`
- Seed: `42`
- Source qubits: `10`
- Compiled qubits: `10`
