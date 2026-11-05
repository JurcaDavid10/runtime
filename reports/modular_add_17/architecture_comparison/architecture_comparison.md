# Modulo-17 adder architecture comparison

These are actual benchmark measurements at a common transpiler seed, basis gates (`u,cx`), and idealized connectivity.

| Level | Baseline CX | RCCX + ancilla CX | CX reduction | Baseline qubits | New qubits |
|---:|---:|---:|---:|---:|---:|
| 0 | 1794 | 1344 | 25.08% | 10 | 13 |
| 1 | 1794 | 1344 | 25.08% | 10 | 13 |
| 2 | 1794 | 1050 | 41.47% | 10 | 13 |
| 3 | 1794 | 1050 | 41.47% | 10 | 13 |

Higher qubit usage may be exchanged for lower two-qubit gate cost.
A negative CX reduction indicates regression, not improvement.
Compilation times are single-run observations, not statistical estimates.
