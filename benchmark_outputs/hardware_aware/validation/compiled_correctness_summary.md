# Hardware-aware compiled-circuit correctness

Exact statevector tests on the *compiled* modulo-17 adders, with the
initial physical-qubit layout and final routing permutation respected.

- Seeds: `42`
- Gate basis: `u, cx`; optimization level: 3
- Basis cases per circuit: **8**
- One coherent two-state superposition per compiled circuit
- Clean work qubits and unchanged `a` are included in the expected output
- Out-of-range `b` values must remain unchanged
- `PASS` means tested output-basis probability and coherent-state fidelity
  both reached at least `1 - tolerance`
- This is **not** an exhaustive unitary equivalence proof, a device run,
  or a noise/fidelity estimate for physical hardware

| Architecture | Topology | Seed | Basis tests | Superposition fidelity | Result |
|---|---|---:|---:|---:|---|
| baseline | unrestricted | 42 | 8 | 1.000000000 | PASS |
| baseline | linear_16 | 42 | 8 | 1.000000000 | PASS |
| baseline | grid_4x4 | 42 | 8 | 1.000000000 | PASS |
| baseline | heavy_hex_19 | 42 | 8 | 1.000000000 | PASS |
| optimized_rccx | unrestricted | 42 | 8 | 1.000000000 | PASS |
| optimized_rccx | linear_16 | 42 | 8 | 1.000000000 | PASS |
| optimized_rccx | grid_4x4 | 42 | 8 | 1.000000000 | PASS |
| optimized_rccx | heavy_hex_19 | 42 | 8 | 1.000000000 | PASS |
