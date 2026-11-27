# Cryptographic Arithmetic Resource Profiling

## Milestone 8.1 and 8.2.1

### Objective

Extend QARW's quantum arithmetic profiling infrastructure toward elliptic-curve cryptography by implementing a classical ECC reference and reversible quantum modular subtraction.

### Classical ECC Reference

The reference implementation uses the finite field F17 and the elliptic curve:

    y² = x³ + 2x + 2 mod 17

- Generator: (5, 1)
- Public key: (0, 6)
- Curve order: 19
- Reference scalar: 7

The implementation supports modular addition, subtraction, multiplication, squaring, inversion, elliptic-curve point addition, point doubling and scalar multiplication.

Nine automated tests validate the finite-field arithmetic and elliptic-curve group properties.

### Quantum Modular Subtraction

Reversible modular subtraction is implemented by inverting the existing baseline and optimized RCCX modular addition circuits.

The implementation preserves the first arithmetic register and subtracts its value from the second register modulo 17.

Both architectures were validated across all 1,024 data-register basis states with clean ancillas.

Validation covers arithmetic correctness, reversible extensions, ancilla cleanup, relative-phase preservation, inverse round trips, superposition correctness and architecture equivalence.

### Compiler Benchmark Configuration

- Modulus: 17
- Register width: 5 qubits
- Gate basis: U/CX
- Connectivity: unrestricted
- Optimization levels: 0 and 3
- Transpiler seeds: 42, 101, 202, 303, 404
- Total benchmark runs: 40

### Optimization Level 3 Results

| Operation | Architecture | Depth | CX count |
|---|---|---:|---:|
| Addition | Baseline | 3307 | 1794 |
| Addition | Optimized RCCX | 1664 | 1050 |
| Subtraction | Baseline | 3035 | 1764 |
| Subtraction | Optimized RCCX | 1645 | 1050 |

The optimized subtraction architecture reduces compiled depth by 45.80% and CX count by 40.48% relative to the baseline subtraction circuit.

All five seeds produced identical depth and CX counts within each benchmark configuration.

### Fault-Tolerant Logical Resources

Addition and subtraction produce identical logical resource estimates under matched inverse-decomposition policies.

| Architecture | MCX policy | Logical qubits | T-count | T-depth |
|---|---|---:|---:|---:|
| Baseline | CCX chain | 13 | 2576 | 1328 |
| Baseline | RCCX chain | 13 | 1712 | 1057 |
| Optimized RCCX | CCX chain | 13 | 1712 | 1057 |
| Optimized RCCX | RCCX chain | 13 | 1712 | 1057 |

RCCX-chain decomposition reduces T-count by 33.54% and T-depth by 20.41% relative to baseline CCX-chain decomposition.

The estimator uses exact recursive gate definitions. T-depth represents ordered T/Tdg layers rather than a globally optimized minimum.

### Key Findings

- Inverse arithmetic circuits can have different compiled resource requirements.
- Matched inverse decompositions preserve logical resource counts in the evaluated configurations.
- RCCX decomposition provides substantial reductions in compiled and logical resource requirements.
- All tested architectures preserve quantum coherence and restore clean ancillas.
- Modular subtraction provides a validated arithmetic primitive for future quantum ECC workloads.

### MLflow Integration

Experiment: QARW-Cryptographic-Arithmetic

Run: mod17_addition_subtraction_comparison

Compiler measurements, logical resource estimates and five comparison visualizations are recorded as MLflow artifacts.

### Regression Validation

- 81 passing tests
- 28 passing subtests
- 14 passing subtraction-specific tests
- No observed regression failures in the executed suite

The optional Qrisp integration test is excluded because Qrisp is not installed in the current environment.

### Limitations

The current experiments use a toy finite field and unrestricted compiler connectivity. Physical surface-code resource estimates and hardware-aware comparisons are not included in this stage.

The results do not directly represent the resource requirements of real-world 256-bit elliptic-curve cryptography.

### Next Step

Implement reversible modular multiplication, followed by modular squaring, inversion and elliptic-curve point operations.