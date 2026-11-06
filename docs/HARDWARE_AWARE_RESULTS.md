# Hardware-aware modulo-17 adder benchmarking — findings

## Experiment

Two modulo-17 quantum adders were compiled to a `u`/`cx` basis using Qiskit optimization level 3.

The experiment compares three bidirectional connectivity topologies:

- Linear topology (16 physical qubits)
- 4×4 grid topology (16 physical qubits)
- Heavy-hex topology (19 physical qubits)

SABRE was used for qubit placement and routing. Unrestricted compilation serves as the reference.

The experiments use five reproducible transpiler seeds: **42, 101, 202, 303, 404**.

Both architectures were compiled with:

```python
qubits_initially_zero=False
```

This setting is important because the quantum arithmetic circuits operate on arbitrary input states, rather than assuming that all input qubits are initialized to zero.

All performance results represent **means across five seeds**. These experiments measure simulated compilation resources, not actual quantum hardware execution.

## Performance results

| Topology | Baseline depth | RCCX depth | Depth reduction | Baseline CX | RCCX CX | CX reduction |
|---|---:|---:|---:|---:|---:|---:|
| Unrestricted | 3375.0 | 1664.0 | 50.7% | 1824.0 | 1050.0 | 42.4% |
| Linear (16) | 6028.0 | 3297.4 | 45.3% | 5100.2 | 3088.2 | 39.4% |
| Grid 4×4 | 4643.4 | 2709.6 | 41.6% | 3314.4 | 2305.0 | 30.5% |
| Heavy-hex (19) | 5577.2 | 3051.8 | 45.3% | 4576.6 | 2532.4 | 44.7% |

The optimized RCCX architecture achieves lower circuit depth and fewer CX operations across all tested topologies.

However, this improvement requires additional logical qubits:

- Baseline architecture: **10 logical qubits**
- Optimized RCCX architecture: **13 logical qubits**

This demonstrates a trade-off between quantum gate efficiency and qubit requirements.

## Net CX increase versus unrestricted compilation

| Topology | Baseline | Optimized RCCX |
|---|---:|---:|
| Linear (16) | +3276.2 | +2038.2 |
| Grid 4×4 | +1490.4 | +1255.0 |
| Heavy-hex (19) | +2752.6 | +1482.4 |

These values represent **net differences in final CX gate counts**, not exact SWAP counts or isolated routing costs.

Qubit placement, routing, and circuit optimization all contribute to the differences.

The optimized RCCX architecture shows lower absolute CX increases than the baseline under every tested connectivity restriction.

## Qubit resources

| Topology | Device capacity | Baseline physical qubits touched (mean) | RCCX physical qubits touched (mean) |
|---|---:|---:|---:|
| Linear (16) | 16 | 10.0 | 13.0 |
| Grid 4×4 | 16 | 11.4 | 14.0 |
| Heavy-hex (19) | 19 | 12.8 | 13.6 |

Four separate resource concepts are distinguished:

- **Logical qubits:** Source circuit's required qubit count.
- **Device capacity:** Total available physical qubits in the selected connectivity graph.
- **Physical qubits touched:** Distinct physical qubit positions involved in compiled operations.
- **Compiled circuit width:** Total wires present in the compiled circuit, potentially including idle wires.

Physical qubit usage and device capacity are not applicable to unrestricted compilation.

The reported standard deviations help identify how physical qubit usage changes across transpilation seeds.

## Functional correctness validation

**Status: PASSED for all tested compiled configurations.**

Functional validation was performed using `validate_hardware_aware_adders.py`, with `qubits_initially_zero=False`.

The tests verified selected computational-basis inputs, including arithmetic behavior, input preservation, clean work registers, and coherence preservation.

### Validation results

| Test | Result |
|---|---|
| Compiled configurations | 8/8 passed |
| Computational-basis test cases | 64/64 passed |
| Coherence tests | 8/8 passed |
| Reported state fidelity | 1.000000000 |
| Connectivity models | Unrestricted, linear, grid, heavy-hex |
| Tested seed | 42 |
| MLflow experiment tracking | Successful |

Both architectures passed all selected-input checks after compilation.

### Compilation issue discovered and resolved

During validation, the baseline circuit initially produced an incorrect result after transpilation:

```text
Input: a=16, b=1
Expected: (16, 0)
Observed: (16, 17)
```

The issue was traced to the default `qubits_initially_zero=True` assumption used during transpilation.

Changing this setting to:

```python
qubits_initially_zero=False
```

restored the correct result across optimization levels 0, 1, and 3 in the diagnostic test.

The original baseline architecture was restored, and both benchmark and validation scripts were updated to use consistent compilation settings.

Following this correction, all selected-input functional validation tests passed.

This highlights the importance of validating compiled circuits rather than relying exclusively on source-circuit correctness.

## Limitations

- These experiments evaluate compiler resource requirements, not physical quantum hardware performance.
- No hardware noise, calibration data, gate errors, or real QPU execution times are included.
- Heavy-hex uses 19 physical qubits, while linear and grid topologies use 16, so this is not a fixed-capacity topology comparison.
- Multi-seed analysis evaluates transpilation variability but does not independently prove circuit equivalence.
- Compiled functional validation used eight representative input pairs and one coherence test per configuration, with seed 42.
- The optimized source circuit additionally passed exhaustive 1,024-input testing, but exhaustive validation of every hardware-compiled configuration has not yet been performed.

## Reproducing the experiments

Run the benchmark:

```powershell
python benchmark_hardware_aware.py --seeds 42 101 202 303 404
```

Run the quick correctness validation:

```powershell
python validate_hardware_aware_adders.py --quick --seeds 42 --topologies unrestricted linear_16
```

Run the complete selected-input validation:

```powershell
python validate_hardware_aware_adders.py --seeds 42
```

Benchmark outputs are generated under:

```text
benchmark_outputs/hardware_aware/
```

Correctness validation reports are saved under:

```text
benchmark_outputs/hardware_aware/validation/
```

Results, individual runs, comparison charts, and reports are also tracked through MLflow.

Selected CSVs and visualizations can be committed using `git add -f` when excluded by `.gitignore`.

## Conclusion

The optimized 13-qubit RCCX modulo-17 adder consistently achieves lower circuit depth and CX gate counts than the 10-qubit baseline under unrestricted, linear, grid, and heavy-hex compilation.

The experiments demonstrate measurable resource improvements while documenting the additional qubit requirements.

Functional validation confirms that both architectures preserve their expected behavior for the tested inputs after hardware-aware compilation.

Together, these results demonstrate the QARW Compiler Profiler's ability to perform reproducible quantum circuit benchmarking, assess compilation constraints, compare circuit architectures, and evaluate resource trade-offs.
