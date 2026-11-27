"""Exhaustive, phase-aware validation of the modulo-17 logical benchmark.

Validate the action of source and lowered arithmetic circuits on all 1024
five-bit input pairs, with work and scratch qubits initially zero. The circuits
must be monomial (one nonzero entry per row and column of each gate matrix).
This permits vectorized basis-state/phase tracking without a dense 13-qubit
unitary. A non-monomial gate fails explicitly rather than being ignored.

Assumes little-endian data registers a=q[0:5], b=q[5:10]. The valid-input
arithmetic oracle is independent of the circuit builders. For out-of-range a,
compare against the baseline's reversible extension, not an invented oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import qiskit
from qiskit.quantum_info import Operator

from benchmark_fault_tolerant import POLICIES, lower_controlled_x
from modular_arithmetic import build_modular_adder_17
from modular_arithmetic_optimized import build_optimized_modular_adder_17

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "benchmark_outputs" / "fault_tolerant" / "phase_validation.json"
TOL = 1e-9
N_DATA = 10
MASK = (1 << N_DATA) - 1


def monomial_table(matrix):
    """Return output index and complex amplitude for each local basis input."""
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Gate matrix must be square")
    n = matrix.shape[0]
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Gate matrix contains nonfinite entries")
    targets = np.argmax(np.abs(matrix), axis=0)
    phases = matrix[targets, np.arange(n)]
    if len(np.unique(targets)) != n or not np.allclose(np.abs(phases), 1, atol=TOL, rtol=0):
        raise ValueError("Gate is not a unit-modulus basis permutation")
    residual = matrix.copy()
    residual[targets, np.arange(n)] = 0
    if np.max(np.abs(residual)) > TOL:
        raise ValueError("Gate contains superposition terms; monomial validator not applicable")
    return targets, phases


def track(circuit):
    """Track every data basis state, including complex phase and ancilla bits."""
    if circuit.num_qubits < N_DATA or circuit.num_qubits > 62 or circuit.num_clbits:
        raise ValueError("Expected a 10+ qubit circuit without classical bits")
    states = np.arange(1 << N_DATA, dtype=np.int64)
    phases = np.full(states.shape, np.exp(1j * float(circuit.global_phase)), dtype=complex)
    qmap = {q: i for i, q in enumerate(circuit.qubits)}
    for inst in circuit.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        if inst.clbits or getattr(op, "condition", None) is not None:
            raise ValueError("Classical instructions are not supported")
        if op.num_qubits > 10:
            raise ValueError(f"Refusing a large dense local matrix for {op.name}")
        try:
            targets, local_phases = monomial_table(Operator(op).data)
        except Exception as exc:
            raise ValueError(f"Cannot validate gate {op.name}: {exc}") from exc
        qs = [qmap[q] for q in inst.qubits]
        local_in = np.zeros_like(states)
        clear_mask = 0
        for i, q in enumerate(qs):
            local_in |= ((states >> q) & 1) << i
            clear_mask |= 1 << q
        local_out = targets[local_in]
        phases *= local_phases[local_in]
        states &= ~clear_mask
        for i, q in enumerate(qs):
            states |= ((local_out >> i) & 1) << q
    return states, phases


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_arithmetic(states, phases, reference=None):
    inputs = np.arange(1 << N_DATA, dtype=np.int64)
    a, b = inputs & 31, inputs >> 5
    valid = (a < 17) & (b < 17)
    expected = a | (((a + b) % 17) << 5)
    require(np.all(states >> N_DATA == 0), "Work/scratch ancilla was not returned to zero")
    require(len(np.unique(states)) == len(inputs), "Output mapping is not reversible")
    require(np.all(states[valid] == expected[valid]), "Modulo-17 arithmetic mismatch (check register layout)")
    require(np.all((states & 31) == a), "Register a was modified")
    require(np.all(states[b >= 17] == inputs[b >= 17]), "Out-of-range b was not preserved")
    require(np.allclose(np.abs(phases), 1, atol=TOL, rtol=0), "Non-unit output amplitude")
    # One common global phase is harmless; a phase depending on input is not.
    normalized = phases / phases[0]
    require(np.allclose(normalized, 1, atol=TOL, rtol=0), "Input-dependent relative phase detected")
    if reference is not None:
        ref_states, ref_phases = reference
        require(np.array_equal(states & MASK, ref_states & MASK), "Reversible extension differs from baseline")
        require(np.allclose(normalized, ref_phases / ref_phases[0], atol=TOL, rtol=0), "Relative phases differ from baseline")
    return {
        "basis_inputs_checked": int(len(inputs)),
        "valid_arithmetic_inputs_checked": int(valid.sum()),
        "out_of_range_b_inputs_checked": int((b >= 17).sum()),
        "clean_ancillas": True,
        "reversible_mapping": True,
        "phase_preserved_up_to_global": True,
        "max_relative_phase_error": float(np.max(np.abs(normalized - 1))),
        "matches_baseline_extension": True if reference is not None else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    circuits = {"baseline": build_modular_adder_17(), "optimized_rccx": build_optimized_modular_adder_17()}
    print("QARW MODULO-17 - EXHAUSTIVE PHASE VALIDATION")
    print("Data layout: a=q[0:5], b=q[5:10]; all other qubits initially zero.")
    for name, circuit in circuits.items():
        print(name, "registers:", [(reg.name, len(reg)) for reg in circuit.qregs])
    reference = track(circuits["baseline"])
    rows = []
    cases = [("source", name, circuit) for name, circuit in circuits.items()]
    cases.extend((policy, name, lower_controlled_x(circuit, policy))
                 for policy in POLICIES for name, circuit in circuits.items())
    try:
        for policy, name, circuit in cases:
            states, phases = reference if (policy, name) == ("source", "baseline") else track(circuit)
            row = check_arithmetic(states, phases, None if (policy, name) == ("source", "baseline") else reference)
            row.update(architecture=name, mcx_policy=policy, logical_qubits=circuit.num_qubits, status="PASS")
            rows.append(row)
            print(f"PASS {policy:12} {name:16} {row['basis_inputs_checked']} inputs | phase error {row['max_relative_phase_error']:.3e}")
    except ValueError as exc:
        print(f"FAIL: {policy} / {name}: {exc}")
        return 1
    report = {"qiskit_version": qiskit.__version__, "method": "exhaustive_monomial_gate_action",
              "data_layout": {"a": list(range(5)), "b": list(range(5, 10))},
              "absolute_tolerance": TOL, "configurations": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(rows)}/{len(cases)} configurations passed. Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
