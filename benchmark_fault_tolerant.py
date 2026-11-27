"""Compare logical Clifford+T resources of the existing modulo-17 adders.

Two explicit clean-ancilla MCX synthesis policies are applied to both source
architectures. No hardware transpilation, approximate rotations or physical
error-correction model is used. Added scratch qubits must start in |0>.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import RCCXGate

from q8s.runtime.qprov.fault_tolerant_resources import estimate_fault_tolerant_resources

POLICIES = ("ccx_chain", "rccx_chain")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "benchmark_outputs" / "fault_tolerant"


def _x_controls(operation):
    """Recognize ancilla-free controlled-X operations, including open controls."""
    k = getattr(operation, "num_ctrl_qubits", None)
    base = getattr(operation, "base_gate", None)
    if isinstance(k, int) and k >= 1 and getattr(base, "name", None) == "x":
        if operation.num_qubits != k + 1:
            raise ValueError(f"Unsupported pre-allocated ancillas in {operation.name}")
        return k, int(operation.ctrl_state)
    return None


def lower_controlled_x(circuit: QuantumCircuit, policy: str) -> QuantumCircuit:
    """Lower top-level controlled-X gates with a reusable clean scratch register.

    RCCX compute steps are undone with their inverses around an exact CCX,
    cancelling relative phases. Existing RCCX gates are preserved, not replaced.
    Other gates are copied; the downstream estimator rejects unsupported gates.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    if circuit.num_clbits:
        raise ValueError("Only circuits without classical bits are supported")
    descriptions = [_x_controls(inst.operation) for inst in circuit.data]
    scratch_count = max((max(0, d[0] - 2) for d in descriptions if d), default=0)
    lowered = QuantumCircuit(circuit.num_qubits + scratch_count, name=circuit.name)
    lowered.global_phase = circuit.global_phase
    scratch = list(range(circuit.num_qubits, lowered.num_qubits))
    index = {q: i for i, q in enumerate(circuit.qubits)}

    for inst, desc in zip(circuit.data, descriptions):
        op = inst.operation
        if inst.clbits or getattr(op, "condition", None) is not None:
            raise ValueError("Classical or conditioned operations are unsupported")
        qs = [index[q] for q in inst.qubits]
        if desc is None:
            lowered.append(op, qs)
            continue
        k, state = desc
        controls, target = qs[:-1], qs[-1]
        open_controls = [q for i, q in enumerate(controls) if not ((state >> i) & 1)]
        for q in open_controls:
            lowered.x(q)
        if k == 1:
            lowered.cx(controls[0], target)
        elif k == 2:
            lowered.ccx(*controls, target)
        else:
            chain = [(controls[0], controls[1], scratch[0])]
            for i in range(1, k - 2):
                chain.append((scratch[i - 1], controls[i + 1], scratch[i]))
            for a, b, c in chain:
                if policy == "ccx_chain":
                    lowered.ccx(a, b, c)
                else:
                    lowered.append(RCCXGate(), [a, b, c])
            lowered.ccx(scratch[k - 3], controls[-1], target)
            for a, b, c in reversed(chain):
                if policy == "ccx_chain":
                    lowered.ccx(a, b, c)
                else:
                    lowered.append(RCCXGate().inverse(), [a, b, c])
        for q in reversed(open_controls):
            lowered.x(q)
    return lowered


def benchmark():
    # Existing builders, not duplicates or hardware-transpiled circuits.
    from modular_arithmetic import build_modular_adder_17
    from modular_arithmetic_optimized import build_optimized_modular_adder_17

    circuits = {
        "baseline": build_modular_adder_17(),
        "optimized_rccx": build_optimized_modular_adder_17(),
    }
    rows = []
    for policy in POLICIES:
        for architecture, source in circuits.items():
            lowered = lower_controlled_x(source, policy)
            resources = estimate_fault_tolerant_resources(lowered)
            rows.append({
                "architecture": architecture,
                "mcx_policy": policy,
                "qiskit_version": qiskit.__version__,
                "source_qubits": source.num_qubits,
                "added_clean_ancillas": lowered.num_qubits - source.num_qubits,
                "source_gate_counts": dict(source.count_ops()),
                **asdict(resources),
            })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    rows = benchmark()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "logical_resources.json"
    csv_path = args.output_dir / "logical_resources.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                             for k, v in row.items()})

    print("\nQARW MODULO-17 - LOGICAL CLIFFORD+T RESOURCES")
    print(f"{'MCX policy':<13} {'Architecture':<17} {'Source':>6} {'Extra':>6} "
          f"{'Logical':>7} {'T-count':>9} {'T-depth':>9} {'Clifford':>9}")
    for row in rows:
        print(f"{row['mcx_policy']:<13} {row['architecture']:<17} "
              f"{row['source_qubits']:>6} {row['added_clean_ancillas']:>6} "
              f"{row['logical_qubits']:>7} {row['t_count']:>9} "
              f"{row['t_depth']:>9} {row['clifford_count']:>9}")
    print("\nMagic-state demand = T-count (one logical T state per T/Tdg).")
    print("No physical qubit estimates; arithmetic phase validation is separate.")
    print("Saved:", json_path, csv_path, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
