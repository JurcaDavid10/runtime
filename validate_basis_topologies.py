
"""Milestone 5: Validate adders across native gate bases and topologies."""

import argparse
import csv
from pathlib import Path

from qiskit import transpile

from benchmark_hardware_aware import (
    make_topologies,
    verify_connectivity,
)
from modular_arithmetic import build_modular_adder_17
from modular_arithmetic_optimized import build_optimized_modular_adder_17
from validate_hardware_aware_adders import (
    QUICK_CASES,
    FULL_CASES,
    get_layout_positions,
    simulate_basis,
    simulate_coherence,
)


BASIS_CONFIGS = {
    "CX": ["u", "cx"],
    "CZ": ["u", "cz"],
    "ECR": ["u", "ecr"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/basis_gate_validation"),
    )
    args = parser.parse_args()

    sources = {
        "baseline": build_modular_adder_17(),
        "optimized_rccx": build_optimized_modular_adder_17(),
    }

    cases = QUICK_CASES if args.quick else FULL_CASES
    topologies = make_topologies()
    tolerance = 1e-7
    rows = []

    print("\nQARW — GATE BASIS CORRECTNESS VALIDATION")
    print("=" * 75)

    for architecture, source in sources.items():
        for topology, coupling in topologies.items():
            for basis_name, basis_gates in BASIS_CONFIGS.items():

                compiled = transpile(
                    source,
                    basis_gates=basis_gates,
                    coupling_map=coupling,
                    optimization_level=3,
                    seed_transpiler=args.seed,
                    layout_method="sabre",
                    routing_method="sabre",
                    qubits_initially_zero=False,
                )

                verify_connectivity(compiled, coupling)

                native_gate = basis_gates[1]
                two_qubit_gates = {
                    inst.operation.name
                    for inst in compiled.data
                    if len(inst.qubits) == 2
                }

                if two_qubit_gates != {native_gate}:
                    raise AssertionError(
                        f"Unexpected native gates: {two_qubit_gates}"
                    )

                initial, final = get_layout_positions(
                    compiled, source.num_qubits
                )

                probabilities = [
                    simulate_basis(
                        compiled, initial, final,
                        a, b, tolerance
                    )
                    for a, b in cases
                ]

                fidelity = simulate_coherence(
                    compiled, initial, final, tolerance
                )

                rows.append({
                    "architecture": architecture,
                    "topology": topology,
                    "basis": basis_name,
                    "seed": args.seed,
                    "basis_cases_passed": len(probabilities),
                    "minimum_probability": min(probabilities),
                    "coherence_fidelity": fidelity,
                    "status": "PASS",
                })

                print(
                    f"PASS: {architecture:<15} "
                    f"{topology:<14} "
                    f"{basis_name:<4} "
                    f"basis_tests={len(probabilities)} "
                    f"fidelity={fidelity:.9f}",
                    flush=True,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "basis_gate_correctness.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSUCCESS: {len(rows)}/18 compiled circuits validated.")
    print(f"Results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
