"""Validate QARW modulo-17 adders AFTER connectivity-aware transpilation.

Place alongside modular_arithmetic.py, modular_arithmetic_optimized.py,
and benchmark_hardware_aware.py in the QARW runtime repository.

Quick check: python validate_hardware_aware_adders.py --quick --seeds 42 --topologies unrestricted linear_16
Full check:  python validate_hardware_aware_adders.py --seeds 42
Extra seeds: python validate_hardware_aware_adders.py --seeds 42 101 202 303 404

For mapped circuits, account for BOTH initial placement and final routing
permutations. This performs exact statevector simulation of selected inputs,
not execution on a quantum processor or an exhaustive proof of equivalence.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from benchmark_hardware_aware import BASIS_GATES, make_topologies, verify_connectivity
from modular_arithmetic import build_modular_adder_17
from modular_arithmetic_optimized import build_optimized_modular_adder_17

MODULUS = 17
NUM_BITS = 5
WIDTH_MASK = (1 << NUM_BITS) - 1
EXPERIMENT = "QARW-Mod17-Compiled-Correctness"

# Valid inputs, modular wraparound, and the deliberately preserved invalid-b domain.
FULL_CASES = (
    (0, 0), (1, 1), (16, 1), (12, 8), (16, 16),
    (1, 17), (9, 31), (31, 16),
)
QUICK_CASES = ((16, 1), (1, 17))


def bit_index(a: int, b: int, positions: list[int]) -> int:
    """Pack little-endian logical a/b bits into their physical positions.

    All remaining source work qubits and added routing ancillas are zero.
    """
    logical_bits = (a & WIDTH_MASK) | ((b & WIDTH_MASK) << NUM_BITS)
    return sum(1 << physical for logical, physical in enumerate(positions)
               if (logical_bits >> logical) & 1)


def expected_b(a: int, b: int) -> int:
    return (a + b) % MODULUS if b < MODULUS else b


def get_layout_positions(compiled, source_width: int) -> tuple[list[int], list[int]]:
    layout = compiled.layout
    if layout is None:  # Unrestricted transpilation without a layout stage.
        initial = list(range(source_width))
        final = list(range(source_width))
    else:
        initial = list(layout.initial_index_layout(filter_ancillas=True))
        final = list(layout.final_index_layout(filter_ancillas=True))
    for label, positions in (("initial", initial), ("final", final)):
        if len(positions) != source_width:
            raise AssertionError(f"Invalid {label} mapping: {positions}")
        if len(set(positions)) != source_width:
            raise AssertionError(f"Repeated {label} physical positions: {positions}")
        if not all(0 <= p < compiled.num_qubits for p in positions):
            raise AssertionError(f"Out-of-range {label} positions: {positions}")
    return initial, final


def simulate_basis(compiled, initial: list[int], final: list[int],
                   a: int, b: int, tolerance: float) -> float:
    start_index = bit_index(a, b, initial)
    target_index = bit_index(a, expected_b(a, b), final)
    state = Statevector.from_int(start_index, 1 << compiled.num_qubits)
    state = state.evolve(compiled)
    probability = float(abs(state.data[target_index]) ** 2)
    if probability < 1.0 - tolerance:
        most_likely = int(np.argmax(abs(state.data) ** 2))
        raise AssertionError(
            f"a={a} b={b}: target |{target_index}> probability={probability:.10f}; "
            f"most likely basis index={most_likely}"
        )
    return probability


def simulate_coherence(compiled, initial: list[int], final: list[int],
                       tolerance: float) -> float:
    """Test the relative phase of |a=0,b=16> + |a=1,b=16>.

    The two outputs must be |0,16> and |1,0> with the same relative phase.
    Comparing fidelity is insensitive to a physically irrelevant global phase.
    """
    b = 16
    prep = QuantumCircuit(compiled.num_qubits)
    for bit in range(NUM_BITS):
        if (b >> bit) & 1:
            prep.x(initial[NUM_BITS + bit])
    prep.h(initial[0])  # a = (|0> + |1>) / sqrt(2)
    actual = Statevector.from_instruction(prep).evolve(compiled)

    target = np.zeros(1 << compiled.num_qubits, dtype=complex)
    target[bit_index(0, expected_b(0, b), final)] = 1 / np.sqrt(2)
    target[bit_index(1, expected_b(1, b), final)] = 1 / np.sqrt(2)
    fidelity = float(abs(np.vdot(target, actual.data)) ** 2)
    if fidelity < 1.0 - tolerance:
        raise AssertionError(f"Superposition fidelity {fidelity:.10f} < {1-tolerance}")
    return fidelity


def configure_mlflow(disabled: bool):
    if disabled:
        return None
    try:
        import mlflow
    except ImportError:
        print("MLflow unavailable; generating local reports only.")
        return None
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def save_reports(rows: list[dict], output: Path, seeds: list[int], quick: bool,
                 optimization_level: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "compiled_correctness.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Hardware-aware compiled-circuit correctness",
        "",
        "Exact statevector tests on the *compiled* modulo-17 adders, with the",
        "initial physical-qubit layout and final routing permutation respected.",
        "",
        f"- Seeds: `{', '.join(map(str, seeds))}`",
        f"- Gate basis: `u, cx`; optimization level: {optimization_level}",
        f"- Basis cases per circuit: **{len(QUICK_CASES if quick else FULL_CASES)}**",
        "- One coherent two-state superposition per compiled circuit",
        "- Clean work qubits and unchanged `a` are included in the expected output",
        "- Out-of-range `b` values must remain unchanged",
        "- `PASS` means tested output-basis probability and coherent-state fidelity",
        "  both reached at least `1 - tolerance`",
        "- This is **not** an exhaustive unitary equivalence proof, a device run,",
        "  or a noise/fidelity estimate for physical hardware",
        "",
        "| Architecture | Topology | Seed | Basis tests | Superposition fidelity | Result |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['architecture']} | {row['topology']} | {row['seed']} | "
            f"{row['basis_cases_passed']} | {row['superposition_fidelity']:.9f} | "
            f"{row['status']} |"
        )
    (output / "compiled_correctness_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--optimization-level", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--quick", action="store_true", help="Test two representative basis inputs")
    parser.add_argument("--topologies", nargs="+",
                        choices=("unrestricted", "linear_16", "grid_4x4", "heavy_hex_19"),
                        default=None, help="Subset of topologies (default: all four)")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmark_outputs/hardware_aware/validation"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if not (0 < args.tolerance < 0.01):
        parser.error("--tolerance must be between 0 and 0.01")

    sources = {
        "baseline": build_modular_adder_17(),
        "optimized_rccx": build_optimized_modular_adder_17(),
    }
    cases = QUICK_CASES if args.quick else FULL_CASES
    all_topologies = {"unrestricted": None, **make_topologies()}
    selected = args.topologies or list(all_topologies)
    topologies = {name: all_topologies[name] for name in selected}
    mlflow = configure_mlflow(args.no_mlflow)
    rows = []
    for architecture, source in sources.items():
        for topology, coupling in topologies.items():
            for seed in args.seeds:
                kwargs = dict(basis_gates=BASIS_GATES,
                              optimization_level=args.optimization_level,
                              seed_transpiler=seed,
                              qubits_initially_zero=False,)
                if coupling is not None:
                    kwargs.update(coupling_map=coupling, layout_method="sabre",
                                  routing_method="sabre")
                compiled = transpile(source, **kwargs)
                if coupling is not None:
                    verify_connectivity(compiled, coupling)
                initial, final = get_layout_positions(compiled, source.num_qubits)
                minimum_probability = min(
                    simulate_basis(compiled, initial, final, a, b, args.tolerance)
                    for a, b in cases
                )
                fidelity = simulate_coherence(compiled, initial, final, args.tolerance)
                row = {
                    "architecture": architecture,
                    "topology": topology,
                    "seed": seed,
                    "logical_qubits": source.num_qubits,
                    "device_capacity_qubits": coupling.size() if coupling else "N/A",
                    "compiled_circuit_width_qubits": compiled.num_qubits,
                    "basis_cases_passed": len(cases),
                    "min_basis_output_probability": round(minimum_probability, 10),
                    "superposition_fidelity": round(fidelity, 10),
                    "connectivity_check": "passed" if coupling else "not_applicable",
                    "status": "PASS",
                }
                rows.append(row)
                print(f"PASS: {architecture:<14} {topology:<13} seed={seed} "
                      f"basis={len(cases)} fidelity={fidelity:.9f}", flush=True)
                if mlflow is not None:
                    with mlflow.start_run(run_name=f"{architecture}_{topology}_seed{seed}"):
                        mlflow.set_tags({"test": "compiled_correctness",
                                         "architecture": architecture, "topology": topology,
                                         "result": "PASS"})
                        mlflow.log_params({"seed": seed, "optimization_level": args.optimization_level,
                                           "logical_qubits": source.num_qubits,
                                           "basis_cases": len(cases)})
                        mlflow.log_metrics({"min_basis_output_probability": minimum_probability,
                                            "superposition_fidelity": fidelity,
                                            "basis_cases_passed": len(cases)})

    save_reports(rows, args.output_dir, args.seeds, args.quick, args.optimization_level)
    if mlflow is not None:
        with mlflow.start_run(run_name="compiled_correctness_summary"):
            mlflow.log_artifacts(str(args.output_dir), artifact_path="correctness_reports")
            mlflow.log_metric("compiled_circuits_passed", len(rows))
    print(f"\nALL {len(rows)} COMPILED CIRCUITS PASSED; "
          f"{len(rows) * len(cases)} basis-case tests and {len(rows)} coherence tests.")
    print(f"Reports saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
