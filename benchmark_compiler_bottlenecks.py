
"""Milestone 6: Compare compiler bottlenecks across gate bases and topologies."""

import argparse
import csv
import json
import os
from pathlib import Path

import mlflow
import qiskit.transpiler
from mlflow.tracking import MlflowClient

from benchmark_hardware_aware import make_topologies, verify_connectivity
from benchmark_hardware_gate_bases import BASIS_CONFIGS
from modular_arithmetic import build_modular_adder_17
from modular_arithmetic_optimized import build_optimized_modular_adder_17
from validate_compiler_bottlenecks import validate_equivalence
from validate_hardware_aware_adders import (
    QUICK_CASES,
    get_layout_positions,
    simulate_basis,
    simulate_coherence,
)
from q8s.runtime.mlflow.qiskit.autologging import autolog


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
)
EXPERIMENT = "QARW-Compiler-Bottleneck-Comparisons"


def validate_compiled(source, compiled, coupling, check_coherence):
    """Apply the existing correctness checks."""

    if coupling is None:
        validate_equivalence(source, compiled)
        return "sampled_equivalence_pass"

    verify_connectivity(compiled, coupling)

    initial, final = get_layout_positions(
        compiled, source.num_qubits
    )
    tolerance = 1e-7

    for a, b in QUICK_CASES[:2]:
        probability = simulate_basis(
            compiled, initial, final, a, b, tolerance
        )
        if probability < 1 - tolerance:
            raise AssertionError(
                f"Incorrect modular addition for ({a}, {b})"
            )

    if check_coherence:
        fidelity = simulate_coherence(
            compiled, initial, final, tolerance
        )
        if fidelity < 1 - tolerance:
            raise AssertionError(
                f"Coherence fidelity too low: {fidelity}"
            )
        return "basis_and_coherence_pass"

    return "sampled_basis_pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures", nargs="+",
        choices=["baseline", "optimized_rccx"],
        default=["baseline"],
    )
    parser.add_argument(
        "--topologies", nargs="+",
        choices=[
            "unrestricted", "linear_16",
            "grid_4x4", "heavy_hex_19",
        ],
        default=["unrestricted", "linear_16"],
    )
    parser.add_argument(
        "--bases", nargs="+",
        choices=list(BASIS_CONFIGS),
        default=["CX", "CZ", "ECR"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42]
    )
    parser.add_argument(
        "--check-coherence", action="store_true"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("benchmark_outputs/compiler_bottlenecks"),
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # Enable QARW instrumentation once for all runs.
    autolog(silent=True)

    client = MlflowClient(tracking_uri=TRACKING_URI)

    sources = {
        "baseline": build_modular_adder_17(),
        "optimized_rccx": build_optimized_modular_adder_17(),
    }
    topologies = make_topologies()

    rows = []

    for architecture in args.architectures:
        source = sources[architecture]

        for topology in args.topologies:
            coupling = (
                None if topology == "unrestricted"
                else topologies[topology]
            )

            if (
                coupling is not None
                and coupling.size() < source.num_qubits
            ):
                raise ValueError(
                    f"{architecture} does not fit {topology}"
                )

            for basis_name in args.bases:
                basis_gates = BASIS_CONFIGS[basis_name]

                for seed in args.seeds:
                    run_name = (
                        f"m6_{architecture}_{topology}_"
                        f"{basis_name}_seed{seed}"
                    )

                    print(f"\nRUN: {run_name}", flush=True)

                    with mlflow.start_run(
                        run_name=run_name
                    ) as run:
                        run_id = run.info.run_id

                        mlflow.set_tags({
                            "milestone": "6",
                            "architecture": architecture,
                            "topology": topology,
                            "basis": basis_name,
                        })

                        manager = (
                            qiskit.transpiler
                            .generate_preset_pass_manager(
                                optimization_level=3,
                                basis_gates=basis_gates,
                                coupling_map=coupling,
                                layout_method=(
                                    "sabre"
                                    if coupling is not None
                                    else None
                                ),
                                routing_method=(
                                    "sabre"
                                    if coupling is not None
                                    else None
                                ),
                                seed_transpiler=seed,
                                qubits_initially_zero=False,
                            )
                        )

                        compiled = manager.run(source)

                        # Check the output native gate basis.
                        two_qubit_names = {
                            inst.operation.name
                            for inst in compiled.data
                            if len(inst.qubits) == 2
                        }

                        if two_qubit_names != {basis_gates[1]}:
                            raise AssertionError(
                                f"Unexpected two-qubit gates: "
                                f"{two_qubit_names}"
                            )

                        validation_status = validate_compiled(
                            source,
                            compiled,
                            coupling,
                            args.check_coherence,
                        )
                        mlflow.set_tag(
                            "correctness_check",
                            validation_status,
                        )

                        depth = compiled.depth()
                        operations = compiled.size()
                        two_qubit_count = sum(
                            len(inst.qubits) == 2
                            for inst in compiled.data
                        )

                    # QARW uploads the reports at MLflow run exit.
                    summary_path = client.download_artifacts(
                        run_id,
                        "compiler_bottlenecks/summary.json",
                    )

                    with open(
                        summary_path, encoding="utf-8"
                    ) as file:
                        summary = json.load(file)

                    if summary["pass_count"] < 2:
                        raise AssertionError(
                            "Insufficient compiler-pass records"
                        )

                    slowest = summary["top_slowest_passes"]
                    stages = summary["stage_summaries"]

                    row = {
                        "architecture": architecture,
                        "topology": topology,
                        "basis": basis_name,
                        "seed": seed,
                        "depth": depth,
                        "operations": operations,
                        "two_qubit_gates": two_qubit_count,
                        "pass_count": summary["pass_count"],
                        "recorded_pass_time_s": summary[
                            "total_recorded_pass_time_s"
                        ],
                        "slowest_pass": (
                            slowest[0]["pass_name"]
                            if slowest else ""
                        ),
                        "slowest_pass_time_s": (
                            slowest[0]["duration_s"]
                            if slowest else None
                        ),
                        "dominant_stage": (
                            stages[0]["stage"]
                            if stages else ""
                        ),
                        "validation": validation_status,
                        "mlflow_run_id": run_id,
                    }

                    rows.append(row)

                    print(
                        f"PASS: depth={depth}, "
                        f"2Q={two_qubit_count}, "
                        f"passes={row['pass_count']}, "
                        f"pass_time={row['recorded_pass_time_s']:.4f}s, "
                        f"slowest={row['slowest_pass']}",
                        flush=True,
                    )

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    output = args.output_dir / "compiler_comparison.csv"

    if rows:
        with output.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(rows[0])
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\nSUCCESS")
    print("Compilation records:", len(rows))
    print("Results:", output.resolve())
    print("MLflow experiment:", EXPERIMENT)


if __name__ == "__main__":
    main()
