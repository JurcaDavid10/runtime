"""Benchmark the ancilla-assisted RCCX modulo-17 adder through QARW/MLflow.

Run one optimization level at a time (recommended):
    python benchmark_modular_arithmetic_optimized.py --level 0

Run all levels in separate subprocesses:
    python benchmark_modular_arithmetic_optimized.py --all

This deliberately matches benchmark_modular_arithmetic.py's basis, seed,
experiment, and two-qubit-depth definition. Results use a separate directory.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import qiskit

from modular_arithmetic_optimized import (
    MODULUS,
    NUM_BITS,
    WORK_QUBITS,
    build_optimized_modular_adder_17,
)

BASIS_GATES = ["u", "cx"]
SEED = 42
EXPERIMENT = "QARW-Modular-Arithmetic-Benchmarks"
BASE_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = (BASE_DIRECTORY / "benchmark_outputs" / "modular_add_17"
                    / "relative_phase_ancilla")


def two_qubit_depth(circuit):
    """Compute depth after filtering out all one-qubit operations."""
    last = {qubit: 0 for qubit in circuit.qubits}
    for instruction in circuit.data:
        qs = instruction.qubits
        if len(qs) == 2:
            depth = 1 + max(last[q] for q in qs)
            for q in qs:
                last[q] = depth
    return max(last.values(), default=0)


def configure_mlflow():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri((BASE_DIRECTORY / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def benchmark_one_level(level):
    # Same order as your working baseline benchmark: patch before importing
    # generate_preset_pass_manager. An isolated Python process per run also
    # avoids retaining mutable q8s provenance state across runs.
    from q8s.runtime.mlflow.qiskit import autolog
    autolog()
    from qiskit.transpiler import generate_preset_pass_manager

    configure_mlflow()
    circuit = build_optimized_modular_adder_17()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68, flush=True)
    print(f"MOD 17 | RELATIVE-PHASE ANCILLA ADDER | OPT LEVEL {level}", flush=True)
    print("Source qubits:", circuit.num_qubits, flush=True)
    print("Source depth:", circuit.depth(), flush=True)
    print("Source operations:", circuit.size(), flush=True)
    print("Basis gates:", BASIS_GATES, flush=True)
    print("=" * 68, flush=True)

    with mlflow.start_run(run_name=f"mod17_rccx_ancilla_opt{level}_seed{SEED}") as run:
        mlflow.set_tags({
            "workload": "modular_addition",
            "arithmetic": "in_place_mod_17",
            "architecture": "rccx_clean_ancilla_3",
            "benchmark_version": "phase2_v2",
        })
        mlflow.log_params({
            "modulus": MODULUS,
            "register_bits": NUM_BITS,
            "work_qubits": WORK_QUBITS,
            "basis_gates": ",".join(BASIS_GATES),
            "connectivity": "all_to_all_idealized",
            "seed_transpiler": SEED,
            "optimization_level": level,
        })
        mlflow.log_metrics({
            "source_qubits": circuit.num_qubits,
            "source_depth": circuit.depth(),
            "source_operations": circuit.size(),
        })

        pm = generate_preset_pass_manager(
            optimization_level=level,
            basis_gates=BASIS_GATES,
            seed_transpiler=SEED,
        )
        started = time.perf_counter()
        compiled = pm.run(circuit)
        elapsed = time.perf_counter() - started
        counts = dict(compiled.count_ops())
        unexpected = set(counts) - set(BASIS_GATES)
        if unexpected:
            raise AssertionError(f"Untranspiled gates remain: {sorted(unexpected)}")

        result = {
            "workload": "modular_add_17",
            "architecture": "rccx_clean_ancilla_3",
            "optimization_level": level,
            "seed_transpiler": SEED,
            "qiskit_version": qiskit.__version__,
            "basis_gates": BASIS_GATES,
            "source_qubits": circuit.num_qubits,
            "source_depth": circuit.depth(),
            "source_operations": circuit.size(),
            "compiled_qubits": compiled.num_qubits,
            "compiled_depth": compiled.depth(),
            "compiled_operations": compiled.size(),
            "compiled_cx_count": counts.get("cx", 0),
            "compiled_u_count": counts.get("u", 0),
            "compiled_two_qubit_gate_count": sum(
                len(item.qubits) == 2 for item in compiled.data
            ),
            "compiled_two_qubit_depth": two_qubit_depth(compiled),
            "compilation_wall_time_s": elapsed,
            "mlflow_run_id": run.info.run_id,
        }
        mlflow.log_metrics({
            key: result[key] for key in (
                "compiled_qubits", "compiled_depth", "compiled_operations",
                "compiled_cx_count", "compiled_u_count",
                "compiled_two_qubit_gate_count", "compiled_two_qubit_depth",
                "compilation_wall_time_s",
            )
        })

        path = OUTPUT_DIRECTORY / f"optimization_level_{level}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(path), artifact_path="benchmark")

    print("\nRESULTS", flush=True)
    print("Compiled qubits:", result["compiled_qubits"], flush=True)
    print("Compiled depth:", result["compiled_depth"], flush=True)
    print("Compiled operations:", result["compiled_operations"], flush=True)
    print("CNOT (CX) gates:", result["compiled_cx_count"], flush=True)
    print("Two-qubit depth:", result["compiled_two_qubit_depth"], flush=True)
    print("Compilation seconds:", round(elapsed, 3), flush=True)
    print("MLflow run ID:", result["mlflow_run_id"], flush=True)
    print("JSON:", path, flush=True)
    return result


def benchmark_all():
    for level in range(4):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--level", str(level)],
            cwd=str(BASE_DIRECTORY), check=True,
        )
    results = []
    for level in range(4):
        path = OUTPUT_DIRECTORY / f"optimization_level_{level}.json"
        results.append(json.loads(path.read_text(encoding="utf-8")))
    columns = ["architecture", "optimization_level", "compiled_qubits",
               "compiled_depth", "compiled_operations", "compiled_cx_count",
               "compiled_two_qubit_depth", "compilation_wall_time_s", "mlflow_run_id"]
    output = OUTPUT_DIRECTORY / "optimization_comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print("Saved:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    option = parser.add_mutually_exclusive_group()
    option.add_argument("--level", type=int, choices=range(4), default=0)
    option.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        benchmark_all()
    else:
        benchmark_one_level(args.level)


if __name__ == "__main__":
    main()
