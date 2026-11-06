"""Benchmark the QARW modulo-17 adder through the existing Qiskit/MLflow profiler.

Place alongside modular_arithmetic.py in the runtime directory.

Quick smoke test:
    python benchmark_modular_arithmetic.py --level 0

Individual levels (recommended for initially large decompositions):
    python benchmark_modular_arithmetic.py --level 1
    python benchmark_modular_arithmetic.py --level 2
    python benchmark_modular_arithmetic.py --level 3

All four levels (separate processes isolate the QARW autolog state):
    python benchmark_modular_arithmetic.py --all
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

from modular_arithmetic import MODULUS, NUM_BITS, build_modular_adder_17


BASIS_GATES = ["u", "cx"]
SEED = int(os.environ.get("QARW_TRANSPILER_SEED", "42"))
EXPERIMENT = "QARW-Modular-Arithmetic-Benchmarks"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "benchmark_outputs" / "modular_add_17"


def two_qubit_depth(circuit):
    """Longest dependency chain of two-qubit gates, ignoring single-qubit gates.

    This is the depth of the circuit after filtering out 1-qubit operations,
    not the depth of a real hardware schedule with gate durations.
    """
    last_layer = {qubit: 0 for qubit in circuit.qubits}
    for instruction in circuit.data:
        qubits = instruction.qubits
        if len(qubits) != 2:
            continue
        next_layer = 1 + max(last_layer[qubit] for qubit in qubits)
        for qubit in qubits:
            last_layer[qubit] = next_layer
    return max(last_layer.values(), default=0)


def configure_mlflow():
    """Use the same local ./mlruns store as the prior benchmarks by default."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        tracking_dir = Path.cwd() / "mlruns"
        mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


def benchmark_one_level(level):
    # Install q8s's monkey-patches BEFORE importing the Qiskit pass-manager
    # factory, so that the patched function is the one called below.
    from q8s.runtime.mlflow.qiskit import autolog

    autolog()
    from qiskit.transpiler import generate_preset_pass_manager

    configure_mlflow()
    original = build_modular_adder_17()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64, flush=True)
    print("MODULAR ADDITION | MOD 17 | OPTIMIZATION LEVEL", level, flush=True)
    print("Source qubits:", original.num_qubits, flush=True)
    print("Source depth:", original.depth(), flush=True)
    print("Source operations:", original.size(), flush=True)
    print("Basis gates:", BASIS_GATES, flush=True)
    print("=" * 64, flush=True)

    # One run per process is intentional: the installed q8s autologging
    # integration retains mutable provenance state between MLflow runs.
    with mlflow.start_run(run_name=f"mod17_add_opt{level}_seed{SEED}") as run:
        mlflow.set_tags({
            "workload": "modular_addition",
            "arithmetic": "in_place_mod_17",
            "benchmark_version": "phase2_v1",
        })
        mlflow.log_params({
            "modulus": MODULUS,
            "register_bits": NUM_BITS,
            "basis_gates": ",".join(BASIS_GATES),
            "connectivity": "all_to_all_idealized",
            "optimization_level": level,
            "seed_transpiler": SEED,
        })
        mlflow.log_metrics({
            "source_qubits": original.num_qubits,
            "source_depth": original.depth(),
            "source_operations": original.size(),
        })

        pass_manager = generate_preset_pass_manager(
            optimization_level=level,
            basis_gates=BASIS_GATES,
            seed_transpiler=SEED,
        )
        start = time.perf_counter()
        compiled = pass_manager.run(original)
        elapsed = time.perf_counter() - start

        counts = dict(compiled.count_ops())
        two_qubit_ops = sum(
            1 for instruction in compiled.data if len(instruction.qubits) == 2
        )
        # Reject incomplete decomposition; otherwise comparisons are misleading.
        unexpected = set(counts) - set(BASIS_GATES)
        if unexpected:
            raise AssertionError(
                "Compiled circuit contains non-basis operations: "
                + ", ".join(sorted(unexpected))
            )
        result = {
            "workload": "modular_add_17",
            "optimization_level": level,
            "seed_transpiler": SEED,
            "qiskit_version": qiskit.__version__,
            "basis_gates": BASIS_GATES,
            "source_qubits": original.num_qubits,
            "source_depth": original.depth(),
            "source_operations": original.size(),
            "compiled_qubits": compiled.num_qubits,
            "compiled_depth": compiled.depth(),
            "compiled_operations": compiled.size(),
            "compiled_cx_count": counts.get("cx", 0),
            "compiled_u_count": counts.get("u", 0),
            "compiled_two_qubit_gate_count": two_qubit_ops,
            "compiled_two_qubit_depth": two_qubit_depth(compiled),
            "compilation_wall_time_s": elapsed,
            "mlflow_run_id": run.info.run_id,
        }
        mlflow.log_metrics({
            key: value for key, value in result.items()
            if key in {
                "compiled_qubits", "compiled_depth", "compiled_operations",
                "compiled_cx_count", "compiled_u_count",
                "compiled_two_qubit_gate_count", "compiled_two_qubit_depth",
                "compilation_wall_time_s",
            }
        })

        artifact_path = OUTPUT_DIRECTORY / (
                f"optimization_level_{level}_seed{SEED}.json"
                if "QARW_TRANSPILER_SEED" in os.environ
                else f"optimization_level_{level}.json"
            )
        artifact_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(artifact_path), artifact_path="benchmark")
        # Exiting the context also lets q8s autolog save qprov_record.json
        # and transpilation_timeline.png to the same MLflow run.

    print("\nRESULTS", flush=True)
    print("Compiled depth:", result["compiled_depth"], flush=True)
    print("Compiled operations:", result["compiled_operations"], flush=True)
    print("CNOT (CX) gates:", result["compiled_cx_count"], flush=True)
    print("Two-qubit depth:", result["compiled_two_qubit_depth"], flush=True)
    print("Compilation seconds:", round(elapsed, 3), flush=True)
    print("MLflow run ID:", result["mlflow_run_id"], flush=True)
    print("JSON:", artifact_path, flush=True)
    return result


def benchmark_all_levels():
    """Launch levels serially, each in a fresh process, then create a CSV."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    for level in range(4):
        print(f"\n--- Starting optimization level {level} ---", flush=True)
        subprocess.run(
            [sys.executable, str(script), "--level", str(level)],
            cwd=str(script.parent),
            check=True,
        )

    results = []
    for level in range(4):
        result_path = OUTPUT_DIRECTORY / (
                f"optimization_level_{level}_seed{SEED}.json"
                if "QARW_TRANSPILER_SEED" in os.environ
                else f"optimization_level_{level}.json"
            )
        results.append(json.loads(result_path.read_text(encoding="utf-8")))

    csv_file = OUTPUT_DIRECTORY / (
        f"comparison_seed{SEED}.csv"
        if "QARW_TRANSPILER_SEED" in os.environ
        else "comparison.csv"
    )
    columns = [
        "optimization_level", "seed_transpiler",
        "source_qubits", "source_depth", "source_operations",
        "compiled_qubits", "compiled_depth", "compiled_operations",
        "compiled_cx_count", "compiled_u_count",
        "compiled_two_qubit_gate_count", "compiled_two_qubit_depth",
        "compilation_wall_time_s", "mlflow_run_id",
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print("\nCompleted all four levels. Comparison saved to:", csv_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--level", type=int, choices=range(4), default=0)
    choice.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        benchmark_all_levels()
    else:
        benchmark_one_level(args.level)


if __name__ == "__main__":
    main()
