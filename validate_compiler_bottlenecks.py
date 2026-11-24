
"""Milestone 6: Validate compiler profiling on a real modulo-17 adder."""

import json
import os

import mlflow
import numpy as np
import qiskit.transpiler
from mlflow.tracking import MlflowClient
from qiskit.quantum_info import Statevector

from modular_arithmetic import build_modular_adder_17
from q8s.runtime.mlflow.qiskit.autologging import autolog


EXPERIMENT = "QARW-Compiler-Profiler"
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)


def validate_equivalence(source, compiled):
    """Check source/compiled equivalence on selected input states."""

    if source.num_qubits != compiled.num_qubits:
        raise AssertionError("Compiled circuit width changed unexpectedly")

    num_qubits = source.num_qubits
    dimension = 2 ** num_qubits

    # Test selected computational-basis inputs.
    inputs = [
        0,
        1,
        dimension // 2,
        dimension - 1,
    ]

    for value in inputs:
        initial = Statevector.from_int(value, dimension)

        expected = initial.evolve(source)
        actual = initial.evolve(compiled)

        if not expected.equiv(actual):
            raise AssertionError(
                f"Compiled circuit differs for input {value}"
            )

    # Also test a complex superposition to detect phase errors.
    rng = np.random.default_rng(42)
    amplitudes = (
        rng.standard_normal(dimension)
        + 1j * rng.standard_normal(dimension)
    )
    amplitudes /= np.linalg.norm(amplitudes)

    initial = Statevector(amplitudes)

    if not initial.evolve(source).equiv(
        initial.evolve(compiled)
    ):
        raise AssertionError("Superposition equivalence failed")

    print("PASS: Sampled functional equivalence")


def validate_mlflow_artifacts(run_id):
    """Inspect uploaded profiler records and reports."""

    client = MlflowClient(tracking_uri=TRACKING_URI)

    artifacts = client.list_artifacts(
        run_id,
        "compiler_bottlenecks",
    )

    paths = {artifact.path for artifact in artifacts}

    print("\nCompiler bottleneck artifacts:")
    for path in sorted(paths):
        print(" -", path)

    required = {
        "compiler_bottlenecks/summary.json",
        "compiler_bottlenecks/slowest_passes.png",
        "compiler_bottlenecks/stage_runtime.png",
    }

    missing = required - paths

    if missing:
        raise AssertionError(f"Missing MLflow artifacts: {missing}")

    # Read the actual compiler provenance record.
    record_path = client.download_artifacts(
        run_id,
        "qprov_record.json",
    )

    with open(record_path, encoding="utf-8") as file:
        record = json.load(file)

    passes = record["compilation"]["passes"]

    if len(passes) < 2:
        raise AssertionError(
            "Expected at least two actual compiler-pass records"
        )

    # Verify recorded resource changes across consecutive passes.
    for previous, current in zip(passes, passes[1:]):
        previous_metadata = previous["pass_metadata"]
        current_metadata = current["pass_metadata"]

        for metric in ("depth", "size"):
            expected_delta = (
                current_metadata[metric]
                - previous_metadata[metric]
            )

            assert (
                current_metadata[f"delta_{metric}"]
                == expected_delta
            )

    summary_path = client.download_artifacts(
        run_id,
        "compiler_bottlenecks/summary.json",
    )

    with open(summary_path, encoding="utf-8") as file:
        summary = json.load(file)

    assert summary["pass_count"] == len(passes)

    print("\nCompiler profiling results:")
    print("Recorded passes:", len(passes))
    print(
        "Accumulated pass time:",
        round(summary["total_recorded_pass_time_s"], 4),
        "seconds",
    )

    if summary["top_slowest_passes"]:
        slowest = summary["top_slowest_passes"][0]
        print("Slowest pass:", slowest["pass_name"])
        print(
            "Slowest pass duration:",
            round(slowest["duration_s"] * 1000, 3),
            "ms",
        )

    print("PASS: Real compiler profiling artifacts validated")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # Enable the repository's existing Qiskit instrumentation.
    autolog(silent=True)

    source = build_modular_adder_17()

    print("\nQARW MILESTONE 6 — REAL CIRCUIT VALIDATION")
    print("-" * 60)
    print("Source qubits:", source.num_qubits)
    print("Source depth:", source.depth())
    print("Source operations:", source.size())

    with mlflow.start_run(
        run_name="milestone6_mod17_baseline_L3_seed42"
    ) as run:

        run_id = run.info.run_id

        mlflow.set_tags({
            "milestone": "6",
            "architecture": "baseline",
            "benchmark": "compiler_bottleneck_validation",
            "topology": "unrestricted",
            "basis": "CX",
        })

        # Access the patched Qiskit function through its module.
        manager = qiskit.transpiler.generate_preset_pass_manager(
            optimization_level=3,
            basis_gates=["u", "cx"],
            seed_transpiler=42,
            qubits_initially_zero=False,
        )

        # Running the staged manager triggers QARW callbacks.
        compiled = manager.run(source)

        print("\nCompiled circuit:")
        print("Qubits:", compiled.num_qubits)
        print("Depth:", compiled.depth())
        print("Operations:", compiled.size())
        print("Gate counts:", dict(compiled.count_ops()))

        validate_equivalence(source, compiled)

    # The patched MLflow run-exit handler uploads profiler artifacts.
    validate_mlflow_artifacts(run_id)

    print("\nSUCCESS: Milestone 6 real-circuit validation passed")
    print("MLflow run ID:", run_id)


if __name__ == "__main__":
    main()
