
import csv
import os
from pathlib import Path
from statistics import median

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from mlflow.tracking import MlflowClient
from q8s.runtime.mlflow.qiskit import autolog

autolog()

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit.transpiler import generate_preset_pass_manager


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "resource_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri((PROJECT_DIR / "mlruns").as_uri())
mlflow.set_experiment("QARW-Quantum-Arithmetic-Benchmarks")

WIDTHS = [2, 4, 6]
LEVELS = [0, 1, 2, 3]
TRIALS = 5
SEED = 42

client = MlflowClient()
results = []


def build_adder(n):
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    helper = QuantumRegister(1, "helper")

    qc = QuantumCircuit(a, b, helper)

    adder = CDKMRippleCarryAdder(
        num_state_qubits=n,
        kind="fixed",
    )

    qc.compose(adder, qubits=qc.qubits, inplace=True)

    return qc.decompose(reps=4)


# Run repeated compilations.
for bits in WIDTHS:
    circuit = build_adder(bits)

    for level in LEVELS:
        print(f"\n{bits}-bit adder | Optimization level {level}")

        for trial in range(1, TRIALS + 1):
            with mlflow.start_run(
                run_name=(
                    f"adder-{bits}bit-opt-{level}"
                    f"-trial-{trial:02d}"
                )
            ) as run:

                mlflow.log_param("register_bits", bits)
                mlflow.log_param("optimization_level", level)
                mlflow.log_param("trial", trial)
                mlflow.log_param("benchmark_type", "repeated")
                mlflow.log_param("seed_transpiler", SEED)

                manager = generate_preset_pass_manager(
                    optimization_level=level,
                    basis_gates=["u", "cx"],
                    seed_transpiler=SEED,
                )

                compiled = manager.run(circuit)
                run_id = run.info.run_id

            saved = client.get_run(run_id)
            metrics = saved.data.metrics

            assert saved.info.status == "FINISHED"

            duration_ms = (
                metrics["transpilation_duration"] * 1000
            )

            results.append({
                "bits": bits,
                "level": level,
                "trial": trial,
                "time_ms": duration_ms,
                "depth": compiled.depth(),
                "cx_count": int(
                    compiled.count_ops().get("cx", 0)
                ),
                "passes": int(metrics["passes_count"]),
                "run_id": run_id,
            })

            print(
                f" Trial {trial}: "
                f"{duration_ms:.3f} ms"
            )


# Summarize repeated measurements.
print("\n" + "=" * 65)
print("REPEATED BENCHMARK SUMMARY")
print("=" * 65)

print(
    f"{'Bits':<8}"
    f"{'Level':<9}"
    f"{'Median(ms)':<14}"
    f"{'Min(ms)':<12}"
    f"{'Max(ms)':<12}"
)

for bits in WIDTHS:
    for level in LEVELS:
        durations = [
            r["time_ms"]
            for r in results
            if r["bits"] == bits
            and r["level"] == level
        ]

        print(
            f"{bits:<8}"
            f"{level:<9}"
            f"{median(durations):<14.3f}"
            f"{min(durations):<12.3f}"
            f"{max(durations):<12.3f}"
        )


# Save all individual trials for later analysis.
output_path = OUTPUT_DIR / "repeated_benchmarks.csv"

with output_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys()),
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved: {output_path}")
print("SUCCESS: Repeated benchmarking completed!")
