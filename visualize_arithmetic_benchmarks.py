
import csv
import os
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient


# 1. Configuration
PROJECT_DIR = Path(__file__).resolve().parent

mlflow.set_tracking_uri(
    (PROJECT_DIR / "mlruns").as_uri()
)

EXPERIMENT_NAME = "QARW-Quantum-Arithmetic-Benchmarks"

WIDTHS = [2, 4, 6]
LEVELS = [0, 1, 2, 3]

OUTPUT_DIR = (
    PROJECT_DIR
    / "resource_visualizations"
    / "arithmetic_benchmarks"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Retrieve the recorded experiments
client = MlflowClient()

experiment = client.get_experiment_by_name(
    EXPERIMENT_NAME
)

assert experiment is not None, "MLflow experiment not found"

runs = client.search_runs(
    [experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=500,
)

required_metrics = {
    "arithmetic_depth",
    "arithmetic_operations",
    "arithmetic_cx_count",
    "arithmetic_two_qubit_depth",
    "passes_count",
    "transpilation_duration",
}

latest = {}

for run in runs:
    if run.info.status != "FINISHED":
        continue

    if not required_metrics.issubset(run.data.metrics):
        continue

    name = run.data.tags.get("mlflow.runName", "")

    for bits in WIDTHS:
        for level in LEVELS:
            key = (bits, level)
            expected = f"adder-{bits}bit-opt-{level}"

            # Results are sorted newest first.
            if name == expected and key not in latest:
                latest[key] = run

missing = [
    (bits, level)
    for bits in WIDTHS
    for level in LEVELS
    if (bits, level) not in latest
]

if missing:
    raise RuntimeError(
        f"Missing benchmark configurations: {missing}"
    )


# 3. Extract resource metrics
results = {}

for bits in WIDTHS:
    for level in LEVELS:
        run = latest[(bits, level)]
        metrics = run.data.metrics

        results[(bits, level)] = {
            "bits": bits,
            "level": level,
            "qubits": 2 * bits + 1,
            "depth": int(metrics["arithmetic_depth"]),
            "operations": int(
                metrics["arithmetic_operations"]
            ),
            "cx": int(
                metrics["arithmetic_cx_count"]
            ),
            "two_qubit_depth": int(
                metrics["arithmetic_two_qubit_depth"]
            ),
            "passes": int(metrics["passes_count"]),
            "time_ms": (
                metrics["transpilation_duration"] * 1000
            ),
            "source_run_id": run.info.run_id,
        }

print(f"Loaded {len(results)} benchmark configurations.")


# 4. Define graphs to generate
charts = [
    ("depth", "Circuit depth", "circuit_depth.png"),
    ("cx", "CX gate count", "cx_gate_count.png"),
    (
        "two_qubit_depth",
        "Two-qubit depth",
        "two_qubit_depth.png",
    ),
    (
        "time_ms",
        "Compilation time (ms) - single run",
        "compilation_time.png",
    ),
]


# 5. Generate comparative graphs
for metric, label, filename in charts:

    fig, ax = plt.subplots(figsize=(10, 5))

    for level in LEVELS:
        values = [
            results[(bits, level)][metric]
            for bits in WIDTHS
        ]

        ax.plot(
            WIDTHS,
            values,
            marker="o",
            linewidth=2,
            label=f"Optimization level {level}",
        )

    ax.set_xlabel("Adder register size (bits)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs. adder register size")

    ax.set_xticks(WIDTHS)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_file = OUTPUT_DIR / filename
    fig.savefig(output_file, dpi=160)
    plt.close(fig)

    print(f"Saved: {filename}")


# 6. Save a reproducible CSV table
csv_path = OUTPUT_DIR / "benchmark_results.csv"

fieldnames = [
    "bits",
    "level",
    "qubits",
    "depth",
    "operations",
    "cx",
    "two_qubit_depth",
    "passes",
    "time_ms",
    "source_run_id",
]

with csv_path.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=fieldnames
    )

    writer.writeheader()

    for result in results.values():
        writer.writerow(result)

print("Saved: benchmark_results.csv")


# 7. Upload the graphs and CSV to a new MLflow run
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(
    run_name="arithmetic-benchmark-summary"
) as summary_run:

    mlflow.log_param("benchmark_family", "CDKM")
    mlflow.log_param("configurations", len(results))
    mlflow.log_param("timing_trials_per_configuration", 1)

    mlflow.set_tag(
        "analysis_type", "single-run-comparison"
    )

    client.log_artifacts(
        summary_run.info.run_id,
        str(OUTPUT_DIR),
        artifact_path="comparative_visualizations",
    )

print("\nSUCCESS: Comparative benchmarks saved to MLflow!")
