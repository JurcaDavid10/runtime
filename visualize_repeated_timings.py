
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import median

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    PROJECT_DIR
    / "resource_visualizations"
    / "repeated_timing_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = (
    PROJECT_DIR
    / "resource_visualizations"
    / "repeated_benchmarks.csv"
)

WIDTHS = [2, 4, 6]
LEVELS = [0, 1, 2, 3]
TRIALS = 5

mlflow.set_tracking_uri(
    (PROJECT_DIR / "mlruns").as_uri()
)

# 1. Read all recorded trials.
groups = defaultdict(list)

with INPUT_CSV.open(
    "r", newline="", encoding="utf-8"
) as file:
    reader = csv.DictReader(file)

    for row in reader:
        key = (int(row["bits"]), int(row["level"]))
        groups[key].append(float(row["time_ms"]))


# 2. Compute statistics.
summary = {}

for bits in WIDTHS:
    for level in LEVELS:
        key = (bits, level)
        values = groups[key]

        assert len(values) == TRIALS, (
            f"Expected {TRIALS} trials for {key}, "
            f"found {len(values)}"
        )

        summary[key] = {
            "bits": bits,
            "level": level,
            "trials": len(values),
            "median_ms": median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }


# 3. Generate the timing chart.
fig, ax = plt.subplots(figsize=(10, 6))

for level in LEVELS:
    values = [
        summary[(bits, level)]
        for bits in WIDTHS
    ]

    medians = [v["median_ms"] for v in values]

    lower = [
        v["median_ms"] - v["min_ms"]
        for v in values
    ]

    upper = [
        v["max_ms"] - v["median_ms"]
        for v in values
    ]

    ax.errorbar(
        WIDTHS,
        medians,
        yerr=[lower, upper],
        marker="o",
        capsize=5,
        linewidth=2,
        label=f"Optimization level {level}",
    )

ax.set_xlabel("Adder register size (bits)")
ax.set_ylabel("Instrumented compilation time (ms)")

ax.set_title(
    "CDKM Adder Compilation Time\n"
    "Median and Observed Min/Max (5 trials)"
)

ax.set_xticks(WIDTHS)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

chart_path = OUTPUT_DIR / "repeated_timing.png"
fig.savefig(chart_path, dpi=160)
plt.close(fig)

print("Saved: repeated_timing.png")


# 4. Save a summary CSV.
csv_path = OUTPUT_DIR / "timing_summary.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "bits",
            "level",
            "trials",
            "median_ms",
            "min_ms",
            "max_ms",
        ],
    )

    writer.writeheader()

    for value in summary.values():
        writer.writerow(value)

print("Saved: timing_summary.csv")


# 5. Upload to the existing MLflow summary run.
client = MlflowClient()

experiment = client.get_experiment_by_name(
    "QARW-Quantum-Arithmetic-Benchmarks"
)

assert experiment is not None

runs = client.search_runs(
    [experiment.experiment_id],
    filter_string=(
        "tags.mlflow.runName = "
        "'arithmetic-benchmark-summary'"
    ),
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

assert runs, "Existing benchmark summary run not found"

client.log_artifacts(
    runs[0].info.run_id,
    str(OUTPUT_DIR),
    artifact_path="repeated_timing",
)

print("\nSUCCESS: Repeated timing analysis saved to MLflow!")
