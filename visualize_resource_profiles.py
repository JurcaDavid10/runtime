
import os
import json
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient


# 1. Connect to our existing MLflow experiments.
project_dir = Path(__file__).resolve().parent
mlflow.set_tracking_uri((project_dir / "mlruns").as_uri())

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "QARW-Compiler-Profiler"
)

assert experiment is not None, "Experiment not found"


# 2. Find the latest optimization-level-1 run.
runs = client.search_runs(
    [experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=100,
)

run = next(
    (
        r for r in runs
        if r.data.tags.get("mlflow.runName")
        == "optimization-level-1"
    ),
    None,
)

assert run is not None, "Optimization level 1 run not found"


# 3. Download our custom profiling information.
record_path = client.download_artifacts(
    run.info.run_id,
    "qprov_record.json",
)

with open(record_path, encoding="utf-8") as file:
    record = json.load(file)


def find_profiles(value):
    if isinstance(value, dict):
        profile = value.get("resource_profile")

        if isinstance(profile, dict):
            yield profile

        for child in value.values():
            yield from find_profiles(child)

    elif isinstance(value, list):
        for child in value:
            yield from find_profiles(child)


profiles = list(find_profiles(record))

assert profiles, "No resource profiles found"

print("Resource profiles found:", len(profiles))


# 4. Generate resource evolution charts.
output_dir = project_dir / "resource_visualizations"
output_dir.mkdir(exist_ok=True)

metrics = [
    ("depth", "Circuit depth"),
    ("two_qubit_gate_count", "Two-qubit gate count"),
    ("two_qubit_depth", "Two-qubit depth"),
]

pass_numbers = list(range(1, len(profiles) + 1))

for metric, label in metrics:
    values = [p.get(metric) for p in profiles]

    if any(value is None for value in values):
        print(f"Skipping {metric}: missing values")
        continue

    plt.figure(figsize=(10, 4))

    plt.plot(
        pass_numbers,
        values,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Recorded compiler pass")
    plt.ylabel(label)
    plt.title(f"{label} across compilation passes")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / f"{metric}.png"
    plt.savefig(output_file, dpi=160)
    plt.close()

    print(f"Saved: {output_file.name}")

# 5. Upload our resource visualizations to MLflow.
print("\nUploading resource visualizations to MLflow...")

for image_file in sorted(output_dir.glob("*.png")):
    client.log_artifact(
        run.info.run_id,
        str(image_file),
        artifact_path="resource_visualizations",
    )

    print(f"Uploaded: {image_file.name}")

print("All resource visualizations saved to MLflow!")

print("\nResource visualization complete!")
