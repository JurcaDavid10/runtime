

import json
import os
from pathlib import Path

# Allow MLflow to use local filesystem storage.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from mlflow.tracking import MlflowClient

# Enable Qubernetes autologging before importing
# Qiskit's pass manager function.
from q8s.runtime.mlflow.qiskit import autolog

autolog()

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager




# 1. Configure local MLflow storage.
tracking_dir = Path(__file__).resolve().parent / "mlruns"
tracking_dir.mkdir(exist_ok=True)

mlflow.set_tracking_uri(tracking_dir.as_uri())
mlflow.set_experiment("QARW-Compiler-Profiler")




# 2. Create a small quantum circuit.
qc = QuantumCircuit(3)

qc.h(0)
qc.cx(0, 1)
qc.cx(0, 1)
qc.h(0)
qc.h(0)
qc.cx(1, 2)
qc.cx(1, 2)
qc.t(2)
qc.tdg(2)
qc.h(1)
qc.cx(0, 2)

print("Original circuit:")
print("Depth:", qc.depth())
print("Operations:", qc.size())


# 3. Run real Qiskit compilation with MLflow enabled.
with mlflow.start_run(run_name="real-qiskit-profile") as run:
    run_id = run.info.run_id

    manager = generate_preset_pass_manager(
        optimization_level=2,
        basis_gates=["u", "cx"],
        seed_transpiler=42,
    )

    compiled = manager.run(qc)

    print("\nCompiled circuit:")
    print("Depth:", compiled.depth())
    print("Operations:", compiled.size())

# Qubernetes should save its profiling data when
# the MLflow run closes.


# 4. Verify that MLflow recorded the compilation.
client = MlflowClient()
saved_run = client.get_run(run_id)

print("\nMLflow run:", run_id)
print("Run status:", saved_run.info.status)
print("Recorded metrics:", saved_run.data.metrics)

artifacts = [
    artifact.path
    for artifact in client.list_artifacts(run_id)
]

print("Recorded artifacts:", artifacts)

required_metrics = {
    "passes_count",
    "transpilation_duration",
    "circuit_depth",
    "circuit_size",
}

assert saved_run.info.status == "FINISHED"
assert required_metrics.issubset(saved_run.data.metrics)
assert saved_run.data.metrics["passes_count"] > 0
assert "qprov_record.json" in artifacts
assert "transpilation_timeline.png" in artifacts


# 5. Verify our custom resource profiles were saved.
record_path = client.download_artifacts(
    run_id, "qprov_record.json"
)

with open(record_path, encoding="utf-8") as file:
    record = json.load(file)


def find_profiles(value):
    profiles = []

    if isinstance(value, dict):
        if isinstance(value.get("resource_profile"), dict):
            profiles.append(value["resource_profile"])

        for child in value.values():
            profiles.extend(find_profiles(child))

    elif isinstance(value, list):
        for child in value:
            profiles.extend(find_profiles(child))

    return profiles


profiles = find_profiles(record)

print("Saved resource profiles:", len(profiles))

assert profiles, "No resource profiles found in MLflow artifact"

print("\nExample saved resource profile:")
print(json.dumps(profiles[0], indent=2))

print("\nSUCCESS: End-to-end MLflow integration verified!")
