import os
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from mlflow.tracking import MlflowClient

from q8s.runtime.mlflow.qiskit import autolog

autolog()

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager


# 1. Configure MLflow.
tracking_dir = Path(__file__).resolve().parent / "mlruns"
tracking_dir.mkdir(exist_ok=True)

mlflow.set_tracking_uri(tracking_dir.as_uri())
mlflow.set_experiment("QARW-Compiler-Profiler")


# 2. Reuse our original demonstration circuit.
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


# 3. Compare optimization levels.
client = MlflowClient()
results = []

for level in range(4):
    print(f"\nRunning optimization level {level}...")

    with mlflow.start_run(
        run_name=f"optimization-level-{level}"
    ) as run:
        mlflow.log_param("optimization_level", level)
        mlflow.log_param("seed_transpiler", 42)

        #Create the pass manager inside the active run.
        manager = generate_preset_pass_manager(
            optimization_level=level,
            basis_gates=["u", "cx"],
            seed_transpiler=42,
        )

        compiled = manager.run(qc)

        depth = compiled.depth()
        size = compiled.size()
        cx_count = int(compiled.count_ops().get("cx", 0))

        mlflow.log_metric("comparison_depth", depth)
        mlflow.log_metric("comparison_size", size)
        mlflow.log_metric("comparison_cx_count", cx_count)

        run_id = run.info.run_id

    # 4. Verify Qubernetes saved this compilation.
    saved_run = client.get_run(run_id)

    artifacts = {
        item.path
        for item in client.list_artifacts(run_id)
    }

    assert saved_run.info.status == "FINISHED"
    assert saved_run.data.metrics["passes_count"] > 0
    assert "qprov_record.json" in artifacts
    assert "transpilation_timeline.png" in artifacts

    result = {
        "level": level,
        "depth": depth,
        "size": size,
        "cx_count": cx_count,
        "passes": int(
            saved_run.data.metrics["passes_count"]
        ),
        "duration": saved_run.data.metrics[
            "transpilation_duration"
        ],
    }

    results.append(result)


# 5. Display comparison.
print("\nOPTIMIZATION COMPARISON")
print("-" * 75)

print(
    f"{'Level':<8}"
    f"{'Depth':<8}"
    f"{'Size':<8}"
    f"{'CX':<8}"
    f"{'Passes':<10}"
    f"{'Time (s)':<12}"
)

for result in results:
    print(
        f"{result['level']:<8}"
        f"{result['depth']:<8}"
        f"{result['size']:<8}"
        f"{result['cx_count']:<8}"
        f"{result['passes']:<10}"
        f"{result['duration']:<12.6f}"
    )

print("\nAll four compilation runs saved to MLflow.")
