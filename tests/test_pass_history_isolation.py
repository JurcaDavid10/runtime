
import json

import mlflow

from mlflow.tracking import MlflowClient
from q8s.runtime.mlflow.qiskit import autolog


def test_compilations_do_not_accumulate_passes(
    tmp_path, monkeypatch
):
    """Each MLflow run must record only its own passes."""

    monkeypatch.setenv(
        "MLFLOW_ALLOW_FILE_STORE", "true"
    )

    previous_uri = mlflow.get_tracking_uri()

    try:
        # Use temporary MLflow storage for this test.
        tracking_dir = tmp_path / "mlruns"
        mlflow.set_tracking_uri(tracking_dir.as_uri())
        mlflow.set_experiment("pass-isolation-test")

        autolog()

        from qiskit import QuantumCircuit
        from qiskit.transpiler import (
            generate_preset_pass_manager,
        )

        client = MlflowClient()
        recorded_counts = []

        # Compile the same circuit in two separate runs.
        for run_number in range(2):

            circuit = QuantumCircuit(3)
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.cx(1, 2)

            with mlflow.start_run(
                run_name=f"isolation-{run_number}"
            ) as run:

                manager = generate_preset_pass_manager(
                    optimization_level=1,
                    basis_gates=["u", "cx"],
                    seed_transpiler=42,
                )

                manager.run(circuit)
                run_id = run.info.run_id

            # Read the saved MLflow record.
            saved_run = client.get_run(run_id)

            record_path = client.download_artifacts(
                run_id,
                "qprov_record.json",
            )

            with open(
                record_path, encoding="utf-8"
            ) as file:
                record = json.load(file)

            passes = record["compilation"]["passes"]

            metric_count = int(
                saved_run.data.metrics["passes_count"]
            )

            assert saved_run.info.status == "FINISHED"
            assert len(passes) == metric_count
            assert metric_count > 0

            recorded_counts.append(metric_count)

        # The second run must not include passes
        # from the first compilation.
        assert recorded_counts[0] == recorded_counts[1]

    finally:
        mlflow.set_tracking_uri(previous_uri)
