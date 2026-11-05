"""Upload generated modulo-17 comparison reports to the existing local MLflow store.

Run from any working directory with the QARW project's virtual environment:
    python log_modular_comparison.py

No MLflow web server or quantum recompilation is needed.
"""

import os
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "benchmark_outputs" / "modular_add_17" / "comparison_reports"
EXPERIMENT_NAME = "QARW-Modular-Arithmetic-Benchmarks"


def main():
    if not REPORT_DIR.is_dir():
        raise FileNotFoundError(
            f"Comparison results not found: {REPORT_DIR}\n"
            "Run 'python compare_modular_optimization.py' first."
        )

    reports = [file for file in REPORT_DIR.iterdir() if file.is_file()]
    if not reports:
        raise RuntimeError(f"Comparison report directory is empty: {REPORT_DIR}")

    # MLflow 3.17 can still use the pre-existing file store in compatibility mode.
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri((ROOT / "mlruns").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="mod17_optimization_comparison") as run:
        mlflow.set_tags({
            "workload": "modular_addition",
            "modulus": "17",
            "report_type": "optimization_levels_0_to_3",
        })
        mlflow.log_artifacts(str(REPORT_DIR), artifact_path="comparison_reports")
        print(f"Logged {len(reports)} comparison report file(s) to MLflow.")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Run ID: {run.info.run_id}")
        print("Open the run's Artifacts tab to view comparison_reports/.")


if __name__ == "__main__":
    main()
