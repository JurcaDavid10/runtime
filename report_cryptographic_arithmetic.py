
"""Visualize and log cryptographic arithmetic benchmarks."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np


OUTPUT_DIR = Path(
    "benchmark_outputs/cryptographic_arithmetic"
)

EXPERIMENT_NAME = "QARW-Cryptographic-Arithmetic"


def load_results():

    with (
        OUTPUT_DIR / "arithmetic_summary.csv"
    ).open(encoding="utf-8") as file:

        compiler_rows = list(csv.DictReader(file))

    with (
        OUTPUT_DIR / "arithmetic_logical_resources.json"
    ).open(encoding="utf-8") as file:

        logical_rows = json.load(file)

    return compiler_rows, logical_rows


def grouped_chart(
    labels,
    addition,
    subtraction,
    title,
    ylabel,
    filename,
):

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_add = ax.bar(
        x - width / 2,
        addition,
        width,
        label="Addition",
        color="#64748b",
    )

    bars_sub = ax.bar(
        x + width / 2,
        subtraction,
        width,
        label="Subtraction",
        color="#2563eb",
    )

    ax.bar_label(
        bars_add,
        fmt="%.0f",
        padding=3,
        fontsize=9,
    )

    ax.bar_label(
        bars_sub,
        fmt="%.0f",
        padding=3,
        fontsize=9,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax.set_ylim(
        0,
        max(addition + subtraction) * 1.18,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_compiler_charts(rows):

    rows = [
        row for row in rows
        if int(row["optimization_level"]) == 3
    ]

    architectures = [
        "baseline",
        "optimized_rccx",
    ]

    labels = [
        "Baseline",
        "Optimized RCCX",
    ]

    metrics = [
        (
            "depth_mean",
            "Compiled circuit depth (L3)",
            "Circuit depth",
            "compiled_depth_comparison.png",
        ),
        (
            "cx_mean",
            "Compiled CX count (L3)",
            "CX gates",
            "compiled_cx_comparison.png",
        ),
    ]

    for metric, title, ylabel, filename in metrics:

        def values(operation):
            return [
                float(next(
                    row[metric]
                    for row in rows
                    if row["operation"] == operation
                    and row["architecture"] == arch
                ))
                for arch in architectures
            ]

        grouped_chart(
            labels,
            values("addition"),
            values("subtraction"),
            title,
            ylabel,
            filename,
        )


def create_logical_charts(rows):

    configurations = list(dict.fromkeys(
        (
            row["architecture"],
            row["mcx_policy"],
        )
        for row in rows
    ))

    labels = [
        f"{arch}\n{policy}"
        for arch, policy in configurations
    ]

    metrics = [
        (
            "t_count",
            "Fault-tolerant T-count",
            "T / Tdg gates",
            "logical_t_count_comparison.png",
        ),
        (
            "t_depth",
            "Fault-tolerant T-depth",
            "T-depth",
            "logical_t_depth_comparison.png",
        ),
        (
            "logical_qubits",
            "Logical qubit requirements",
            "Logical qubits",
            "logical_qubits_comparison.png",
        ),
    ]

    for metric, title, ylabel, filename in metrics:

        def values(operation):
            return [
                next(
                    row[metric]
                    for row in rows
                    if row["operation"] == operation
                    and row["architecture"] == arch
                    and row["mcx_policy"] == policy
                )
                for arch, policy in configurations
            ]

        grouped_chart(
            labels,
            values("addition"),
            values("subtraction"),
            title,
            ylabel,
            filename,
        )


def log_to_mlflow(
    tracking_uri,
    compiler_rows,
    logical_rows,
):

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="mod17_addition_subtraction_comparison"
    ):

        mlflow.log_params({
            "modulus": 17,
            "register_width": 5,
            "operations": "addition,subtraction",
            "compiler_basis": "u,cx",
            "connectivity": "unrestricted",
        })

        for row in compiler_rows:

            prefix = (
                f"{row['operation']}."
                f"{row['architecture']}."
                f"L{row['optimization_level']}"
            )

            for metric in [
                "depth_mean",
                "cx_mean",
                "operations_mean",
                "two_qubit_depth_mean",
            ]:
                mlflow.log_metric(
                    f"{prefix}.{metric}",
                    float(row[metric]),
                )

        for row in logical_rows:

            prefix = (
                f"logical.{row['operation']}."
                f"{row['architecture']}."
                f"{row['mcx_policy']}"
            )

            for metric in [
                "logical_qubits",
                "t_count",
                "t_depth",
            ]:
                mlflow.log_metric(
                    f"{prefix}.{metric}",
                    float(row[metric]),
                )

        mlflow.log_artifacts(
            str(OUTPUT_DIR),
            artifact_path="cryptographic_arithmetic",
        )

        print("MLflow logging completed.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-mlflow",
        action="store_true",
    )

    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
    )

    args = parser.parse_args()

    compiler_rows, logical_rows = load_results()

    create_compiler_charts(compiler_rows)
    create_logical_charts(logical_rows)

    print("Generated 5 benchmark visualizations.")
    print(f"Output directory: {OUTPUT_DIR}")

    if args.log_mlflow:
        log_to_mlflow(
            args.tracking_uri,
            compiler_rows,
            logical_rows,
        )


if __name__ == "__main__":
    main()
