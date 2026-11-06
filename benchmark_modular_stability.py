"""Repeat QARW modulo-17 transpilation benchmarks across multiple seeds.

Run from the QARW runtime directory (with MLflow server already running):

    $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    python benchmark_modular_stability.py

By default, runs seeds 42, 101, 202, 303 and 404 for BOTH architectures,
at optimization levels 0-3. Fully completed seed/architecture combinations
are reused (including the seed-101 data generated previously). Add --force
to rerun them and create new MLflow runs.

All graphs and summary CSV files go in benchmark_outputs/modular_add_17/stability.
A new MLflow run stores those summary artifacts. This benchmark measures
compilation resource counts, not hardware execution or functional correctness.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots without requiring a graphical display.
import matplotlib.pyplot as plt
import mlflow

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "benchmark_outputs" / "modular_add_17"
SUMMARY_DIR = OUTPUT_ROOT / "stability"
EXPERIMENT = "QARW-Modular-Arithmetic-Benchmarks"
DEFAULT_SEEDS = (42, 101, 202, 303, 404)
LEVELS = range(4)

ARCHITECTURES = {
    "baseline": {
        "script": ROOT / "benchmark_modular_arithmetic.py",
        "directory": OUTPUT_ROOT,
        "csv_prefix": "comparison",
    },
    "optimized": {
        "script": ROOT / "benchmark_modular_arithmetic_optimized.py",
        "directory": OUTPUT_ROOT / "relative_phase_ancilla",
        "csv_prefix": "optimization_comparison",
    },
}

METRICS = (
    "compiled_depth",
    "compiled_cx_count",
    "compiled_two_qubit_depth",
    "compiled_operations",
    "compilation_wall_time_s",
)
PLOT_METRICS = (
    ("compiled_depth", "Circuit depth"),
    ("compiled_cx_count", "CX gates"),
    ("compiled_two_qubit_depth", "Two-qubit depth"),
)


def json_path(architecture: str, level: int, seed: int) -> Path:
    return (
        ARCHITECTURES[architecture]["directory"]
        / f"optimization_level_{level}_seed{seed}.json"
    )


def comparison_path(architecture: str, seed: int) -> Path:
    config = ARCHITECTURES[architecture]
    return config["directory"] / f"{config['csv_prefix']}_seed{seed}.csv"


def load_results(architecture: str, seed: int) -> list[dict]:
    """Read the four per-level JSON files, checking for complete seed data."""
    results = []
    for level in LEVELS:
        record = json.loads(json_path(architecture, level, seed).read_text(encoding="utf-8"))
        if record.get("seed_transpiler") != seed or record.get("optimization_level") != level:
            raise ValueError(f"Seed/level mismatch: {architecture}, level={level}, seed={seed}")
        if record.get("basis_gates") != ["u", "cx"]:
            raise ValueError(f"Unexpected basis gates: {architecture}, seed={seed}, level={level}")
        if architecture == "optimized" and record.get("architecture") != "rccx_clean_ancilla_3":
            raise ValueError(f"Unexpected optimized architecture in {json_path(architecture, level, seed)}")
        if not isinstance(record.get("mlflow_run_id"), str) or not record["mlflow_run_id"]:
            raise ValueError(f"Missing MLflow run ID: {architecture}, seed={seed}, level={level}")
        for metric in METRICS + ("compiled_qubits",):
            if not isinstance(record.get(metric), (int, float)):
                raise ValueError(f"Missing metric {metric}: {architecture}, seed={seed}, level={level}")
        record["architecture"] = architecture  # The baseline JSON has no architecture field.
        results.append(record)
    return results


def is_complete(architecture: str, seed: int) -> bool:
    """Skip already-completed work only if BOTH the CSV and JSONs are valid."""
    csv_path = comparison_path(architecture, seed)
    if not csv_path.is_file():
        return False
    try:
        results = load_results(architecture, seed)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return (
            len(rows) == len(results)
            and {int(row["optimization_level"]) for row in rows} == set(LEVELS)
            and all(int(row["seed_transpiler"]) == seed for row in rows)
            and all(
                next(row for row in rows if int(row["optimization_level"]) == r["optimization_level"])["mlflow_run_id"]
                == r["mlflow_run_id"]
                for r in results
            )
        )
    except (OSError, KeyError, ValueError, TypeError, StopIteration, json.JSONDecodeError):
        return False


def run_benchmarks(seeds: tuple[int, ...], force: bool, tracking_uri: str) -> list[dict]:
    environment = os.environ.copy()
    environment["MLFLOW_TRACKING_URI"] = tracking_uri
    rows = []
    for seed in seeds:
        for architecture, config in ARCHITECTURES.items():
            if not config["script"].is_file():
                raise FileNotFoundError(f"Missing existing benchmark script: {config['script']}")
            if not force and is_complete(architecture, seed):
                print(f"REUSE: {architecture}, seed {seed}: 4 previously completed levels", flush=True)
            else:
                print(f"\nRUN: {architecture}, seed {seed}: levels 0-3", flush=True)
                environment["QARW_TRANSPILER_SEED"] = str(seed)
                subprocess.run(
                    [sys.executable, str(config["script"]), "--all"],
                    cwd=str(ROOT),
                    env=environment,
                    check=True,
                )
                if not is_complete(architecture, seed):
                    raise RuntimeError(f"No valid results CSV/JSON after {architecture}, seed {seed}")
            rows.extend(load_results(architecture, seed))
    return rows


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_statistics(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    lookup = {}
    for row in rows:
        groups[(row["architecture"], row["optimization_level"])].append(row)
        lookup[(row["architecture"], row["optimization_level"], row["seed_transpiler"])] = row

    summaries = []
    for (architecture, level), records in sorted(groups.items()):
        summary = {
            "architecture": architecture,
            "optimization_level": level,
            "n_seeds": len(records),
            "compiled_qubits": records[0]["compiled_qubits"],
        }
        for metric in METRICS:
            values = [float(r[metric]) for r in records]
            avg, sd = mean_sd(values)
            summary.update({
                f"{metric}_mean": avg,
                f"{metric}_std": sd,
                f"{metric}_min": min(values),
                f"{metric}_max": max(values),
            })
        summaries.append(summary)

    paired_improvements = []
    for level in LEVELS:
        for metric in METRICS:
            paired = []
            for seed in sorted({r["seed_transpiler"] for r in rows}):
                base = float(lookup[("baseline", level, seed)][metric])
                improved = float(lookup[("optimized", level, seed)][metric])
                if base == 0:
                    raise ValueError(f"Cannot calculate improvement for zero baseline: {metric}")
                paired.append(100.0 * (base - improved) / base)
            avg, sd = mean_sd(paired)
            paired_improvements.append({
                "optimization_level": level,
                "metric": metric,
                "n_seeds": len(paired),
                "reduction_percent_mean": avg,
                "reduction_percent_std": sd,
                "reduction_percent_min": min(paired),
                "reduction_percent_max": max(paired),
            })
    return summaries, paired_improvements


def draw_metric_chart(summaries: list[dict], metric: str, label: str) -> Path:
    lookup = {(r["architecture"], r["optimization_level"]): r for r in summaries}
    x = list(LEVELS)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    for architecture, offset in (("baseline", -width / 2), ("optimized", width / 2)):
        means = [lookup[(architecture, level)][f"{metric}_mean"] for level in x]
        sds = [lookup[(architecture, level)][f"{metric}_std"] for level in x]
        ax.bar([i + offset for i in x], means, width, label=architecture.title(), yerr=sds, capsize=4)
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlabel("Qiskit optimization level")
    ax.set_ylabel(label)
    ax.set_title(f"Modulo-17 adders: {label} across seeds (mean ± SD)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = SUMMARY_DIR / f"stability_{metric}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def print_level3(summaries: list[dict], improvements: list[dict]) -> None:
    indexed = {(r["architecture"], r["optimization_level"]): r for r in summaries}
    indexed_improvements = {(r["optimization_level"], r["metric"]): r for r in improvements}
    print("\nLEVEL 3 STABILITY SUMMARY (MEAN OVER SEEDS)")
    print("-" * 82)
    for metric, label in PLOT_METRICS:
        a = indexed[("baseline", 3)][f"{metric}_mean"]
        b = indexed[("optimized", 3)][f"{metric}_mean"]
        benefit = indexed_improvements[(3, metric)]
        print(f"{label:20} baseline={a:9.2f}  optimized={b:9.2f}  "
              f"paired reduction={benefit['reduction_percent_mean']:6.2f}% "
              f"± {benefit['reduction_percent_std']:.2f} pp")
    print("Qubits: baseline=10, optimized=13 (additional qubit cost: +3)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Transpiler seeds; default: 42 101 202 303 404")
    parser.add_argument("--force", action="store_true",
                        help="Rerun completed seed/architecture combinations (creates additional MLflow runs)")
    args = parser.parse_args()
    seeds = tuple(dict.fromkeys(args.seeds))  # Preserve order, ignore duplicates.
    if not seeds:
        parser.error("At least one seed is required")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        # Fail early rather than running dozens of benchmarks with an unavailable server.
        mlflow.get_experiment_by_name(EXPERIMENT)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot contact MLflow at {tracking_uri}; start the server and/or set "
            "MLFLOW_TRACKING_URI in this terminal before running benchmarks."
        ) from exc
    print("MLflow tracking URI:", tracking_uri)
    print("Seeds:", seeds)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    rows = run_benchmarks(seeds, args.force, tracking_uri)
    versions = {row["qiskit_version"] for row in rows}
    if len(versions) != 1:
        raise RuntimeError(f"Results were compiled with different Qiskit versions: {versions}")
    rows.sort(key=lambda x: (x["seed_transpiler"], x["optimization_level"], x["architecture"]))
    run_columns = [
        "architecture", "seed_transpiler", "optimization_level", "qiskit_version",
        "source_qubits", "compiled_qubits", "compiled_depth", "compiled_cx_count",
        "compiled_two_qubit_depth", "compiled_operations", "compiled_u_count",
        "compiled_two_qubit_gate_count", "compilation_wall_time_s", "mlflow_run_id",
    ]
    write_csv(SUMMARY_DIR / "modular_stability_runs.csv", rows, run_columns)

    summaries, improvements = create_statistics(rows)
    summary_columns = ["architecture", "optimization_level", "n_seeds", "compiled_qubits"]
    for metric in METRICS:
        summary_columns.extend(f"{metric}_{stat}" for stat in ("mean", "std", "min", "max"))
    write_csv(SUMMARY_DIR / "modular_stability_summary.csv", summaries, summary_columns)
    write_csv(
        SUMMARY_DIR / "modular_stability_improvements.csv", improvements,
        ["optimization_level", "metric", "n_seeds", "reduction_percent_mean",
         "reduction_percent_std", "reduction_percent_min", "reduction_percent_max"],
    )
    for metric, label in PLOT_METRICS:
        draw_metric_chart(summaries, metric, label)

    # Store all aggregate files under ONE MLflow run; the 40 individual trials
    # are already recorded by their respective benchmark scripts.
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"mod17_stability_summary_{len(seeds)}seeds") as run:
        mlflow.set_tags({
            "workload": "modular_addition",
            "analysis": "repeated_transpilation_stability",
            "architectures": "baseline,optimized",
        })
        mlflow.log_params({
            "seed_values": ",".join(map(str, seeds)),
            "seed_count": len(seeds),
            "optimization_levels": "0,1,2,3",
            "basis_gates": "u,cx",
            "qiskit_version": next(iter(versions)),
            "connectivity": "all_to_all_idealized",
        })
        for record in improvements:
            if record["optimization_level"] == 3:
                mlflow.log_metric(
                    f"level3_{record['metric']}_reduction_pct_mean",
                    record["reduction_percent_mean"],
                )
        mlflow.log_artifacts(str(SUMMARY_DIR), artifact_path="stability")
        summary_run_id = run.info.run_id

    print_level3(summaries, improvements)
    print("\nResults saved in:", SUMMARY_DIR)
    print("MLflow summary run ID:", summary_run_id)
    print("Total individual benchmark records:", len(rows))


if __name__ == "__main__":
    main()
