"""Report validated logical Clifford+T resources; optionally log to MLflow.

Reads existing benchmark and phase-validation JSONs, never reruns Qiskit.
These are logical gate-level costs, not surface-code or physical estimates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "benchmark_outputs" / "fault_tolerant"
ARCHITECTURES = ("baseline", "optimized_rccx")
POLICIES = ("ccx_chain", "rccx_chain")
KEYS = tuple((p, a) for p in POLICIES for a in ARCHITECTURES)
VALIDATION_KEYS = tuple((p, a) for p in ("source", *POLICIES) for a in ARCHITECTURES)
METRICS = ("source_qubits", "added_clean_ancillas", "logical_qubits", "t_count",
           "t_depth", "clifford_count", "magic_states")
LABELS = {"logical_qubits": "Logical qubits", "t_count": "T-count (T + Tdg)",
          "t_depth": "Dependency-aware T-depth", "clifford_count": "Clifford gates",
          "magic_states": "Logical T-state demand"}
EXPERIMENT = "QARW-Fault-Tolerant-Resources"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_rows(rows, expected):
    require(isinstance(rows, list), "Expected a list of configuration records")
    out = {}
    for row in rows:
        require(isinstance(row, dict), "Invalid configuration record")
        key = (row["mcx_policy"], row["architecture"])
        require(key in expected and key not in out, f"Unexpected or duplicate configuration: {key}")
        out[key] = row
    require(set(out) == set(expected), "Missing configuration(s)")
    return out


def load_inputs(input_dir):
    """Check consistency and validation status; not a cryptographic provenance check."""
    resources = index_rows(read_json(input_dir / "logical_resources.json"), KEYS)
    report = read_json(input_dir / "phase_validation.json")
    validation = index_rows(report["configurations"], VALIDATION_KEYS)
    version = report["qiskit_version"]
    require(isinstance(version, str) and bool(version), "Invalid Qiskit version")
    require(report["method"] == "exhaustive_monomial_gate_action", "Unexpected validation method")
    require(report["data_layout"] == {"a": list(range(5)), "b": list(range(5, 10))}, "Unexpected data layout")
    tol = report["absolute_tolerance"]
    require(type(tol) in (float, int) and math.isfinite(tol) and 0 < tol <= 1e-9, "Invalid validation tolerance")
    for key, row in validation.items():
        require(row["status"] == "PASS", f"Validation did not pass: {key}")
        require(row["basis_inputs_checked"] == 1024 and row["valid_arithmetic_inputs_checked"] == 289
                and row["out_of_range_b_inputs_checked"] == 480, f"Incomplete validation: {key}")
        require(all(row[k] is True for k in ("clean_ancillas", "reversible_mapping", "phase_preserved_up_to_global")),
                f"Validation flags failed: {key}")
        if key != ("source", "baseline"):
            require(row["matches_baseline_extension"] is True, f"Baseline equivalence missing: {key}")
        error = row["max_relative_phase_error"]
        require(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= tol,
                f"Invalid phase error: {key}")
    for key, row in resources.items():
        require(row["qiskit_version"] == version, f"Qiskit version mismatch: {key}")
        for metric in METRICS:
            require(type(row[metric]) is int and row[metric] >= 0, f"Invalid {metric}: {key}")
        require(row["logical_qubits"] == row["source_qubits"] + row["added_clean_ancillas"], f"Qubit accounting mismatch: {key}")
        require(row["logical_qubits"] == validation[key]["logical_qubits"], f"Validated logical width mismatch: {key}")
        require(row["source_qubits"] == validation[("source", key[1])]["logical_qubits"], f"Source width mismatch: {key}")
        require(row["magic_states"] == row["t_count"] and row["t_depth"] <= row["t_count"], f"T-resource accounting mismatch: {key}")
        counts = row["gate_counts"]
        require(isinstance(counts, dict) and all(type(v) is int and v >= 0 for v in counts.values()), f"Invalid gate counts: {key}")
        require(counts.get("t", 0) + counts.get("tdg", 0) == row["t_count"] and sum(counts.values()) == row["t_count"] + row["clifford_count"], f"Gate count mismatch: {key}")
    return resources, validation, report


def reduction(before, after):
    return 100 * (before - after) / before if before else None


def comparisons(resources):
    pairs = (("architecture_at_ccx", KEYS[0], KEYS[1]),
             ("architecture_at_rccx", KEYS[2], KEYS[3]),
             ("synthesis_on_baseline", KEYS[0], KEYS[2]),
             ("synthesis_on_optimized", KEYS[1], KEYS[3]))
    return {name: {metric: reduction(resources[a][metric], resources[b][metric])
                   for metric in LABELS} for name, a, b in pairs}


def build_summary(resources, validation, report):
    return {"workload": "modular_add_17", "qiskit_version": report["qiskit_version"],
            "model": "logical_clifford_t", "validation_method": report["method"],
            "validation_scope": "all five-bit input pairs with clean ancillas; up to global phase",
            "validation_configurations_passed": len(validation),
            "validation_basis_inputs_total": sum(r["basis_inputs_checked"] for r in validation.values()),
            "valid_arithmetic_inputs_total": sum(r["valid_arithmetic_inputs_checked"] for r in validation.values()),
            "max_relative_phase_error": max(r["max_relative_phase_error"] for r in validation.values()),
            "comparisons_percent_reduction": comparisons(resources),
            "limitations": ["No physical qubits, code distance, factory count or runtime estimate.",
                            "T-depth is dependency-aware, not globally minimized.",
                            "Magic-state demand assumes one logical T state per T/Tdg.",
                            "CCX policy preserves pre-existing RCCX gates in the optimized source.",
                            "Input consistency checks do not prove the saved files came from the same circuit revision."]}


def save_plots(resources, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    names = ["Baseline\nCCX chain", "Optimized\nCCX chain", "Baseline\nRCCX chain", "Optimized\nRCCX chain"]
    paths = []
    for key in ("logical_qubits", "t_count", "t_depth", "clifford_count"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        values = [resources[k][key] for k in KEYS]
        bars = ax.bar(names, values, color=["#64748b", "#0f766e", "#64748b", "#0f766e"], width=.65)
        ax.bar_label(bars, fmt="%d", padding=4)
        ax.set_ylim(0, max(values + [1]) * 1.18)
        ax.set_ylabel(LABELS[key]); ax.set_title("Modulo-17 adder | " + LABELS[key])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        fig.tight_layout(); path = out / (key + ".png"); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    source = [resources[k]["source_qubits"] for k in KEYS]
    extra = [resources[k]["added_clean_ancillas"] for k in KEYS]
    ax.bar(names, source, label="Source qubits", color="#64748b")
    ax.bar(names, extra, bottom=source, label="Added clean scratch", color="#d97706")
    ax.set_ylim(0, max([a+b for a,b in zip(source, extra)] + [1]) * 1.25)
    ax.set_ylabel("Logical qubits"); ax.set_title("Logical width accounting"); ax.legend(); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(); path = out / "qubit_accounting.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ("t_count", "t_depth", "clifford_count")
    series = summary["comparisons_percent_reduction"]
    for i, (name, label) in enumerate((("architecture_at_ccx", "Architecture change | CCX policy"),
                                      ("architecture_at_rccx", "Architecture change | RCCX policy"),
                                      ("synthesis_on_baseline", "Synthesis change | baseline"),
                                      ("synthesis_on_optimized", "Synthesis change | optimized"))):
        vals = [series[name][m] for m in metrics]
        for j, v in enumerate(vals):
            if v is not None:
                ax.bar(j + (i-1.5)*.19, v, width=.18, label=label if j == 0 else None,
                       color=["#64748b", "#94a3b8", "#0f766e", "#d97706"][i])
            else:
                ax.text(j + (i-1.5)*.19, 0, "N/A", rotation=90, fontsize=8)
    ax.set_xticks(range(len(metrics)), [LABELS[m] for m in metrics]); ax.set_ylabel("Reduction (%)")
    ax.axhline(0, color="#333333", linewidth=.8); ax.set_title("Architecture and synthesis effects (matched comparisons)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); path = out / "reductions.png"; fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    return paths


def write_report(resources, validation, report, out):
    out.mkdir(parents=True, exist_ok=True)
    summary = build_summary(resources, validation, report)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["comparison", "metric", "reduction_percent"]
    with (out / "reductions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for name, metrics in summary["comparisons_percent_reduction"].items():
            for metric, value in metrics.items():
                writer.writerow(dict(comparison=name, metric=metric, reduction_percent=value))
    lines = ["# Modulo-17 logical Clifford+T resource comparison", "", f"Qiskit: {report['qiskit_version']}", "",
             "| Policy | Architecture | Source qubits | Added scratch | Logical qubits | T-count | T-depth | Clifford gates | T states |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p, a in KEYS:
        lines.append(f"| {p} | {a} | " + " | ".join(str(resources[p,a][m]) for m in METRICS) + " |")
    lines.extend(["", "## Matched comparisons (% reduction)", "", "Positive means reduction; negative means increase.", "",
                  "| Comparison | " + " | ".join(LABELS.values()) + " |", "|---|" + "---:|"*len(LABELS)])
    for name, values in summary["comparisons_percent_reduction"].items():
        lines.append("| " + name + " | " + " | ".join("N/A" if v is None else f"{v:.2f}%" for v in values.values()) + " |")
    lines.extend(["", "## Phase validation", "", f"{len(validation)} configurations passed; {summary['validation_basis_inputs_total']} input/configuration checks.",
                  f"Of these, {summary['valid_arithmetic_inputs_total']} are valid modulo-17 arithmetic inputs.",
                  f"Maximum relative phase error: {summary['max_relative_phase_error']:.3e}.", "", "## Limitations", ""])
    lines.extend("- " + s for s in summary["limitations"])
    (out / "logical_resource_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, save_plots(resources, summary, out)


def log_mlflow(resources, validation, summary, input_dir, out, tracking_uri, experiment):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="mod17_logical_resource_comparison") as parent:
        mlflow.log_params({"workload": summary["workload"], "model": summary["model"], "qiskit_version": summary["qiskit_version"], "validation_method": summary["validation_method"]})
        mlflow.log_metrics({k: summary[k] for k in ("validation_configurations_passed", "validation_basis_inputs_total", "valid_arithmetic_inputs_total", "max_relative_phase_error")})
        mlflow.log_metrics({f"{name}.{metric}.reduction_pct": v for name, metrics in summary["comparisons_percent_reduction"].items() for metric, v in metrics.items() if v is not None})
        for name in ("logical_resources.json", "logical_resources.csv", "phase_validation.json"):
            if (input_dir / name).is_file():
                mlflow.log_artifact(str(input_dir / name), artifact_path="fault_tolerant/inputs")
        for name in ("summary.json", "reductions.csv", "logical_resource_report.md", "logical_qubits.png", "t_count.png", "t_depth.png", "clifford_count.png", "qubit_accounting.png", "reductions.png"):
            mlflow.log_artifact(str(out / name), artifact_path="fault_tolerant/reports")
        for p, a in KEYS:
            row, val = resources[p, a], validation[p, a]
            with mlflow.start_run(run_name=f"{a}__{p}", nested=True):
                mlflow.log_params({"workload": summary["workload"], "architecture": a, "mcx_policy": p,
                                   "qiskit_version": row["qiskit_version"], "decomposition_method": row["decomposition_method"]})
                mlflow.log_metrics({**{k: row[k] for k in METRICS}, "phase_validation_passed": 1,
                                    "basis_inputs_checked": val["basis_inputs_checked"], "max_relative_phase_error": val["max_relative_phase_error"]})
        return parent.info.run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-mlflow", action="store_true", help="Also publish one parent and four child runs")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default=EXPERIMENT)
    args = parser.parse_args(argv)
    out = args.output_dir or args.input_dir / "reports"
    try:
        resources, validation, report = load_inputs(args.input_dir)
        summary, plots = write_report(resources, validation, report, out)
        print(f"Validated {len(validation)} configurations; {summary['validation_basis_inputs_total']} input/configuration checks")
        print(f"Saved {len(plots)} plots and 3 report files to {out.resolve()}")
        if args.log_mlflow:
            run_id = log_mlflow(resources, validation, summary, args.input_dir, out, args.tracking_uri, args.experiment)
            print(f"MLflow experiment: {args.experiment}; parent run: {run_id}")
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
