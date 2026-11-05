"""Compare previously saved QARW modulo-17 modular-adder transpilation results.

Place this file in the runtime/ folder next to benchmark_modular_arithmetic.py.
Run from an activated Python environment:

    python compare_modular_optimization.py

This does NOT run Qiskit or MLflow. It reads the four JSON files that the
benchmark has already saved and produces a CSV, a Markdown report, and plots.

Optional:
    python compare_modular_optimization.py --input-dir PATH --output-dir PATH
    python compare_modular_optimization.py --no-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "benchmark_outputs" / "modular_add_17"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "comparison_reports"
LEVELS = (0, 1, 2, 3)

COLUMNS = (
    "optimization_level",
    "source_qubits",
    "source_depth",
    "source_operations",
    "compiled_qubits",
    "compiled_depth",
    "compiled_operations",
    "compiled_cx_count",
    "compiled_u_count",
    "compiled_two_qubit_gate_count",
    "compiled_two_qubit_depth",
    "compilation_wall_time_s",
    "seed_transpiler",
    "basis_gates",
    "qiskit_version",
    "mlflow_run_id",
)

NUMERIC_FIELDS = (
    "source_qubits",
    "source_depth",
    "source_operations",
    "compiled_qubits",
    "compiled_depth",
    "compiled_operations",
    "compiled_cx_count",
    "compiled_u_count",
    "compiled_two_qubit_gate_count",
    "compiled_two_qubit_depth",
    "compilation_wall_time_s",
)

# Identical settings are required for a scientifically meaningful comparison.
COMPARABLE_FIELDS = (
    "workload",
    "source_qubits",
    "source_depth",
    "source_operations",
    "compiled_qubits",
    "seed_transpiler",
    "basis_gates",
    "qiskit_version",
)

METRICS = (
    ("compiled_depth", "Compiled circuit depth", "circuit_depth.png"),
    ("compiled_operations", "Compiled total operations", "total_operations.png"),
    ("compiled_cx_count", "CX (two-qubit) gate count", "cx_gate_count.png"),
    ("compiled_two_qubit_depth", "Two-qubit circuit depth", "two_qubit_depth.png"),
    ("compiled_u_count", "Single-qubit U gate count", "u_gate_count.png"),
    ("compilation_wall_time_s", "Compilation wall time (seconds)", "compilation_time.png"),
)


class DataError(ValueError):
    """The saved benchmark files are missing or not comparable."""


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    """Load and validate four independent, identical-configuration benchmarks."""
    results: list[dict[str, Any]] = []
    missing = [
        input_dir / f"optimization_level_{level}.json"
        for level in LEVELS
        if not (input_dir / f"optimization_level_{level}.json").is_file()
    ]
    if missing:
        paths = "\n  ".join(str(path) for path in missing)
        raise DataError(
            "Missing saved benchmark JSON file(s):\n  "
            + paths
            + "\nRun the missing benchmark level(s) first."
        )

    for level in LEVELS:
        path = input_dir / f"optimization_level_{level}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DataError(f"Unable to read JSON in {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise DataError(f"Expected a JSON object in {path}")
        required = set(COLUMNS) | {"workload"}
        absent = sorted(required - data.keys())
        if absent:
            raise DataError(f"Missing JSON keys in {path}: {', '.join(absent)}")

        if data["optimization_level"] != level:
            raise DataError(
                f"{path.name} reports optimization_level="
                f"{data['optimization_level']!r} (expected {level})"
            )
        for key in NUMERIC_FIELDS:
            value = data[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise DataError(f"Invalid nonnegative numeric metric {key}={value!r} in {path}")
        if data["compiled_operations"] != data["compiled_cx_count"] + data["compiled_u_count"]:
            raise DataError(
                f"{path.name}: compiled gate counts do not add up to compiled_operations."
            )
        if data["compiled_two_qubit_gate_count"] != data["compiled_cx_count"]:
            raise DataError(
                f"{path.name}: expected only CX two-qubit gates under the u/cx basis."
            )
        if data["basis_gates"] != ["u", "cx"]:
            raise DataError(f"{path.name}: expected basis_gates=['u', 'cx'].")
        results.append(data)

    first = results[0]
    for result in results[1:]:
        for key in COMPARABLE_FIELDS:
            if result[key] != first[key]:
                raise DataError(
                    f"The four runs have inconsistent {key}: "
                    f"level 0={first[key]!r}, "
                    f"level {result['optimization_level']}={result[key]!r}."
                )
    return results


def percentage_reduction(baseline: float, current: float) -> float:
    """Positive values mean less of the metric than optimization level 0."""
    if baseline == 0:
        return 0.0 if current == 0 else float("nan")
    return 100.0 * (baseline - current) / baseline


def format_change(baseline: float, current: float) -> str:
    change = percentage_reduction(baseline, current)
    if math.isnan(change):
        return "N/A (baseline zero)"
    if abs(change) < 0.00005:
        return "0.0%"
    if change > 0:
        return f"{change:.2f}% reduction"
    return f"{-change:.2f}% increase"


def format_number(value: int | float, *, seconds: bool = False) -> str:
    if seconds:
        return f"{value:.3f}"
    return f"{value:,}" if isinstance(value, int) else f"{value:,.0f}"


def save_csv(results: list[dict[str, Any]], output_dir: Path) -> Path:
    target = output_dir / "modular_optimization_comparison.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            exported = dict(row)
            exported["basis_gates"] = ",".join(row["basis_gates"])
            writer.writerow(exported)
    return target


def build_report(results: list[dict[str, Any]]) -> str:
    baseline, final = results[0], results[-1]
    lines = [
        "# QARW — Modulo-17 quantum adder: optimization comparison",
        "",
        "Compiled with basis gates `u,cx`, idealized all-to-all connectivity; "
        "the same Qiskit version and transpiler seed for all levels.",
        "",
        "| Metric | Level 0 | Level 1 | Level 2 | Level 3 | Change L0 → L3 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, label, _ in METRICS:
        seconds = key == "compilation_wall_time_s"
        formatted = [format_number(result[key], seconds=seconds) for result in results]
        lines.append(
            f"| {label} | " + " | ".join(formatted) + " | "
            + format_change(baseline[key], final[key]) + " |"
        )
    lines.extend(
        [
            "",
            "## Key observations",
            "",
            f"- Circuit depth: {format_change(baseline['compiled_depth'], final['compiled_depth'])} "
            "from level 0 to level 3.",
            f"- Total operations: {format_change(baseline['compiled_operations'], final['compiled_operations'])}.",
            f"- CX gate count: {format_change(baseline['compiled_cx_count'], final['compiled_cx_count'])}.",
            f"- Two-qubit depth: "
            f"{format_change(baseline['compiled_two_qubit_depth'], final['compiled_two_qubit_depth'])}.",
            f"- Single-qubit U gates: "
            f"{format_change(baseline['compiled_u_count'], final['compiled_u_count'])}.",
        ]
    )
    if all(
        results[2][key] == results[3][key]
        for key in ("compiled_depth", "compiled_operations", "compiled_cx_count", "compiled_two_qubit_depth")
    ):
        lines.append(
            "- Levels 2 and 3 have identical measured structural resource metrics "
            "in these runs (an observed plateau)."
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "- The source circuit contains high-level multi-controlled gates; "
            "its depth/operation count should not be compared directly with "
            "the compiled `u,cx` circuit as if the gate primitives were equivalent.",
            "- Each level's compilation time is one measured run, not a mean. "
            "Repeat runs before drawing statistical timing conclusions.",
            "- These are compiler resource estimates, not hardware performance "
            "or hardware-aware routing results.",
            "- The benchmark itself transpiles the circuit; functional correctness "
            "was tested separately.",
            "",
            "## Experimental settings",
            "",
            f"- Workload: `{baseline['workload']}`",
            f"- Qiskit version: `{baseline['qiskit_version']}`",
            f"- Seed: `{baseline['seed_transpiler']}`",
            f"- Source qubits: `{baseline['source_qubits']}`",
            f"- Compiled qubits: `{baseline['compiled_qubits']}`",
            "",
        ]
    )
    return "\n".join(lines)


def save_plots(results: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for the graphs. "
            "Install it with 'python -m pip install matplotlib' "
            "or run with --no-plots."
        ) from exc

    levels = list(LEVELS)
    plot_paths = []
    for key, label, filename in METRICS:
        values = [record[key] for record in results]
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        bars = ax.bar(levels, values, width=0.58)
        ax.set_xticks(levels)
        ax.set_xticklabels([f"Level {level}" for level in levels])
        ax.set_ylabel(label)
        ax.set_title(f"Modulo-17 adder — {label}")
        ax.set_ylim(0, max(values) * 1.18 if any(values) else 1)
        ax.yaxis.grid(True, alpha=0.18)
        ax.set_axisbelow(True)
        if key != "compilation_wall_time_s":
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        for bar, value in zip(bars, values):
            shown = f"{value:.3f}s" if key == "compilation_wall_time_s" else f"{int(value):,}"
            ax.annotate(
                shown,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(path)
    return plot_paths


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nQARW MODULO-17 ADDER — COMPARISON")
    print("-" * 65)
    print(f"{'Metric':<27} {'L0':>9} {'L1':>9} {'L2':>9} {'L3':>9}")
    print("-" * 65)
    for key, label, _ in METRICS:
        label_short = label.replace("Compiled ", "").replace(" (seconds)", " (s)")
        if len(label_short) > 27:
            label_short = label_short[:27]
        values = [format_number(r[key], seconds=key == "compilation_wall_time_s") for r in results]
        print(f"{label_short:<27} " + " ".join(f"{v:>9}" for v in values))
    print("-" * 65)
    first, last = results[0], results[-1]
    print("L0 to L3 depth:", format_change(first["compiled_depth"], last["compiled_depth"]))
    print("L0 to L3 operations:", format_change(first["compiled_operations"], last["compiled_operations"]))
    print("L0 to L3 CX gates:", format_change(first["compiled_cx_count"], last["compiled_cx_count"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Folder containing optimization_level_0.json through optimization_level_3.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write the CSV/report/graphs (default: INPUT_DIR/comparison_reports)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip charts; write CSV and report only")
    args = parser.parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "comparison_reports"
    )
    try:
        results = load_results(input_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_csv(results, output_dir)
        report_path = output_dir / "modular_optimization_report.md"
        report_path.write_text(build_report(results), encoding="utf-8")
        plot_paths = [] if args.no_plots else save_plots(results, output_dir)
    except (DataError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_summary(results)
    print("\nSaved files:")
    print("  CSV:", csv_path)
    print("  Report:", report_path)
    for path in plot_paths:
        print("  Chart:", path)
    print("\nNo quantum compilation or MLflow server was needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
