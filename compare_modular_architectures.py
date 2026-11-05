"""Compare baseline and relative-phase-ancilla modulo-17 benchmark results.

Reads actual JSON metrics; never recompiles. Works even if only one new
optimization level is available. Creates CSV, Markdown and four PNG charts.
    python compare_modular_architectures.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent / "benchmark_outputs" / "modular_add_17"
ALT = BASE / "relative_phase_ancilla"
OUTPUT = BASE / "architecture_comparison"
METRICS = {
    "compiled_qubits": "Qubits",
    "compiled_depth": "Circuit depth",
    "compiled_cx_count": "CX gate count",
    "compiled_two_qubit_depth": "Two-qubit depth",
    "compiled_operations": "Total operations",
    "compilation_wall_time_s": "Compilation time (s)",
}


def read_result(directory, level):
    path = directory / f"optimization_level_{level}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["optimization_level"] != level or data["basis_gates"] != ["u", "cx"]:
        raise ValueError(f"Wrong level or gate basis in {path}")
    return data


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for level in range(4):
        before = read_result(BASE, level)
        after = read_result(ALT, level)
        if before is None or after is None:
            continue
        if before["seed_transpiler"] != after["seed_transpiler"]:
            raise ValueError(f"Different random seeds at level {level}")
        if before.get("qiskit_version") != after.get("qiskit_version"):
            raise ValueError(f"Different Qiskit versions at level {level}")
        row = {"optimization_level": level}
        for key in METRICS:
            row[f"baseline_{key}"] = before[key]
            row[f"optimized_{key}"] = after[key]
        old_cx = before["compiled_cx_count"]
        row["cx_reduction_percent"] = (
            100 * (old_cx - after["compiled_cx_count"]) / old_cx
            if old_cx else 0.0
        )
        comparisons.append(row)

    if not comparisons:
        raise SystemExit(
            "No matching benchmark files found. First run the optimized "
            "benchmark at level 0 or above."
        )

    csv_path = OUTPUT / "architecture_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    print("\nMODULO-17: BASELINE vs CLEAN-ANCILLA RCCX")
    print(f"{'Level':>5}  {'CX (old)':>9}  {'CX (new)':>9}  {'CX reduction':>12}  {'Qubits old/new':>16}")
    for row in comparisons:
        print(f"{row['optimization_level']:>5}  {row['baseline_compiled_cx_count']:>9}  "
              f"{row['optimized_compiled_cx_count']:>9}  "
              f"{row['cx_reduction_percent']:>11.2f}%  "
              f"{row['baseline_compiled_qubits']:>6}/{row['optimized_compiled_qubits']:<6}")

    for key, title, filename in [
        ("compiled_cx_count", "CNOT (CX) gates", "architecture_cx.png"),
        ("compiled_depth", "Compiled circuit depth", "architecture_depth.png"),
        ("compiled_two_qubit_depth", "Two-qubit depth", "architecture_2q_depth.png"),
        ("compiled_qubits", "Qubit count", "architecture_qubits.png"),
    ]:
        x = [row["optimization_level"] for row in comparisons]
        base = [row[f"baseline_{key}"] for row in comparisons]
        opt = [row[f"optimized_{key}"] for row in comparisons]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([level - 0.18 for level in x], base, width=0.36, label="Baseline")
        ax.bar([level + 0.18 for level in x], opt, width=0.36, label="RCCX + ancillas")
        ax.set_xticks(x, [f"L{level}" for level in x])
        ax.set_xlabel("Qiskit optimization level")
        ax.set_ylabel(title)
        ax.set_title(f"Modulo-17 adder: {title}")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUTPUT / filename, dpi=160)
        plt.close(fig)

    report = ["# Modulo-17 adder architecture comparison", "",
              "These are actual benchmark measurements at a common transpiler "
              "seed, basis gates (`u,cx`), and idealized connectivity.", "",
              "| Level | Baseline CX | RCCX + ancilla CX | CX reduction | Baseline qubits | New qubits |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        report.append(
            f"| {row['optimization_level']} | {row['baseline_compiled_cx_count']} | "
            f"{row['optimized_compiled_cx_count']} | {row['cx_reduction_percent']:.2f}% | "
            f"{row['baseline_compiled_qubits']} | {row['optimized_compiled_qubits']} |"
        )
    report.extend(["", "Higher qubit usage may be exchanged for lower two-qubit gate cost.",
                   "A negative CX reduction indicates regression, not improvement.",
                   "Compilation times are single-run observations, not statistical estimates.", ""])
    (OUTPUT / "architecture_comparison.md").write_text("\n".join(report), encoding="utf-8")
    print("\nSaved CSV, Markdown report and four PNG charts to:", OUTPUT)


if __name__ == "__main__":
    main()
