"""QARW Milestone 5: compare basis gates across connectivity constraints.

Example smoke run (without MLflow):
    python benchmark_hardware_gate_bases.py --seeds 42 --no-mlflow \
        --output-dir benchmark_outputs/basis_gate_smoke

Full run (with MLflow tracking configured):
    python benchmark_hardware_gate_bases.py --seeds 42 101 202 303 404 \
        --output-dir benchmark_outputs/basis_gate_full

These are abstract compilation targets, not actual device calibrations.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import transpile

from benchmark_hardware_aware import make_topologies, verify_connectivity
from modular_arithmetic import build_modular_adder_17
from modular_arithmetic_optimized import build_optimized_modular_adder_17
from routing_profiler import RoutingProfiler

EXPERIMENT = "QARW-Hardware-Gate-Basis-Benchmarks"
BASIS_CONFIGS = {
    "CX": ["u", "cx"],
    "CZ": ["u", "cz"],
    "ECR": ["u", "ecr"],
}
TOPOLOGY_NAMES = ("unrestricted", "linear_16", "grid_4x4", "heavy_hex_19")
ARCHITECTURE_NAMES = ("baseline", "optimized_rccx")

NUMERIC_METRICS = (
    "depth",
    "native_2q_count",
    "two_qubit_depth",
    "single_qubit_depth",
    "u_count",
    "total_operations",
    "compiled_width_qubits",
    "physical_qubits_touched",
    "transpilation_seconds",
    "routing_swaps",
    "routing_depth_delta",
    "routing_two_qubit_ops_delta",
    "routing_pass_seconds",
)
RAW_COLUMNS = (
    "architecture", "topology", "basis", "seed", "optimization_level",
    "logical_qubits", "device_capacity_qubits", "native_gate",
    "routing_pass_name", *NUMERIC_METRICS,
)


def compile_trial(source, coupling, basis_name: str, seed: int, level: int) -> dict:
    basis_gates = BASIS_CONFIGS[basis_name]
    native_gate = basis_gates[1]
    profiler = RoutingProfiler() if coupling is not None else None

    kwargs = {
        "basis_gates": basis_gates,
        "optimization_level": level,
        "seed_transpiler": seed,
        "qubits_initially_zero": False,
    }
    if coupling is not None:
        kwargs.update({
            "coupling_map": coupling,
            "layout_method": "sabre",
            "routing_method": "sabre",
            "callback": profiler,
        })

    started = time.perf_counter()
    compiled = transpile(source, **kwargs)
    elapsed = time.perf_counter() - started

    if coupling is not None:
        verify_connectivity(compiled, coupling)

    two_qubit_names = {
        instruction.operation.name
        for instruction in compiled.data
        if len(instruction.qubits) == 2
    }
    if two_qubit_names != {native_gate}:
        raise AssertionError(
            f"Expected only native two-qubit gate {native_gate}, "
            f"found {two_qubit_names}"
        )

    counts = compiled.count_ops()
    touched = {
        compiled.find_bit(q).index
        for inst in compiled.data
        if inst.operation.name not in ("barrier", "delay")
        for q in inst.qubits
    }
    routing = profiler.metrics() if profiler is not None else {
        "routing_pass_name": None,
        "routing_swaps": None,
        "routing_depth_delta": None,
        "routing_two_qubit_ops_delta": None,
        "routing_pass_seconds": None,
    }

    return {
        "native_gate": native_gate,
        "depth": int(compiled.depth()),
        "native_2q_count": int(counts.get(native_gate, 0)),
        "two_qubit_depth": int(compiled.depth(lambda op: len(op.qubits) == 2)),
        "single_qubit_depth": int(compiled.depth(lambda op: len(op.qubits) == 1)),
        "u_count": int(counts.get("u", 0)),
        "total_operations": int(compiled.size()),
        "compiled_width_qubits": int(compiled.num_qubits),
        "physical_qubits_touched": len(touched) if coupling is not None else None,
        "transpilation_seconds": round(elapsed, 5),
        **routing,
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["architecture"], row["topology"], row["basis"])].append(row)

    summary = []
    for (architecture, topology, basis), members in sorted(groups.items()):
        output = {
            "architecture": architecture,
            "topology": topology,
            "basis": basis,
            "runs": len(members),
            "logical_qubits": members[0]["logical_qubits"],
            "device_capacity_qubits": members[0]["device_capacity_qubits"],
        }
        for metric in NUMERIC_METRICS:
            values = [float(row[metric]) for row in members if row[metric] is not None]
            output[f"{metric}_mean"] = (
                round(statistics.mean(values), 4) if values else None
            )
            output[f"{metric}_stdev"] = (
                round(statistics.stdev(values), 4) if len(values) > 1
                else 0.0 if values else None
            )
        summary.append(output)
    return summary


def write_csv(path: Path, records: list[dict], fields=None) -> None:
    if not records:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def save_chart(summary: list[dict], architecture: str, metric: str,
               label: str, path: Path, routing_only: bool = False) -> None:
    lookup = {
        (entry["architecture"], entry["topology"], entry["basis"]): entry
        for entry in summary
    }
    topologies = TOPOLOGY_NAMES[1:] if routing_only else TOPOLOGY_NAMES
    positions = list(range(len(topologies)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.2))

    for index, basis in enumerate(BASIS_CONFIGS):
        means = [lookup[(architecture, topo, basis)][f"{metric}_mean"]
                 for topo in topologies]
        errors = [lookup[(architecture, topo, basis)][f"{metric}_stdev"]
                  for topo in topologies]
        ax.bar([x + (index - 1) * width for x in positions], means,
               yerr=errors, capsize=3, width=width, label=basis)

    ax.set_xticks(positions, [topo.replace("_", " ") for topo in topologies])
    ax.set_ylabel(label)
    ax.set_title(f"{architecture}: {label} (mean ± SD across seeds)")
    ax.legend(title="Native two-qubit gate")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_definitions(path: Path) -> None:
    path.write_text(
        """# QARW Milestone 5 — Metric definitions

- Each row compares a **gate basis**, **coupling map**, **adder architecture**,
  **transpiler seed**, and **optimization level**.
- **native_2q_count**: Number of final native CX, CZ, or ECR gates.
  A count of different native gate types does not imply equal physical cost.
- **two_qubit_depth**: Depth of the final circuit after ignoring non-2Q gates.
- **single_qubit_depth**: Depth after ignoring non-1Q gates. This and
  two_qubit_depth do not generally add up to the total circuit depth.
- **u_count**: Number of final single-qubit U operations.
- **routing_swaps**: Change in explicitly visible SWAP operations across the
  observed combined layout/routing pass, not inferred from native gate counts.
- **routing_depth_delta**: DAG depth difference across the complete observed
  routing-related pass; not the isolated cost of SWAP gates.
- **routing_two_qubit_ops_delta**: 2Q operation count difference across that pass.
- **routing_pass_seconds**: Qiskit-reported time for that observed pass.
- **transpilation_seconds**: End-to-end time with instrumentation for mapped
  circuits; execution time is sensitive to the machine and profiling overhead.
- Unrestricted results have no device capacity or measured routing overhead.
- All three coupling maps are symmetric, abstract connectivity models;
  no actual native gate timing, calibration, device errors, or noise is modeled.
- Means and sample standard deviations are computed across supplied seeds.
  An SD of zero for one seed is a display convention, not evidence of stability.
""",
        encoding="utf-8",
    )


def log_mlflow(rows: list[dict], summary: list[dict], folder: Path, args) -> None:
    if args.no_mlflow:
        return
    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)

    for row in rows:
        run_name = (
            f"{row['architecture']}_{row['topology']}_"
            f"{row['basis']}_seed{row['seed']}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "benchmark": "hardware_gate_bases_mod17",
                "architecture": row["architecture"],
                "topology": row["topology"],
                "gate_basis": row["basis"],
                "routing_pass": row["routing_pass_name"] or "not_applicable",
            })
            mlflow.log_params({
                "seed": row["seed"],
                "optimization_level": row["optimization_level"],
                "logical_qubits": row["logical_qubits"],
                "device_capacity_qubits": (
                    row["device_capacity_qubits"]
                    if row["device_capacity_qubits"] is not None else "N/A"
                ),
            })
            mlflow.log_metrics({
                metric: float(row[metric]) for metric in NUMERIC_METRICS
                if row[metric] is not None
            })

    with mlflow.start_run(run_name=f"basis_comparison_L{args.optimization_level}"):
        mlflow.set_tag("benchmark", "hardware_gate_bases_mod17_summary")
        mlflow.log_params({
            "seeds": ",".join(str(seed) for seed in args.seeds),
            "optimization_level": args.optimization_level,
            "single_qubit_basis": "u",
        })
        for group in summary:
            prefix = (f"{group['architecture']}_{group['topology']}_"
                      f"{group['basis']}")
            for metric in ("depth", "native_2q_count", "u_count", "routing_swaps"):
                value = group[f"{metric}_mean"]
                if value is not None:
                    mlflow.log_metric(f"{prefix}_{metric}_mean", value)
        mlflow.log_artifacts(str(folder), artifact_path="basis_gate_reports")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmark_outputs/basis_gate_comparison"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    sources = {
        "baseline": build_modular_adder_17(),
        "optimized_rccx": build_optimized_modular_adder_17(),
    }
    maps = make_topologies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print("\nQARW — HARDWARE GATE-BASIS BENCHMARK")
    print("-" * 88)
    for architecture in ARCHITECTURE_NAMES:
        source = sources[architecture]
        for topology in TOPOLOGY_NAMES:
            coupling = maps[topology] if topology != "unrestricted" else None
            if coupling is not None and coupling.size() < source.num_qubits:
                raise ValueError(f"{topology} cannot fit {architecture}")
            for basis in BASIS_CONFIGS:
                for seed in args.seeds:
                    measurements = compile_trial(
                        source, coupling, basis, seed, args.optimization_level
                    )
                    row = {
                        "architecture": architecture,
                        "topology": topology,
                        "basis": basis,
                        "seed": seed,
                        "optimization_level": args.optimization_level,
                        "logical_qubits": source.num_qubits,
                        "device_capacity_qubits": (
                            coupling.size() if coupling is not None else None
                        ),
                        **measurements,
                    }
                    rows.append(row)
                    swaps = row["routing_swaps"]
                    print(
                        f"{architecture:<15} {topology:<14} {basis:<3} "
                        f"seed={seed:<3} depth={row['depth']:<6} "
                        f"2Q={row['native_2q_count']:<6} "
                        f"U={row['u_count']:<6} "
                        f"SWAP_delta={swaps if swaps is not None else 'N/A'}",
                        flush=True,
                    )

    summary = summarize(rows)
    write_csv(args.output_dir / "basis_gate_raw.csv", rows, RAW_COLUMNS)
    write_csv(args.output_dir / "basis_gate_summary.csv", summary)
    for architecture in ARCHITECTURE_NAMES:
        for metric, label in (
            ("depth", "Final circuit depth"),
            ("native_2q_count", "Native two-qubit gate count"),
            ("two_qubit_depth", "Two-qubit depth"),
            ("u_count", "Single-qubit U gate count"),
            ("routing_swaps", "SWAP count change across routing pass"),
            ("transpilation_seconds", "Instrumented transpilation time (s)"),
        ):
            save_chart(
                summary, architecture, metric, label,
                args.output_dir / f"{architecture}_{metric}.png",
                routing_only=(metric == "routing_swaps"),
            )
    save_definitions(args.output_dir / "metric_definitions.md")
    log_mlflow(rows, summary, args.output_dir, args)

    print(f"\nSUCCESS: {len(rows)} compilation records.")
    print(f"Results: {args.output_dir.resolve()}")
    if not args.no_mlflow:
        print(f"MLflow experiment: {EXPERIMENT}")
    print("Functional correctness must be validated separately for each seed.")


if __name__ == "__main__":
    main()
