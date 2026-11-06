"""Hardware-aware benchmarks for the two QARW modulo-17 adders.

Place in the QARW runtime repository alongside:
    modular_arithmetic.py
    modular_arithmetic_optimized.py

Quick smoke run:
    python benchmark_hardware_aware.py --seeds 42

Full, reproducible multi-seed run:
    python benchmark_hardware_aware.py --seeds 42 101 202 303 404

Runs are *connectivity-constrained simulations of compilation*, not QPU jobs.
The benchmark reports NET changes in compiled CX count and circuit depth relative
to unrestricted transpilation with the same seed and gate basis. Such changes
reflect layout, routing, and subsequent optimization; they are NOT measured
SWAP counts or a breakdown of costs exclusively attributable to routing.

The source's logical qubit count is distinct from the chosen device graph's
capacity, the output circuit width, and the number of physical qubits that
actually participate in operations. Unrestricted references have no physical
device, so their device capacity and physical usage are not applicable.
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

matplotlib.use("Agg")  # Save PNGs on servers and in non-GUI terminals.
import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit.transpiler import CouplingMap

from modular_arithmetic import build_modular_adder_17
from modular_arithmetic_optimized import build_optimized_modular_adder_17


BASIS_GATES = ["u", "cx"]  # Same basis as our existing QARW benchmarks.
EXPERIMENT = "QARW-Hardware-Aware-Modular-Benchmarks"
TOPOLOGY_NAMES = ("unrestricted", "linear_16", "grid_4x4", "heavy_hex_19")
ARCHITECTURE_NAMES = ("baseline", "optimized_rccx")

RAW_COLUMNS = (
    "architecture", "topology", "seed", "optimization_level",
    "logical_qubits", "device_capacity_qubits", "physical_qubits_touched",
    "compiled_circuit_width_qubits", "depth", "cx_count", "two_qubit_depth",
    "total_operations", "transpilation_seconds", "unrestricted_depth",
    "unrestricted_cx", "unrestricted_two_qubit_depth",
    "depth_delta_vs_unrestricted", "cx_delta_vs_unrestricted",
    "two_qubit_depth_delta_vs_unrestricted",
    "depth_change_pct_vs_unrestricted", "cx_change_pct_vs_unrestricted",
    "connectivity_check",
)


def make_topologies() -> dict[str, CouplingMap]:
    """Bidirectional graphs remove native CX-direction as a confounding factor."""
    return {
        "linear_16": CouplingMap.from_line(16, bidirectional=True),
        "grid_4x4": CouplingMap.from_grid(4, 4, bidirectional=True),
        "heavy_hex_19": CouplingMap.from_heavy_hex(3, bidirectional=True),
    }


def circuit_metrics(circuit) -> dict:
    """Count the resources of the final u/cx circuit."""
    counts = circuit.count_ops()
    touched = {
        circuit.find_bit(q).index
        for inst in circuit.data
        if inst.operation.name not in ("barrier", "delay")
        for q in inst.qubits
    }
    return {
        "depth": int(circuit.depth()),
        "cx_count": int(counts.get("cx", 0)),
        "two_qubit_depth": int(circuit.depth(lambda inst: len(inst.qubits) == 2)),
        "total_operations": int(circuit.size()),
        "compiled_circuit_width_qubits": int(circuit.num_qubits),
        # Wires on which at least one non-barrier/non-delay operation occurs.
        # This becomes a *physical* usage measure only for mapped circuits.
        "qubits_touched": len(touched),
    }


def verify_connectivity(compiled, coupling: CouplingMap) -> None:
    """Reject output with two-qubit operations not allowed by the device graph."""
    edges = set(coupling.get_edges())
    for inst in compiled.data:
        if len(inst.qubits) != 2:
            continue
        first, second = (compiled.find_bit(q).index for q in inst.qubits)
        if (first, second) not in edges:
            raise AssertionError(
                f"Disallowed 2-qubit interaction ({first}, {second}) "
                f"for {inst.operation.name}"
            )


def transpile_and_measure(source, seed: int, opt_level: int,
                          coupling: CouplingMap | None = None):
    kwargs = dict(
        basis_gates=BASIS_GATES,
        optimization_level=opt_level,
        seed_transpiler=seed,
    )
    if coupling is not None:
        kwargs.update(
            coupling_map=coupling,
            layout_method="sabre",
            routing_method="sabre",
        )
    started = time.perf_counter()
    compiled = transpile(source, **kwargs)
    elapsed = time.perf_counter() - started
    if coupling is not None:
        verify_connectivity(compiled, coupling)
    result = circuit_metrics(compiled)
    result["transpilation_seconds"] = round(elapsed, 5)
    return result


def create_row(arch: str, topology: str, seed: int, level: int, logical: int,
               available: int | None, result: dict, reference: dict) -> dict:
    """Produce one well-labeled trial row.

    'unrestricted' has no device capacity or physical-qubit assignment.
    CX/depth deltas are net changes in final compiled circuits, not SWAP counts.
    """
    is_mapped = topology != "unrestricted"
    if is_mapped and available is None:
        raise ValueError(f"Missing device capacity for {topology}")
    if is_mapped and result["compiled_circuit_width_qubits"] > available:
        raise AssertionError(f"Compiled circuit exceeds {topology} capacity")
    depth_delta = result["depth"] - reference["depth"]
    cx_delta = result["cx_count"] - reference["cx_count"]
    twoq_delta = result["two_qubit_depth"] - reference["two_qubit_depth"]
    qubits_touched = result["qubits_touched"]
    if qubits_touched > result["compiled_circuit_width_qubits"]:
        raise AssertionError("Touched qubits exceed compiled circuit width")
    return {
        "architecture": arch,
        "topology": topology,
        "seed": seed,
        "optimization_level": level,
        "logical_qubits": logical,
        "device_capacity_qubits": available if is_mapped else None,
        "physical_qubits_touched": qubits_touched if is_mapped else None,
        "compiled_circuit_width_qubits": result["compiled_circuit_width_qubits"],
        "depth": result["depth"],
        "cx_count": result["cx_count"],
        "two_qubit_depth": result["two_qubit_depth"],
        "total_operations": result["total_operations"],
        "transpilation_seconds": result["transpilation_seconds"],
        "unrestricted_depth": reference["depth"],
        "unrestricted_cx": reference["cx_count"],
        "unrestricted_two_qubit_depth": reference["two_qubit_depth"],
        "depth_delta_vs_unrestricted": depth_delta,
        "cx_delta_vs_unrestricted": cx_delta,
        "two_qubit_depth_delta_vs_unrestricted": twoq_delta,
        "depth_change_pct_vs_unrestricted": round(100 * depth_delta / reference["depth"], 3),
        "cx_change_pct_vs_unrestricted": round(100 * cx_delta / reference["cx_count"], 3),
        "connectivity_check": "passed" if is_mapped else "not_applicable",
    }


def summarize(rows: list[dict]) -> list[dict]:
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["architecture"], row["topology"])].append(row)

    summary = []
    for (arch, topology), members in sorted(by_group.items()):
        out = {
            "architecture": arch,
            "topology": topology,
            "runs": len(members),
            "logical_qubits": members[0]["logical_qubits"],
            "device_capacity_qubits": members[0]["device_capacity_qubits"],
        }
        for metric in (
            "depth", "cx_count", "two_qubit_depth", "physical_qubits_touched",
            "compiled_circuit_width_qubits", "depth_delta_vs_unrestricted",
            "cx_delta_vs_unrestricted", "depth_change_pct_vs_unrestricted",
            "cx_change_pct_vs_unrestricted", "transpilation_seconds",
        ):
            numbers = [float(m[metric]) for m in members if m[metric] is not None]
            out[f"{metric}_mean"] = (
                round(statistics.mean(numbers), 3) if numbers else None
            )
            out[f"{metric}_stdev"] = (
                round(statistics.stdev(numbers), 3) if len(numbers) >= 2
                else 0.0 if numbers else None
            )
        summary.append(out)
    return summary


def save_csv(path: Path, rows: list[dict], columns=None) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    columns = columns or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_chart(summary: list[dict], metric: str, ylabel: str,
               title: str, path: Path, show_error: bool = True) -> None:
    """One chart per metric, bars for baseline vs optimized on each topology."""
    lookup = {(r["architecture"], r["topology"]): r for r in summary}
    topologies = TOPOLOGY_NAMES
    labels = ("Unrestricted", "Linear (16)", "Grid 4x4", "Heavy hex (19)")
    bar_width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for j, arch in enumerate(ARCHITECTURE_NAMES):
        means = [lookup[(arch, topo)][f"{metric}_mean"] for topo in topologies]
        errors = [lookup[(arch, topo)][f"{metric}_stdev"] for topo in topologies]
        positions = [i + (j - 0.5) * bar_width for i in range(len(topologies))]
        ax.bar(
            positions, means, width=bar_width,
            yerr=errors if show_error else None,
            capsize=3 if show_error else 0,
            label="Baseline (10q)" if j == 0 else "Optimized RCCX (13q)",
        )
    ax.set_xticks(range(len(topologies)), labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_qubit_resources_chart(summary: list[dict], path: Path) -> None:
    """Show logical qubits, mapped physical usage, and device capacity distinctly."""
    topologies = TOPOLOGY_NAMES[1:]  # No physical device for unrestricted.
    lookup = {(r["architecture"], r["topology"]): r for r in summary}
    labels = []
    logical_counts, touched_counts, touched_errors, capacities = [], [], [], []
    for topo, short_name in zip(topologies, ("Linear", "Grid", "Heavy-hex")):
        for arch, arch_short in (("baseline", "Base"), ("optimized_rccx", "RCCX")):
            entry = lookup[(arch, topo)]
            labels.append(f"{short_name}\n{arch_short}")
            logical_counts.append(entry["logical_qubits"])
            touched_counts.append(entry["physical_qubits_touched_mean"])
            touched_errors.append(entry["physical_qubits_touched_stdev"])
            capacities.append(entry["device_capacity_qubits"])

    x = list(range(len(labels)))
    bar_width = 0.32
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar([v - bar_width / 2 for v in x], logical_counts, width=bar_width,
           label="Logical qubits (minimum required)")
    ax.bar([v + bar_width / 2 for v in x], touched_counts, width=bar_width,
           yerr=touched_errors, capsize=3,
           label="Physical qubits touched (seed mean ± SD)")
    ax.scatter(x, capacities, marker="_", s=230, linewidths=2.5,
               c="black", label="Device capacity (not usage)", zorder=5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(capacities) + 3)
    ax.set_ylabel("Number of qubits")
    ax.set_title("Qubit counts: logical requirements vs physical usage vs capacity")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_metric_definitions(path: Path) -> None:
    path.write_text(
        """# QARW Hardware-Aware Benchmark — metric definitions

- **logical_qubits**: Source circuit's logical width; the minimum number of
  available device qubits needed to accommodate the circuit, not necessarily
  the number of qubits activated by the compiled circuit.
- **device_capacity_qubits**: Number of physical vertices in the supplied
  coupling map (16, 16, or 19). **N/A for unrestricted**, which uses no device.
- **physical_qubits_touched**: Distinct physical qubit indices participating
  in at least one non-barrier, non-delay compiled instruction. The figure
  reflects the selected mapping and possible use of routing ancillas; it does
  not measure peak simultaneous usage or hardware errors. **N/A for unrestricted.**
- **compiled_circuit_width_qubits**: Number of wires in the transpiled circuit;
  it can include idle wires and may equal the full coupling-map capacity.
- **cx_delta_vs_unrestricted**, **depth_delta_vs_unrestricted**, and
  **two_qubit_depth_delta_vs_unrestricted**: CONSTRAINED compiled metric minus
  the same architecture's UNRESTRICTED compiled metric at the same seed,
  optimization level, and gate basis. These are **net compilation differences**,
  combining effects of initial layout, routing, and optimization. They are
  **NOT SWAP counts** and cannot isolate the cost of routing alone.
- **cx_change_pct_vs_unrestricted**, **depth_change_pct_vs_unrestricted**:
  Corresponding net differences divided by the unrestricted metric, times 100.
- **connectivity_check**: `passed` means the compiled two-qubit operations
  were checked against directed coupling-map edges; `not_applicable` for the
  unrestricted reference, which has no coupling constraints.

The experiment compares connectivity-constrained circuit compilations. It is
not actual device execution, and it does not model gate duration, fidelity,
errors, calibration, or physical runtime. SWAP gates are normally translated
into the selected `u`/`cx` basis, and subsequent optimizations can change CX
counts; do not infer the number of inserted SWAP gates from CX deltas / 3.
""",
        encoding="utf-8",
    )


def mlflow_module(disabled: bool):
    if disabled:
        return None
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed: CSV and PNG files will still be generated.")
        return None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def record_trial(mlflow, row: dict) -> None:
    if mlflow is None:
        return
    label = f"{row['architecture']}_{row['topology']}_seed{row['seed']}"
    with mlflow.start_run(run_name=label):
        mlflow.set_tags({
            "benchmark": "hardware_aware_mod17",
            "architecture": row["architecture"],
            "topology": row["topology"],
            "gate_basis": ",".join(BASIS_GATES),
            "routing_method": "sabre" if row["topology"] != "unrestricted" else "none",
            "cx_delta_interpretation": "net_compilation_change_not_swap_count",
            "connectivity_check": row["connectivity_check"],
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
        metric_names = (
            "depth", "cx_count", "two_qubit_depth", "total_operations",
            "physical_qubits_touched", "compiled_circuit_width_qubits",
            "transpilation_seconds", "cx_delta_vs_unrestricted",
            "depth_delta_vs_unrestricted", "cx_change_pct_vs_unrestricted",
            "depth_change_pct_vs_unrestricted",
        )
        mlflow.log_metrics({
            key: float(row[key])
            for key in metric_names if row[key] is not None
        })


def record_summary(mlflow, folder: Path, summary: list[dict], args) -> None:
    if mlflow is None:
        return
    with mlflow.start_run(run_name=f"hardware_comparison_L{args.optimization_level}"):
        mlflow.set_tag("benchmark", "hardware_aware_mod17_summary")
        mlflow.log_params({
            "optimization_level": args.optimization_level,
            "seeds": ",".join(map(str, args.seeds)),
            "basis_gates": ",".join(BASIS_GATES),
            "routing_method": "sabre",
        })
        for item in summary:
            prefix = f"{item['architecture']}_{item['topology']}"
            for key in (
                "depth_mean", "cx_count_mean", "cx_change_pct_vs_unrestricted_mean",
                "cx_delta_vs_unrestricted_mean", "physical_qubits_touched_mean",
            ):
                if item[key] is not None:
                    mlflow.log_metric(f"{prefix}_{key}", item[key])
        for artifact in sorted(folder.glob("*")):
            if artifact.is_file() and artifact.suffix in (".csv", ".png", ".md"):
                mlflow.log_artifact(str(artifact), artifact_path="hardware_aware_reports")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    parser.add_argument("--optimization-level", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmark_outputs/hardware_aware"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds should not contain duplicates")

    sources = {
        "baseline": build_modular_adder_17(),
        "optimized_rccx": build_optimized_modular_adder_17(),
    }
    maps = make_topologies()
    for arch, source in sources.items():
        print(f"Source circuit: {arch}, {source.num_qubits} logical qubits")
        for name, coupling in maps.items():
            if coupling.size() < source.num_qubits:
                raise ValueError(f"{name} has too few qubits for {arch}")

    mlflow = mlflow_module(args.no_mlflow)
    raw = []
    for arch, source in sources.items():
        for seed in args.seeds:
            reference = transpile_and_measure(source, seed, args.optimization_level)
            baseline_row = create_row(
                arch, "unrestricted", seed, args.optimization_level,
                source.num_qubits, None, reference, reference,
            )
            raw.append(baseline_row)
            record_trial(mlflow, baseline_row)
            print(
                f"[{arch}] seed={seed} unrestricted: "
                f"logical={source.num_qubits} device_capacity=N/A "
                f"physical_touched=N/A "
                f"depth={reference['depth']} CX={reference['cx_count']}",
                flush=True,
            )
            for topology, coupling in maps.items():
                result = transpile_and_measure(
                    source, seed, args.optimization_level, coupling
                )
                row = create_row(
                    arch, topology, seed, args.optimization_level,
                    source.num_qubits, coupling.size(), result, reference,
                )
                raw.append(row)
                record_trial(mlflow, row)
                print(
                    f"  {topology}: depth={row['depth']} "
                    f"CX={row['cx_count']} "
                    f"CX_delta_vs_unrestricted={row['cx_delta_vs_unrestricted']:+} "
                    f"logical={row['logical_qubits']} "
                    f"device_capacity={row['device_capacity_qubits']} "
                    f"physical_touched={row['physical_qubits_touched']}",
                    flush=True,
                )

    summary = summarize(raw)
    save_csv(args.output_dir / "hardware_aware_raw.csv", raw, RAW_COLUMNS)
    save_csv(args.output_dir / "hardware_aware_summary.csv", summary)
    save_chart(summary, "depth", "Circuit depth", "Depth by device topology",
               args.output_dir / "depth_comparison.png")
    save_chart(summary, "cx_count", "CX gates", "CX gates by device topology",
               args.output_dir / "cx_comparison.png")
    save_chart(summary, "cx_delta_vs_unrestricted",
               "Net CX change vs unrestricted (not SWAP count)",
               "Connectivity-constrained compilation: net CX difference",
               args.output_dir / "routing_overhead_cx_delta.png")
    save_qubit_resources_chart(
        summary, args.output_dir / "physical_qubit_usage.png"
    )
    save_metric_definitions(args.output_dir / "metric_definitions.md")
    record_summary(mlflow, args.output_dir, summary, args)

    print("\n--- Average results across seeds ---")
    for item in summary:
        print(
            f"{item['architecture']:<15} {item['topology']:<14} "
            f"depth={item['depth_mean']:>8.1f} "
            f"CX={item['cx_count_mean']:>8.1f} "
            f"CX_delta_vs_unrestricted={item['cx_delta_vs_unrestricted_mean']:>+8.1f} "
            f"logical={item['logical_qubits']:>2} "
            f"capacity={item['device_capacity_qubits'] if item['device_capacity_qubits'] is not None else 'N/A':>3} "
            f"physical_touched_mean={item['physical_qubits_touched_mean'] if item['physical_qubits_touched_mean'] is not None else 'N/A':>4}"
        )
    print("Note: CX deltas are NET changes vs unrestricted compilation, NOT SWAP counts.")
    print("Qubit fields: logical = circuit requirement; capacity = device size; ")
    print("physical_touched = mapped qubits used; unrestricted physical metrics = N/A.")
    print(f"\nCSV + charts + metric definitions saved to: {args.output_dir.resolve()}")
    if mlflow is not None:
        print(f"MLflow experiment: {EXPERIMENT}")


if __name__ == "__main__":
    main()
