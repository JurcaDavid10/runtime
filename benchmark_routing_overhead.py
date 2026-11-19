
"""QARW Milestone 4: Measure routing-pass resource changes."""

import argparse
import csv
from pathlib import Path

from qiskit import transpile

from benchmark_hardware_aware import (
    BASIS_GATES,
    make_topologies,
    verify_connectivity,
)
from modular_arithmetic import build_modular_adder_17
from modular_arithmetic_optimized import build_optimized_modular_adder_17


ROUTING_PASSES = {
    "SabreLayout",
    "SabreSwap",
    "BasicSwap",
    "LookaheadSwap",
}


def snapshot(dag):
    counts = dag.count_ops()

    return {
        "depth": int(dag.depth()),
        "swaps": int(counts.get("swap", 0)),
        "two_qubit_ops": sum(
            1 for node in dag.op_nodes()
            if len(node.qargs) == 2
        ),
    }


def analyze(source, coupling, seed, optimization_level):
    previous = None
    routing_events = []

    def observe_pass(**kwargs):
        nonlocal previous

        name = type(kwargs["pass_"]).__name__
        current = snapshot(kwargs["dag"])

        if name in ROUTING_PASSES and previous is not None:
            routing_events.append({
                "routing_pass": name,
                "routing_pass_seconds": round(
                    float(kwargs["time"]), 5
                ),
                "swaps_before": previous["swaps"],
                "swaps_after": current["swaps"],
                "swap_delta": (
                    current["swaps"] - previous["swaps"]
                ),
                "depth_before": previous["depth"],
                "depth_after": current["depth"],
                "depth_delta": (
                    current["depth"] - previous["depth"]
                ),
                "two_qubit_ops_before": previous["two_qubit_ops"],
                "two_qubit_ops_after": current["two_qubit_ops"],
                "two_qubit_ops_delta": (
                    current["two_qubit_ops"]
                    - previous["two_qubit_ops"]
                ),
            })

        previous = current

    compiled = transpile(
        source,
        basis_gates=BASIS_GATES,
        coupling_map=coupling,
        optimization_level=optimization_level,
        seed_transpiler=seed,
        layout_method="sabre",
        routing_method="sabre",
        qubits_initially_zero=False,
        callback=observe_pass,
    )

    verify_connectivity(compiled, coupling)

    for event in routing_events:
        event["final_depth"] = int(compiled.depth())
        event["final_cx_count"] = int(
            compiled.count_ops().get("cx", 0)
        )

    return routing_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42]
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=(0, 1, 2, 3),
        default=3,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs/routing_overhead"),
    )
    args = parser.parse_args()

    sources = {
        "baseline": build_modular_adder_17(),
        "optimized_rccx": build_optimized_modular_adder_17(),
    }

    topologies = make_topologies()
    rows = []

    print("\nQARW — ROUTING OVERHEAD BENCHMARK")
    print("-" * 65)

    for architecture, source in sources.items():
        for topology, coupling in topologies.items():
            for seed in args.seeds:
                events = analyze(
                    source,
                    coupling,
                    seed,
                    args.optimization_level,
                )

                if not events:
                    print(
                        f"{architecture} | {topology} | seed={seed}: "
                        "no named routing pass observed"
                    )
                    continue

                for event in events:
                    row = {
                        "architecture": architecture,
                        "topology": topology,
                        "seed": seed,
                        "optimization_level": args.optimization_level,
                        **event,
                    }

                    rows.append(row)

                    print(
                        f"{architecture:<15} "
                        f"{topology:<14} "
                        f"seed={seed} "
                        f"pass={event['routing_pass']} "
                        f"SWAP_delta={event['swap_delta']:+} "
                        f"depth_delta={event['depth_delta']:+} "
                        f"final_CX={event['final_cx_count']}"
                    )

    if not rows:
        raise RuntimeError("No routing events were captured.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "routing_pass_metrics.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nRouting events recorded:", len(rows))
    print("Results saved to:", output.resolve())


if __name__ == "__main__":
    main()
