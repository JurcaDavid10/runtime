
"""Benchmark quantum cryptographic arithmetic workloads."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from qiskit import transpile

from modular_arithmetic import (
    build_modular_adder_17,
)
from modular_arithmetic_optimized import (
    build_optimized_modular_adder_17,
)
from modular_subtraction import (
    build_modular_subtractor_17,
    build_optimized_modular_subtractor_17,
)


OUTPUT_DIR = Path(
    "benchmark_outputs/cryptographic_arithmetic"
)

BASIS_GATES = ["u", "cx"]

BUILDERS = {
    ("addition", "baseline"):
        build_modular_adder_17,

    ("addition", "optimized_rccx"):
        build_optimized_modular_adder_17,

    ("subtraction", "baseline"):
        build_modular_subtractor_17,

    ("subtraction", "optimized_rccx"):
        build_optimized_modular_subtractor_17,
}


def benchmark_circuit(
    operation,
    architecture,
    builder,
    optimization_level,
    seed,
):
    source = builder()

    start = perf_counter()

    compiled = transpile(
        source,
        basis_gates=BASIS_GATES,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )

    duration = perf_counter() - start

    gate_counts = compiled.count_ops()

    return {
        "operation": operation,
        "architecture": architecture,
        "optimization_level": optimization_level,
        "seed": seed,

        "source_qubits": source.num_qubits,
        "source_depth": source.depth(),
        "source_operations": source.size(),

        "compiled_qubits": compiled.num_qubits,
        "compiled_depth": compiled.depth(),
        "compiled_operations": compiled.size(),

        "cx_count": int(gate_counts.get("cx", 0)),
        "u_count": int(gate_counts.get("u", 0)),

        "two_qubit_depth": compiled.depth(
            filter_function=lambda instruction:
                len(instruction.qubits) == 2
        ),

        "transpilation_seconds": duration,
    }


def summarize(results):
    groups = {}

    for row in results:
        key = (
            row["operation"],
            row["architecture"],
            row["optimization_level"],
        )

        groups.setdefault(key, []).append(row)

    summary = []

    for key, rows in groups.items():
        operation, architecture, level = key

        summary.append({
            "operation": operation,
            "architecture": architecture,
            "optimization_level": level,
            "num_seeds": len(rows),
            "qubits": rows[0]["compiled_qubits"],

            "depth_mean": mean(
                r["compiled_depth"] for r in rows
            ),

            "depth_min": min(
                r["compiled_depth"] for r in rows
            ),

            "depth_max": max(
                r["compiled_depth"] for r in rows
            ),

            "operations_mean": mean(
                r["compiled_operations"] for r in rows
            ),

            "cx_mean": mean(
                r["cx_count"] for r in rows
            ),

            "two_qubit_depth_mean": mean(
                r["two_qubit_depth"] for r in rows
            ),

            "transpilation_seconds_mean": mean(
                r["transpilation_seconds"]
                for r in rows
            ),
        })

    return summary


def save_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
    )

    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        choices=[0, 1, 2, 3],
        default=[0, 3],
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print("=" * 72)
    print("QARW - CRYPTOGRAPHIC ARITHMETIC BENCHMARK")
    print("=" * 72)

    for (operation, architecture), builder in BUILDERS.items():
        for level in args.levels:
            for seed in args.seeds:

                result = benchmark_circuit(
                    operation,
                    architecture,
                    builder,
                    level,
                    seed,
                )

                results.append(result)

                print(
                    f"{operation:<12} | "
                    f"{architecture:<15} | "
                    f"L{level} | "
                    f"seed={seed:<3} | "
                    f"depth={result['compiled_depth']:<5} | "
                    f"CX={result['cx_count']:<5} | "
                    f"time={result['transpilation_seconds']:.4f}s"
                )

    summary = summarize(results)

    save_csv(
        OUTPUT_DIR / "arithmetic_resources.csv",
        results,
    )

    save_csv(
        OUTPUT_DIR / "arithmetic_summary.csv",
        summary,
    )

    with (
        OUTPUT_DIR / "arithmetic_resources.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            {
                "configuration": {
                    "basis_gates": BASIS_GATES,
                    "seeds": args.seeds,
                    "optimization_levels": args.levels,
                },
                "results": results,
                "summary": summary,
            },
            file,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for row in summary:
        print(
            f"{row['operation']:<12} | "
            f"{row['architecture']:<15} | "
            f"L{row['optimization_level']} | "
            f"mean depth={row['depth_mean']:.1f} | "
            f"mean CX={row['cx_mean']:.1f}"
        )

    print(f"\nCompleted benchmark runs: {len(results)}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
