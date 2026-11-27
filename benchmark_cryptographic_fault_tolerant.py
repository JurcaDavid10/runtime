
"""Fault-tolerant profiling of cryptographic arithmetic."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import qiskit

from benchmark_fault_tolerant import (
    POLICIES,
    lower_controlled_x,
    estimate_fault_tolerant_resources,
)

from modular_arithmetic import (
    build_modular_adder_17,
)

from modular_arithmetic_optimized import (
    build_optimized_modular_adder_17,
)


OUTPUT_DIR = Path(
    "benchmark_outputs/cryptographic_arithmetic"
)

BUILDERS = {
    "baseline": build_modular_adder_17,
    "optimized_rccx": build_optimized_modular_adder_17,
}


def benchmark():
    rows = []

    for architecture, builder in BUILDERS.items():

        source_adder = builder()
        source_subtractor = source_adder.inverse()

        for policy in POLICIES:

            start = perf_counter()

            lowered_adder = lower_controlled_x(
                source_adder,
                policy,
            )

            lowering_seconds = perf_counter() - start

            lowered_subtractor = (
                lowered_adder.inverse()
            )

            circuits = {
                "addition": (
                    source_adder,
                    lowered_adder,
                ),
                "subtraction": (
                    source_subtractor,
                    lowered_subtractor,
                ),
            }

            for operation, (
                source,
                lowered,
            ) in circuits.items():

                start = perf_counter()

                resources = (
                    estimate_fault_tolerant_resources(
                        lowered
                    )
                )

                estimation_seconds = (
                    perf_counter() - start
                )

                row = {
                    "operation": operation,
                    "architecture": architecture,
                    "mcx_policy": policy,
                    "qiskit_version": qiskit.__version__,

                    "source_qubits": source.num_qubits,
                    "source_depth": source.depth(),
                    "source_operations": source.size(),

                    "added_clean_ancillas": (
                        lowered.num_qubits
                        - source.num_qubits
                    ),

                    "source_gate_counts": dict(
                        source.count_ops()
                    ),

                    "shared_lowering_seconds": (
                        lowering_seconds
                    ),

                    "estimation_seconds": (
                        estimation_seconds
                    ),

                    **asdict(resources),
                }

                rows.append(row)

                print(
                    f"{operation:<12} | "
                    f"{architecture:<15} | "
                    f"{policy:<18} | "
                    f"Q={resources.logical_qubits:<3} | "
                    f"T={resources.t_count:<6} | "
                    f"T-depth={resources.t_depth}"
                )

    return rows


def save_results(rows):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        OUTPUT_DIR
        / "arithmetic_logical_resources.json"
    )

    csv_path = (
        OUTPUT_DIR
        / "arithmetic_logical_resources.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(rows, file, indent=2)

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()

        for row in rows:
            csv_row = {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }

            writer.writerow(csv_row)


def compare_inverse_resources(rows):

    print("\n" + "=" * 72)
    print("ADDITION VS SUBTRACTION")
    print("=" * 72)

    for architecture in BUILDERS:
        for policy in POLICIES:

            addition = next(
                row for row in rows
                if row["operation"] == "addition"
                and row["architecture"] == architecture
                and row["mcx_policy"] == policy
            )

            subtraction = next(
                row for row in rows
                if row["operation"] == "subtraction"
                and row["architecture"] == architecture
                and row["mcx_policy"] == policy
            )

            t_delta = (
                subtraction["t_count"]
                - addition["t_count"]
            )

            depth_delta = (
                subtraction["t_depth"]
                - addition["t_depth"]
            )

            print(
                f"{architecture:<15} | "
                f"{policy:<18} | "
                f"T-count delta={t_delta:+d} | "
                f"T-depth delta={depth_delta:+d}"
            )


def main():

    print("=" * 72)
    print("QARW - FAULT-TOLERANT CRYPTOGRAPHIC ARITHMETIC")
    print("=" * 72)

    rows = benchmark()

    save_results(rows)
    compare_inverse_resources(rows)

    print(f"\nCompleted configurations: {len(rows)}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
