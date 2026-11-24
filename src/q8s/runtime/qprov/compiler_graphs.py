
"""Visualizations for compiler bottleneck analysis."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .compiler_analysis import summarize_compiler_passes
from .record import PassProvenance


def plot_compiler_bottlenecks(
    passes: list[PassProvenance],
    top_n: int = 5,
) -> dict[str, Figure]:
    """Generate compiler timing and resource-change figures."""

    summary = summarize_compiler_passes(passes, top_n=top_n)
    figures = {}

    def draw_chart(rows, value_key, title, xlabel):
        labels = [
            f"{row['pass_index']}: {row['pass_name']}"
            for row in rows
        ]
        values = [row[value_key] for row in rows]

        fig, ax = plt.subplots(
            figsize=(11, max(4, len(rows) * 0.55))
        )

        ax.barh(labels, values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        if any(value < 0 for value in values):
            ax.axvline(0, color="black", linewidth=0.8)

        fig.tight_layout()
        return fig

    # 1. Slowest individual compiler passes.
    slowest = summary["top_slowest_passes"]

    if slowest:
        timed_rows = [
            {
                **row,
                "duration_ms": row["duration_s"] * 1000,
            }
            for row in reversed(slowest)
        ]

        figures["slowest_passes"] = draw_chart(
            timed_rows,
            "duration_ms",
            "Slowest Compiler Passes",
            "Execution time (ms)",
        )

    # 2. Accumulated execution time per compiler stage.
    stages = summary["stage_summaries"]

    if summary["total_recorded_pass_time_s"] > 0:
        stage_rows = [
            {
                "pass_index": "",
                "pass_name": row["stage"],
                "duration_ms": row["recorded_pass_time_s"] * 1000,
            }
            for row in reversed(stages)
        ]

        figures["stage_runtime"] = draw_chart(
            stage_rows,
            "duration_ms",
            "Recorded Runtime by Compiler Stage",
            "Accumulated pass time (ms)",
        )

    # 3 and 4. Resource changes associated with compiler passes.
    resource_metrics = [
        (
            "delta_depth",
            "top_depth_increases",
            "top_depth_reductions",
            "depth_changes",
            "Circuit Depth Changes",
            "Change in circuit depth",
        ),
        (
            "delta_two_qubit_gate_count",
            "top_two_qubit_increases",
            "top_two_qubit_reductions",
            "two_qubit_changes",
            "Two-Qubit Gate Count Changes",
            "Change in two-qubit gate count",
        ),
    ]

    for (
        metric,
        increases_key,
        reductions_key,
        filename,
        title,
        xlabel,
    ) in resource_metrics:

        selected = {}

        for row in (
            summary[increases_key] + summary[reductions_key]
        ):
            selected[row["pass_index"]] = row

        if not selected:
            continue

        ranked = sorted(
            selected.values(),
            key=lambda row: row[metric],
        )

        figures[filename] = draw_chart(
            ranked,
            metric,
            title,
            xlabel,
        )

    return figures
