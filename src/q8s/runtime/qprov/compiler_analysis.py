
"""Analyze Qiskit compiler-pass performance and resource changes."""

from typing import Any

from .record import PassProvenance


def summarize_compiler_passes(
    passes: list[PassProvenance],
    top_n: int = 5,
) -> dict[str, Any]:
    """Summarize recorded pass runtimes and resource changes."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    rows = []
    stages = {}
    total_time = 0.0

    for recorded_pass in passes:
        metadata = recorded_pass.pass_metadata
        stage = metadata.get("stage") or "unknown"
        duration = recorded_pass.pass_duration_s

        if duration is not None and duration < 0:
            raise ValueError("Pass duration cannot be negative")

        row = {
            "pass_index": recorded_pass.pass_index,
            "pass_name": recorded_pass.pass_name,
            "stage": stage,
            "duration_s": duration,
            "delta_depth": metadata.get("delta_depth"),
            "delta_size": metadata.get("delta_size"),
            "delta_two_qubit_gate_count": metadata.get(
                "delta_two_qubit_gate_count"
            ),
            "delta_two_qubit_depth": metadata.get(
                "delta_two_qubit_depth"
            ),
            "delta_gate_counts": metadata.get("delta_gate_counts"),
        }
        rows.append(row)

        if stage not in stages:
            stages[stage] = {
                "stage": stage,
                "pass_count": 0,
                "timed_pass_count": 0,
                "recorded_pass_time_s": 0.0,
            }

        stages[stage]["pass_count"] += 1

        if duration is not None:
            stages[stage]["timed_pass_count"] += 1
            stages[stage]["recorded_pass_time_s"] += duration
            total_time += duration

    stage_summaries = list(stages.values())

    for stage in stage_summaries:
        stage["time_share_pct"] = (
            100 * stage["recorded_pass_time_s"] / total_time
            if total_time > 0
            else 0.0
        )

    stage_summaries.sort(
        key=lambda item: item["recorded_pass_time_s"],
        reverse=True,
    )

    timed_passes = [
        row for row in rows if row["duration_s"] is not None
    ]

    slowest_passes = sorted(
        timed_passes,
        key=lambda item: item["duration_s"],
        reverse=True,
    )[:top_n]

    def rank_resource_changes(metric: str, increase: bool):
        candidates = [
            row
            for row in rows
            if isinstance(row[metric], (int, float))
            and (
                row[metric] > 0
                if increase
                else row[metric] < 0
            )
        ]

        return sorted(
            candidates,
            key=lambda item: item[metric],
            reverse=increase,
        )[:top_n]

    return {
        "pass_count": len(rows),
        "total_recorded_pass_time_s": total_time,
        "stage_summaries": stage_summaries,
        "top_slowest_passes": slowest_passes,
        "top_depth_increases": rank_resource_changes(
            "delta_depth", True
        ),
        "top_depth_reductions": rank_resource_changes(
            "delta_depth", False
        ),
        "top_two_qubit_increases": rank_resource_changes(
            "delta_two_qubit_gate_count", True
        ),
        "top_two_qubit_reductions": rank_resource_changes(
            "delta_two_qubit_gate_count", False
        ),
    }
