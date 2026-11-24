
import pytest

from q8s.runtime.qprov.compiler_analysis import (
    summarize_compiler_passes,
)
from q8s.runtime.qprov.record import PassProvenance


def test_compiler_bottleneck_summary():
    passes = [
        PassProvenance(
            pass_name="SetLayout",
            pass_index=1,
            pass_duration_s=0.10,
            pass_metadata={"stage": "layout"},
        ),
        PassProvenance(
            pass_name="SabreSwap",
            pass_index=2,
            pass_duration_s=0.45,
            pass_metadata={
                "stage": "routing",
                "delta_depth": 12,
                "delta_two_qubit_gate_count": 8,
            },
        ),
        PassProvenance(
            pass_name="Optimize1qGates",
            pass_index=3,
            pass_duration_s=0.20,
            pass_metadata={
                "stage": "optimization",
                "delta_depth": -4,
                "delta_two_qubit_gate_count": -2,
            },
        ),
        PassProvenance(
            pass_name="ConsolidateBlocks",
            pass_index=4,
            pass_duration_s=0.05,
            pass_metadata={
                "stage": "optimization",
                "delta_depth": -3,
                "delta_two_qubit_gate_count": -4,
            },
        ),
    ]

    summary = summarize_compiler_passes(passes)

    assert summary["pass_count"] == 4
    assert summary["total_recorded_pass_time_s"] == pytest.approx(0.8)

    assert summary["top_slowest_passes"][0]["pass_name"] == "SabreSwap"

    routing = next(
        stage
        for stage in summary["stage_summaries"]
        if stage["stage"] == "routing"
    )
    assert routing["recorded_pass_time_s"] == pytest.approx(0.45)
    assert routing["time_share_pct"] == pytest.approx(56.25)

    assert summary["top_depth_increases"][0]["delta_depth"] == 12
    assert summary["top_depth_reductions"][0]["delta_depth"] == -4

    assert (
        summary["top_two_qubit_increases"][0][
            "delta_two_qubit_gate_count"
        ]
        == 8
    )

    assert (
        summary["top_two_qubit_reductions"][0][
            "delta_two_qubit_gate_count"
        ]
        == -4
    )


def test_empty_compiler_summary():
    summary = summarize_compiler_passes([])

    assert summary["pass_count"] == 0
    assert summary["total_recorded_pass_time_s"] == 0
    assert summary["stage_summaries"] == []
    assert summary["top_slowest_passes"] == []


def test_missing_pass_duration():
    passes = [
        PassProvenance(
            pass_name="UnknownPass",
            pass_index=1,
            pass_duration_s=None,
            pass_metadata={},
        ),
    ]

    summary = summarize_compiler_passes(passes)

    assert summary["pass_count"] == 1
    assert summary["total_recorded_pass_time_s"] == 0
    assert summary["top_slowest_passes"] == []
    assert summary["stage_summaries"][0]["stage"] == "unknown"
    assert summary["stage_summaries"][0]["timed_pass_count"] == 0
