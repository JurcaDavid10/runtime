
import matplotlib.pyplot as plt

from q8s.runtime.qprov.compiler_graphs import (
    plot_compiler_bottlenecks,
)
from q8s.runtime.qprov.record import PassProvenance


def test_compiler_bottleneck_figures():
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
            pass_name="OptimizeGates",
            pass_index=3,
            pass_duration_s=0.20,
            pass_metadata={
                "stage": "optimization",
                "delta_depth": -5,
                "delta_two_qubit_gate_count": -3,
            },
        ),
    ]

    figures = plot_compiler_bottlenecks(passes)

    try:
        assert set(figures) == {
            "slowest_passes",
            "stage_runtime",
            "depth_changes",
            "two_qubit_changes",
        }

        for figure in figures.values():
            assert figure.axes

    finally:
        for figure in figures.values():
            plt.close(figure)


def test_empty_compiler_bottleneck_figures():
    figures = plot_compiler_bottlenecks([])

    assert figures == {}


def test_missing_timing_and_resource_changes():
    passes = [
        PassProvenance(
            pass_name="UnknownPass",
            pass_index=1,
            pass_duration_s=None,
            pass_metadata={},
        ),
    ]

    figures = plot_compiler_bottlenecks(passes)

    assert figures == {}
