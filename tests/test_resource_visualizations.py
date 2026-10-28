import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from q8s.runtime.qprov.graphs import plot_resource_profiles
from q8s.runtime.qprov.record import (
    PassProvenance,
    CompilationProvenance,
    QProvRecord,
)
from q8s.runtime.mlflow.qiskit import utils


def make_pass(index, profile=None):
    metadata = {
        "depth": 8,
        "size": 11,
        "stage": "init",
    }

    if profile is not None:
        metadata["resource_profile"] = profile

    return PassProvenance(
        pass_name="ExamplePass",
        pass_index=index,
        pass_duration_s=0.001,
        pass_metadata=metadata,
    )


def collect_logged_artifacts(record):
    """Capture artifact contents without contacting MLflow."""

    captured = {}

    def capture(directory):
        root = Path(directory)

        for file in root.rglob("*"):
            if file.is_file():
                name = file.relative_to(root).as_posix()
                captured[name] = file.read_bytes()

    with (
        patch.object(
            utils.mlflow, "active_run", return_value=object()
        ),
        patch.object(utils.mlflow, "log_metric"),
        patch.object(
            utils.mlflow,
            "log_artifacts",
            side_effect=capture,
        ),
    ):
        utils.log_to_mlflow(record)

    return captured


class TestResourceVisualizations(unittest.TestCase):

    def tearDown(self):
        plt.close("all")

    def test_passes_are_sorted(self):
        passes = [
            make_pass(10, {
                "depth": 2,
                "two_qubit_gate_count": 1,
                "two_qubit_depth": 1,
            }),
            make_pass(2, {
                "depth": 8,
                "two_qubit_gate_count": 5,
                "two_qubit_depth": 5,
            }),
        ]

        figures = plot_resource_profiles(passes)

        self.assertEqual(
            set(figures),
            {
                "depth",
                "two_qubit_gate_count",
                "two_qubit_depth",
            },
        )

        line = figures["depth"].axes[0].lines[0]

        self.assertEqual(list(line.get_xdata()), [2, 10])
        self.assertEqual(list(line.get_ydata()), [8, 2])

        gate_line = (
            figures["two_qubit_gate_count"].axes[0].lines[0]
        )

        self.assertEqual(
            list(gate_line.get_ydata()), [5, 1]
        )

    def test_missing_metrics_and_zero_values(self):
        passes = [
            make_pass(0, {"depth": 0}),
            make_pass(4, {
                "two_qubit_gate_count": 0,
                "two_qubit_depth": 0,
            }),
        ]

        figures = plot_resource_profiles(passes)

        self.assertEqual(len(figures), 3)

        depth_line = figures["depth"].axes[0].lines[0]

        self.assertEqual(
            list(depth_line.get_xdata()), [0]
        )
        self.assertEqual(
            list(depth_line.get_ydata()), [0]
        )

        gate_line = (
            figures["two_qubit_gate_count"].axes[0].lines[0]
        )

        self.assertEqual(
            list(gate_line.get_xdata()), [4]
        )
        self.assertEqual(
            list(gate_line.get_ydata()), [0]
        )

    def test_no_resource_profiles(self):
        passes = [
            make_pass(0),
            make_pass(1),
        ]

        figures = plot_resource_profiles(passes)

        self.assertEqual(figures, {})
        self.assertEqual(
            plot_resource_profiles([]), {}
        )

    def test_mlflow_uploads_resource_graphs(self):
        compilation = CompilationProvenance(
            duration_s=0.02,
            passes=[
                make_pass(0, {
                    "depth": 8,
                    "two_qubit_gate_count": 5,
                    "two_qubit_depth": 5,
                }),
                make_pass(1, {
                    "depth": 2,
                    "two_qubit_gate_count": 1,
                    "two_qubit_depth": 1,
                }),
            ],
        )

        record = QProvRecord(compilation=compilation)
        artifacts = collect_logged_artifacts(record)

        expected = {
            "qprov_record.json",
            "transpilation_timeline.png",
            "resource_visualizations/depth.png",
            "resource_visualizations/two_qubit_depth.png",
            "resource_visualizations/two_qubit_gate_count.png",
        }

        self.assertEqual(set(artifacts), expected)

        png_header = b"\x89PNG\r\n\x1a\n"

        for name in expected:
            self.assertGreater(len(artifacts[name]), 0)

            if name.endswith(".png"):
                self.assertTrue(
                    artifacts[name].startswith(png_header)
                )

    def test_original_artifacts_without_profiles(self):
        compilation = CompilationProvenance(
            duration_s=0.01,
            passes=[make_pass(0)],
        )

        record = QProvRecord(compilation=compilation)
        artifacts = collect_logged_artifacts(record)

        self.assertEqual(
            set(artifacts),
            {
                "qprov_record.json",
                "transpilation_timeline.png",
            },
        )


if __name__ == "__main__":
    unittest.main()
