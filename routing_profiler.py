
"""Capture routing-pass metrics from Qiskit transpilation."""

ROUTING_PASSES = {
    "SabreLayout",
    "SabreSwap",
    "BasicSwap",
    "LookaheadSwap",
}


class RoutingProfiler:
    def __init__(self):
        self.previous = None
        self.events = []

    @staticmethod
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

    def __call__(self, **kwargs):
        name = type(kwargs["pass_"]).__name__
        current = self.snapshot(kwargs["dag"])

        if name in ROUTING_PASSES and self.previous is not None:
            self.events.append({
                "routing_pass_name": name,
                "routing_swaps": (
                    current["swaps"] - self.previous["swaps"]
                ),
                "routing_depth_delta": (
                    current["depth"] - self.previous["depth"]
                ),
                "routing_two_qubit_ops_delta": (
                    current["two_qubit_ops"]
                    - self.previous["two_qubit_ops"]
                ),
                "routing_pass_seconds": round(
                    float(kwargs["time"]), 5
                ),
            })

        self.previous = current

    def metrics(self):
        # This benchmark currently expects one combined
        # layout/routing pass for each constrained compilation.
        if len(self.events) != 1:
            raise RuntimeError(
                "Expected exactly one routing-related pass, "
                f"observed {len(self.events)}: {self.events}"
            )

        return self.events[0]
