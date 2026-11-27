"""Dependency-aware logical Clifford+T resource estimates.

This is a gate-level estimator, NOT a surface-code physical resource model.
Unsupported operations are rejected rather than silently assigned zero T cost.
"""

from collections import Counter
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit


# Primitive operations treated as exact Clifford gates.
CLIFFORD_GATES = frozenset({
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg",
    "cx", "cy", "cz", "swap", "ecr", "dcx",
})
T_GATES = frozenset({"t", "tdg"})
# Do not recurse into general rotations: exact Clifford+T representation is
# not guaranteed. Such gates need a stated approximation accuracy and method.
UNSUPPORTED_ROTATIONS = frozenset({
    "u", "u1", "u2", "u3", "p", "rx", "ry", "rz",
})


@dataclass(frozen=True)
class FaultTolerantResources:
    logical_qubits: int
    t_count: int
    t_depth: int
    clifford_count: int
    magic_states: int
    gate_counts: dict[str, int]
    decomposition_method: str = "exact_recursive_gate_definitions"


def estimate_fault_tolerant_resources(
    circuit: QuantumCircuit,
    max_decomposition_levels: int = 32,
) -> FaultTolerantResources:
    """Estimate logical resources using exact Qiskit gate definitions.

    T-depth is the number of ordered T/Tdg layers in the expanded circuit,
    respecting cross-qubit dependencies through Clifford gates. It is not a
    globally optimized T-depth. Magic-state demand assumes one state per T.

    Raise ValueError if any gate cannot be exactly expanded into the supported
    Clifford+T primitives. Circuit global phase is irrelevant to gate counts.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a qiskit.QuantumCircuit")
    if max_decomposition_levels < 1:
        raise ValueError("max_decomposition_levels must be positive")

    t_layers = [0] * circuit.num_qubits
    counts: Counter[str] = Counter()

    def visit(operation, mapped_qubits: tuple[int, ...], level: int) -> None:
        name = operation.name

        # Barriers synchronize dependencies; they consume no logical T states.
        if name == "barrier":
            at_layer = max((t_layers[q] for q in mapped_qubits), default=0)
            for q in mapped_qubits:
                t_layers[q] = at_layer
            return

        if name in CLIFFORD_GATES or name in T_GATES:
            layer = max((t_layers[q] for q in mapped_qubits), default=0)
            if name in T_GATES:
                layer += 1
            for q in mapped_qubits:
                t_layers[q] = layer
            counts[name] += 1
            return

        if name in UNSUPPORTED_ROTATIONS:
            raise ValueError(
                f"Gate '{name}' is not an exact supported Clifford+T primitive. "
                "First synthesize it to Clifford+T with documented precision."
            )

        if getattr(operation, "condition", None) is not None:
            raise ValueError(f"Conditioned gate '{name}' is not yet supported")
        if level >= max_decomposition_levels:
            raise ValueError(f"Decomposition limit reached at gate '{name}'")

        definition = operation.definition
        if definition is None or not definition.data:
            raise ValueError(
                f"Gate '{name}' has no usable exact Clifford+T definition. "
                "Resource estimate aborted rather than undercounted."
            )
        if len(definition.qubits) != len(mapped_qubits):
            raise ValueError(f"Qubit mapping mismatch for gate '{name}'")

        local_map = dict(zip(definition.qubits, mapped_qubits))
        for instruction in definition.data:
            if instruction.clbits:
                raise ValueError(
                    f"Gate '{name}' uses classical bits; not supported in this estimator"
                )
            visit(
                instruction.operation,
                tuple(local_map[q] for q in instruction.qubits),
                level + 1,
            )

    top_level_map = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    for instruction in circuit.data:
        if instruction.clbits:
            raise ValueError(
                "Measurement or classical-bit instructions are not yet supported"
            )
        visit(
            instruction.operation,
            tuple(top_level_map[q] for q in instruction.qubits),
            0,
        )

    t_count = sum(counts[gate] for gate in T_GATES)
    return FaultTolerantResources(
        logical_qubits=circuit.num_qubits,
        t_count=t_count,
        t_depth=max(t_layers, default=0),
        clifford_count=sum(counts[gate] for gate in CLIFFORD_GATES),
        magic_states=t_count,  # logical T-state demand, NOT factory count
        gate_counts=dict(sorted(counts.items())),
    )
