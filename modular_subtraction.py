
"""Reversible quantum modular subtraction circuits."""

from qiskit import QuantumCircuit

from modular_arithmetic import (
    build_modular_adder_17,
)
from modular_arithmetic_optimized import (
    build_optimized_modular_adder_17,
)


MODULUS = 17
NUM_BITS = 5


def build_modular_subtractor_17() -> QuantumCircuit:
    """Baseline reversible modulo-17 subtraction.

    |a>|b> -> |a>|(b-a) mod 17>
    """

    circuit = build_modular_adder_17().inverse()
    circuit.name = "modular_subtractor_17"

    return circuit


def build_optimized_modular_subtractor_17() -> QuantumCircuit:
    """RCCX-optimized reversible modulo-17 subtraction.

    |a>|b>|000> -> |a>|(b-a) mod 17>|000>
    """

    circuit = build_optimized_modular_adder_17().inverse()
    circuit.name = "optimized_modular_subtractor_17"

    return circuit


if __name__ == "__main__":
    architectures = {
        "Baseline": build_modular_subtractor_17,
        "Optimized RCCX": (
            build_optimized_modular_subtractor_17
        ),
    }

    print("=" * 60)
    print("QARW - QUANTUM MODULAR SUBTRACTION")
    print("=" * 60)

    for name, builder in architectures.items():
        circuit = builder()

        print(f"\nArchitecture: {name}")
        print(f"Qubits: {circuit.num_qubits}")
        print(f"Source depth: {circuit.depth()}")
        print(f"Source operations: {circuit.size()}")
        print(f"Gate counts: {dict(circuit.count_ops())}")
