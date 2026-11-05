"""Tests for the 13-qubit, clean-ancilla modulo-17 adder.

Exhaustively tests all 32x32 input basis states with an independent
basis-and-phase gate simulator, and tests Qiskit statevector coherence.
No MLflow server is required.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from modular_arithmetic_optimized import (
    MODULUS,
    NUM_BITS,
    WORK_QUBITS,
    build_optimized_modular_adder_17,
)


A_MASK = (1 << NUM_BITS) - 1
B_SHIFT = NUM_BITS
WORK_SHIFT = 2 * NUM_BITS
TOTAL_QUBITS = 2 * NUM_BITS + WORK_QUBITS


def exact_basis_simulation(circuit, initial_index):
    """Simulate X/CX/CCX/RCCX on basis states, including RCCX phases.

    Only valid for these gates, intentionally fails if the circuit changes.
    Qiskit's RCCX has: |11,0> -> +i|11,1>,
    |11,1> -> -i|11,0>, and a -1 phase on |10,1>
    with (control0,control1,target) bit ordering.
    """
    index = initial_index
    phase = 1 + 0j
    mapping = {qubit: i for i, qubit in enumerate(circuit.qubits)}

    for item in circuit.data:
        name = item.operation.name
        positions = [mapping[q] for q in item.qubits]
        bits = [(index >> pos) & 1 for pos in positions]
        if name == "x":
            index ^= 1 << positions[0]
        elif name == "cx":
            if bits[0]:
                index ^= 1 << positions[1]
        elif name == "ccx":
            if bits[0] and bits[1]:
                index ^= 1 << positions[2]
        elif name == "rccx":
            if bits[0] and bits[1]:
                phase *= 1j if not bits[2] else -1j
                index ^= 1 << positions[2]
            elif bits[0] and (not bits[1]) and bits[2]:
                phase *= -1
        else:
            raise AssertionError(f"Unexpected source operation: {name}")
    return index, phase


def prepare_input(a, b):
    state = QuantumCircuit(TOTAL_QUBITS)
    for bit in range(NUM_BITS):
        if (a >> bit) & 1:
            state.x(bit)
        if (b >> bit) & 1:
            state.x(B_SHIFT + bit)
    return state


def test_exhaustive_basis():
    circuit = build_optimized_modular_adder_17()
    assert circuit.num_qubits == TOTAL_QUBITS
    for a in range(1 << NUM_BITS):
        for b in range(1 << NUM_BITS):
            initial = a | (b << B_SHIFT)
            index, phase = exact_basis_simulation(circuit, initial)
            expected_b = (a + b) % MODULUS if b < MODULUS else b
            expected = a | (expected_b << B_SHIFT)
            assert index == expected, (a, b, index, expected)
            assert phase == 1 + 0j, (a, b, phase)
    print("PASS: all 1,024 basis inputs, unchanged a, clean ancillas, exact phases")


def test_qiskit_statevectors():
    circuit = build_optimized_modular_adder_17()
    cases = [(0, 0), (1, 1), (5, 8), (12, 8), (16, 1), (16, 16),
             (10, 15), (9, 31), (1, 17), (31, 16)]
    for a, b in cases:
        prep = prepare_input(a, b)
        prep.compose(circuit, inplace=True)
        state = Statevector.from_instruction(prep)
        expected_b = (a + b) % MODULUS if b < MODULUS else b
        expected_index = a | (expected_b << B_SHIFT)
        assert np.isclose(state.data[expected_index], 1 + 0j, atol=1e-8), (
            a, b, state.data[expected_index]
        )
        assert np.isclose(np.linalg.norm(state.data), 1.0, atol=1e-8)
    print("PASS: 10 Qiskit statevector basis checks")


def test_coherent_superposition():
    circuit = build_optimized_modular_adder_17()
    prep = QuantumCircuit(TOTAL_QUBITS)
    prep.h(0)   # a in |0> + |1>
    prep.h(5)   # b in |0> + |1>
    prep.compose(circuit, inplace=True)
    actual = Statevector.from_instruction(prep)
    expected = np.zeros(1 << TOTAL_QUBITS, dtype=complex)
    for a in (0, 1):
        for b in (0, 1):
            expected[a | (((a + b) % MODULUS) << B_SHIFT)] = 0.5
    assert np.allclose(actual.data, expected, atol=1e-8), (
        "Unexpected phase or dirty ancillas in coherent superposition"
    )
    print("PASS: coherent superposition (no relative phases or garbage)")


def test_inverse():
    circuit = build_optimized_modular_adder_17()
    prep = prepare_input(12, 8)
    prep.compose(circuit, inplace=True)
    prep.compose(circuit.inverse(), inplace=True)
    state = Statevector.from_instruction(prep)
    assert np.isclose(state.data[12 | (8 << B_SHIFT)], 1 + 0j, atol=1e-8)
    print("PASS: circuit inverse restores the input")


if __name__ == "__main__":
    test_exhaustive_basis()
    test_qiskit_statevectors()
    test_coherent_superposition()
    test_inverse()
    print("\nALL OPTIMIZED MODULO-17 TESTS PASSED")
