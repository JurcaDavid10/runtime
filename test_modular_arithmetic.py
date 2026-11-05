"""Basic correctness and reversibility tests for the modulo-17 adder."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from modular_arithmetic import MODULUS, NUM_BITS, build_modular_adder_17


def evaluate_basis_input(circuit, a, b):
    """Return (a_out, b_out) from an exact statevector simulation."""
    if not (0 <= a < (1 << NUM_BITS) and 0 <= b < (1 << NUM_BITS)):
        raise ValueError("Inputs must fit in five bits")

    initialized = QuantumCircuit(2 * NUM_BITS)
    for bit in range(NUM_BITS):
        if (a >> bit) & 1:
            initialized.x(bit)
        if (b >> bit) & 1:
            initialized.x(NUM_BITS + bit)

    initialized.compose(circuit, inplace=True)
    result = Statevector.from_instruction(initialized)
    probabilities = np.abs(result.data) ** 2
    index = int(np.argmax(probabilities))
    if not np.isclose(probabilities[index], 1.0, atol=1e-9):
        raise AssertionError("Expected a deterministic computational-basis output")

    mask = (1 << NUM_BITS) - 1
    return index & mask, (index >> NUM_BITS) & mask


def run_tests():
    adder = build_modular_adder_17()
    cases = [
        (0, 0), (1, 1), (5, 8), (12, 8), (16, 1),
        (10, 15), (16, 16), (4, 13), (8, 9), (7, 7),
    ]

    for a, b in cases:
        actual_a, actual_b = evaluate_basis_input(adder, a, b)
        expected_b = (a + b) % MODULUS
        assert actual_a == a, f"Input a changed: {a} -> {actual_a}"
        assert actual_b == expected_b, (
            f"FAIL: ({a} + {b}) mod 17 expected {expected_b}, got {actual_b}"
        )
        print(f"PASS: ({a} + {b}) mod 17 = {expected_b}")

    # The out-of-range b states must stay out of the valid arithmetic domain.
    for a, b in [(1, 17), (16, 20), (9, 31)]:
        actual = evaluate_basis_input(adder, a, b)
        assert actual == (a, b), f"Invalid b state changed: {(a,b)} -> {actual}"
        print(f"PASS: invalid b={b} is preserved")

    # A reversible circuit must allow exact recovery via its inverse.
    for a, b in [(3, 6), (16, 16), (12, 8)]:
        prepare_and_uncompute = QuantumCircuit(2 * NUM_BITS)
        for bit in range(NUM_BITS):
            if (a >> bit) & 1:
                prepare_and_uncompute.x(bit)
            if (b >> bit) & 1:
                prepare_and_uncompute.x(NUM_BITS + bit)
        prepare_and_uncompute.compose(adder, inplace=True)
        prepare_and_uncompute.compose(adder.inverse(), inplace=True)
        output = Statevector.from_instruction(prepare_and_uncompute)
        target_index = a | (b << NUM_BITS)
        assert np.isclose(abs(output.data[target_index]) ** 2, 1.0, atol=1e-9)
        print(f"PASS: inverse restores |{a}>|{b}>")

    print("\nAll modular arithmetic tests passed.")


if __name__ == "__main__":
    run_tests()
