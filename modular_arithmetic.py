"""Reversible arithmetic modulo 17 for QARW compiler benchmarks.

The in-place adder acts on two 5-qubit registers with little-endian encoding:

    |a>|b> -> |a>|(a+b) mod 17>, for 0 <= a,b < 17.

No measurements, resets or extra work qubits are used. For b >= 17, the
second register is left unchanged, making the transformation a permutation
on the entire 10-qubit computational basis (and therefore unitary).
"""

from qiskit import QuantumCircuit, QuantumRegister

MODULUS = 17
NUM_BITS = 5


def _toggle_if(qc, controls, target, required_bits):
    """Flip target only when control qubits match the 0/1 pattern.

    Qiskit interprets ctrl_state as a little-endian mask over controls:
    controls[0] corresponds to bit 0.
    """
    if len(controls) != len(required_bits):
        raise ValueError("Need one required bit per control qubit")
    ctrl_state = sum(bit << i for i, bit in enumerate(required_bits))
    qc.mcx(controls, target, ctrl_state=ctrl_state)


def _controlled_increment_mod17(qc, enable, b):
    """When enable=1, add 1 modulo 17 to valid values of b.

    First increment the low four bits (only if b[4]=0), then swap basis
    states 00000 (0) and 10000 (16). States 17..31 are unaffected.
    """
    # 4-bit increment, from most significant to least significant bit.
    for i in range(3, -1, -1):
        controls = [enable, b[4], *b[:i]]
        required = [1, 0, *([1] * i)]
        _toggle_if(qc, controls, b[i], required)

    # If the low 4 bits are zero, exchange 0 <-> 16.
    _toggle_if(qc, [enable, *b[:4]], b[4], [1, 0, 0, 0, 0])


def _controlled_decrement_mod17(qc, enable, b):
    """When enable=1, subtract 1 modulo 17 from valid values of b."""
    # Inverse of the increment: exchange 0 <-> 16 first.
    _toggle_if(qc, [enable, *b[:4]], b[4], [1, 0, 0, 0, 0])

    # 4-bit decrement, from most significant to least significant bit.
    for i in range(3, -1, -1):
        controls = [enable, b[4], *b[:i]]
        required = [1, 0, *([0] * i)]
        _toggle_if(qc, controls, b[i], required)


def build_modular_adder_17():
    """Return a 10-qubit in-place adder for inputs in the range 0..16.

    The five bits of a have weights 1, 2, 4, 8, 16. We implement the
    first four as repeated conditional increments. Since 16 == -1
    modulo 17, the final bit instead triggers a single decrement.
    """
    a = QuantumRegister(NUM_BITS, "a")
    b = QuantumRegister(NUM_BITS, "b")
    qc = QuantumCircuit(a, b, name="add_mod_17")

    for i in range(4):
        for _ in range(1 << i):
            _controlled_increment_mod17(qc, a[i], b)

    _controlled_decrement_mod17(qc, a[4], b)
    return qc


if __name__ == "__main__":
    circuit = build_modular_adder_17()
    print("Modulus:", MODULUS)
    print("Qubits:", circuit.num_qubits)
    print("Circuit depth (before decomposition):", circuit.depth())
    print("Operations (before decomposition):", circuit.size())
    print("Operation counts:", dict(circuit.count_ops()))
    print("Run 'python test_modular_arithmetic.py' to check correctness.")
