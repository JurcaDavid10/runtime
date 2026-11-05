"""Clean-ancilla, relative-phase implementation of in-place addition modulo 17.

Qubit layout: a[0..4], b[0..4], work[0..2], all little-endian.
For a in 0..31 and b in 0..16, with work initialized to |000>:
    |a>|b>|000> -> |a>|(a+b) mod 17>|000>
For b in 17..31, b is unchanged. The transformation on the initialized
ancilla subspace is exact and phase-free, including on superpositions.

The arithmetic network is intentionally the same as modular_arithmetic.py;
the difference is the way 3- to 5-controlled X gates are synthesized.
This isolates a gate-decomposition/ancilla tradeoff from algorithm changes.
"""

from qiskit import QuantumCircuit, QuantumRegister

MODULUS = 17
NUM_BITS = 5
WORK_QUBITS = 3


def _toggle_with_clean_ancillas(qc, controls, target, required_bits, work):
    """Apply an exact multi-controlled X via an RCCX compute/uncompute ladder.

    Negated controls are handled with X conjugations. Temporary AND values
    are computed using relative-phase RCCX gates, followed by an exact CCX
    flip of the target, and then uncomputed. The relative phases cancel.
    """
    controls = list(controls)
    required_bits = list(required_bits)
    if len(controls) != len(required_bits):
        raise ValueError("One required bit per control is needed")
    if any(bit not in (0, 1) for bit in required_bits):
        raise ValueError("Control patterns must contain only 0 or 1")
    if len(set(controls)) != len(controls) or target in controls:
        raise ValueError("Controls must be distinct from each other and target")
    if any(q in controls or q == target for q in work):
        raise ValueError("Work qubits must be disjoint from controls and target")

    for control, bit in zip(controls, required_bits):
        if bit == 0:
            qc.x(control)

    k = len(controls)
    if k == 0:
        qc.x(target)
    elif k == 1:
        qc.cx(controls[0], target)
    elif k == 2:
        qc.ccx(controls[0], controls[1], target)
    else:
        if len(work) < k - 2:
            raise ValueError(f"{k} controls require {k - 2} clean work qubits")
        # work[0] = controls[0] AND controls[1] (up to relative phase).
        qc.rccx(controls[0], controls[1], work[0])
        # Extend the running conjunction one control at a time.
        for j in range(1, k - 2):
            qc.rccx(work[j - 1], controls[j + 1], work[j])

        # Flip target exactly, conditioned on all k controls.
        qc.ccx(work[k - 3], controls[k - 1], target)

        # Uncompute the AND chain. These gates cancel RCCX phases exactly.
        for j in range(k - 3, 0, -1):
            qc.rccx(work[j - 1], controls[j + 1], work[j])
        qc.rccx(controls[0], controls[1], work[0])

    for control, bit in zip(controls, required_bits):
        if bit == 0:
            qc.x(control)


def _controlled_increment_mod17(qc, enable, b, work):
    # Use the original modular network to make comparisons meaningful.
    for i in range(3, -1, -1):
        controls = [enable, b[4], *b[:i]]
        required = [1, 0, *([1] * i)]
        _toggle_with_clean_ancillas(qc, controls, b[i], required, work)
    _toggle_with_clean_ancillas(
        qc, [enable, *b[:4]], b[4], [1, 0, 0, 0, 0], work
    )


def _controlled_decrement_mod17(qc, enable, b, work):
    _toggle_with_clean_ancillas(
        qc, [enable, *b[:4]], b[4], [1, 0, 0, 0, 0], work
    )
    for i in range(3, -1, -1):
        controls = [enable, b[4], *b[:i]]
        required = [1, 0, *([0] * i)]
        _toggle_with_clean_ancillas(qc, controls, b[i], required, work)


def build_optimized_modular_adder_17():
    """Build 13-qubit modulo-17 adder, with work register returned to zero."""
    a = QuantumRegister(NUM_BITS, "a")
    b = QuantumRegister(NUM_BITS, "b")
    work = QuantumRegister(WORK_QUBITS, "work")
    qc = QuantumCircuit(a, b, work, name="add_mod17_rccx_anc")

    for i in range(4):
        for _ in range(1 << i):
            _controlled_increment_mod17(qc, a[i], b, work)
    _controlled_decrement_mod17(qc, a[4], b, work)
    return qc


if __name__ == "__main__":
    circuit = build_optimized_modular_adder_17()
    print("Architecture: clean-ancilla RCCX compute/uncompute")
    print("Qubits:", circuit.num_qubits)
    print("Depth before basis decomposition:", circuit.depth())
    print("Operations before basis decomposition:", circuit.size())
    print("Operations by type:", dict(circuit.count_ops()))
    print("Next: python test_modular_arithmetic_optimized.py")
