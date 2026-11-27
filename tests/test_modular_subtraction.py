
import numpy as np
import pytest

from qiskit.quantum_info import Operator, Statevector

from ecc_reference import mod_sub

from modular_arithmetic import (
    build_modular_adder_17,
)
from modular_arithmetic_optimized import (
    build_optimized_modular_adder_17,
)
from modular_subtraction import (
    build_modular_subtractor_17,
    build_optimized_modular_subtractor_17,
)


MODULUS = 17
NUM_BITS = 5
REGISTER_SIZE = 2**NUM_BITS
TOLERANCE = 1e-10


def simulate_monomial(circuit, inputs):
    """Track basis-state indices and complex phases."""

    states = np.asarray(inputs, dtype=np.int64).copy()

    phases = np.full(
        states.shape,
        np.exp(1j * float(circuit.global_phase)),
        dtype=complex,
    )

    for instruction in circuit.data:
        operation = instruction.operation

        if operation.name == "barrier":
            continue

        positions = [
            circuit.find_bit(qubit).index
            for qubit in instruction.qubits
        ]

        # Efficient simulation of controlled-X gates.
        base_gate = getattr(operation, "base_gate", None)
        num_controls = getattr(
            operation, "num_ctrl_qubits", 0
        )

        if (
            getattr(base_gate, "name", None) == "x"
            and num_controls + 1 == len(positions)
        ):
            active = np.ones(states.shape, dtype=bool)
            control_state = operation.ctrl_state

            for i, position in enumerate(positions[:-1]):
                expected_bit = (control_state >> i) & 1
                actual_bit = (states >> position) & 1

                active &= actual_bit == expected_bit

            states[active] ^= 1 << positions[-1]
            continue

        # Inspect the matrix of smaller gates,
        # including relative-phase Toffoli gates.
        if len(positions) > 5:
            raise ValueError(
                f"Unsupported large gate: {operation.name}"
            )

        matrix = Operator(operation).data
        nonzero = np.abs(matrix) > TOLERANCE

        # A monomial gate has exactly one nonzero
        # element in every row and column.
        assert np.all(nonzero.sum(axis=0) == 1)
        assert np.all(nonzero.sum(axis=1) == 1)

        permutation = np.argmax(
            np.abs(matrix), axis=0
        )

        coefficients = matrix[
            permutation,
            np.arange(matrix.shape[1]),
        ]

        np.testing.assert_allclose(
            np.abs(coefficients),
            1.0,
            atol=TOLERANCE,
            rtol=0,
        )

        local_inputs = np.zeros_like(states)

        for i, position in enumerate(positions):
            local_inputs |= (
                ((states >> position) & 1) << i
            )

        local_outputs = permutation[local_inputs]
        phases *= coefficients[local_inputs]

        mask = sum(
            1 << position for position in positions
        )

        states &= ~mask

        for i, position in enumerate(positions):
            states |= (
                ((local_outputs >> i) & 1) << position
            )

    return states, phases


@pytest.fixture(
    scope="module",
    params=["baseline", "optimized_rccx"],
)
def architecture(request):
    if request.param == "baseline":
        adder = build_modular_adder_17()
        subtractor = build_modular_subtractor_17()
    else:
        adder = build_optimized_modular_adder_17()
        subtractor = (
            build_optimized_modular_subtractor_17()
        )

    return adder, subtractor


@pytest.fixture(scope="module")
def exhaustive_results(architecture):
    _, subtractor = architecture

    # All 1024 data-register states.
    # Ancillas are initialized to zero.
    inputs = np.arange(
        REGISTER_SIZE**2,
        dtype=np.int64,
    )

    outputs, phases = simulate_monomial(
        subtractor, inputs
    )

    return inputs, outputs, phases


def test_valid_arithmetic(exhaustive_results):
    inputs, outputs, _ = exhaustive_results

    for a in range(MODULUS):
        for b in range(MODULUS):
            index = a | (b << NUM_BITS)

            expected = a | (
                mod_sub(b, a) << NUM_BITS
            )

            assert inputs[index] == index
            assert outputs[index] == expected, (
                f"Failed for a={a}, b={b}"
            )



def test_invalid_inputs_and_ancillas(exhaustive_results):
    inputs, outputs, _ = exhaustive_results

    for index in inputs:
        a = index & (REGISTER_SIZE - 1)
        b = index >> NUM_BITS

        output = int(outputs[index])

        output_a = output & (REGISTER_SIZE - 1)
        output_b = (
            output >> NUM_BITS
        ) & (REGISTER_SIZE - 1)

        input_valid = (
            a < MODULUS and b < MODULUS
        )

        output_valid = (
            output_a < MODULUS
            and output_b < MODULUS
        )

        # Valid and invalid subspaces must remain separate.
        assert input_valid == output_valid, (
            f"Subspace violation for a={a}, b={b}"
        )

    # All ancillas must return to zero.
    assert np.all(
        (outputs >> (2 * NUM_BITS)) == 0
    ), "Ancilla cleanup failed"

    # The data-register transformation must be bijective.
    assert len(np.unique(outputs)) == len(inputs)

    # Every data-register basis state must occur once.
    np.testing.assert_array_equal(
        np.sort(outputs),
        inputs,
    )



def test_relative_phase_preservation(exhaustive_results):
    _, _, phases = exhaustive_results

    # All inputs must acquire the same phase.
    # A common global phase is physically irrelevant.
    np.testing.assert_allclose(
        phases,
        phases[0],
        atol=TOLERANCE,
        rtol=0,
    )


def test_inverse_round_trip(
    architecture,
    exhaustive_results,
):
    adder, _ = architecture
    inputs, outputs, phases = exhaustive_results

    restored, addition_phases = simulate_monomial(
        adder, outputs
    )

    np.testing.assert_array_equal(
        restored, inputs
    )

    np.testing.assert_allclose(
        phases * addition_phases,
        1.0,
        atol=TOLERANCE,
        rtol=0,
    )


def test_superposition_correctness(architecture):
    _, subtractor = architecture

    initial = np.zeros(
        2**subtractor.num_qubits,
        dtype=complex,
    )

    expected = np.zeros_like(initial)

    cases = [
        (0, 0, 0.5),
        (8, 3, 0.5j),
        (16, 0, -0.5),
        (5, 16, -0.5j),
    ]

    for a, b, amplitude in cases:
        input_index = a | (b << NUM_BITS)

        output_index = a | (
            mod_sub(b, a) << NUM_BITS
        )

        initial[input_index] = amplitude
        expected[output_index] = amplitude

    actual = Statevector(initial).evolve(
        subtractor
    ).data

    fidelity = abs(
        np.vdot(expected, actual)
    )**2

    assert fidelity == pytest.approx(
        1.0, abs=TOLERANCE
    )


def test_architecture_equivalence():
    baseline = build_modular_subtractor_17()
    optimized = build_optimized_modular_subtractor_17()

    inputs = np.arange(
        REGISTER_SIZE**2,
        dtype=np.int64,
    )

    baseline_outputs, baseline_phases = (
        simulate_monomial(baseline, inputs)
    )

    optimized_outputs, optimized_phases = (
        simulate_monomial(optimized, inputs)
    )

    # Both architectures must implement the same permutation.
    np.testing.assert_array_equal(
        baseline_outputs,
        optimized_outputs,
    )

    # Compare phases up to a common global phase.
    np.testing.assert_allclose(
        baseline_phases / baseline_phases[0],
        optimized_phases / optimized_phases[0],
        atol=TOLERANCE,
        rtol=0,
    )


def test_inverse_rccx_fault_tolerant_resources():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RCCXGate

    from benchmark_fault_tolerant import (
        estimate_fault_tolerant_resources,
    )

    forward = QuantumCircuit(3)
    forward.append(RCCXGate(), [0, 1, 2])

    inverse = forward.inverse()

    forward_resources = (
        estimate_fault_tolerant_resources(forward)
    )

    inverse_resources = (
        estimate_fault_tolerant_resources(inverse)
    )

    assert forward_resources.t_count == 4
    assert inverse_resources.t_count == 4

    assert (
        forward_resources.logical_qubits
        == inverse_resources.logical_qubits
        == 3
    )


def test_fault_tolerant_lowering_equivalence(
    architecture,
):
    from benchmark_fault_tolerant import (
        POLICIES,
        lower_controlled_x,
    )

    adder, subtractor = architecture

    inputs = np.arange(
        REGISTER_SIZE**2,
        dtype=np.int64,
    )

    expected_outputs, expected_phases = (
        simulate_monomial(subtractor, inputs)
    )

    for policy in POLICIES:
        lowered_adder = lower_controlled_x(
            adder,
            policy,
        )

        lowered_subtractor = lowered_adder.inverse()

        actual_outputs, actual_phases = (
            simulate_monomial(
                lowered_subtractor,
                inputs,
            )
        )

        # Correct arithmetic and clean ancillas.
        np.testing.assert_array_equal(
            actual_outputs,
            expected_outputs,
            err_msg=f"Failed policy: {policy}",
        )

        # Phase equivalence up to global phase.
        np.testing.assert_allclose(
            actual_phases / actual_phases[0],
            expected_phases / expected_phases[0],
            atol=TOLERANCE,
            rtol=0,
            err_msg=f"Failed policy: {policy}",
        )
