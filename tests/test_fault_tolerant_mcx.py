"""Tests for the explicit clean-ancilla MCX lowering policies."""
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit.quantum_info import Statevector

from benchmark_fault_tolerant import POLICIES, lower_controlled_x
from q8s.runtime.qprov.fault_tolerant_resources import estimate_fault_tolerant_resources


class TestFaultTolerantMCX(unittest.TestCase):
    def test_exact_actions_and_phases(self):
        for k in (3, 4):
            for state in ((1 << k) - 1, 1):
                source = QuantumCircuit(k + 1)
                source.append(MCXGate(k, ctrl_state=state), range(k + 1))
                for policy in POLICIES:
                    with self.subTest(k=k, state=state, policy=policy):
                        lowered = lower_controlled_x(source, policy)
                        self.assertEqual(lowered.num_qubits, 2 * k - 1)
                        for b in range(1 << (k + 1)):
                            expected = Statevector.from_int(b, 1 << (k + 1)).evolve(source)
                            actual = Statevector.from_int(b, 1 << lowered.num_qubits).evolve(lowered)
                            padded = np.zeros(1 << lowered.num_qubits, dtype=complex)
                            padded[:len(expected.data)] = expected.data
                            # No per-input global phase alignment: detects relative phases
                            # and nonzero scratch output amplitudes.
                            np.testing.assert_allclose(actual.data, padded, atol=1e-10)

    def test_t_costs(self):
        for k in (3, 4, 5):
            source = QuantumCircuit(k + 1)
            source.append(MCXGate(k), range(k + 1))
            for policy, expected in (("ccx_chain", 7 * (2*k - 3)),
                                     ("rccx_chain", 8*(k - 2) + 7)):
                with self.subTest(k=k, policy=policy):
                    result = estimate_fault_tolerant_resources(lower_controlled_x(source, policy))
                    self.assertEqual(result.t_count, expected)
                    self.assertEqual(result.magic_states, expected)
                    self.assertEqual(result.logical_qubits, 2*k - 1)

    def test_reuses_scratch_and_preserves_existing_rccx(self):
        source = QuantumCircuit(5)
        source.append(MCXGate(4), range(5))
        source.append(MCXGate(3), range(4))
        source.rccx(0, 1, 2)
        for policy in POLICIES:
            lowered = lower_controlled_x(source, policy)
            self.assertEqual(lowered.num_qubits, 7)
            result = estimate_fault_tolerant_resources(lowered)
            expected = 7*(5+3)+4 if policy == "ccx_chain" else 23+15+4
            self.assertEqual(result.t_count, expected)


if __name__ == "__main__":
    unittest.main()
