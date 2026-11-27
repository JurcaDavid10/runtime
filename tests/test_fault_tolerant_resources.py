"""Milestone 7.1: exact Clifford+T resource-estimator regression tests."""

import unittest

from qiskit import QuantumCircuit
from q8s.runtime.qprov.fault_tolerant_resources import (
    estimate_fault_tolerant_resources,
)


class TestFaultTolerantResources(unittest.TestCase):
    def test_t_layers_respect_clifford_dependencies(self):
        circuit = QuantumCircuit(3)
        circuit.t(0)
        circuit.tdg(1)
        circuit.cx(0, 1)
        circuit.t(1)

        resources = estimate_fault_tolerant_resources(circuit)
        self.assertEqual(resources.logical_qubits, 3)
        self.assertEqual(resources.t_count, 3)
        self.assertEqual(resources.t_depth, 2)
        self.assertEqual(resources.magic_states, 3)
        self.assertEqual(resources.clifford_count, 1)

    def test_only_clifford_gates_require_no_magic_states(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        resources = estimate_fault_tolerant_resources(circuit)
        self.assertEqual(resources.t_count, 0)
        self.assertEqual(resources.t_depth, 0)
        self.assertEqual(resources.magic_states, 0)

    def test_ccx_exact_decomposition(self):
        circuit = QuantumCircuit(3)
        circuit.ccx(0, 1, 2)
        resources = estimate_fault_tolerant_resources(circuit)
        self.assertEqual(resources.t_count, 7)
        self.assertGreater(resources.t_depth, 0)

    def test_rccx_exact_decomposition(self):
        circuit = QuantumCircuit(3)
        circuit.rccx(0, 1, 2)
        resources = estimate_fault_tolerant_resources(circuit)
        self.assertEqual(resources.t_count, 4)
        self.assertGreater(resources.t_depth, 0)

    def test_unsupported_u_and_rotation_are_not_undercounted(self):
        for name in ("u", "rz"):
            with self.subTest(gate=name):
                circuit = QuantumCircuit(1)
                if name == "u":
                    circuit.u(0.1, 0.2, 0.3, 0)
                else:
                    circuit.rz(0.123, 0)
                with self.assertRaisesRegex(ValueError, "Clifford\\+T"):
                    estimate_fault_tolerant_resources(circuit)


if __name__ == "__main__":
    unittest.main()
