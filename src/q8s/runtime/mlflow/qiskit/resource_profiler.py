
from qiskit.circuit import QuantumCircuit
from qiskit.converters import dag_to_circuit
from qiskit.dagcircuit import DAGCircuit


def analyze_circuit(circuit: QuantumCircuit) -> dict:
    """Collect resource metrics from a Qiskit quantum circuit."""

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("Expected a Qiskit QuantumCircuit")

    # Identify two-qubit operations, excluding directives.
    def is_two_qubit_operation(instruction):
        return (
            len(instruction.qubits) == 2
            and not instruction.is_directive()
        )

    two_qubit_count = sum(
        is_two_qubit_operation(instruction)
        for instruction in circuit.data
    )

    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "operation_count": circuit.size(),
        "gate_counts": dict(circuit.count_ops()),
        "two_qubit_gate_count": two_qubit_count,
        "two_qubit_depth": circuit.depth(
            filter_function=is_two_qubit_operation
        ),
    }


def analyze_dag(dag: DAGCircuit) -> dict:
    """Collect resource metrics from a Qiskit DAG."""

    if not isinstance(dag, DAGCircuit):
        raise TypeError("Expected a Qiskit DAGCircuit")

    circuit = dag_to_circuit(dag)

    return analyze_circuit(circuit)
