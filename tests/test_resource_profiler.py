
from qiskit import QuantumCircuit

from q8s.runtime.mlflow.qiskit.resource_profiler import analyze_circuit


from qiskit.converters import circuit_to_dag
from q8s.runtime.mlflow.qiskit.resource_profiler import analyze_dag


def test_bell_circuit_metrics():
    circuit = QuantumCircuit(2)

    circuit.h(0)
    circuit.cx(0, 1)

    metrics = analyze_circuit(circuit)

    assert metrics["num_qubits"] == 2
    assert metrics["depth"] == 2
    assert metrics["operation_count"] == 2
    assert metrics["gate_counts"] == {"h": 1, "cx": 1}
    assert metrics["two_qubit_gate_count"] == 1
    assert metrics["two_qubit_depth"] == 1


def test_parallel_two_qubit_gates():
    circuit = QuantumCircuit(4)

    circuit.cx(0, 1)
    circuit.cx(2, 3)

    metrics = analyze_circuit(circuit)

    assert metrics["two_qubit_gate_count"] == 2
    assert metrics["two_qubit_depth"] == 1


def test_empty_circuit():
    circuit = QuantumCircuit(2)

    metrics = analyze_circuit(circuit)

    assert metrics["operation_count"] == 0
    assert metrics["depth"] == 0
    assert metrics["two_qubit_gate_count"] == 0

def test_dag_metrics_match_circuit_metrics():
    circuit = QuantumCircuit(3)

    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    dag = circuit_to_dag(circuit)

    circuit_metrics = analyze_circuit(circuit)
    dag_metrics = analyze_dag(dag)

    assert circuit_metrics == dag_metrics