
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from q8s.runtime.mlflow.qiskit.resource_profiler import (
    analyze_circuit,
    analyze_dag,
)

# 1. Create a sample quantum circuit
circuit = QuantumCircuit(3)

circuit.h(0)
circuit.cx(0, 1)
circuit.cx(0, 1)
circuit.ccx(0, 1, 2)
circuit.swap(0, 2)

# 2. Store the resource metrics after each compiler pass
history = []


def profiler_callback(**kwargs):
    dag = kwargs["dag"]
    compiler_pass = kwargs["pass_"]

    metrics = analyze_dag(dag)

    history.append({
        "pass_name": compiler_pass.__class__.__name__,
        "pass_index": kwargs["count"],
        **metrics,
    })


# 3. Configure Qiskit's compiler
manager = generate_preset_pass_manager(
    optimization_level=2,
    basis_gates=["u", "cx"],
    seed_transpiler=42,
)

# 4. Compile the circuit and collect metrics
optimized_circuit = manager.run(
    circuit,
    callback=profiler_callback,
)

# 5. Display the results
print("\nINITIAL CIRCUIT:")
print(analyze_circuit(circuit))

print("\nOPTIMIZED CIRCUIT:")
print(analyze_circuit(optimized_circuit))

print("\nCOMPILER PASS HISTORY:")
print("Number of recorded passes:", len(history))

for entry in history[:10]:
    print(
        f"Pass {entry['pass_index']}: "
        f"{entry['pass_name']} | "
        f"Depth={entry['depth']} | "
        f"Operations={entry['operation_count']} | "
        f"Two-qubit gates={entry['two_qubit_gate_count']}"
    )

assert history, "No compiler passes were recorded"
