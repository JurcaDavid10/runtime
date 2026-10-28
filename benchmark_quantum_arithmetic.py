
import os
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from q8s.runtime.mlflow.qiskit import autolog
from q8s.runtime.mlflow.qiskit.resource_profiler import (
    analyze_circuit,
)

# Enable Qubernetes profiling before importing
# Qiskit's transpiler function.
autolog()

import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager


# 1. Experiment configuration
PROJECT_DIR = Path(__file__).resolve().parent
TRACKING_DIR = PROJECT_DIR / "mlruns"

TRACKING_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
mlflow.set_experiment("QARW-Quantum-Arithmetic-Benchmarks")

WIDTHS = [2, 4, 6]
OPTIMIZATION_LEVELS = [0, 1, 2, 3]
SEED = 42
BASIS_GATES = ["u", "cx"]


# 2. Construct a reversible quantum adder
def build_adder(n, a=0, b=0):
    limit = 1 << n

    if not (0 <= a < limit and 0 <= b < limit):
        raise ValueError("Input value exceeds register size")

    reg_a = QuantumRegister(n, "a")
    reg_b = QuantumRegister(n, "b")
    helper = QuantumRegister(1, "helper")

    qc = QuantumCircuit(
        reg_a, reg_b, helper,
        name=f"cdkm_adder_{n}bit"
    )

    # Initialize binary input registers.
    for i in range(n):
        if (a >> i) & 1:
            qc.x(reg_a[i])

        if (b >> i) & 1:
            qc.x(reg_b[i])

    # Add b <- (a + b) modulo 2^n.
    adder = CDKMRippleCarryAdder(
        num_state_qubits=n,
        kind="fixed",
    )

    qc.compose(adder, qubits=qc.qubits, inplace=True)

    # Expand the adder into elementary gates.
    # This avoids counting the whole adder as one gate.
    qc = qc.decompose(reps=4)

    assert all(
        len(instruction.qubits) <= 2
        for instruction in qc.data
    ), "Adder still contains undecomposed multi-qubit gates"

    return qc


# 3. Verify arithmetic correctness on a simulator
def verify_adder(n):
    mask = (1 << n) - 1

    cases = [
        (1, 1),
        (mask, 1),  # Test modular overflow
    ]

    for a, b in cases:
        circuit = build_adder(n, a, b)
        state = Statevector.from_instruction(circuit)

        probabilities = np.abs(state.data) ** 2
        basis_index = int(np.argmax(probabilities))

        assert probabilities[basis_index] > 1 - 1e-10

        # Qiskit uses little-endian qubit indexing.
        result_a = basis_index & mask
        result_b = (basis_index >> n) & mask
        result_helper = (basis_index >> (2 * n)) & 1

        expected = (a + b) % (1 << n)

        assert result_a == a
        assert result_b == expected
        assert result_helper == 0

        print(
            f"PASS: {n}-bit adder | "
            f"{a} + {b} mod {1 << n} = {result_b}"
        )


# 4. Run the compilation benchmarks
client = MlflowClient()
results = []

for n in WIDTHS:
    print(f"\n{'=' * 60}")
    print(f"BENCHMARKING {n}-BIT RIPPLE-CARRY ADDER")
    print(f"{'=' * 60}")

    verify_adder(n)

    # Benchmark the arithmetic itself, without input setup.
    source_circuit = build_adder(n)

    print("Circuit qubits:", source_circuit.num_qubits)
    print("Source depth:", source_circuit.depth())
    print("Source operations:", source_circuit.size())

    for level in OPTIMIZATION_LEVELS:
        print(f"\nOptimization level {level}...")

        with mlflow.start_run(
            run_name=f"adder-{n}bit-opt-{level}"
        ) as run:

            mlflow.log_param("circuit_family", "CDKM")
            mlflow.log_param("register_bits", n)
            mlflow.log_param("adder_kind", "fixed")
            mlflow.log_param("basis_gates", "u,cx")
            mlflow.log_param(
                "qiskit_version", qiskit.__version__
            )

            # Must be created inside the active MLflow run.
            manager = generate_preset_pass_manager(
                optimization_level=level,
                basis_gates=BASIS_GATES,
                seed_transpiler=SEED,
            )

            compiled = manager.run(source_circuit)

            # Reuse our custom circuit analyzer.
            profile = analyze_circuit(compiled)

            depth = profile["depth"]
            size = profile["operation_count"]
            twoq_count = profile["two_qubit_gate_count"]
            twoq_depth = profile["two_qubit_depth"]
            cx_count = int(
                profile["gate_counts"].get("cx", 0)
            )

            mlflow.log_metric("arithmetic_depth", depth)
            mlflow.log_metric("arithmetic_operations", size)
            mlflow.log_metric(
                "arithmetic_two_qubit_count", twoq_count
            )
            mlflow.log_metric(
                "arithmetic_two_qubit_depth", twoq_depth
            )
            mlflow.log_metric(
                "arithmetic_cx_count", cx_count
            )

            run_id = run.info.run_id

        # Verify the automatically generated artifacts.
        saved_run = client.get_run(run_id)

        artifacts = {
            item.path
            for item in client.list_artifacts(run_id)
        }

        expected_artifacts = {
            "qprov_record.json",
            "transpilation_timeline.png",
            "resource_visualizations",
        }

        assert saved_run.info.status == "FINISHED"
        assert expected_artifacts.issubset(artifacts)

        metrics = saved_run.data.metrics

        results.append({
            "bits": n,
            "qubits": compiled.num_qubits,
            "level": level,
            "depth": depth,
            "size": size,
            "cx": cx_count,
            "twoq_depth": twoq_depth,
            "passes": int(metrics["passes_count"]),
            "time_ms": (
                metrics["transpilation_duration"] * 1000
            ),
        })

        print(
            f"Depth={depth}, "
            f"CX={cx_count}, "
            f"2Q depth={twoq_depth}, "
            f"Passes={int(metrics['passes_count'])}"
        )


# 5. Display the final results
print("\n" + "=" * 85)
print("QUANTUM ARITHMETIC BENCHMARK RESULTS")
print("=" * 85)

print(
    f"{'Bits':<7}"
    f"{'Qubits':<9}"
    f"{'Level':<8}"
    f"{'Depth':<9}"
    f"{'Ops':<8}"
    f"{'CX':<8}"
    f"{'2QDepth':<10}"
    f"{'Passes':<9}"
    f"{'Time(ms)':<12}"
)

for r in results:
    print(
        f"{r['bits']:<7}"
        f"{r['qubits']:<9}"
        f"{r['level']:<8}"
        f"{r['depth']:<9}"
        f"{r['size']:<8}"
        f"{r['cx']:<8}"
        f"{r['twoq_depth']:<10}"
        f"{r['passes']:<9}"
        f"{r['time_ms']:<12.3f}"
    )

print("\nSUCCESS: Arithmetic benchmarks saved to MLflow!")
