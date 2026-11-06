# QARW Hardware-Aware Benchmark — metric definitions

- **logical_qubits**: Source circuit's logical width; the minimum number of
  available device qubits needed to accommodate the circuit, not necessarily
  the number of qubits activated by the compiled circuit.
- **device_capacity_qubits**: Number of physical vertices in the supplied
  coupling map (16, 16, or 19). **N/A for unrestricted**, which uses no device.
- **physical_qubits_touched**: Distinct physical qubit indices participating
  in at least one non-barrier, non-delay compiled instruction. The figure
  reflects the selected mapping and possible use of routing ancillas; it does
  not measure peak simultaneous usage or hardware errors. **N/A for unrestricted.**
- **compiled_circuit_width_qubits**: Number of wires in the transpiled circuit;
  it can include idle wires and may equal the full coupling-map capacity.
- **cx_delta_vs_unrestricted**, **depth_delta_vs_unrestricted**, and
  **two_qubit_depth_delta_vs_unrestricted**: CONSTRAINED compiled metric minus
  the same architecture's UNRESTRICTED compiled metric at the same seed,
  optimization level, and gate basis. These are **net compilation differences**,
  combining effects of initial layout, routing, and optimization. They are
  **NOT SWAP counts** and cannot isolate the cost of routing alone.
- **cx_change_pct_vs_unrestricted**, **depth_change_pct_vs_unrestricted**:
  Corresponding net differences divided by the unrestricted metric, times 100.
- **connectivity_check**: `passed` means the compiled two-qubit operations
  were checked against directed coupling-map edges; `not_applicable` for the
  unrestricted reference, which has no coupling constraints.

The experiment compares connectivity-constrained circuit compilations. It is
not actual device execution, and it does not model gate duration, fidelity,
errors, calibration, or physical runtime. SWAP gates are normally translated
into the selected `u`/`cx` basis, and subsequent optimizations can change CX
counts; do not infer the number of inserted SWAP gates from CX deltas / 3.
