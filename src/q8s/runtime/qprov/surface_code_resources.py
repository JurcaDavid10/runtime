"""Conditional surface-code scenario model, not a hardware architecture estimator."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class SurfaceCodeAssumptions:
    physical_error_rate: float = 1e-3
    failure_budget: float = 0.01
    cycle_time_us: float = 1.0
    factories: int = 4
    factory_patches: int = 11
    factory_steps: int = 11
    residual_magic_error: float = 1e-7
    extra_data_patches_per_qubit: int = 1
    threshold: float = 0.01
    prefactor: float = 0.1
    max_distance: int = 99

    def __post_init__(self):
        for name in ('physical_error_rate', 'failure_budget', 'cycle_time_us', 'threshold', 'prefactor', 'residual_magic_error'):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v):
                raise ValueError(f'Invalid {name}')
        if not 0 < self.physical_error_rate < self.threshold < 1:
            raise ValueError('Physical error rate must be positive and below threshold')
        if not 0 < self.failure_budget < 1 or self.cycle_time_us <= 0 or not 0 < self.prefactor <= 1:
            raise ValueError('Invalid error budget, cycle time or prefactor')
        if not 0 <= self.residual_magic_error < 1:
            raise ValueError('Invalid residual magic-state error')
        for name in ('factories', 'factory_patches', 'factory_steps', 'extra_data_patches_per_qubit', 'max_distance'):
            v = getattr(self, name)
            if type(v) is not int or v < (0 if name in ('factories', 'extra_data_patches_per_qubit') else 1):
                raise ValueError(f'Invalid {name}')
        if self.max_distance < 3 or self.max_distance % 2 == 0:
            raise ValueError('max_distance must be odd and >= 3')


def estimate_surface_code(logical_qubits, t_count, t_depth, clifford_count, assumptions=SurfaceCodeAssumptions()):
    """Serial, non-overlapping gate/factory scenario; all reserved patches accrue error.

    Residual magic-state error is an external assumption, excluding modeled patch
    faults. No factory yield, injection circuitry or routed schedule is established.
    Raises ValueError if the scenario cannot meet the specified model budget.
    """
    for name, value in (('logical_qubits', logical_qubits), ('t_count', t_count), ('t_depth', t_depth), ('clifford_count', clifford_count)):
        if type(value) is not int or value < 0:
            raise ValueError(f'Invalid {name}')
    if logical_qubits == 0 or t_depth > t_count or (t_count > 0 and not t_depth):
        raise ValueError('Inconsistent logical resources')
    a = assumptions
    if t_count and not a.factories:
        raise ValueError('T gates require at least one factory in this model')
    f = min(a.factories, t_count)
    batches = (t_count + f - 1) // f if f else 0
    data_patches = logical_qubits * (1 + a.extra_data_patches_per_qubit)
    factory_patches = f * a.factory_patches
    buffer_patches = f
    patches = data_patches + factory_patches + buffer_patches
    residual = t_count * a.residual_magic_error
    if residual >= a.failure_budget:
        raise ValueError('Residual magic-state error exhausts model failure budget')
    for d in range(3, a.max_distance + 1, 2):
        gate_cycles = (t_count + clifford_count) * d
        factory_cycles = batches * a.factory_steps * d
        cycles = gate_cycles + factory_cycles
        p_logical = a.prefactor * (a.physical_error_rate / a.threshold) ** ((d + 1) / 2)
        patch_bound = patches * cycles * p_logical
        combined = patch_bound + residual
        if combined <= a.failure_budget:
            per_patch = 2 * d * d - 1
            return {**asdict(a), 'logical_qubits': logical_qubits, 't_count': t_count,
                    't_depth': t_depth, 'clifford_count': clifford_count,
                    'code_distance': d, 'active_factories': f, 'factory_batches': batches,
                    'data_patches': data_patches, 'factory_reserved_patches': factory_patches,
                    'buffer_patches': buffer_patches, 'total_patches': patches,
                    'physical_qubits_per_patch': per_patch,
                    'data_physical_qubits': data_patches * per_patch,
                    'factory_physical_qubits': factory_patches * per_patch,
                    'buffer_physical_qubits': buffer_patches * per_patch,
                    'total_physical_qubits': patches * per_patch,
                    'logical_error_per_patch_cycle': p_logical,
                    'patch_failure_union_term': patch_bound, 'magic_failure_union_term': residual,
                    'modeled_failure_union_bound': combined,
                    'gate_cycles': gate_cycles, 'factory_cycles': factory_cycles,
                    'total_cycles': cycles, 'serial_runtime_s': cycles * a.cycle_time_us * 1e-6,
                    'idealized_t_layer_time_s': t_depth * d * a.cycle_time_us * 1e-6}
    raise ValueError('No code distance satisfies the model budget up to max_distance')
