import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
import sys
from q8s.runtime.qprov.surface_code_resources import SurfaceCodeAssumptions, estimate_surface_code
import benchmark_surface_code as bench


class TestSurfaceCodeResources(unittest.TestCase):
    def setUp(self):
        self.a = SurfaceCodeAssumptions()

    def estimate(self, a=None, optimized=False):
        return estimate_surface_code(13, *( (1712,1057,2348) if optimized else (2576,1328,3212)), a or self.a)

    def test_patch_and_cycle_accounting(self):
        r=self.estimate(); d=r['code_distance']
        self.assertEqual(r['physical_qubits_per_patch'],2*d*d-1)
        self.assertEqual(r['total_patches'],26+44+4)
        self.assertEqual(r['total_physical_qubits'],74*(2*d*d-1))
        self.assertEqual(r['total_cycles'],(2576+3212+644*11)*d)
        self.assertAlmostEqual(r['serial_runtime_s'],r['total_cycles']*1e-6)

    def test_minimum_odd_distance(self):
        r=self.estimate(); d=r['code_distance']; self.assertEqual(d%2,1)
        self.assertLessEqual(r['modeled_failure_union_bound'],self.a.failure_budget)
        if d>3:
            with self.assertRaisesRegex(ValueError,'No code distance'):
                self.estimate(replace(self.a,max_distance=d-2))

    def test_lower_noise_and_optimization(self):
        r=self.estimate(); quiet=self.estimate(replace(self.a,physical_error_rate=1e-4)); opt=self.estimate(optimized=True)
        self.assertLessEqual(quiet['code_distance'],r['code_distance'])
        self.assertLess(opt['serial_runtime_s'],r['serial_runtime_s'])
        self.assertLessEqual(opt['total_physical_qubits'],r['total_physical_qubits'])

    def test_factory_tradeoff(self):
        one=self.estimate(replace(self.a,factories=1)); many=self.estimate(replace(self.a,factories=8))
        self.assertLess(many['factory_batches'],one['factory_batches'])
        self.assertGreater(many['factory_reserved_patches'],one['factory_reserved_patches'])

    def test_no_t_gates(self):
        r=estimate_surface_code(2,0,0,8,replace(self.a,factories=0))
        self.assertEqual(r['factory_cycles'],0); self.assertEqual(r['buffer_patches'],0)
        self.assertEqual(r['magic_failure_union_term'],0)

    def test_invalid_assumptions(self):
        for kwargs in ({'physical_error_rate':.01},{'cycle_time_us':0},{'max_distance':4},{'residual_magic_error':float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.a,**kwargs)
        with self.assertRaisesRegex(ValueError,'exhausts'):
            self.estimate(replace(self.a,residual_magic_error=.001))
        with self.assertRaisesRegex(ValueError,'at least one factory'):
            self.estimate(replace(self.a,factories=0))

    def test_sweep_reports_and_mocked_mlflow(self):
        resources={k:dict(logical_qubits=13,t_count=2576 if k==bench.KEYS[0] else 1712,t_depth=1328 if k==bench.KEYS[0] else 1057,clifford_count=3212 if k==bench.KEYS[0] else 2348) for k in bench.KEYS}
        rows=bench.sweep(resources,[1e-4,1e-3,3e-3],[1,4,8]); self.assertEqual(len(rows),36)
        self.assertTrue(all(r['status']=='OK' for r in rows))
        self.assertTrue(all(r['status']=='INFEASIBLE' for r in bench.sweep(resources,[1e-3],[0])))
        with TemporaryDirectory() as tmp:
            out=Path(tmp); plots=bench.write_outputs(rows,out)
            self.assertEqual(len(plots),3)
            self.assertTrue(all(p.read_bytes().startswith(b'\x89PNG') for p in plots))
            self.assertTrue((out/'physical_resources.csv').is_file())
            mlflow=MagicMock(); mlflow.start_run.return_value.__enter__.return_value.info.run_id='test'
            with patch.dict(sys.modules,{'mlflow':mlflow}):
                self.assertEqual(bench.log_mlflow(rows,out,out,'http://127.0.0.1:5000',bench.EXPERIMENT),'test')
            self.assertEqual(mlflow.start_run.call_count,37)

if __name__=='__main__':
    unittest.main()
