"""Reporting tests using synthetic records matching the benchmark JSON schema."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import sys

import report_fault_tolerant as reporting


class TestFaultTolerantReporting(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for p, a in reporting.KEYS:
            source = 10 if a == 'baseline' else 13
            t, d, c = (2576, 1328, 3212) if (p, a) == reporting.KEYS[0] else (1712, 1057, 2348)
            self.rows.append(dict(mcx_policy=p, architecture=a, qiskit_version='test', source_qubits=source,
                added_clean_ancillas=13-source, logical_qubits=13, t_count=t, t_depth=d, clifford_count=c,
                magic_states=t, gate_counts={'t':t//2, 'tdg':t-t//2, 'cx':c}, decomposition_method='exact_recursive_gate_definitions'))
        self.report = dict(qiskit_version='test', method='exhaustive_monomial_gate_action',
            data_layout={'a':list(range(5)), 'b':list(range(5,10))}, absolute_tolerance=1e-9, configurations=[])
        for p,a in reporting.VALIDATION_KEYS:
            self.report['configurations'].append(dict(mcx_policy=p, architecture=a, logical_qubits=10 if (p,a)==('source','baseline') else 13,
                status='PASS', basis_inputs_checked=1024, valid_arithmetic_inputs_checked=289, out_of_range_b_inputs_checked=480,
                clean_ancillas=True, reversible_mapping=True, phase_preserved_up_to_global=True,
                matches_baseline_extension=None if (p,a)==('source','baseline') else True,
                max_relative_phase_error=3.09e-14 if (p,a)==('source','baseline') else 0.0))
        self.save_inputs()

    def save_inputs(self):
        (self.root/'logical_resources.json').write_text(json.dumps(self.rows), encoding='utf-8')
        (self.root/'phase_validation.json').write_text(json.dumps(self.report), encoding='utf-8')

    def test_summary_and_matched_comparisons(self):
        res, val, rep = reporting.load_inputs(self.root)
        summary = reporting.build_summary(res,val,rep)
        self.assertEqual(summary['validation_basis_inputs_total'],6144)
        self.assertEqual(summary['valid_arithmetic_inputs_total'],1734)
        reductions = summary['comparisons_percent_reduction']
        self.assertAlmostEqual(reductions['synthesis_on_baseline']['t_count'], 100*864/2576)
        self.assertEqual(reductions['architecture_at_rccx']['t_count'],0)
        self.assertIsNone(reporting.reduction(0,0))

    def test_rejects_failed_validation(self):
        self.report['configurations'][0]['status']='FAIL'
        self.save_inputs()
        with self.assertRaisesRegex(ValueError,'did not pass'):
            reporting.load_inputs(self.root)

    def test_rejects_inconsistent_resource_counts(self):
        self.rows[0]['magic_states']+=1
        self.save_inputs()
        with self.assertRaisesRegex(ValueError,'T-resource'):
            reporting.load_inputs(self.root)

    def test_rejects_duplicate_configuration(self):
        self.rows.append(self.rows[0])
        self.save_inputs()
        with self.assertRaisesRegex(ValueError,'duplicate'):
            reporting.load_inputs(self.root)

    def test_reports_and_mocked_mlflow_logging(self):
        res,val,rep = reporting.load_inputs(self.root)
        out = self.root/'reports'
        summary, plots = reporting.write_report(res,val,rep,out)
        self.assertEqual(len(plots),6)
        for path in plots:
            self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
        for name in ('summary.json','reductions.csv','logical_resource_report.md'):
            self.assertTrue((out/name).is_file())
        mlflow = MagicMock()
        mlflow.start_run.return_value.__enter__.return_value.info.run_id = 'parent-id'
        with patch.dict(sys.modules, {'mlflow':mlflow}):
            run_id=reporting.log_mlflow(res,val,summary,self.root,out,'http://127.0.0.1:5000',reporting.EXPERIMENT)
        self.assertEqual(run_id,'parent-id')
        self.assertEqual(mlflow.start_run.call_count,5)
        self.assertEqual(mlflow.log_artifact.call_count,11)
        self.assertEqual(mlflow.log_metrics.call_count,6)


if __name__ == '__main__':
    unittest.main()
