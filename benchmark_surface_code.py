"""Sweep conditional physical-resource scenarios using validated QARW logical data."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from dataclasses import asdict
from itertools import product
from report_fault_tolerant import load_inputs, KEYS, DEFAULT_INPUT, EXPERIMENT
from q8s.runtime.qprov.surface_code_resources import SurfaceCodeAssumptions, estimate_surface_code

MODEL = 'surface_code_serial_scenario_v1'


def sweep(resources, errors, factories, **kwargs):
    rows = []
    for p, f in product(errors, factories):
        a = SurfaceCodeAssumptions(physical_error_rate=p, factories=f, **kwargs)
        for policy, arch in KEYS:
            r = resources[policy, arch]
            row = dict(model=MODEL, architecture=arch, mcx_policy=policy)
            try:
                estimate = estimate_surface_code(*(r[k] for k in ('logical_qubits', 't_count', 't_depth', 'clifford_count')), a)
            except ValueError as exc:
                row.update(asdict(a), status='INFEASIBLE', reason=str(exc))
            else:
                row.update(estimate, status='OK', reason='')
            rows.append(row)
    return rows


def write_outputs(rows, out):
    out.mkdir(parents=True, exist_ok=True)
    (out/'physical_resources.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out/'physical_resources.csv').open('w', encoding='utf-8', newline='') as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
    lines = ['# Conditional surface-code scenarios', '', 'Not a hardware prediction. See docs/FAULT_TOLERANT_RESULTS.md for assumptions and limitations.', '', '| p | Factories | Architecture | Policy | Distance | Physical qubits | Serial time (s) | Status |', '|---:|---:|---|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['physical_error_rate']:g} | {r['factories']} | {r['architecture']} | {r['mcx_policy']} | {r.get('code_distance','-')} | {r.get('total_physical_qubits','-')} | {r['serial_runtime_s']:.6g} | OK |" if r['status']=='OK' else f"| {r['physical_error_rate']:g} | {r['factories']} | {r['architecture']} | {r['mcx_policy']} | - | - | - | {r['reason']} |")
    (out/'physical_resource_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = []
    # Plot each policy/architecture as its own series; overlapping series are intentional.
    for metric, label in [('total_physical_qubits','Modeled physical qubits'), ('serial_runtime_s','Serial scenario runtime (s)'), ('code_distance','Code distance')]:
        errors = sorted({r['physical_error_rate'] for r in rows})
        fig, axes = plt.subplots(1, len(errors), figsize=(max(5, 4*len(errors)), 4.5), squeeze=False)
        for ax, p in zip(axes[0], errors):
            for i, (policy, arch) in enumerate(KEYS):
                group = sorted((r for r in rows if r['physical_error_rate']==p and (r['mcx_policy'],r['architecture'])==(policy,arch)), key=lambda r:r['factories'])
                ax.plot([r['factories'] for r in group], [r.get(metric, float('nan')) for r in group], marker=['o','s','^','x'][i], linestyle=['-','--',':','-.'][i], label=f'{arch} / {policy}')
            ax.set_title(f'p = {p:g}'); ax.set_xlabel('Requested factories'); ax.set_ylabel(label); ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7); fig.suptitle(label+' | conditional model'); fig.tight_layout()
        path = out/(metric+'.png'); fig.savefig(path,dpi=150); plt.close(fig); plots.append(path)
    return plots


def log_mlflow(rows, input_dir, out, tracking_uri, experiment):
    import mlflow
    mlflow.set_tracking_uri(tracking_uri); mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name='mod17_surface_code_sensitivity') as parent:
        mlflow.log_params({'model':MODEL,'workload':'modular_add_17','runtime_model':'serial_no_overlap'})
        mlflow.log_metrics({'scenarios':len(rows),'feasible_scenarios':sum(r['status']=='OK' for r in rows)})
        for name in ('logical_resources.json','phase_validation.json'):
            mlflow.log_artifact(str(input_dir/name),artifact_path='surface_code/inputs')
        for name in ('physical_resources.json','physical_resources.csv','physical_resource_report.md','total_physical_qubits.png','serial_runtime_s.png','code_distance.png'):
            mlflow.log_artifact(str(out/name),artifact_path='surface_code/reports')
        for r in rows:
            if r['status'] != 'OK':
                continue
            with mlflow.start_run(run_name=f"{r['architecture']}__{r['mcx_policy']}__p{r['physical_error_rate']:g}__f{r['factories']}",nested=True):
                mlflow.log_params({**{k:r[k] for k in asdict(SurfaceCodeAssumptions())},'model':MODEL,'architecture':r['architecture'],'mcx_policy':r['mcx_policy']})
                mlflow.log_metrics({k:r[k] for k in ('code_distance','total_physical_qubits','serial_runtime_s','modeled_failure_union_bound','t_count','t_depth','logical_qubits','factory_cycles','gate_cycles')})
        return parent.info.run_id


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=DEFAULT_INPUT); p.add_argument('--output-dir',type=Path)
    p.add_argument('--physical-error-rates',type=float,nargs='+',default=[1e-4,1e-3,3e-3]); p.add_argument('--factories',type=int,nargs='+',default=[1,4,8])
    p.add_argument('--failure-budget',type=float,default=.01); p.add_argument('--cycle-time-us',type=float,default=1.)
    p.add_argument('--factory-patches',type=int,default=11); p.add_argument('--factory-steps',type=int,default=11)
    p.add_argument('--residual-magic-error',type=float,default=1e-7); p.add_argument('--extra-data-patches-per-qubit',type=int,default=1)
    p.add_argument('--log-mlflow',action='store_true'); p.add_argument('--tracking-uri',default='http://127.0.0.1:5000'); p.add_argument('--experiment',default=EXPERIMENT)
    a=p.parse_args(argv); out=a.output_dir or a.input_dir/'physical'
    try:
        resources, _, _=load_inputs(a.input_dir)
        rows=sweep(resources,sorted(set(a.physical_error_rates)),sorted(set(a.factories)),failure_budget=a.failure_budget,cycle_time_us=a.cycle_time_us,factory_patches=a.factory_patches,factory_steps=a.factory_steps,residual_magic_error=a.residual_magic_error,extra_data_patches_per_qubit=a.extra_data_patches_per_qubit)
        write_outputs(rows,out)
        print('QARW CONDITIONAL SURFACE-CODE SCENARIOS (not hardware predictions)')
        for r in rows:
            print(f"p={r['physical_error_rate']:g} f={r['factories']} {r['architecture']:15} {r['mcx_policy']:10}", f"d={r['code_distance']:2} qubits={r['total_physical_qubits']:7} serial={r['serial_runtime_s']:.6f}s" if r['status']=='OK' else r['reason'])
        print(f"{sum(r['status']=='OK' for r in rows)}/{len(rows)} scenarios meet modeled budget; saved {out.resolve()}")
        if a.log_mlflow:
            print('MLflow parent:',log_mlflow(rows,a.input_dir,out,a.tracking_uri,a.experiment))
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        p.exit(1,f'ERROR: {exc}\n')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
