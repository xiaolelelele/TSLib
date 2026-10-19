"""Separate CPU/GPU original-formula drift from true-anchor addition repair.

Uses the saved CPU branch traces, the original GPU raw outputs, and an
independent NumPy scorer. No neural training or additional model selection.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_anchor64_repair import score


def symbols(x,z):
    return np.sign(np.diff(np.concatenate((x[:,-1:],z),axis=1),axis=1)).astype(int)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--repair',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run);repair=Path(args.repair);rows=[]
    for shared in sorted(repair.glob('*/shared_seed*')):
        if not (shared/'result.json').exists():continue
        domain=shared.parent.name;seed=shared.name.split('seed')[1]
        cases=[p for p in sorted((run/domain).glob('*_seed'+seed))
               if not p.name.startswith('shared') and (p/'result.json').exists()]
        if not cases:continue
        with np.load(cases[0]/'predictions.npz') as a:common={k:a[k] for k in a.files}
        with np.load(run/domain/shared.name/'predictions.npz') as a:original={k:a[k] for k in a.files}
        with np.load(shared/'signed_trace.npz') as a:trace={k:a[k] for k in a.files}
        meta=json.loads((cases[0]/'validation_lock.json').read_text())['metadata']
        def bound(z):
            if meta['spec']['grid_mode']=='native':
                return np.maximum(z,-np.asarray(meta['mean'])/np.asarray(meta['std']))
            return z
        for split in ('validation','test'):
            x,y=common[split+'__x'],common[split+'__y']
            for variant in (3,4):
                method='carets%d_phase'%variant;key=split+'__'+method
                gpu=bound(original[key].astype(np.float64))
                cpu=bound(trace[key+'__raw32'].astype(np.float64))
                corrected=bound(trace[key+'__unbounded_anchor64'])
                gs,cs,rs=[symbols(x,z) for z in (gpu,cpu,corrected)]
                scores=[score(x,y,z) for z in (gpu,cpu,corrected)]
                rows.append({'domain':domain,'seed':int(seed),'split':split,'method':method,
                    'shape':list(gpu.shape),'max_abs_cpu_gpu_original_path_error':float(np.max(np.abs(cpu-gpu))),
                    'cpu_gpu_original_symbols_differ':int(np.count_nonzero(gs!=cs)),
                    'cpu_gpu_original_first_step_symbols_differ':int(np.count_nonzero(gs[:,0]!=cs[:,0])),
                    'cpu_anchor_repair_symbols_differ':int(np.count_nonzero(cs!=rs)),
                    'cpu_anchor_repair_first_step_symbols_differ':int(np.count_nonzero(cs[:,0]!=rs[:,0])),
                    'raw_gpu_original':scores[0]['raw_accuracy'],
                    'raw_cpu_original':scores[1]['raw_accuracy'],
                    'raw_cpu_anchor64':scores[2]['raw_accuracy'],
                    'cpu_gpu_raw_change_pp':100*(scores[1]['raw_accuracy']-scores[0]['raw_accuracy']),
                    'cpu_anchor_raw_change_pp':100*(scores[2]['raw_accuracy']-scores[1]['raw_accuracy'])})
    report={'rows':rows,'shared_seeds':len(rows)//4,
            'total_cpu_gpu_original_symbols_differ':sum(r['cpu_gpu_original_symbols_differ'] for r in rows),
            'total_cpu_anchor_repair_symbols_differ':sum(r['cpu_anchor_repair_symbols_differ'] for r in rows),
            'scope':'Diagnostic comparison without selection: original saved GPU formula versus independently reloaded CPU original formula, then identical CPU signed32 with true anchor64. Not bitwise GPU reproduction; no retraining.'}
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
