"""Declared numerical diagnostic for CaReTS's observed-anchor semantics.

Does not modify trained checkpoints, frozen selections or evaluation labels.
Correction replaces the float32 history anchor by the original observed
float64 normalized anchor in the SAME learned relative-deviation formula.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_runner_outputs import classes,metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for shared in sorted(run.glob('*/shared_seed*')):
        if not (shared/'result.json').exists():continue
        seed=int(shared.name.split('seed')[1]);case=shared.parent/('dlinear_seed'+str(seed))
        if not (case/'result.json').exists():continue
        lock=json.loads((case/'validation_lock.json').read_text());meta=lock['metadata']
        with np.load(case/'predictions.npz') as a:case_arrays={k:a[k] for k in a.files}
        with np.load(shared/'predictions.npz') as a:shared_arrays={k:a[k] for k in a.files}
        saved={}
        for split in ('validation','test'):
            x,y=case_arrays[split+'__x'],case_arrays[split+'__y']
            anchor=x[:,-1:,:];cast=anchor.astype(np.float32).astype(np.float64)
            for method in ('carets3_phase','carets4_phase'):
                raw=shared_arrays[split+'__'+method].astype(np.float64)
                corrected=anchor+(raw-cast)
                original=raw
                if meta['spec']['grid_mode']=='native':
                    lower=-np.array(meta['mean'])/np.array(meta['std'])
                    original=np.maximum(original,lower);corrected=np.maximum(corrected,lower)
                before,after=metrics(x,y,original),metrics(x,y,corrected)
                row={'dataset':shared.parent.name,'seed':seed,'split':split,'method':method,
                     'float32_zero_deviation_first_steps':int(np.count_nonzero(raw[:,0,:]==cast[:,0,:])),
                     'first_step_symbols_changed':int(np.count_nonzero(classes(x,original)[:,0]!=classes(x,corrected)[:,0])),
                     'later_step_symbols_changed':int(np.count_nonzero(classes(x,original)[:,1:]!=classes(x,corrected)[:,1:])),
                     'max_abs_path_change':float(np.max(np.abs(original-corrected))),
                     'raw_change_pp':float(100*(after['raw_accuracy']-before['raw_accuracy'])),
                     'mse_change_pct':float(100*(after['mse']/before['mse']-1)),
                     'original':before,'observed_anchor_corrected':after}
                # Convert NumPy scalar/list values for reproducible JSON.
                row=json.loads(json.dumps(row,default=lambda v:v.item() if isinstance(v,np.generic) else v))
                rows.append(row);saved[split+'__'+method+'_observed_anchor']=corrected
        np.savez_compressed(output/(shared.parent.name+'_seed'+str(seed)+'.npz'),**saved)
    record={'client_date':'2026-09-30','run':str(run.resolve()),'status':'post hoc numerical baseline diagnostic, not retuning',
            'formula':'x_last_float64 + (CaReTS_path_float32 - x_last_float32), then identical predeclared physical bounds',
            'unchanged':'Original saved checkpoints, predictions, validation locks and true labels',
            'rows':rows}
    (output/'results.json').write_text(json.dumps(record,indent=2))
    print(json.dumps({'rows':len(rows),'changed_first_step_symbols':sum(r['first_step_symbols_changed'] for r in rows),
                      'changed_later_step_symbols':sum(r['later_step_symbols_changed'] for r in rows)}))


if __name__=='__main__':main()
