"""Descriptive supplements only: no new paths, fits, or selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_direction_mean_coherence import bootstrap, classes


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run=Path(args.run);out=Path(args.output)
    files=list((out/'cases').glob('*/*/*.npz'))
    assert len(files)==24
    inputs={str(p):sha(p) for p in files}
    inputs.update({str(p):sha(p) for p in (run/'anchor64_repair_v2').glob('*/[di]*_seed*/predictions.npz') if p.parent.parent.name in ('carpark','finland')})
    write(out/'supplement_scope.json',{'client_date':'2026-10-08',
        'scope':'Post-controls descriptive coverage of reliable whole-horizon necessary moment bounds and fixed-reference differences; no new transformations, fitting, or selection.',
        'necessary_conditions':['mu_h >= p_up_h','mu_(h-1) >= p_down_h; mu_0=true observed count'],
        'these_are_not_sufficient_conditions':True,'prior_controls_rule_sha256':sha(out/'controls_rule_lock.json'),
        'code_sha256':sha(Path(__file__)),'inputs_sha256':inputs})
    rows=[];paired={};case_ref_rows=[];group_rows=[]
    for file in sorted(files):
        domain=file.parent.parent.name;case=file.parent.name;bone,seed=case.rsplit('_seed',1);seed=int(seed);stage=file.stem
        meta=json.loads((run/'results_v2'/domain/case/'validation_lock.json').read_text())['metadata']
        mean,std=np.array(meta['mean']),np.array(meta['std'])
        with np.load(file) as a:
            x,y,mu,p,origins=[a[k] for k in ('x','y','mean_original','p_normalized','origins')]
            raw_mu=mu*std+mean;anchor=np.rint(x[:,-1]*std+mean)
            previous=np.concatenate((anchor[:,None,:],raw_mu[:,:-1]),1)
            up_gap=np.maximum(p[...,2]-raw_mu,0.)
            down_gap=np.maximum(p[...,0]-previous,0.)
            up_bad=up_gap>1e-9;down_bad=down_gap>1e-9;bad=up_bad|down_bad
            curves=bad.any(1)
            rows.append({'stage':stage,'domain':domain,'backbone':bone,'seed':seed,
                'forecast_cells':bad.size,'curves':curves.size,'up_violation_cells':int(up_bad.sum()),
                'down_violation_cells':int(down_bad.sum()),'necessary_bad_cells':int(bad.sum()),
                'necessary_bad_cell_fraction':float(bad.mean()),'necessary_bad_curves':int(curves.sum()),
                'necessary_bad_curve_fraction':float(curves.mean()),'max_up_gap_counts':float(up_gap.max()),
                'max_down_gap_counts':float(down_gap.max())})
            truth=classes(x,y);prediction=classes(x,mu)
            for label,mask in [('necessary_bad',bad),('no_detected_necessary_failure',~bad)]:
                n=int(mask.sum())
                group_rows.append({'stage':stage,'domain':domain,'backbone':bone,'seed':seed,'group':label,'cells':n,
                    'mu_mse_standardized':float(((y-mu)**2)[mask].mean()) if n else None,
                    'mu_raw_accuracy':float((truth==prediction)[mask].mean()) if n else None,
                    'head_accuracy':float((truth==p.argmax(-1))[mask].mean()) if n else None})
            paths={k[6:]:a[k] for k in a.files if k.startswith('path__')}
            with np.load(run/'anchor64_repair_v2'/domain/case/'predictions.npz') as r:
                paths['competitor_corrected']=r[('validation' if stage=='validation' else 'test')+'__competitor_corrected']
            metrics={m:np.stack([np.mean(truth==classes(x,z),(1,2)),np.mean((y-z)**2,(1,2))],-1) for m,z in paths.items()}
        key=stage+'/'+domain+'/'+bone
        times=pd.to_datetime(pd.read_csv(meta['csv'],usecols=['date'])['date'])
        days=times.iloc[origins].to_numpy().astype('datetime64[D]')
        paired.setdefault(key,{'days':days,'seeds':[]})
        assert np.array_equal(paired[key]['days'],days)
        paired[key]['seeds'].append(metrics)
        for method in ('boundary_only','mean_only_repair','joint_repair','mean_fixed_raw'):
            for ref in ('normalized_only','competitor_corrected','mean_raw'):
                delta=metrics[method]-metrics[ref]
                case_ref_rows.append({'stage':stage,'domain':domain,'backbone':bone,'seed':seed,'method':method,'reference':ref,
                    'accuracy_pp':float(100*delta[:,0].mean()),
                    'mse_percent':float(100*delta[:,1].mean()/metrics[ref][:,1].mean())})
    aggregate={}
    for key,entry in paired.items():
        avg={m:np.mean([s[m] for s in entry['seeds']],0) for m in entry['seeds'][0]}
        group={}
        for method in ('boundary_only','mean_only_repair','joint_repair','mean_fixed_raw'):
            obj={}
            for ref in ('normalized_only','competitor_corrected','mean_raw'):
                delta=avg[method]-avg[ref]
                obj['vs_'+ref]={'accuracy_pp':float(100*delta[:,0].mean()),
                    'mse_percent':float(100*delta[:,1].mean()/avg[ref][:,1].mean()),
                    'one_day':bootstrap(entry['days'],delta,avg[ref][:,1],1),
                    'three_day':bootstrap(entry['days'],delta,avg[ref][:,1],3)}
            group[method]=obj
        aggregate[key]=group
    pd.DataFrame(rows).to_csv(out/'whole_horizon_necessary_conditions.csv',index=False)
    pd.DataFrame(group_rows).to_csv(out/'whole_horizon_conditional_errors.csv',index=False)
    pd.DataFrame(case_ref_rows).to_csv(out/'supplement_case_comparisons.csv',index=False)
    write(out/'supplement_paired_comparisons.json',aggregate)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=json.loads((out/'paired_differences.json').read_text())
    names=['Carpark / DLinear','Carpark / iTransformer','Finland / DLinear','Finland / iTransformer']
    suffix=['carpark/dlinear','carpark/itransformer','finland/dlinear','finland/itransformer']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for stage,color,offset,label in [('validation','#1F6E8C',-.10,'Validation development'),('test_legacy','#D17A22',.10,'Legacy test (exploratory)')]:
        for i,key in enumerate(suffix):
            d=original[stage+'/'+key]['joint_repair']['vs_original_replay']
            for ax,metric,ci in [(axes[0],'accuracy_pp','raw_accuracy_pp_95ci'),(axes[1],'mse_percent','mse_percent_95ci')]:
                limits=d['three_day'][ci];value=d[metric]
                ax.errorbar(value,i+offset,xerr=np.array([[value-limits[0]],[limits[1]-value]]),
                            fmt='o',color=color,capsize=3,label=label if i==0 else None)
    for ax in axes:
        ax.set_yticks(range(4),names);ax.invert_yaxis();ax.axvline(0,color='#888888',linewidth=.7);ax.grid(axis='x',alpha=.2)
    axes[0].set_xlabel('Raw accuracy change vs original DP (pp)')
    axes[1].set_xlabel('MSE change vs original DP (%)')
    axes[0].set_title('First-step joint repair: direction')
    axes[1].set_title('First-step joint repair: point error')
    axes[0].legend(loc='lower left',fontsize=8)
    fig.suptitle('Fixed rules; three-day paired 95% intervals; no new model fits',fontsize=11)
    fig.tight_layout();fig.savefig(out/'coherence_fixed_repair_effects.png',dpi=180);plt.close(fig)
    for name,digest in inputs.items():assert sha(Path(name))==digest
    write(out/'supplement_completion.json',{'stage_cases':24,'new_fits':0,'new_paths':0,
        'whole_horizon_checks':'necessary only, not an incompatibility completeness test',
        'source_hash_checks':len(inputs),'source_failures':0})


if __name__=='__main__':main()
