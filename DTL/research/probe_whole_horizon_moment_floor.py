"""Post-diagnostic validation-only floor; no fitting or model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from validate_direction_mean_coherence import metrics, classes, bootstrap


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();run=Path(a.run);out=Path(a.output);dest=out/'moment_floor_validation'
    sys.path.insert(0,str(run/'source/v2'))
    from trend_lab.trajectory_decision import decode_trajectory,make_common_grids
    protocol=out/'moment_floor_validation_protocol.md';assert protocol.exists()
    files=sorted((out/'cases').glob('*/*/validation.npz'));assert len(files)==12
    source={str(f):sha(f) for f in files};source[str(protocol)]=sha(protocol)
    write(dest/'rule_lock.json',{'client_date':'2026-10-08','posthoc':True,'validation_only':True,
        'no_fitting':True,'no_selection':True,'original_grid_lambda_preserved':True,
        'code_sha256':sha(Path(__file__)),'source_sha256':source})
    records=[];pairs={};checks=[]
    for f in files:
        domain=f.parent.parent.name;case=f.parent.name;bone,seed=case.rsplit('_seed',1);seed=int(seed)
        lock=json.loads((run/'results_v2'/domain/case/'validation_lock.json').read_text());meta=lock['metadata']
        mean,std=np.array(meta['mean']),np.array(meta['std'])
        with np.load(f) as arr:
            x,y,mu,p,origins=[arr[k] for k in ('x','y','mean_original','p_masked','origins')]
            b=arr['path__cosdir'];lower=p[...,2].copy()
            lower[:,:-1]=np.maximum(lower[:,:-1],p[:,1:,:,0])
            lower[:,0]=np.maximum(lower[:,0],arr['mean_lower'])
            raw=mu*std+mean;changed=raw<lower-1e-9
            fixed=np.where(changed,(lower-mean)/std,mu)
            choice=lock['choices']['posterior'];strength=float(choice.rsplit('_',1)[1]) if choice else None
            grids,_=make_common_grids(x,b,mu,mode='native',step=.1,padding=.25,
                normalizer_mean=mean,normalizer_std=std,native_step=1.,native_origin=0.,
                lower=-mean/std,max_states=257,coarsen=True)
            z=decode_trajectory(x,b,fixed,p,strength,grids)[0] if strength is not None else b.copy()
            paths={'whole_floor_dp':z,'whole_floor_raw':fixed,'first_joint':arr['path__joint_repair'],
                'original_posterior':arr['path__original_replay'],'mean_raw':arr['path__mean_raw']}
            checks.append(np.array_equal(fixed[~changed],mu[~changed]))
            checks.append(bool(np.all(fixed*std+mean>=lower-1e-9)))
            (dest/domain).mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/domain/(case+'.npz'),x=x,y=y,origins=origins,mean_fixed=fixed,
                                lower=lower,changed=changed,**{'path__'+k:v for k,v in paths.items()})
            scores={m:metrics(x,y,path) for m,path in paths.items()}
            write(dest/domain/(case+'.json'),{'domain':domain,'backbone':bone,'seed':seed,
                'changed_mean_cells':int(changed.sum()),'total_cells':changed.size,
                'changed_curves':int(changed.any(1).sum()),'max_change_counts':float(np.max(fixed*std+mean-raw)),
                'old_choice':choice,'metrics':scores})
            for m,v in scores.items():records.append({'domain':domain,'backbone':bone,'seed':seed,'method':m,**{k:a for k,a in v.items() if k!='confusion'}})
            times=pd.to_datetime(pd.read_csv(meta['csv'],usecols=['date'])['date'])
            days=times.iloc[origins].to_numpy().astype('datetime64[D]')
            om={m:np.stack([np.mean(classes(x,y)==classes(x,path),(1,2)),np.mean((y-path)**2,(1,2))],-1) for m,path in paths.items()}
            pairs.setdefault(domain+'/'+bone,{'days':days,'seeds':[]})['seeds'].append(om)
    assert all(checks)
    aggregate={}
    for key,item in pairs.items():
        averaged={m:np.mean([s[m] for s in item['seeds']],0) for m in item['seeds'][0]}
        group={}
        for method in ('whole_floor_dp','whole_floor_raw'):
            row={}
            for ref in ('first_joint','original_posterior','mean_raw'):
                delta=averaged[method]-averaged[ref]
                row['vs_'+ref]={'accuracy_pp':float(100*delta[:,0].mean()),
                    'mse_percent':float(100*delta[:,1].mean()/averaged[ref][:,1].mean()),
                    'one_day':bootstrap(item['days'],delta,averaged[ref][:,1],1),
                    'three_day':bootstrap(item['days'],delta,averaged[ref][:,1],3)}
            group[method]=row
        aggregate[key]=group
    write(dest/'paired_differences.json',aggregate)
    pd.DataFrame(records).to_csv(dest/'per_case_metrics.csv',index=False)
    for f,digest in source.items():assert sha(Path(f))==digest
    write(dest/'completion.json',{'cases':12,'validation_only':True,'posthoc':True,'model_fits':0,'new_selections':0,'internal_checks':len(checks),'failures':checks.count(False)})
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
