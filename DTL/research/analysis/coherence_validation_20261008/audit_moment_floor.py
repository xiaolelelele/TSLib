"""Independent full-H necessary-conditions and validation-only floor audit.
Imports only earlier independent audit helpers, never experimental functions.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_controls import score,label,grid_original,dense_dp,objective
ROOT=Path(__file__).parent;RUN=Path('/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ci_independent(days,delta,reference,block):
    dates=np.asarray(days,dtype='datetime64[D]');group=((dates-dates.min()).astype(int)//block);ids=np.unique(group)
    membership=(group[None,:]==ids[:,None]).astype(float)
    sizes=membership.sum(1);dif=membership@delta;base=membership@reference
    draws=np.random.RandomState(90817).randint(0,len(ids),size=(2000,len(ids)))
    weights=np.zeros((2000,len(ids)),int)
    for row in range(2000):weights[row]=np.bincount(draws[row],minlength=len(ids))
    denominators=weights@sizes;d=(weights@dif)/denominators[:,None];b=(weights@base)/denominators
    return {'blocks':len(ids),'days_per_block':block,'raw_accuracy_pp_95ci':np.percentile(d[:,0]*100,[2.5,97.5]),'mse_percent_95ci':np.percentile(d[:,1]/b*100,[2.5,97.5])}
def main():
    checks=[];rows=[];groups={};expectednecessary=[];expectedconditional=[]
    def ck(name,ok,detail=None):checks.append({'name':name,'pass':bool(ok),'detail':detail})
    def close(name,a,b):
        if a is None or b is None:ck(name,a is b);return
        aa,bb=np.asarray(a),np.asarray(b);ck(name,np.allclose(aa,bb,atol=2e-10,rtol=1e-9),float(np.max(np.abs(aa.astype(float)-bb.astype(float)))) if aa.size else 0.)
    scope=json.loads((ROOT/'supplement_scope.json').read_text());ck('supplement_code_hash',sha(Path('/home/lgc/TSLib/DTL/research/summarize_direction_mean_coherence.py'))==scope['code_sha256']);ck('necessary_only_scope',scope['these_are_not_sufficient_conditions'] is True)
    for p,digest in scope['inputs_sha256'].items():ck('supplement_source/'+p,sha(Path(p))==digest)
    table=pd.read_csv(ROOT/'whole_horizon_necessary_conditions.csv');conditional=pd.read_csv(ROOT/'whole_horizon_conditional_errors.csv')
    for file in sorted((ROOT/'cases').glob('*/*/*.npz')):
        domain=file.parent.parent.name;case=file.parent.name;stage=file.stem;bone,seed=case.rsplit('_seed',1);seed=int(seed)
        meta=json.loads((RUN/'results_v2'/domain/case/'validation_lock.json').read_text())['metadata'];m,s=np.array(meta['mean']),np.array(meta['std'])
        with np.load(file) as a:x,y,mu,p=[a[k] for k in ('x','y','mean_original','p_normalized')]
        raw=mu*s+m;anchor=np.rint(x[:,-1,:]*s+m);prev=np.empty_like(raw);prev[:,0]=anchor;prev[:,1:]=raw[:,:-1]
        upgap=np.maximum(p[...,2]-raw,0);downgap=np.maximum(p[...,0]-prev,0);upbad=upgap>1e-9;downbad=downgap>1e-9;bad=upbad|downbad;curve=bad.any(1)
        expected={'forecast_cells':bad.size,'curves':curve.size,'up_violation_cells':int(upbad.sum()),'down_violation_cells':int(downbad.sum()),'necessary_bad_cells':int(bad.sum()),'necessary_bad_cell_fraction':float(bad.mean()),'necessary_bad_curves':int(curve.sum()),'necessary_bad_curve_fraction':float(curve.mean()),'max_up_gap_counts':float(upgap.max()),'max_down_gap_counts':float(downgap.max())}
        selected=table[(table.stage==stage)&(table.domain==domain)&(table.backbone==bone)&(table.seed==seed)];name=f'necessary/{stage}/{domain}/{case}';ck(name+'/one_row',len(selected)==1)
        for k,z in expected.items():close(name+'/'+k,selected.iloc[0][k],z)
        correct=label(x,y)==label(x,mu);hcorrect=label(x,y)==p.argmax(-1)
        for g,mask in [('necessary_bad',bad),('no_detected_necessary_failure',~bad)]:
            rr=conditional[(conditional.stage==stage)&(conditional.domain==domain)&(conditional.backbone==bone)&(conditional.seed==seed)&(conditional.group==g)].iloc[0]
            vals={'cells':int(mask.sum()),'mu_mse_standardized':float(((y-mu)**2)[mask].mean()) if mask.any() else None,'mu_raw_accuracy':float(correct[mask].mean()) if mask.any() else None,'head_accuracy':float(hcorrect[mask].mean()) if mask.any() else None}
            for k,z in vals.items():close(name+'/'+g+'/'+k,None if pd.isna(rr[k]) else rr[k],z)
    ck('expected_24_necessary_stage_cases',len(table)==24);ck('expected_48_conditional_group_rows',len(conditional)==48)
    dest=ROOT/'moment_floor_validation';rule=json.loads((dest/'rule_lock.json').read_text());ck('floor_code_hash',sha(Path('/home/lgc/TSLib/DTL/research/probe_whole_horizon_moment_floor.py'))==rule['code_sha256']);ck('floor_validation_only',rule['validation_only'] and rule['no_fitting'] and rule['no_selection'] and rule['posthoc']);ck('floor_frozen_grid_lambda_flag',rule['original_grid_lambda_preserved'])
    for p,digest in rule['source_sha256'].items():ck('floor_source/'+p,sha(Path(p))==digest)
    scientific_code=Path('/home/lgc/TSLib/DTL/research/probe_whole_horizon_moment_floor.py').read_text()
    ck('floor_no_test_arrays_read',"glob('*/*/validation.npz')" in scientific_code and 'test__' not in scientific_code and 'test_legacy' not in scientific_code)
    for file in sorted((dest).glob('*/*.npz')):
        domain=file.parent.name;case=file.stem;bone,seed=case.rsplit('_seed',1);seed=int(seed);name=f'floor/{domain}/{case}'
        lock=json.loads((RUN/'results_v2'/domain/case/'validation_lock.json').read_text());meta=lock['metadata'];m,s=np.array(meta['mean']),np.array(meta['std']);choice=lock['choices']['posterior'];lam=float(choice.rsplit('_',1)[1]) if choice else None
        sourcefile=ROOT/'cases'/domain/case/'validation.npz'
        with np.load(sourcefile) as a:source={k:a[k] for k in a.files}
        with np.load(file) as a:out={k:a[k] for k in a.files}
        x,y,mu,p,b=[source[k] for k in ('x','y','mean_original','p_masked','path__cosdir')]
        raw=mu*s+m;lo=np.empty_like(raw)
        for h in range(raw.shape[1]):lo[:,h]=np.maximum(p[:,h,:,2],p[:,h+1,:,0]) if h+1<raw.shape[1] else p[:,h,:,2]
        anchor=np.rint(x[:,-1]*s+m);firstlo=p[:,0,:,1]*anchor+p[:,0,:,2]*(anchor+1);lo[:,0]=np.maximum(lo[:,0],firstlo)
        changed=raw<lo-1e-9;fixed=mu.copy();fixed[changed]=np.broadcast_to((lo-m)/s,mu.shape)[changed]
        for k,z in [('x',x),('y',y),('origins',source['origins']),('lower',lo),('changed',changed),('mean_fixed',fixed)]:ck(name+'/'+k,np.array_equal(out[k],z))
        ck(name+'/unchanged_elements_exact',np.array_equal(out['mean_fixed'][~changed],mu[~changed]));ck(name+'/necessary_floor_satisfied',np.all(out['mean_fixed']*s+m>=lo-1e-9))
        info=json.loads(file.with_suffix('.json').read_text());ck(name+'/old_choice',info['old_choice']==choice)
        for k,z in [('changed_mean_cells',int(changed.sum())),('total_cells',changed.size),('changed_curves',int(changed.any(1).sum())),('max_change_counts',float(np.max(fixed*s+m-raw)))]:close(name+'/'+k,info[k],z)
        for k,z in [(kk[6:],zz) for kk,zz in out.items() if kk.startswith('path__')]:
            for field,v in score(x,y,z).items():close(name+'/score/'+k+'/'+field,info['metrics'][k][field],v)
        ck(name+'/raw_fixed_path',np.array_equal(out['path__whole_floor_raw'],fixed));ck(name+'/old_paths_exact',np.array_equal(out['path__first_joint'],source['path__joint_repair']) and np.array_equal(out['path__original_posterior'],source['path__original_replay']) and np.array_equal(out['path__mean_raw'],source['path__mean_raw']))
        for n in sorted(set([0,len(x)//2,len(x)-1])):
            g=[grid_original(x[n,-1,c],b[n,:,c],mu[n,:,c],m[c],s[c]) for c in range(x.shape[2])]
            if lam is None:z=b[n]
            else:
                z=np.stack([dense_dp(x[n,-1,c],fixed[n,:,c],p[n,:,c],g[c],lam) for c in range(x.shape[2])],-1)
                ob=np.mean([objective(x[n,-1,c],b[n,:,c],fixed[n,:,c],p[n,:,c],lam) for c in range(x.shape[2])]);oz=np.mean([objective(x[n,-1,c],z[:,c],fixed[n,:,c],p[n,:,c],lam) for c in range(x.shape[2])]);z=b[n] if ob<=oz else z
            ck(name+f'/dense_dp_origin{n}',np.array_equal(out['path__whole_floor_dp'][n],z))
        times=pd.to_datetime(pd.read_csv(meta['csv'],usecols=['date'])['date']);days=times.iloc[out['origins']].to_numpy().astype('datetime64[D]')
        perorigin={k[6:]:np.stack([np.mean(label(x,y)==label(x,z),(1,2)),np.mean((y-z)**2,(1,2))],-1) for k,z in out.items() if k.startswith('path__')}
        key=domain+'/'+bone;groups.setdefault(key,{'days':days,'seeds':[]});ck(name+'/same_days_across_seeds',np.array_equal(groups[key]['days'],days));groups[key]['seeds'].append(perorigin)
        rows.append({'case':name,'changed_cells':int(changed.sum()),'total_cells':int(changed.size),'changed_channel_curves':int(changed.any(1).sum()),'lambda':lam,'sha_source':sha(sourcefile),'sha_output':sha(file)})
    expected=json.loads((dest/'paired_differences.json').read_text());comparisons=[]
    for key,g in groups.items():
        ck(key+'/three_seeds',len(g['seeds'])==3);avg={m:np.mean([r[m] for r in g['seeds']],0) for m in g['seeds'][0]}
        for method in ('whole_floor_dp','whole_floor_raw'):
            for ref in ('first_joint','original_posterior','mean_raw'):
                d=avg[method]-avg[ref];q=expected[key][method]['vs_'+ref]
                close(key+'/'+method+'/'+ref+'/accuracy_pp',q['accuracy_pp'],100*d[:,0].mean());close(key+'/'+method+'/'+ref+'/mse_percent',q['mse_percent'],100*d[:,1].mean()/avg[ref][:,1].mean())
                for block,labelkey in [(1,'one_day'),(3,'three_day')]:
                    rebuilt=ci_independent(g['days'],d,avg[ref][:,1],block)
                    for f,v in rebuilt.items():close(key+'/'+method+'/'+ref+'/'+labelkey+'/'+f,q[labelkey][f],v)
                comparisons.append({'setting':key,'method':method,'reference':ref,'accuracy_pp':q['accuracy_pp'],'mse_percent':q['mse_percent']})
    ck('expected_12_floor_cases',len(rows)==12)
    res={'scope':'Independent all-H necessary moment diagnostics across 24 stage-cases, plus posthoc validation-only floor across 12 cases. No experimental scoring/projection/bootstrap/DP imported. Dense DP verifies 3 fixed origins per case with all channels and whole-base comparison. Necessity is not joint-distribution sufficiency; this rule is not pre-registered or tested on new test data.','check_count':len(checks),'failure_count':sum(not z['pass'] for z in checks),'checks':checks,'floor_cases':rows,'comparisons':comparisons,'independent_dense_dp_origin_checks':36,'bootstrap_review':'Independently recreated 2000 block draws and multiplicity-weighted block sums for 1/3 day intervals, averaging the same three fitted seeds before resampling. Conditional on fitted models, few calendar blocks, not seed population variability or safety guarantee.'}
    (ROOT/'audit_moment_floor.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n');print(json.dumps({'check_count':len(checks),'failure_count':res['failure_count'],'failed':[z for z in checks if not z['pass']]},indent=2))
if __name__=='__main__':main()
