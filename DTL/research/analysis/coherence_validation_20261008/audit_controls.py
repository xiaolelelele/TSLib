"""Independent check of fixed compatibility controls; imports no experiment code.
Rebuilds projections/metrics and a dense O(H M^2) reference DP for fixed origins.
This consumes already-inspected legacy test solely to audit fixed artifacts.
"""
import hashlib,json
from pathlib import Path
import numpy as np
RUN=Path('/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930')
OUT=Path(__file__).parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def label(x,z):
    prev=np.empty_like(z);prev[:,0,:]=x[:,-1,:];prev[:,1:,:]=z[:,:-1,:]
    return (np.sign(z-prev)+1).astype(np.int64)
def score(x,y,z):
    t=label(x,y);pred=label(x,z);conf=np.zeros((3,3),np.int64)
    for i in range(3):
        for j in range(3):conf[i,j]=int(np.count_nonzero((t==i)&(pred==j)))
    diag=np.diag(conf);den=conf.sum(0)+conf.sum(1);f1=np.divide(2*diag,den,out=np.zeros(3),where=den>0)
    ok=t==pred;zero=t==1;err=(z-y)**2
    return {'mse':float(err.mean()),'mae':float(np.abs(z-y).mean()),'raw_accuracy':float(ok.mean()),'macro_f1':float(f1.mean()),'first_mse':float(err[:,0].mean()),'first_raw_accuracy':float(ok[:,0].mean()),'rest_mse':float(err[:,1:].mean()),'rest_raw_accuracy':float(ok[:,1:].mean()),'zero_recall':float(ok[zero].mean()) if zero.any() else None,'nonzero_accuracy':float(ok[~zero].mean()) if (~zero).any() else None,'confusion':conf.tolist()}
def head(t,p):
    nll=-np.log(np.maximum(np.take_along_axis(p,t[...,None],-1)[...,0],1e-300)).mean()
    return {'accuracy':float((p.argmax(-1)==t).mean()),'nll':float(nll),'brier':float(np.mean(np.sum((p-np.eye(3)[t])**2,-1)))}
def grid_original(anchor,b,mu,m,s):
    # Fixed v2 native definition, inferred only from old observable inputs.
    lower=-m/s;origin=lower;delta=1/s
    lo=max(float(np.min(np.r_[anchor,b,mu]))-.25,lower);hi=float(np.max(np.r_[anchor,b,mu]))+.25
    first=int(np.floor((lo-origin)/delta));last=int(np.ceil((hi-origin)/delta));n=last-first+1
    ai=int(np.rint((anchor-origin)/delta));keep=first<=ai<=last and (ai-m)/s==anchor
    if n>257:
        slots=257-int(keep);indices=np.array([first+i*(last-first)//(slots-1) for i in range(slots)],float)
        if keep:indices=np.unique(np.r_[indices,ai])
    else:indices=np.arange(first,last+1,dtype=float)
    g=(indices-m)/s;return g[g>=lower]
def dense_dp(anchor,mu,p,g,strength):
    h=len(mu);c=(np.sign(g-anchor)+1).astype(int)
    costs=(g-mu[0])**2-strength*p[0,c];back=[]
    trans=(np.sign(g[None,:]-g[:,None])+1).astype(int)
    for t in range(1,h):
        allcost=costs[:,None]-strength*p[t,trans]
        prev=np.argmin(allcost,axis=0);costs=allcost[prev,np.arange(len(g))]+(g-mu[t])**2;back.append(prev)
    state=int(costs.argmin());path=np.empty(h);path[-1]=g[state]
    for t in range(h-2,-1,-1):state=int(back[t][state]);path[t]=g[state]
    return path
def objective(anchor,z,mu,p,strength):
    cls=(np.sign(z-np.r_[anchor,z[:-1]])+1).astype(int)
    return float(np.mean((z-mu)**2)-strength*np.mean(p[np.arange(len(z)),cls]))
def main():
    checks=[];rows=[];supplement=[];source_repair={};dp_checks=0
    def ck(name,ok,detail=None):checks.append({'name':name,'pass':bool(ok),'detail':detail})
    def close(name,a,b):
        if a is None or b is None:ck(name,a is b);return
        aa,bb=np.asarray(a),np.asarray(b);ck(name,np.allclose(aa,bb,atol=2e-12,rtol=1e-10),float(np.max(np.abs(aa.astype(float)-bb.astype(float)))) if aa.size else 0)
    rule=json.loads((OUT/'controls_rule_lock.json').read_text())
    ck('protocol_hash',sha(Path(rule['protocol']))==rule['protocol_sha256']);ck('script_hash',sha(Path('/home/lgc/TSLib/DTL/research/validate_direction_mean_coherence.py'))==rule['script_sha256'])
    ck('no_new_fitting_or_selection',not rule['new_fitting'] and not rule['new_selection']);ck('fixed_lambda_and_grid',rule['old_lambda_inherited'] and rule['old_common_grid_fixed'])
    for f,digest in rule['source_sha256'].items():ck('source/'+f,sha(RUN/f)==digest)
    for domain in ('carpark','finland'):
        for bone in ('dlinear','itransformer'):
            for seed in (2024,2025,2026):
                olddir=RUN/'results_v2'/domain/f'{bone}_seed{seed}';folder=OUT/'cases'/domain/olddir.name
                lock=json.loads((olddir/'validation_lock.json').read_text());meta=lock['metadata'];m,s=np.array(meta['mean']),np.array(meta['std']);choice=lock['choices']['posterior'];lam=float(choice.rsplit('_',1)[1]) if choice else None
                repairfile=RUN/'anchor64_repair_v2'/domain/olddir.name/'predictions.npz';source_repair[str(repairfile)]=sha(repairfile)
                for stage,oldstage in [('validation','validation'),('test_legacy','test')]:
                    name=f'{stage}/{domain}/{olddir.name}'
                    with np.load(olddir/'predictions.npz') as old, np.load(folder/(stage+'.npz')) as new:
                        a={k:new[k] for k in new.files};x,y,mu,p,b=[old[oldstage+'__'+k] for k in ('x','y','mean','probabilities','cosdir')]
                        orig=old[oldstage+'__'+(choice if oldstage=='validation' and choice else 'posterior_selected' if oldstage=='test' else 'cosdir')]
                        for nk,ok in [('x','x'),('y','y'),('origins','origins'),('p_original','probabilities'),('mean_original','mean')]:ck(name+'/source_array_'+nk,np.array_equal(a[nk],old[oldstage+'__'+ok]))
                    result=json.loads((folder/(stage+'.json')).read_text());ck(name+'/old_lambda',result['strength']==lam and result['old_choice']==choice)
                    ck(name+'/old_source_sha',result['original_prediction_sha256']==sha(olddir/'predictions.npz') and result['original_lock_sha256']==sha(olddir/'validation_lock.json'))
                    xr=np.rint(x[:,-1,:]*s+m);raw_mu=mu[:,0,:]*s+m
                    pn=p.astype(float);pn/=pn.sum(-1,keepdims=True);pm=pn.copy();boundary=xr==0
                    pm[:,0,:,0]=np.where(boundary,0,pm[:,0,:,0]);den=pm[:,0].sum(-1,keepdims=True);pm[:,0]=np.where(boundary[...,None],pm[:,0]/den,pm[:,0])
                    lo=pm[:,0,:,1]*xr+pm[:,0,:,2]*(xr+1);up=np.where(pm[:,0,:,2]>0,np.inf,pm[:,0,:,0]*np.maximum(xr-1,0)+pm[:,0,:,1]*xr)
                    bad=(raw_mu<lo-1e-9)|(raw_mu>up+1e-9);mf=mu.copy();mf[:,0]=np.where(bad,(np.clip(raw_mu,lo,up)-m)/s,mu[:,0]);gap=np.abs(np.clip(raw_mu,lo,up)-raw_mu)
                    for k,z in [('anchor',xr),('boundary',boundary),('mean_violation',bad),('mean_lower',lo),('mean_upper',up),('mean_raw_first',raw_mu),('mean_gap',gap),('p_normalized',pn),('p_masked',pm),('mean_fixed',mf)]:
                        if np.isinf(z).any():ck(name+'/'+k,np.array_equal(a[k],z))
                        else:close(name+'/'+k,a[k],z)
                    ck(name+'/future_mean_exact',np.array_equal(a['mean_fixed'][:,1:],mu[:,1:]));ck(name+'/unchanged_mean_exact',np.array_equal(a['mean_fixed'][:,0][~bad],mu[:,0][~bad]));ck(name+'/future_p_exact',np.array_equal(a['p_masked'][:,1:],pn[:,1:]));ck(name+'/original_replay_exact',np.array_equal(a['path__original_replay'],orig))
                    t=label(x,y);d=result['diagnostic']
                    for k,z in [('first_cells',boundary.size),('boundary_cells',int(boundary.sum())),('boundary_fraction',float(boundary.mean())),('impossible_probability_mass_mean',float(np.mean(pn[:,0,:,0]*boundary))),('impossible_probability_mass_on_boundary',float(pn[:,0,:,0][boundary].mean()) if boundary.any() else None),('mean_violation_cells',int(bad.sum())),('mean_violation_fraction',float(bad.mean())),('mean_gap_original_units_mean',float(gap.mean())),('mean_gap_standardized_mean',float((gap/s).mean())),('mean_gap_original_units_conditional',float(gap[bad].mean()) if bad.any() else None),('mean_gap_original_units_max',float(gap.max())),('upper_bound_finite_cells',int(np.isfinite(up).sum())),('probability_normalization_max_change',float(np.max(np.abs(p.astype(float)-pn)))),('true_down_at_known_zero',int(np.count_nonzero(boundary&(t[:,0]==0))))]:close(name+'/diagnostic/'+k,d[k],z)
                    for k,tt,pp in [('head_original_all',t,pn),('head_masked_all',t,pm),('head_original_first',t[:,0],pn[:,0]),('head_masked_first',t[:,0],pm[:,0])]:
                        for field,v in head(tt,pp).items():close(name+'/diagnostic/'+k+'/'+field,d[k][field],v)
                    for k,z in [(kk[len('path__'):],zz) for kk,zz in a.items() if kk.startswith('path__')]:
                        expected=score(x,y,z)
                        for field,v in expected.items():close(name+'/metrics/'+k+'/'+field,result['metrics'][k][field],v)
                    with np.load(repairfile) as r:comp=r[oldstage+'__competitor_corrected']
                    if stage=='test_legacy':ck(name+'/corrected_competitor_array',np.array_equal(a['path__competitor_corrected'],comp))
                    # Independently supply missing normalized-only and validation
                    # competitor contrasts, without modifying scientific controls.
                    for target in ['boundary_only','mean_only_repair','joint_repair','mean_fixed_raw']:
                        for ref,zref in [('normalized_only',a['path__normalized_only']),('competitor_corrected',comp)]:
                            q=score(x,y,a['path__'+target]);rr=score(x,y,zref)
                            supplement.append({'stage':stage,'domain':domain,'backbone':bone,'seed':seed,'method':target,'reference':ref,'accuracy_pp':100*(q['raw_accuracy']-rr['raw_accuracy']),'mse_percent':100*(q['mse']/rr['mse']-1)})
                    # At three fixed origin positions, reconstruct all component
                    # channels and dense DP before selecting the same whole base.
                    for n in sorted(set([0,len(x)//2,len(x)-1])):
                        grids=[grid_original(x[n,-1,c],b[n,:,c],mu[n,:,c],m[c],s[c]) for c in range(x.shape[2])]
                        for method,est,prob in [('original_replay',mu,p),('normalized_only',mu,pn),('boundary_only',mu,pm),('mean_only_repair',mf,pn),('joint_repair',mf,pm)]:
                            if lam is None:expected=b[n].copy()
                            else:
                                z=np.stack([dense_dp(x[n,-1,c],est[n,:,c],prob[n,:,c],grids[c],lam) for c in range(x.shape[2])],-1)
                                ob=np.mean([objective(x[n,-1,c],b[n,:,c],est[n,:,c],prob[n,:,c],lam) for c in range(x.shape[2])]);oz=np.mean([objective(x[n,-1,c],z[:,c],est[n,:,c],prob[n,:,c],lam) for c in range(x.shape[2])]);expected=b[n] if ob<=oz else z
                            ck(name+f'/dense_dp_{method}_origin{n}',np.array_equal(a['path__'+method][n],expected),float(np.max(np.abs(a['path__'+method][n]-expected))));dp_checks+=1
                    rows.append({'case':name,'cells':int(boundary.size),'boundary_cells':int(boundary.sum()),'mean_conflict_cells':int(bad.sum()),'metrics_checked':len(result['metrics']),'normalization_path_identical':bool(np.array_equal(a['path__normalized_only'],a['path__original_replay'])),'sha_output_npz':sha(folder/(stage+'.npz'))})
    ck('expected_24_stage_cases',len(rows)==24,len(rows))
    valflag=OUT/'validation_diagnostics_complete.json';ck('validation_before_legacy_test',all((OUT/'cases'/d/f'{b}_seed{s}'/'test_legacy.npz').stat().st_mtime>=valflag.stat().st_mtime for d in ('carpark','finland') for b in ('dlinear','itransformer') for s in (2024,2025,2026)))
    ck('rule_before_validation',all((OUT/'cases'/d/f'{b}_seed{s}'/'validation.npz').stat().st_mtime>=(OUT/'controls_rule_lock.json').stat().st_mtime for d in ('carpark','finland') for b in ('dlinear','itransformer') for s in (2024,2025,2026)))
    res={'scope':'Independent fixed-array/strict-score/projection/source audit across 12 cases x validation + inspected legacy test; dense O(H M^2) DP at three fixed origins per stage. No experiment code imported. No training or selection. Check counts are program assertions, not experiment samples.','checks':checks,'check_count':len(checks),'failure_count':sum(not c['pass'] for c in checks),'dp_origin_control_checks':dp_checks,'normalization_bitexact_stage_cases':sum(r['normalization_path_identical'] for r in rows),'cases':rows,'supplemental_repair_input_sha256':source_repair,'supplemental_fixed_comparisons':supplement}
    (OUT/'audit_controls.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
    import pandas as pd
    pd.DataFrame(supplement).to_csv(OUT/'audit_supplemental_comparisons.csv',index=False)
    print(json.dumps({'check_count':len(checks),'failure_count':res['failure_count'],'dp_origin_control_checks':dp_checks,'failed':[c for c in checks if not c['pass']]},indent=2))
if __name__=='__main__':main()
