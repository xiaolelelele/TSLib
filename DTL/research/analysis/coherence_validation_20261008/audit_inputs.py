"""Independent frozen-input and first-step count-support audit, validation only.
No primary experimental metrics/projection code is imported. This script reads
old validation arrays only and performs no tuning, training or test inspection.
"""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930/results_v2')
OUT=Path('/home/lgc/TSLib/DTL/research/analysis/coherence_validation_20261008')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def mean_or_none(x):return float(np.mean(x)) if x.size else None

def count_first_bounds(x,p):
    """Complete first-moment bounds for nonnegative q=1 integer support.
    Return impossible directional mass, lower mean, upper mean. Positive up
    probability makes the upper support unbounded. At x=0, any positive down
    probability is infeasible independently of the value of the mean.
    """
    x=np.asarray(x,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    if np.any(x<0) or np.any(x!=np.rint(x)):raise ValueError('nonnegative integer anchor required')
    if np.any(p<0) or not np.allclose(p.sum(-1),1,atol=1e-12,rtol=0):raise ValueError('normalized probabilities required')
    pd,pf,pu=[p[...,i] for i in range(3)]
    impossible=np.where(x==0,pd,0.)
    low=pf*x+pu*(x+1)
    high=np.where(pu>0,np.inf,pd*np.maximum(x-1,0)+pf*x)
    return impossible,low,high

def boundary_mask(x,p):
    q=p.copy();q[...,0]=np.where(x==0,0.,q[...,0]);den=q.sum(-1,keepdims=True)
    if np.any(den<=0):raise ValueError('zero allowed mass cannot be renormalized')
    return q/den

def main():
    checks=[];rows=[]
    def check(name,ok,details=None):checks.append({'name':name,'pass':bool(ok),'details':details})
    # Exact support corners, including cases where a lower-only test is incomplete.
    for x,p,expected in [(0,[0,1,0],(0,0)),(2,[0,1,0],(2,2)),(2,[1,0,0],(0,1)),(2,[.5,.5,0],(1,1.5)),(0,[0,0,1],(1,np.inf)),(3,[.2,.3,.5],(2.9,np.inf))]:
        bad,low,high=count_first_bounds(np.array(x),np.array(p));check('moment_corner_'+str((x,p)),abs(float(low)-expected[0])<1e-12 and ((np.isinf(high) and np.isinf(expected[1])) or abs(float(high)-expected[1])<1e-12))
    for domain in ('carpark','finland'):
        meta=json.loads((ROOT/domain/'dataset.json').read_text());m,s=np.array(meta['mean']),np.array(meta['std'])
        for case in sorted((ROOT/domain).glob('*_seed*')):
            if case.name.startswith('shared'):continue
            with np.load(case/'predictions.npz') as a:
                x=a['validation__x'][:,-1,:]*s+m;y=a['validation__y'][:,0,:]*s+m
                mu=a['validation__mean'][:,0,:]*s+m
                p=a['validation__probabilities'][:,0,:,:].astype(np.float64)
            lock=json.loads((case/'validation_lock.json').read_text())
            xr=np.rint(x);yr=np.rint(y);p/=p.sum(-1,keepdims=True)
            name=f'{domain}/{case.name}'
            check(name+'/integer_anchor',np.max(np.abs(x-xr))<1e-9,float(np.max(np.abs(x-xr))))
            check(name+'/integer_truth',np.max(np.abs(y-yr))<1e-9,float(np.max(np.abs(y-yr))))
            check(name+'/nonnegative',np.min(xr)>=0 and np.min(yr)>=0)
            check(name+'/declared_native_q1',meta['native_integer_training'] and meta['spec']['grid_mode']=='native')
            check(name+'/p_source_temperature',lock['temperature'] in [.5,1,2,4],lock['temperature'])
            check(name+'/mu_already_nonnegative',np.min(mu)>-1e-10,float(np.min(mu)))
            check(name+'/no_test_selection',lock['test_labels_used_for_selection'] is False)
            pfix=boundary_mask(xr,p);bad,low,high=count_first_bounds(xr,p);_,lf,hf=count_first_bounds(xr,pfix)
            b=xr==0;lb=mu<lf-1e-9
            labels=np.where(yr<xr,0,np.where(yr>xr,2,1))
            nll=-np.log(np.take_along_axis(p,labels[...,None],-1)[...,0]);nllfix=-np.log(np.take_along_axis(pfix,labels[...,None],-1)[...,0])
            check(name+'/boundary_no_true_down',not np.any(labels[b]==0),int(b.sum()))
            check(name+'/mask_exact_expected_nll',np.allclose((nllfix-nll)[b],np.log1p(-p[...,0][b]),atol=1e-12,rtol=1e-10))
            rows.append({'case':name,'sha_npz':sha(case/'predictions.npz'),'sha_lock':sha(case/'validation_lock.json'),'temperature':lock['temperature'],'firststep_cells':int(xr.size),'boundary_cells':int(b.sum()),'boundary_fraction':float(b.mean()),'boundary_down_mass_mean':mean_or_none(p[...,0][b]),'boundary_down_mass_sum':float(p[...,0][b].sum()),'impossible_mass_all_cells_mean':float(bad.mean()),'mask_nll_delta_all':float(np.mean(nllfix-nll)),'mask_nll_delta_boundary':mean_or_none((nllfix-nll)[b]),'mean_lower_violation_after_mask_fraction':float(lb.mean()),'mean_lower_violation_after_mask_cells':int(lb.sum()),'mean_lower_gap_units_mean':float(np.maximum(lf-mu,0).mean()),'mean_lower_gap_on_violations_units_mean':mean_or_none((lf-mu)[lb]),'finite_upper_bound_cells':int(np.isfinite(hf).sum()),'max_integer_reconstruction_error':float(np.maximum(np.abs(x-xr).max(),np.abs(y-yr).max()))})
    check('expected_12_native_cases',len(rows)==12,len(rows))
    result={'scope':'Frozen native-count validation arrays; no test arrays read and no new fitting/selection. Independently recomputed first-step moment bounds. Counts are checks, not statistical samples.','theory':'Unbounded nonnegative integer support q=1. Impossible mass at x=0 is pd. Low=pf*x+pu*(x+1). High=inf when pu>0; otherwise pd*(x-1)+pf*x for x>=1; flat-only is exactly x.','checks':checks,'check_count':len(checks),'failure_count':sum(not z['pass'] for z in checks),'cases':rows}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'audit_inputs.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'case_count':len(rows),'check_count':len(checks),'failure_count':result['failure_count'],'cases':rows},indent=2))
if __name__=='__main__':main()
