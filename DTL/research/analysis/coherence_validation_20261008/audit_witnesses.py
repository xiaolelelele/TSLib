"""Verify every saved numerical chain witness without importing a solver.
Independently reconstruct dense flows from sparse edges and check moments,
directions, continuity, initial point mass and root control-array identities.
"""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
RUN=Path('/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    d=json.loads((ROOT/'real_flow_certificates.json').read_text());checks=[];records=[];necessary=[];maxerrs={'flow':0.,'probability':0.,'mean_raw':0.,'normalization':0.};active=0
    def ck(name,ok,detail=None):checks.append({'name':name,'pass':bool(ok),'detail':detail})
    for p,digest in d['source_hashes'].items():ck('source/'+p,sha(Path(p))==digest)
    for p,digest in d['code_hashes'].items():ck('code/'+p,sha(ROOT/p)==digest)
    for row in d['rows']:
        n=row['origin_array_index'];c=row['channel_index'];case=row['case'];variant=row['variant'];name=f'{case}/o{n}/c{c}/{variant}'
        domain,bonecase=case.split('/');meta=json.loads((RUN/'results_v2'/domain/'dataset.json').read_text());mean=np.array(meta['mean']);std=np.array(meta['std'])
        with np.load(ROOT/'cases'/domain/bonecase/'validation.npz') as a:
            anchor=int(np.rint(a['x'][n,-1,c]*std[c]+mean[c]));oldp=a['p_normalized'][n,:,c].copy();oldmu=a['mean_original'][n,:,c].copy()*std[c]+mean[c]
            p=a['p_masked'][n,:,c].copy() if variant=='simple_repair' else oldp
            mu=a['mean_fixed'][n,:,c].copy()*std[c]+mean[c] if variant=='simple_repair' else oldmu
            originrow=int(a['origins'][n]);count=len(a['x'])
        ck(name+'/fixed_origin_positions',n in [0,count//2]);ck(name+'/origin_row',originrow==row['origin_row']);ck(name+'/anchor',anchor==row['anchor'])
        if variant=='simple_repair' and (np.max(np.abs(oldp-p))>0 or np.max(np.abs(oldmu-mu))>0):active+=1
        if row['status']=='CERTIFIED_FEASIBLE_NUMERICAL':
            file=Path(row['certificate_path']);ck(name+'/certificate_sha',sha(file)==row['certificate_sha256'])
            with np.load(file) as a:cert={k:a[k] for k in a.files}
            g=cert['grid'];index=cert['edge_index'];values=cert['edge_value'];v=cert['v'];h=len(mu);m=len(g)
            ck(name+'/native_support',np.issubdtype(g.dtype,np.integer) and np.all(g>=0) and len(np.unique(g))==m and np.all(np.diff(g)>0) and anchor in g)
            ck(name+'/input_p_matches_root',np.allclose(cert['probabilities'],p,atol=1e-14,rtol=1e-14));ck(name+'/input_mu_matches_root',np.allclose(cert['mean_raw'],mu,atol=1e-12,rtol=1e-14));ck(name+'/input_std',float(cert['train_std'])==float(std[c]));ck(name+'/input_anchor',int(cert['anchor'])==anchor)
            ck(name+'/edge_indices_valid',index.ndim==2 and index.shape[1]==3 and len(index)==len(values) and np.all(index[:,0]>=0) and np.all(index[:,0]<h) and np.all(index[:,1:]>=0) and np.all(index[:,1:]<m))
            ck(name+'/nonnegative_finite',np.isfinite(values).all() and np.isfinite(v).all() and values.min(initial=0)>=-1e-12 and v.min(initial=0)>=-1e-12)
            edges=np.zeros((h,m,m));np.add.at(edges,tuple(index.T),values)
            source=edges.sum(2);dest=edges.sum(1);init=np.zeros(m);init[int(np.flatnonzero(g==anchor)[0])]=1.
            flowerr=float(max(np.abs(source-v[:-1]).max(),np.abs(dest-v[1:]).max(),np.abs(v[0]-init).max()))
            direction=(np.sign(g[None,:]-g[:,None])+1).astype(int);phat=np.array([[edges[t][direction==k].sum() for k in range(3)] for t in range(h)])
            perr=float(np.abs(phat-p).max());merr=float(np.abs(v[1:]@g-mu).max());nerr=float(np.abs(v.sum(1)-1).max())
            ck(name+'/flow_continuity',flowerr<=1e-8,flowerr);ck(name+'/direction_probability',perr<=1e-8,perr);ck(name+'/raw_means',merr<=max(1,std[c])*1e-8,merr);ck(name+'/normalization',nerr<=1e-8,nerr)
            for k,z in [('flow',flowerr),('probability',perr),('mean_raw',merr),('normalization',nerr)]:maxerrs[k]=max(maxerrs[k],z)
            records.append({'name':name,'support_size':m,'support_max':int(g[-1]),'flow_error':flowerr,'probability_error':perr,'mean_raw_error':merr,'normalization_error':nerr,'sha_certificate':sha(file)})
        elif row['status']=='NECESSARY_INCOMPATIBLE':
            # Independently verify the exact necessary counterexamples used in
            # this run; these conditions are valid on the full unbounded support.
            violations=[]
            for t in range(len(mu)):
                if mu[t]<p[t,2]-1e-9:violations.append({'condition':'current_mean_at_least_up_mass','step':t+1,'gap':float(p[t,2]-mu[t])})
                previous=float(anchor) if t==0 else mu[t-1]
                if previous<p[t,0]-1e-9:violations.append({'condition':'previous_mean_at_least_down_mass','step':t+1,'gap':float(p[t,0]-previous)})
            ck(name+'/native_necessary_conditions_violate',len(violations)>0,violations)
            ck(name+'/named_violations_equal',violations==row['native_necessary_violations'])
            necessary.append({'name':name,'violations':violations})
        else:ck(name+'/unexpected_status',False,row['status'])
    ck('expected_216_variants',len(d['rows'])==216);ck('expected_214_certificates',len(records)==214,len(records));ck('expected_two_necessary_variants',len(necessary)==2,len(necessary))
    res={'scope':'All 214 saved floating point witnesses independently reconstructed; 108 fixed validation curves, two variants each. No test labels, LP solver or experiment implementation imported. Accepted residuals are numerical evidence, not exact rational/interval formal proofs. No full-data incompatibility-rate inference.','check_count':len(checks),'failure_count':sum(not z['pass'] for z in checks),'witness_count':len(records),'sample_curve_count':108,'simple_repair_active_sample_curves':active,'max_residuals':maxerrs,'checks':checks,'certificates':records,'necessary_incompatible':necessary}
    (ROOT/'audit_witnesses.json').write_text(json.dumps(res,indent=2,allow_nan=False)+'\n');print(json.dumps({k:res[k] for k in ['check_count','failure_count','witness_count','simple_repair_active_sample_curves','max_residuals','necessary_incompatible']},indent=2))
if __name__=='__main__':main()
