"""Independent finite-support chain feasibility audit.
Uses two distinct LP formulations (full-path masses and transition flows).
It proves neither unbounded-support infeasibility nor a novel theorem.
"""
from itertools import product
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
OUT=Path(__file__).parent/'audit_flow.json'

def classify(a,b):return 0 if b<a else 2 if b>a else 1

def feasible_paths(grid,anchor,p,mean):
    grid=np.asarray(grid,float);p=np.asarray(p,float);mean=np.asarray(mean,float);h=len(mean)
    paths=np.array(list(product(grid,repeat=h)),float)
    rows=[np.ones(len(paths))];rhs=[1.]
    for t in range(h):
        prev=np.full(len(paths),anchor) if t==0 else paths[:,t-1]
        cls=np.where(paths[:,t]<prev,0,np.where(paths[:,t]>prev,2,1))
        for k in range(3):rows.append((cls==k).astype(float));rhs.append(p[t,k])
        rows.append(paths[:,t]);rhs.append(mean[t])
    res=linprog(np.zeros(len(paths)),A_eq=np.array(rows),b_eq=np.array(rhs),bounds=(0,None),method='highs')
    return {'feasible':bool(res.success),'status':int(res.status),'variables':int(len(paths)),'max_residual':float(np.abs(np.array(rows)@res.x-rhs).max()) if res.success else None}

def feasible_flows(grid,anchor,p,mean):
    grid=np.asarray(grid,float);p=np.asarray(p,float);mean=np.asarray(mean,float);h=len(mean);m=len(grid)
    if not np.any(grid==anchor):raise ValueError('anchor must be in finite grid for this formulation')
    a=int(np.flatnonzero(grid==anchor)[0]);size=h*m*m;rows=[];rhs=[]
    def row():return np.zeros((h,m,m),float)
    # Initial outflow is a point mass at the observed real anchor.
    for i in range(m):
        r=row();r[0,i,:]=1;rows.append(r.ravel());rhs.append(float(i==a))
    # Destination mass of one transition is source mass of the next transition.
    for t in range(1,h):
        for i in range(m):
            r=row();r[t,i,:]=1;r[t-1,:,i]=-1;rows.append(r.ravel());rhs.append(0.)
    for t in range(h):
        for k in range(3):
            r=row()
            for i in range(m):
                for j in range(m):r[t,i,j]=float(classify(grid[i],grid[j])==k)
            rows.append(r.ravel());rhs.append(p[t,k])
        r=row();r[t,:,:]=grid[None,:];rows.append(r.ravel());rhs.append(mean[t])
    res=linprog(np.zeros(size),A_eq=np.array(rows),b_eq=np.array(rhs),bounds=(0,None),method='highs')
    return {'feasible':bool(res.success),'status':int(res.status),'variables':int(size),'max_residual':float(np.abs(np.array(rows)@res.x-rhs).max()) if res.success else None}

def main():
    rows=[];checks=[]
    def check(name,ok,details=None):checks.append({'name':name,'pass':bool(ok),'details':details})
    cases=[
        ('legal_deterministic',[0,1,2],0,[[0,0,1],[0,0,1]],[1,2],True),
        ('illegal_boundary_down',[0,1,2],0,[[.1,.9,0]],[0],False),
        ('illegal_first_moment',[0,1,2],0,[[0,.5,.5]],[.4],False),
        ('first_step_compatible_does_not_certify_future',[0,1,2],0,[[0,0,1],[1,0,0]],[1,1],False),
        ('truncated_support_false_problem',[0,1,2],0,[[0,0,1]],[4],False),
        ('expanded_support_resolves_problem',[0,1,2,3,4],0,[[0,0,1]],[4],True),
    ]
    for name,g,x,p,mu,expected in cases:
        a=feasible_paths(g,x,p,mu);b=feasible_flows(g,x,p,mu)
        check(name+'/path_expected',a['feasible']==expected);check(name+'/flow_expected',b['feasible']==expected)
        rows.append({'name':name,'grid':g,'anchor':x,'p':p,'mean':mu,'expected_within_this_grid':expected,'paths':a,'flow':b})
    rng=np.random.default_rng(20261008)
    for i in range(60):
        h=int(rng.integers(1,4));m=int(rng.integers(2,5));grid=np.arange(m,dtype=float);anchor=int(rng.integers(m))
        paths=np.array(list(product(grid,repeat=h)));w=rng.dirichlet(np.ones(len(paths)));mu=w@paths
        p=np.zeros((h,3))
        for t in range(h):
            prev=np.full(len(paths),anchor) if t==0 else paths[:,t-1]
            c=np.where(paths[:,t]<prev,0,np.where(paths[:,t]>prev,2,1))
            for k in range(3):p[t,k]=w[c==k].sum()
        a=feasible_paths(grid,anchor,p,mu);b=feasible_flows(grid,anchor,p,mu)
        check(f'generated_{i}/both_feasible',a['feasible'] and b['feasible'])
        check(f'generated_{i}/residual',a['max_residual']<1e-7 and b['max_residual']<1e-7)
        # Independently perturb one mean; compare feasibility, not presumed status.
        shifted=mu.copy();shifted[int(rng.integers(h))]+=float(rng.uniform(-m,m))
        a2=feasible_paths(grid,anchor,p,shifted);b2=feasible_flows(grid,anchor,p,shifted)
        check(f'perturbed_{i}/same_feasibility',a2['feasible']==b2['feasible'])
    result={'scope':'Independent small finite-support mechanism checks only; 60 generated chains are numerical tests, not data or statistical replications. Infeasibility is conditional on the stated finite support.','cases':rows,'checks':checks,'check_count':len(checks),'failure_count':sum(not r['pass'] for r in checks),'root_primary_code_imported':False}
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'check_count':len(checks),'failure_count':result['failure_count'],'artifact':str(OUT)}))
if __name__=='__main__':main()
