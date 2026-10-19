"""Independent dense transition audit of frozen TIME trajectory artifacts.

No imports of the decoder/runner.  Full saved objectives are recomputed; dense
O(H*M^2) reference solutions are compared at two declared origin indices across
all channels for selected controls and lambda=0.  Training is never rerun.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

FAMILIES = ('posterior', 'top1_gap', 'mean_is_base', 'training_prior', 'greedy')


def symbols(x,z):
    previous=np.concatenate((x[:,-1:,:],z[:,:-1,:]),1)
    return np.where(z<previous,0,np.where(z>previous,2,1))


def rewards(p,mode):
    if mode=='posterior':return p
    out=np.zeros_like(p)
    top=np.argmax(p,-1);gap=np.sort(p,axis=-1)[...,-1]-np.sort(p,axis=-1)[...,-2]
    np.put_along_axis(out,top[...,None],gap[...,None],axis=-1)
    return out


def objective(x,z,mu,p,strength,mode='posterior'):
    cls=symbols(x,z);utility=np.take_along_axis(rewards(p,mode),cls[...,None],axis=-1)[...,0].mean(1)
    squared=((z-mu)**2).mean(1)
    return squared-strength*utility,squared,utility


def grid_reference(x,b,mu,meta,info):
    """Materialize the declared covering lattice and its deterministic subset."""
    n,_,c=b.shape;grid=[];counts=[];anchor=[];sparse=[];full=[]
    std=np.asarray(meta['std'],np.float64);mean=np.asarray(meta['mean'],np.float64)
    native=info['mode']=='native';step=np.asarray(info['normalized_step'],np.float64)
    origin=np.asarray(info['normalized_origin'],np.float64);padding=np.asarray(info['padding'],np.float64)
    limit=info['max_states'];include=info['include_anchor']
    for i in range(n):
        row=[];countrow=[];anchorrow=[];sparserow=[];fullrow=[]
        for j in range(c):
            a=x[i,-1,j];visible=np.concatenate(([a],b[i,:,j],mu[i,:,j]))
            lo=visible.min()-padding[j];hi=visible.max()+padding[j]
            lower=-mean[j]/std[j] if native else None
            if native:lo=max(lo,lower)
            first=int(np.floor((lo-origin[j])/step[j]));last=int(np.ceil((hi-origin[j])/step[j]));size=last-first+1
            ai=int(np.rint((a-origin[j])/step[j]))
            av=(ai-mean[j])/std[j] if native else origin[j]+ai*step[j]
            has_ai=bool(first<=ai<=last and av==a)
            reserve=int(include or has_ai)
            is_sparse=size+(int(include) if not has_ai else 0)>limit
            if is_sparse:
                slots=limit-reserve
                indices=np.asarray([first+(k*(last-first))//(slots-1) for k in range(slots)],np.float64)
                if has_ai:indices=np.unique(np.r_[indices,ai])
            else:indices=np.arange(first,last+1,dtype=np.float64)
            values=(indices-mean[j])/std[j] if native else origin[j]+indices*step[j]
            if native:values=values[values>=lower]
            if include and (not native or a>=lower):values=np.unique(np.r_[values,a])
            row.append(values);countrow.append(len(values));anchorrow.append(bool((values==a).any()))
            sparserow.append(is_sparse);fullrow.append(size)
        grid.append(row);counts.append(countrow);anchor.append(anchorrow);sparse.append(sparserow);full.append(fullrow)
    return grid,counts,anchor,sparse,full


def dense_dp(anchor,mean,p,strength,grid,mode='posterior'):
    """All predecessor transitions, avoiding prefix/suffix machinery."""
    p=np.asarray(p,np.float64);g=np.asarray(grid,np.float64)
    if mode=='top1_gap':p=rewards(p,'top1_gap')
    h,m=len(mean),len(g);back=np.empty((h,m),np.int64)
    cls=np.where(g<anchor,0,np.where(g>anchor,2,1))
    value=(g-mean[0])**2-strength*p[0,cls];back[0]=-1
    transition_class=np.where(g[None,:]<g[:,None],0,np.where(g[None,:]>g[:,None],2,1))
    columns=np.arange(m)
    for t in range(1,h):
        complete=value[:,None]-strength*p[t,transition_class]
        predecessor=complete.argmin(0)
        back[t]=predecessor
        value=complete[predecessor,columns]+(g-mean[t])**2
    pos=int(value.argmin());path=np.empty(h,np.float64)
    for t in range(h-1,-1,-1):path[t]=g[pos];pos=int(back[t,pos])
    return path


class Audit:
    def __init__(self):self.checks=[];self.case_rows=[];self.dense_cells=0
    def compare(self,name,actual,expected,tol=1e-10):
        a,b=np.asarray(actual),np.asarray(expected)
        if a.shape!=b.shape:ok=False;error=None
        elif a.size:
            error=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))));ok=np.isfinite(error) and error<=tol
        else:error=0.;ok=True
        self.checks.append({'name':name,'pass':bool(ok),'max_abs_error':error})
    def case(self,case):
        begin=len(self.checks);dense_begin=self.dense_cells
        lock=json.loads((case/'validation_lock.json').read_text());diagnostics=json.loads((case/'decoder_diagnostics.json').read_text())
        meta=lock['metadata'];prior=np.asarray(lock['prior'],np.float64)
        archive=np.load(case/'predictions.npz');tag=str(case)
        for split in ('validation','test'):
            x=archive[split+'__x'];b=archive[split+'__cosdir'];mu=archive[split+'__mean'];p=np.asarray(archive[split+'__probabilities'],np.float64)
            info=diagnostics[split]['grid'];grids,counts,anchor,sparse,full=grid_reference(x,b,mu,meta,info)
            self.compare(tag+' '+split+' float64 x',str(x.dtype)=='float64',True,0.)
            self.compare(tag+' '+split+' finite probability',np.isfinite(p).all(),True,0.)
            self.compare(tag+' '+split+' probability sum',p.sum(-1),np.ones(p.shape[:-1]),1e-6)
            for key,ref in [('grid_state_counts',counts),('anchor_reachable',anchor),('coarsened',sparse),('full_covering_lattice_state_counts',full)]:
                self.compare(tag+' '+split+' reconstructed '+key,info[key],ref,0.)
            if info['mode']=='native':
                raw_anchor=x[:,-1,:]*np.asarray(meta['std'])+np.asarray(meta['mean'])
                self.compare(tag+' '+split+' native historical integer anchor',raw_anchor,np.rint(raw_anchor),1e-9)
                self.compare(tag+' '+split+' native reconstructed anchor exactly equals x',
                             (np.rint(raw_anchor)-np.asarray(meta['mean']))/np.asarray(meta['std']),x[:,-1,:],0.)
            indices=sorted(set([0,len(x)//2]))
            dense_keys={'mean_only_grid'}
            for family in FAMILIES:
                selected=lock['choices'][family]
                if selected is not None:dense_keys.add(selected)
                if split=='validation':dense_keys.add(family+'_0.0')
                if split=='test':
                    chosen=archive[split+'__'+family+'_selected']
                    if selected is None:self.compare(tag+' '+split+' fallback '+family,chosen,b,0.)
                    else:self.compare(tag+' '+split+' selected '+family,chosen,archive[split+'__'+selected],0.)
            for key,diag in diagnostics[split].items():
                if key=='grid' or key.startswith('soft_'):continue
                z=archive[split+'__'+key]
                family='mean_only_grid' if key=='mean_only_grid' else key.rsplit('_',1)[0]
                estimate=b if family=='mean_is_base' else mu
                prob=np.broadcast_to(prior,p.shape) if family=='training_prior' else p
                mode='top1_gap' if family=='top1_gap' else 'posterior'
                strength=diag['strength']
                score,error,utility=objective(x,z,estimate,prob,strength,mode)
                bscore,berror,butility=objective(x,b,estimate,prob,strength,mode)
                for label,actual in [('objective',score.mean(1)),('objective_by_channel',score),('base_objective',bscore.mean(1)),
                                     ('estimated_risk_delta',(error-berror).mean(1)),('estimated_utility_delta',(utility-butility).mean(1))]:
                    self.compare(tag+' '+split+' '+key+' '+label,actual,diag[label])
                self.compare(tag+' '+split+' '+key+' same grid counts',diag['grid_state_counts'],counts,0.)
                self.compare(tag+' '+split+' '+key+' same state work',diag['state_updates'],sum(sum(row) for row in counts)*z.shape[1],0.)
                selected_base=np.asarray(diag['selected_base'],bool)
                self.compare(tag+' '+split+' '+key+' full base objective comparison',selected_base,
                             bscore.mean(1)<=np.asarray(diag['grid_objective']),0.)
                self.compare(tag+' '+split+' '+key+' selected complete base',z[selected_base],b[selected_base],0.)
                # Every non-base output remains on the identical explicitly declared library.
                member=True
                for i in np.flatnonzero(~selected_base):
                    for c,g in enumerate(grids[i]):
                        location=np.searchsorted(g,z[i,:,c]);safe=np.clip(location,0,len(g)-1)
                        member=member and bool(np.all(location<len(g)) and np.array_equal(g[safe],z[i,:,c]))
                self.compare(tag+' '+split+' '+key+' output candidate membership',member,True,0.)
                if key not in dense_keys:continue
                sample_x=x[indices];sample_b=b[indices];sample_mu=estimate[indices];sample_p=prob[indices]
                dense=np.empty_like(sample_b)
                for si,i in enumerate(indices):
                    for c in range(z.shape[2]):
                        dense[si,:,c]=dense_dp(x[i,-1,c],estimate[i,:,c],prob[i,:,c],strength,grids[i][c],mode)
                        self.dense_cells+=1
                dscore,_,_=objective(sample_x,dense,sample_mu,sample_p,strength,mode)
                if family!='greedy':
                    self.compare(tag+' '+split+' '+key+' independent dense grid objective',dscore.mean(1),np.asarray(diag['grid_objective'])[indices])
                    fallback=bscore.mean(1)[indices]<=dscore.mean(1)
                    dense[fallback]=sample_b[fallback]
                    self.compare(tag+' '+split+' '+key+' independent dense complete path',dense,z[indices],0.)
                else:
                    # Greedy is compared with the optimal DP at its OWN lambda.
                    best=np.minimum(dscore.mean(1),bscore.mean(1)[indices])
                    self.compare(tag+' '+split+' '+key+' greedy cannot beat exact same-objective DP',
                                 np.all(score.mean(1)[indices]>=best-1e-10),True,0.)
                if strength==0:
                    # Direction probabilities do not affect a zero-strength decision.
                    expected=np.empty_like(sample_b)
                    for si,i in enumerate(indices):
                        for c in range(z.shape[2]):
                            g=grids[i][c];dist=(g[None,:]-estimate[i,:,c,None])**2
                            expected[si,:,c]=g[dist.argmin(1)]
                    expected_score,_,_=objective(sample_x,expected,sample_mu,sample_p,0.,mode)
                    fallback=bscore.mean(1)[indices]<=expected_score.mean(1)
                    expected[fallback]=sample_b[fallback]
                    self.compare(tag+' '+split+' '+key+' zero-lambda independent nearest-mean decision',z[indices],expected,0.)
        failures=sum(not r['pass'] for r in self.checks[begin:])
        self.case_rows.append({'case':tag,'checks':len(self.checks)-begin,'failures':failures,'independent_dense_channel_paths':self.dense_cells-dense_begin})
        archive.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.results);audit=Audit();start=time.perf_counter()
    cases=sorted(path.parent for path in root.glob('*/*/result.json') if (path.parent/'decoder_diagnostics.json').exists())
    for path in cases:audit.case(path)
    files=[path/file for path in cases for file in ['predictions.npz','decoder_diagnostics.json','validation_lock.json']]
    result={'results_root':str(root.resolve()),'audit_source':'independent objectives/grids + dense O(H*M^2) transitions, no solver import',
            'cases':len(cases),'checks':len(audit.checks),'failures':sum(not row['pass'] for row in audit.checks),
            'sample_rule':'origin indices 0 and floor(N/2), every channel, selected controls plus validation lambda=0; all saved objectives recomputed',
            'independent_dense_channel_paths':audit.dense_cells,'elapsed_seconds':time.perf_counter()-start,
            'case_rows':audit.case_rows,'details':audit.checks,
            'artifact_sha256':{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['cases','checks','failures','independent_dense_channel_paths','elapsed_seconds']}))
    if result['failures']:raise AssertionError('decoder audit failures')


if __name__=='__main__':main()
