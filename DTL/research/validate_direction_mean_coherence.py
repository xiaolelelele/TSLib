"""Fixed exploratory compatibility controls. No fitting or model selection."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classes(x, z):
    previous = np.concatenate((x[:, -1:, :], z[:, :-1, :]), 1)
    return np.where(z < previous, 0, np.where(z > previous, 2, 1))


def metrics(x, y, z):
    true, pred = classes(x, y), classes(x, z)
    conf = np.bincount((3*true+pred).ravel(), minlength=9).reshape(3, 3)
    diag, nt, npred = np.diag(conf), conf.sum(1), conf.sum(0)
    f1 = np.divide(2*diag, nt+npred, out=np.zeros(3), where=(nt+npred)>0)
    zero = true == 1
    return {'mse':float(np.mean((y-z)**2)), 'mae':float(np.mean(np.abs(y-z))),
            'raw_accuracy':float(np.mean(true==pred)), 'macro_f1':float(f1.mean()),
            'first_mse':float(np.mean((y[:,0]-z[:,0])**2)),
            'first_raw_accuracy':float(np.mean(true[:,0]==pred[:,0])),
            'rest_mse':float(np.mean((y[:,1:]-z[:,1:])**2)),
            'rest_raw_accuracy':float(np.mean(true[:,1:]==pred[:,1:])),
            'zero_recall':float(np.mean((true==pred)[zero])) if zero.any() else None,
            'nonzero_accuracy':float(np.mean((true==pred)[~zero])) if (~zero).any() else None,
            'confusion':conf.tolist()}


def head_metrics(true, p):
    target=np.eye(3)[true]
    chosen=np.take_along_axis(p, true[...,None], -1)[...,0]
    return {'accuracy':float(np.mean(p.argmax(-1)==true)),
            'nll':float(-np.log(np.maximum(chosen, 1e-300)).mean()),
            'brier':float(np.sum((p-target)**2,-1).mean())}


def repair_first(x, mu, p, mean, std):
    # The previous observation is known only for h=1.
    raw_anchor=x[:,-1,:]*std+mean
    anchor=np.rint(raw_anchor)
    assert np.max(np.abs(raw_anchor-anchor)) < 1e-8
    assert np.min(anchor) >= 0
    original=np.asarray(p, dtype=np.float64)
    normalized=original/original.sum(-1,keepdims=True)
    masked=normalized.copy();zero=anchor==0
    first=masked[:,0,:,:]
    first[...,0][zero]=0.
    first[zero]=first[zero]/first[zero].sum(-1,keepdims=True)
    pdn,pflat,pup=first[...,0],first[...,1],first[...,2]
    lower=pflat*anchor+pup*(anchor+1.)
    upper=np.where(pup>0, np.inf, pdn*(anchor-1.)+pflat*anchor)
    raw_mean=mu[:,0,:]*std+mean
    bad=(raw_mean<lower-1e-9)|(raw_mean>upper+1e-9)
    fixed=mu.copy()
    fixed[:,0,:]=np.where(bad,(np.clip(raw_mean,lower,upper)-mean)/std,mu[:,0,:])
    return normalized,masked,fixed,{'anchor':anchor,'boundary':zero,
        'mean_violation':bad,'mean_lower':lower,'mean_upper':upper,
        'mean_raw_first':raw_mean,'mean_gap':np.abs(np.clip(raw_mean,lower,upper)-raw_mean),
        'probability_normalization_max_change':float(np.max(np.abs(original-normalized)))}


def diagnostic(x,y,mu,pnorm,pmask,fixed,info,meta):
    true=classes(x,y)
    boundary=info['boundary'];bad=info['mean_violation'];gap=info['mean_gap']
    std=np.asarray(meta['std']);mean=np.asarray(meta['mean'])
    impossible=pnorm[:,0,:,0]*boundary
    tfirst=y[:,0,:]*std+mean
    mufirst=mu[:,0,:]*std+mean
    rows=[]
    for name,mask in [('boundary',boundary),('mean_violation',bad),('mean_compatible',~bad)]:
        count=int(mask.sum())
        rows.append({'group':name,'count':count,
            'first_mu_mse_standardized':float(((y[:,0]-mu[:,0])**2)[mask].mean()) if count else None,
            'first_mu_mse_original_units':float(((tfirst-mufirst)**2)[mask].mean()) if count else None,
            'head_first_accuracy':float((pnorm[:,0].argmax(-1)==true[:,0])[mask].mean()) if count else None,
            'mean_repair_mse_delta_standardized':float((((fixed[:,0]-y[:,0])**2-(mu[:,0]-y[:,0])**2)[mask]).mean()) if count else None})
    channels=[]
    for c,label in enumerate(meta['columns']):
        channels.append({'channel':label,'first_cells':len(x),'boundary_cells':int(boundary[:,c].sum()),
            'impossible_probability_mass_mean':float(impossible[:,c].mean()),
            'mean_violation_cells':int(bad[:,c].sum()),
            'mean_gap_original_units':float(gap[:,c].mean())})
    return {'first_cells':int(boundary.size),'boundary_cells':int(boundary.sum()),
        'boundary_fraction':float(boundary.mean()),'impossible_probability_mass_mean':float(impossible.mean()),
        'impossible_probability_mass_on_boundary':float(pnorm[:,0,:,0][boundary].mean()) if boundary.any() else None,
        'mean_violation_cells':int(bad.sum()),'mean_violation_fraction':float(bad.mean()),
        'mean_gap_original_units_mean':float(gap.mean()),'mean_gap_standardized_mean':float((gap/std).mean()),
        'mean_gap_original_units_conditional':float(gap[bad].mean()) if bad.any() else None,
        'mean_gap_original_units_max':float(gap.max()),
        'upper_bound_finite_cells':int(np.isfinite(info['mean_upper']).sum()),
        'probability_normalization_max_change':info['probability_normalization_max_change'],
        'true_down_at_known_zero':int(np.count_nonzero(boundary & (true[:,0]==0))),
        'head_original_all':head_metrics(true,pnorm),'head_masked_all':head_metrics(true,pmask),
        'head_original_first':head_metrics(true[:,0],pnorm[:,0]),
        'head_masked_first':head_metrics(true[:,0],pmask[:,0]),'conditional_groups':rows,'channels':channels}


def bootstrap(days,delta,base_error,block,seed=90817):
    dates=np.asarray(days,dtype='datetime64[D]')
    groups=((dates-dates.min()).astype(int)//block)
    ids=np.unique(groups);count=np.array([(groups==g).sum() for g in ids])
    sums=np.array([delta[groups==g].sum(0) for g in ids])
    bases=np.array([base_error[groups==g].sum() for g in ids])
    rng=np.random.RandomState(seed);draw=rng.randint(0,len(ids),(2000,len(ids)))
    est=sums[draw].sum(1)/count[draw].sum(1)[:,None]
    b=bases[draw].sum(1)/count[draw].sum(1)
    return {'blocks':len(ids),'days_per_block':block,
            'raw_accuracy_pp_95ci':np.percentile(100*est[:,0],[2.5,97.5]).tolist(),
            'mse_percent_95ci':np.percentile(100*est[:,1]/b,[2.5,97.5]).tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run=Path(args.run).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=Path(__file__).with_name('coherence_validation_protocol_2026-10-08.md')
    assert protocol.exists()
    frozen=run/'source/v2';sys.path.insert(0,str(frozen))
    from trend_lab.trajectory_decision import make_common_grids,decode_trajectory
    start=time.perf_counter();cases=[];records=[];diagnostics=[];pair={};identities=[]
    for domain in ('carpark','finland'):
        for bone in ('dlinear','itransformer'):
            for seed in (2024,2025,2026):
                folder=run/'results_v2'/domain/(bone+'_seed'+str(seed))
                assert (folder/'result.json').exists()
                cases.append((domain,bone,seed,folder));identities.append(domain+'/'+folder.name)
    source={str(p.relative_to(run)):sha(p) for _,_,_,f in cases for p in (f/'predictions.npz',f/'validation_lock.json',f/'result.json')}
    source['source/v2/trend_lab/trajectory_decision.py']=sha(frozen/'trend_lab/trajectory_decision.py')
    rule={'client_date':'2026-10-08','scope':'validation development and already inspected legacy test exploratory evaluation only',
          'protocol':str(protocol),'protocol_sha256':sha(protocol),'script_sha256':sha(__file__),
          'old_lambda_inherited':True,'old_common_grid_fixed':True,'new_fitting':False,'new_selection':False,
          'cases':identities,'source_sha256':source}
    dump(out/'controls_rule_lock.json',rule)
    for stage in ('validation','test_legacy'):
        oldstage='test' if stage=='test_legacy' else stage
        for domain,bone,seed,folder in cases:
            lock=json.loads((folder/'validation_lock.json').read_text());meta=lock['metadata']
            choice=lock['choices']['posterior'];strength=float(choice.rsplit('_',1)[1]) if choice else None
            mean,std=np.asarray(meta['mean']),np.asarray(meta['std'])
            with np.load(folder/'predictions.npz') as a:
                x,y,b,mu,p,origins=[a[oldstage+'__'+k] for k in ('x','y','cosdir','mean','probabilities','origins')]
                old=a[oldstage+'__'+(choice if oldstage=='validation' and choice else 'posterior_selected' if oldstage=='test' else 'cosdir')]
                existing={m:a[oldstage+'__'+m] for m in ('cosdir','cosdir_rounded','mean_raw')}
                if oldstage=='test':existing['competitor_corrected']=np.load(run/'anchor64_repair_v2'/domain/folder.name/'predictions.npz')['test__competitor_corrected']
            pnorm,pmask,fixed,info=repair_first(x,mu,p,mean,std)
            grids,ginfo=make_common_grids(x,b,mu,mode='native',step=.1,padding=.25,
                normalizer_mean=mean,normalizer_std=std,native_step=1.,native_origin=0.,
                lower=-mean/std,max_states=257,coarsen=True)
            control_specs={'original_replay':(mu,p),'normalized_only':(mu,pnorm),
                'boundary_only':(mu,pmask),'mean_only_repair':(fixed,pnorm),'joint_repair':(fixed,pmask)}
            paths={};solver={}
            for method,(estimate,prob) in control_specs.items():
                if strength is None:
                    paths[method]=b.copy();solver[method]={'old_fallback_preserved':True}
                else:
                    paths[method],d=decode_trajectory(x,b,estimate,prob,strength,grids)
                    solver[method]={'strength':strength,'selected_base_samples':int(sum(d['selected_base'])),
                        'max_dp_objective_recompute_error':d['max_dp_objective_recompute_error'],
                        'elapsed_seconds':d['elapsed_seconds']}
            assert np.array_equal(paths['original_replay'],old), (domain,bone,seed,stage,'replay mismatch')
            paths.update(existing);paths['mean_fixed_raw']=fixed
            # First-step mean repair has no reason to change future means.
            assert np.array_equal(fixed[:,1:,:],mu[:,1:,:])
            assert np.array_equal(pmask[:,1:,:,:],pnorm[:,1:,:,:])
            d=diagnostic(x,y,mu,pnorm,pmask,fixed,info,meta)
            assert d['true_down_at_known_zero']==0
            case_out=out/'cases'/domain/folder.name
            arrays={'x':x,'y':y,'origins':origins,'p_original':p,'p_normalized':pnorm,
                'p_masked':pmask,'mean_original':mu,'mean_fixed':fixed,
                **{k:v for k,v in info.items() if isinstance(v,np.ndarray)},
                **{'path__'+k:v for k,v in paths.items()}}
            case_out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(case_out/(stage+'.npz'),**arrays)
            metrics_rows={m:metrics(x,y,z) for m,z in paths.items()}
            dump(case_out/(stage+'.json'),{'domain':domain,'backbone':bone,'seed':seed,'stage':stage,
                'old_choice':choice,'strength':strength,'same_grid':True,'diagnostic':d,'metrics':metrics_rows,'solver':solver,
                'original_prediction_sha256':source[str((folder/'predictions.npz').relative_to(run))],
                'original_lock_sha256':source[str((folder/'validation_lock.json').relative_to(run))]})
            for m,v in metrics_rows.items():records.append({'domain':domain,'backbone':bone,'seed':seed,'stage':stage,'method':m,**{k:x for k,x in v.items() if k!='confusion'}})
            diagnostics.append({'domain':domain,'backbone':bone,'seed':seed,'stage':stage,
                                **{k:v for k,v in d.items() if not isinstance(v,(dict,list))}})
            times=pd.to_datetime(pd.read_csv(meta['csv'],usecols=['date'])['date'])
            days=times.iloc[origins].to_numpy().astype('datetime64[D]')
            metric_origin={m:np.stack([np.mean(classes(x,y)==classes(x,z),(1,2)),np.mean((y-z)**2,(1,2))],-1) for m,z in paths.items()}
            key=(stage,domain,bone)
            pair.setdefault(key,{'days':days,'seeds':[],'metrics':[]})
            assert np.array_equal(pair[key]['days'],days)
            pair[key]['seeds'].append(seed);pair[key]['metrics'].append(metric_origin)
            print('DONE',stage,domain,bone,seed,'boundary',d['boundary_cells'],'mean_conflict',d['mean_violation_cells'],flush=True)
        if stage=='validation':dump(out/'validation_diagnostics_complete.json',{'cases':12,'selection_changed':False,'script_sha256':sha(__file__),'rule_sha256':sha(out/'controls_rule_lock.json')})
    frame=pd.DataFrame(records);frame.to_csv(out/'per_case_metrics.csv',index=False)
    frame.groupby(['stage','domain','backbone','method']).mean(numeric_only=True).reset_index().to_csv(out/'mean_metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'per_case_diagnostics.csv',index=False)
    aggregates={}
    for (stage,domain,bone),entry in pair.items():
        averaged={m:np.mean([s[m] for s in entry['metrics']],0) for m in entry['metrics'][0]}
        group={}
        for method in ('normalized_only','boundary_only','mean_only_repair','joint_repair','mean_fixed_raw'):
            refs=['original_replay','mean_raw','cosdir_rounded']
            if stage=='test_legacy':refs.append('competitor_corrected')
            item={'raw_accuracy':float(averaged[method][:,0].mean()),'mse':float(averaged[method][:,1].mean())}
            for ref in refs:
                baseline=averaged[ref];delta=averaged[method]-baseline
                item['vs_'+ref]={'accuracy_pp':float(100*delta[:,0].mean()),
                    'mse_percent':float(100*delta[:,1].mean()/baseline[:,1].mean()),
                    'joint':bool(delta[:,0].mean()>0 and delta[:,1].mean()<=1e-8),
                    'one_day':bootstrap(entry['days'],delta,baseline[:,1],1),
                    'three_day':bootstrap(entry['days'],delta,baseline[:,1],3)}
            group[method]=item
        aggregates[stage+'/'+domain+'/'+bone]=group
    dump(out/'paired_differences.json',aggregates)
    for name,digest in source.items():assert sha(run/name)==digest,name
    result={'client_date':'2026-10-08','cases':12,'stages':['validation','test_legacy'],
        'fixed_rule_paths':12*2*5,'model_fits':0,'new_hyperparameter_selection':False,
        'metric_records':len(records),'original_replays_bit_exact':24,'source_hash_checks':len(source),
        'source_hash_failures':0,'seconds':time.perf_counter()-start,'legacy_test_is_confirmation':False}
    dump(out/'controls_completion.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
