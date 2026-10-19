"""Post-lock descriptive analysis; this file never selects models or lambdas."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def classes(x,z):
    previous=np.concatenate([x[:,-1:,:],z[:,:-1,:]],1)
    return np.where(z<previous,0,np.where(z>previous,2,1))


def origin_metrics(x,y,z):
    return np.mean(classes(x,y)==classes(x,z),(1,2)),np.mean((y-z)**2,(1,2))


def intervals(days,delta,baseline_error,block_days,seed=90731):
    # Days are sampled as contiguous groups; preserve unequal origin counts.
    days=np.asarray(days,dtype='datetime64[D]')
    offset=((days-days.min()).astype(int)//block_days)
    groups=np.unique(offset)
    counts=np.array([(offset==g).sum() for g in groups])
    sums=np.array([delta[offset==g].sum(0) for g in groups])
    base=np.array([baseline_error[offset==g].sum() for g in groups])
    rng=np.random.RandomState(seed)
    draws=rng.randint(0,len(groups),(2000,len(groups)))
    estimate=sums[draws].sum(1)/counts[draws].sum(1)[:,None]
    base_est=base[draws].sum(1)/counts[draws].sum(1)
    return {'blocks':len(groups),'days_per_block':block_days,
            'accuracy_pp_95ci':np.percentile(100*estimate[:,0],[2.5,97.5]).tolist(),
            'mse_percent_95ci':np.percentile(100*estimate[:,1]/base_est,[2.5,97.5]).tolist()}


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    run=Path(args.run);root=run/'results_v2';out=run/'analysis';out.mkdir(exist_ok=True)
    repair=run/'anchor64_repair_v2'
    completion=json.loads((root/'completion.json').read_text())
    assert completion['cases']==18
    assert sum(len(list(repair.glob('*/'+b+'_seed*/result.json'))) for b in ('dlinear','itransformer'))==18
    assert sum(len(list((run/'precision_controls').glob('*/'+b+'_seed*/result.json'))) for b in ('dlinear','itransformer'))==12
    rows=[];locks={};repair_locks={};results={};checks=[];aggregate={};details=[];costs=[];origin_averages={};origin_days={};strata=[];applications=[];thresholds={};inputs={}
    sys.path.insert(0,str(run/'source/v2'))
    from trend_lab.trajectory_data import load_trajectory_data
    methods=['cosdir','posterior_selected','top1_gap_selected','mean_is_base_selected',
             'training_prior_selected','greedy_selected','mean_only_grid','mean_raw',
             'soft_selected','competitor_selected','mse_base','tdalign','residual_capacity',
             'residual_direction_capacity','oce_fixed_support','carets3_phase','carets4_phase',
             'cosdir_rounded','last_value','seasonal_naive','carets3_phase_anchor64',
             'carets4_phase_anchor64','competitor_corrected']
    native_controls=[m+'_native_round' for m in ('mean_raw','residual_capacity','residual_direction_capacity','mse_base','cosdir','tdalign')]+['residual_capacity_shared_grid','residual_direction_capacity_shared_grid']
    for domain in ('nem','carpark','finland'):
        meta=json.loads((root/domain/'dataset.json').read_text())
        domain_methods=methods+native_controls if meta['spec']['grid_mode']=='native' else methods
        raw_frame=pd.read_csv(meta['csv']);times=pd.to_datetime(raw_frame.pop('date'));raw_values=raw_frame.to_numpy(dtype=np.float64)
        split_data,_,_=load_trajectory_data(meta['spec'])
        train=split_data['train'];changes=np.stack([np.abs(np.diff(raw_values[t:t+meta['horizon']+1],axis=0)) for t in train.origins])
        cutoffs=np.array([np.quantile(changes[:,:,c][changes[:,:,c]>0],[.25,.75]) for c in range(changes.shape[-1])])
        thresholds[domain]=dict(columns=meta['columns'],nonzero_train_delta_q25_q75=cutoffs.tolist(),definition='fixed training-window quantiles in original channel units')
        del split_data,train,changes
        for bone in ('dlinear','itransformer'):
            paired={m:[] for m in domain_methods};seed_rows=[]
            for seed in (2024,2025,2026):
                case=root/domain/(bone+'_seed'+str(seed))
                result=json.loads((case/'result.json').read_text());lock=json.loads((case/'validation_lock.json').read_text())
                repaired_case=repair/domain/(bone+'_seed'+str(seed))
                repaired=json.loads((repaired_case/'result.json').read_text())
                repair_locks[domain+'/'+bone+'/'+str(seed)]=json.loads((repaired_case/'repair_lock.json').read_text())
                for folder,filenames in ((case,('result.json','validation_lock.json','predictions.npz')),(repaired_case,('result.json','repair_lock.json','predictions.npz'))):
                    for filename in filenames:
                        p=folder/filename;inputs[str(p.relative_to(run))]=hashlib.sha256(p.read_bytes()).hexdigest()
                result['test'].update(repaired['test'])
                if domain in ('carpark','finland'):
                    control_case=run/'precision_controls'/domain/(bone+'_seed'+str(seed))
                    controls=json.loads((control_case/'result.json').read_text())
                    for filename in ('result.json','predictions.npz'):
                        p=control_case/filename;inputs[str(p.relative_to(run))]=hashlib.sha256(p.read_bytes()).hexdigest()
                    result['test'].update(controls['stages']['test']['metrics'])
                key=domain+'/'+bone+'/'+str(seed);locks[key]=lock;results[key]=result
                with np.load(case/'predictions.npz') as original,np.load(repaired_case/'predictions.npz') as fixed:
                    ar={k:original[k] for k in original.files if k.startswith('test__')}
                    ar.update({k:fixed[k] for k in fixed.files if k.startswith('test__')})
                    if domain in ('carpark','finland'):
                        with np.load(control_case/'predictions.npz') as control_arrays:
                            ar.update({k:control_arrays[k] for k in control_arrays.files if k.startswith('test__')})
                    x,y=ar['test__x'],ar['test__y'];truth=classes(x,y)
                    origins=ar['test__origins'];days=times.iloc[origins].to_numpy().astype('datetime64[D]')
                    phase=times.iloc[origins].dt.hour.to_numpy()//6
                    amplitude=np.stack([np.abs(np.diff(raw_values[t:t+meta['horizon']+1],axis=0)) for t in origins])
                    masks={'zero':amplitude==0,'nonzero_low':(amplitude>0)&(amplitude<=cutoffs[:,0]),
                           'nonzero_middle':(amplitude>cutoffs[:,0])&(amplitude<=cutoffs[:,1]),'nonzero_high':amplitude>cutoffs[:,1]}
                    for method in domain_methods:
                        z=ar['test__'+method];accuracy,error=origin_metrics(x,y,z)
                        paired[method].append(np.stack([accuracy,error],-1))
                        metrics=result['test'][method]
                        row=dict(domain=domain,backbone=bone,seed=seed,method=method,**{k:metrics.get(k) for k in ['mse','mae','raw_accuracy','macro_f1','true_zero_fraction','predicted_zero_fraction','nonzero_accuracy']})
                        row.update(zero_precision=metrics['precision'][1],zero_recall=metrics['recall'][1],head_accuracy=result['head_accuracy'])
                        rows.append(row)
                        checks.append(abs(accuracy.mean()-metrics['raw_accuracy'])<1e-12 and abs(error.mean()-metrics['mse'])<1e-12)
                        pred=classes(x,z);squared=(y-z)**2
                        for label,mask in masks.items():
                            strata.append(dict(domain=domain,backbone=bone,seed=seed,method=method,stratum='target_amplitude',label=label,count=int(mask.sum()),
                                               raw_accuracy=float((truth==pred)[mask].mean()) if mask.any() else None,mse=float(squared[mask].mean()) if mask.any() else None))
                        for group in range(4):
                            mask=phase==group
                            strata.append(dict(domain=domain,backbone=bone,seed=seed,method=method,stratum='origin_hour',label=str(6*group)+'-'+str(6*group+5),count=int(mask.sum()),
                                               raw_accuracy=float((truth==pred)[mask].mean()) if mask.any() else None,mse=float(squared[mask].mean()) if mask.any() else None))
                        true_terminal=np.sign(y[:,-1,:]-x[:,-1,:]);pred_terminal=np.sign(z[:,-1,:]-x[:,-1,:])
                        def longest_flat(cls):
                            current=np.zeros((len(cls),cls.shape[-1]),int);longest=current.copy()
                            for h in range(cls.shape[1]):
                                current=np.where(cls[:,h]==1,current+1,0);longest=np.maximum(current,longest)
                            return longest
                        true_flat,pred_flat=longest_flat(truth),longest_flat(pred)
                        minutes=pd.Timedelta(meta['spec']['frequency']).total_seconds()/60
                        for c,label in enumerate(meta['columns']):
                            true_down=true_terminal[:,c]<0;pred_down=pred_terminal[:,c]<0
                            applications.append(dict(domain=domain,backbone=bone,seed=seed,method=method,channel=label,
                                                     terminal_direction_accuracy=float((true_terminal[:,c]==pred_terminal[:,c]).mean()),
                                                     terminal_down_precision=float((true_down&pred_down).sum()/pred_down.sum()) if pred_down.any() else None,
                                                     terminal_down_recall=float((true_down&pred_down).sum()/true_down.sum()) if true_down.any() else None,
                                                     terminal_change_mae_original_units=float((np.abs(y[:,-1,c]-z[:,-1,c])*meta['std'][c]).mean()),
                                                     longest_true_flat_steps=float(true_flat[:,c].mean()),longest_predicted_flat_steps=float(pred_flat[:,c].mean()),
                                                     longest_flat_mae_minutes=float((np.abs(true_flat[:,c]-pred_flat[:,c])*minutes).mean())))
                    proposed=result['test']['posterior_selected'];base=result['test']['cosdir'];comp=result['test']['competitor_selected'];corrected=result['test']['competitor_corrected']
                    seed_rows.append({'seed':seed,'choice':result['choices']['posterior'],'competitor':result['choices']['competitor'],
                                      'joint_vs_base':proposed['raw_accuracy']>base['raw_accuracy'] and proposed['mse']<=base['mse']+1e-8,
                                      'corrected_competitor':repaired['choices']['competitor_corrected'],
                                      'joint_vs_original_competitor':proposed['raw_accuracy']>comp['raw_accuracy'] and proposed['mse']<=comp['mse']+1e-8,
                                      'joint_vs_competitor':proposed['raw_accuracy']>corrected['raw_accuracy'] and proposed['mse']<=corrected['mse']+1e-8})
                    groups={'all':list(range(len(meta['columns'])))}
                    groups.update({name:[i] for i,name in enumerate(meta['columns'])})
                    if domain=='nem':groups.update(demand=list(range(5)),price=list(range(5,10)))
                    for method in domain_methods:
                        z=ar['test__'+method];pred=classes(x,z)
                        for label,channels in groups.items():
                            yy=truth[:,:,channels];zz=pred[:,:,channels]
                            zero=yy==1;nonzero=~zero
                            details.append(dict(domain=domain,backbone=bone,seed=seed,method=method,group=label,
                                           raw_accuracy=float((yy==zz).mean()),mse=float(((y[:,:,channels]-z[:,:,channels])**2).mean()),
                                           zero_fraction=float(zero.mean()),predicted_zero_fraction=float((zz==1).mean()),
                                           zero_recall=float((yy==zz)[zero].mean()) if zero.any() else None,
                                           nonzero_accuracy=float((yy==zz)[nonzero].mean()) if nonzero.any() else None))
                diag=json.loads((case/'decoder_diagnostics.json').read_text())
                d=diag['test'];chosen=lock['choices']['posterior']
                costs.append(dict(domain=domain,backbone=bone,seed=seed,
                                 training_seconds=sum(t['seconds'] for t in result['training']),
                                 p_plus_mean_parameters=result['p_plus_mean_parameters'],
                                 residual_capacity_parameters=result['residual_capacity_parameters'],
                                 selected_decoder=chosen,selected_diagnostics=d.get(chosen) if chosen else None,
                                 grid_diagnostics=d['grid']))
            averaged={m:np.mean(v,0) for m,v in paired.items()}
            origin_averages[domain+'/'+bone]=averaged;origin_days[domain]=days
            entry={'seeds':seed_rows,'methods':{}}
            for method,values in averaged.items():
                row={'accuracy':float(values[:,0].mean()),'mse':float(values[:,1].mean())}
                refs=['cosdir','competitor_selected','competitor_corrected']
                if method=='posterior_selected':refs+=['top1_gap_selected','greedy_selected','mean_raw','mean_only_grid','training_prior_selected','residual_direction_capacity']
                if method=='posterior_selected' and domain in ('carpark','finland'):refs+=native_controls
                for ref in refs:
                    reference=averaged[ref];delta=values-reference
                    row['vs_'+ref]={'accuracy_pp':float(100*delta[:,0].mean()),
                                   'mse_percent':float(100*delta[:,1].mean()/reference[:,1].mean()),
                                   'joint_improvement':bool(delta[:,0].mean()>0 and delta[:,1].mean()<=1e-8),
                                   'one_day_bootstrap':intervals(days,delta,reference[:,1],1),
                                   'three_day_bootstrap':intervals(days,delta,reference[:,1],3)}
                entry['methods'][method]=row
            aggregate[domain+'/'+bone]=entry
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_case_methods.csv',index=False)
    pd.DataFrame(details).to_csv(out/'per_channel_methods.csv',index=False)
    pd.DataFrame(strata).to_csv(out/'phase_amplitude_methods.csv',index=False)
    pd.DataFrame(applications).to_csv(out/'net_change_plateau_methods.csv',index=False)
    write(out/'train_amplitude_thresholds.json',thresholds)
    averages=frame.groupby(['domain','backbone','method']).mean(numeric_only=True).reset_index()
    averages.to_csv(out/'mean_methods.csv',index=False)
    write(out/'paired_bootstrap.json',aggregate);write(out/'costs.json',costs)
    shared_training_seconds=sum(t['seconds'] for p in root.glob('*/shared_seed*/result.json')
                                for t in json.loads(p.read_text())['training'])
    cost_summary={'matrix_wall_seconds':completion['seconds'],
                  'recorded_training_seconds':sum(c['training_seconds'] for c in costs)+shared_training_seconds,
                  'peak_cuda_allocated_bytes':completion['peak_cuda_allocated_bytes'],'domains':{}}
    for domain in ('nem','carpark','finland'):
        cs=[c for c in costs if c['domain']==domain]
        state_counts=np.concatenate([np.asarray(c['grid_diagnostics']['grid_state_counts']).ravel() for c in cs])
        coarsened=np.concatenate([np.asarray(c['grid_diagnostics']['coarsened']).ravel() for c in cs])
        elapsed=[c['selected_diagnostics']['elapsed_seconds'] for c in cs if c['selected_diagnostics']]
        cost_summary['domains'][domain]={'test_states_min':int(state_counts.min()),'test_states_max':int(state_counts.max()),
            'coarsened_sample_channel_fraction':float(coarsened.mean()),
            'selected_test_decoder_batch_seconds':elapsed,
            'p_plus_mean_parameters':cs[0]['p_plus_mean_parameters'],
            'residual_capacity_parameters':cs[0]['residual_capacity_parameters']}
    write(out/'cost_summary.json',cost_summary)
    # Fully expose the competing methods and their validation Pareto set.
    frontiers={}
    for key,lock in locks.items():
        candidates=lock['selection']['competitor']['candidates']
        frontiers[key]=[r for r in candidates if not any(s['mse']<=r['mse'] and s['raw_accuracy']>=r['raw_accuracy']
                         and (s['mse']<r['mse'] or s['raw_accuracy']>r['raw_accuracy']) for s in candidates)]
    write(out/'validation_competitor_frontiers.json',frontiers)
    corrected_frontiers={}
    for key,lock in repair_locks.items():
        candidates=lock['selection']['candidates']
        assert len(candidates)==13
        assert all(r['key'] not in ('carets3_phase','carets4_phase') for r in candidates)
        corrected_frontiers[key]=[r for r in candidates if not any(s['mse']<=r['mse'] and s['raw_accuracy']>=r['raw_accuracy']
                                  and (s['mse']<r['mse'] or s['raw_accuracy']>r['raw_accuracy']) for s in candidates)]
    write(out/'validation_competitor_corrected_frontiers.json',corrected_frontiers)
    write(out/'input_artifact_manifest.json',inputs)
    pooled={}
    for domain in ('nem','carpark','finland'):
        values={m:np.mean([origin_averages[domain+'/'+b][m] for b in ('dlinear','itransformer')],0) for m in methods}
        proposed=values['posterior_selected'];pooled[domain]={}
        for ref in ('cosdir','competitor_selected','competitor_corrected','top1_gap_selected','greedy_selected','mean_raw','mean_only_grid'):
            delta=proposed-values[ref]
            pooled[domain]['vs_'+ref]={'accuracy_pp':float(100*delta[:,0].mean()),
                                      'mse_percent':float(100*delta[:,1].mean()/values[ref][:,1].mean()),
                                      'joint_improvement':bool(delta[:,0].mean()>0 and delta[:,1].mean()<=1e-8),
                                      'one_day_bootstrap':intervals(origin_days[domain],delta,values[ref][:,1],1),
                                      'three_day_bootstrap':intervals(origin_days[domain],delta,values[ref][:,1],3)}
    write(out/'pooled_task_bootstrap.json',pooled)
    summary={'case_metric_checks':len(checks),'metric_failures':checks.count(False),
             'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'fits':sum(len(d['training']) for d in results.values())+27,'cases':len(results),
             'joint_cases_vs_base':sum(s['joint_vs_base'] for a in aggregate.values() for s in a['seeds']),
             'joint_cases_vs_original_competitor':sum(s['joint_vs_original_competitor'] for a in aggregate.values() for s in a['seeds']),
             'joint_cases_vs_competitor':sum(s['joint_vs_competitor'] for a in aggregate.values() for s in a['seeds']),
             'posterior_fallback_cases':sum(s['choice'] is None for a in aggregate.values() for s in a['seeds']),
             'task_joint_pooled':{domain:pooled[domain]['vs_competitor_corrected']['joint_improvement'] for domain in pooled},
             'task_joint_on_both_backbones':{domain:all(aggregate[domain+'/'+b]['methods']['posterior_selected']['vs_competitor_corrected']['joint_improvement'] for b in ('dlinear','itransformer')) for domain in ('nem','carpark','finland')},
             'interpretation':'descriptive test comparisons, no test-dependent selection or parameter changes'}
    assert summary['metric_failures']==0
    write(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=False)
    visible=['cosdir','top1_gap_selected','mean_only_grid','mean_raw','residual_direction_capacity','competitor_corrected','cosdir_rounded','greedy_selected','posterior_selected']
    names=dict(cosdir='CosDir',posterior_selected='Full posterior DP',top1_gap_selected='Top1-gap DP',
               mean_only_grid='Mean grid',mean_raw='Mean raw',residual_direction_capacity='Direction residual',
               competitor_corrected='Selected competitor',cosdir_rounded='CosDir rounding',greedy_selected='Greedy')
    colors={m:plt.cm.tab10(i) for i,m in enumerate(visible)}
    for ax,domain in zip(axes,('nem','carpark','finland')):
        for bi,bone in enumerate(('dlinear','itransformer')):
            rows=aggregate[domain+'/'+bone]['methods']
            for method in visible:
                row=rows[method]['vs_cosdir'];marker='o' if bi==0 else '^'
                ax.scatter(row['mse_percent'],row['accuracy_pp'],marker=marker,color=colors[method],s=38,
                           edgecolors='white',linewidths=.4,zorder=4 if method=='posterior_selected' else 2)
                if method=='posterior_selected':ax.annotate('DP '+bone,(row['mse_percent'],row['accuracy_pp']),fontsize=7,xytext=(3,-11),textcoords='offset points')
        ax.axhline(0,color='grey',linewidth=.7);ax.axvline(0,color='grey',linewidth=.7);ax.set_title(domain)
        ax.set_xlabel('MSE change vs CosDir (%)');ax.set_ylabel('Raw accuracy change (pp)')
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',linestyle='',color=colors[m],label=names[m]) for m in visible]
    handles+=[Line2D([0],[0],marker='o',linestyle='',color='black',label='DLinear'),
              Line2D([0],[0],marker='^',linestyle='',color='black',label='iTransformer')]
    fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=7)
    fig.tight_layout(rect=(0,.22,1,1));fig.savefig(out/'direction_mse_tradeoff.png',dpi=180);plt.close(fig)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}
    write(out/'analysis_manifest.json',manifest)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
