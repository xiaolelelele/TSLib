"""Independent NumPy re-score and selector/lock audit for saved pilot cases.

Uses no trajectory_run metrics/selection implementation. Does not retrain or
claim to independently verify GPU kernels or the entire finite-grid optimum.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def classes(x,z):
    previous=np.concatenate((np.asarray(x,dtype=np.float64)[:,-1:],np.asarray(z,dtype=np.float64)[:,:-1]),axis=1)
    return np.sign(np.asarray(z,dtype=np.float64)-previous).astype(int)+1


def metrics(x,y,z):
    y,z=np.asarray(y,dtype=np.float64),np.asarray(z,dtype=np.float64)
    truth,pred=classes(x,y),classes(x,z)
    matrix=np.zeros((3,3),dtype=np.int64)
    for t in range(3):
        for p in range(3):matrix[t,p]=np.count_nonzero((truth==t)&(pred==p))
    correct=np.diag(matrix);actual=matrix.sum(1);predicted=matrix.sum(0)
    f1=np.array([2*correct[k]/(actual[k]+predicted[k]) if actual[k]+predicted[k] else 0. for k in range(3)])
    return {'mse':np.square(z-y).mean(),'mae':np.abs(z-y).mean(),
            'raw_accuracy':(truth==pred).mean(),'macro_f1':f1.mean(),
            'precision':[correct[k]/predicted[k] if predicted[k] else None for k in range(3)],
            'recall':[correct[k]/actual[k] if actual[k] else None for k in range(3)],
            'confusion':matrix.tolist(),'true_zero_fraction':(truth==1).mean(),
            'predicted_zero_fraction':(pred==1).mean(),
            'nonzero_accuracy':(truth==pred)[truth!=1].mean() if np.any(truth!=1) else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint-audit',action='store_true')
    parser.add_argument('--expected-cases',type=int)
    args=parser.parse_args();run=Path(args.run);checks=0;case_rows=[]
    reconstructed={};shared_audited=set();cpu_rows=[];dtype_label_rows=[]
    frozen=run.parent/'source/v2'
    if args.checkpoint_audit:
        import torch
        torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        sys.path.insert(0,str(frozen if frozen.exists() else Path(__file__).resolve().parents[3]))
        from trend_lab.models import DirectionHead,ResidualAdapter
        from trend_lab.recent_methods import backbone
        from trend_lab.trajectory_baselines import OCETimeSeries,CaReTSPhaseTransformer
    def same(actual,expected,name):
        nonlocal checks
        checks+=1
        if actual is None or expected is None:
            if actual is not None or expected is not None:raise AssertionError(name)
        elif isinstance(actual,(dict,str,bool)) or (isinstance(actual,list) and actual and isinstance(actual[0],str)):
            if actual!=expected:raise AssertionError(name)
        else:np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12,err_msg=name)
    def check_score(x,y,z,record,label):
        for key,value in metrics(x,y,z).items():
            if key in ('precision','recall'):
                for k,v in enumerate(value):same(v,record[key][k],label+'/'+key+'/'+str(k))
            else:same(value,record[key],label+'/'+key)
    def data_reconstruction(metadata):
        key=metadata['csv']
        if key in reconstructed:return reconstructed[key]
        frame=pd.read_csv(key);time=pd.to_datetime(frame.pop('date'));raw=frame.to_numpy(dtype=np.float64)
        cuts=metadata['cuts'];spec=metadata['spec'];train=raw[:cuts[1]]
        mean=np.nanmean(train,0);std=np.nanstd(train,0);values=(raw-mean)/std
        same(mean,metadata['mean'],'training mean');same(std,metadata['std'],'training std')
        same(np.nanmin(values[:cuts[1]],axis=0),metadata['train_min'],'training minimum')
        same(np.nanmax(values[:cuts[1]],axis=0),metadata['train_max'],'training maximum')
        same(frame.columns.tolist(),metadata['columns'],'fixed channels')
        computed_cuts=[int(time.searchsorted(pd.Timestamp(v))) for v in spec['boundaries']]
        same(computed_cuts,cuts,'timestamp cuts')
        if not np.all(np.diff(time.astype('int64'))==pd.Timedelta(spec['frequency']).value):raise AssertionError('compressed/misaligned time axis')
        nonlocal checks
        checks+=1
        historical=values.copy()
        if spec.get('history_policy','complete')=='causal_ffill':
            # Independent forward-only scalar loop, preserving leading NaN.
            for c in range(historical.shape[1]):
                last=np.nan
                for t in range(len(historical)):
                    if np.isfinite(historical[t,c]):last=historical[t,c]
                    else:historical[t,c]=last
        valid=np.isfinite(values).all(1);history_valid=np.isfinite(historical).all(1)
        L,H=metadata['lookback'],metadata['horizon'];splits={}
        for i,name in enumerate(('train','earlystop','calibration','validation','test')):
            start,end=cuts[i:i+2];stride=spec['train_stride'] if name=='train' else spec['evaluation_stride']
            eligible=[o for o in range(0,len(raw),stride) if o>=L-1 and o+1>=start and o+H+1<=end]
            origins=np.array([o for o in eligible if valid[o] and valid[o+1:o+H+1].all() and history_valid[o-L+1:o+1].all()],dtype=np.int64)
            same(len(eligible),metadata['splits'][name]['eligible'],name+' eligible')
            same(len(origins),metadata['splits'][name]['kept'],name+' kept')
            same(len(eligible)-len(origins),metadata['splits'][name]['discarded'],name+' dropped')
            x=np.stack([historical[o-L+1:o+1] for o in origins]);y=np.stack([values[o+1:o+H+1] for o in origins])
            # Encoder/loss inputs are float32, while saved raw evaluation uses
            # original float64 observations. Count any label change separately
            # for every split rather than assuming train agreement generalizes.
            truth64=classes(x,y)
            truth32=classes(x.astype(np.float32),y.astype(np.float32))
            dtype_label_rows.append({'csv':key,'split':name,
                'labels':int(truth64.size),
                'float32_float64_labels_differ':int(np.count_nonzero(truth32!=truth64)),
                'carets_relative_anchor_float32_float64_labels_differ':int(np.count_nonzero(
                    (y<x[:,-1:])!=(y.astype(np.float32)<x[:,-1:].astype(np.float32))))})
            if 'filled_history_fraction' in metadata['splits'][name]:
                fraction=np.mean(np.stack([~np.isfinite(values[o-L+1:o+1]) for o in origins]))
                same(fraction,metadata['splits'][name]['filled_history_fraction'],name+' historical fill fraction')
            if np.any(origins+1<start) or np.any(origins+H>=end):raise AssertionError('future targets across split boundary')
            checks+=1
            same(str(time.iloc[origins[0]]),metadata['splits'][name]['first_origin'],name+' first origin')
            same(str(time.iloc[origins[-1]]),metadata['splits'][name]['last_origin'],name+' last origin')
            splits[name]=(x,y,origins)
        reconstructed[key]=splits
        return splits
    for lock_path in sorted(run.glob('*/*/validation_lock.json')):
        case=lock_path.parent
        if not (case/'result.json').exists():continue
        lock=json.loads(lock_path.read_text());result=json.loads((case/'result.json').read_text())
        with np.load(case/'predictions.npz') as archive:arrays={k:archive[k] for k in archive.files}
        if args.checkpoint_audit:
            rebuilt=data_reconstruction(lock['metadata'])
            for split in ('validation','test'):
                for part,actual in zip(('x','y','origins'),rebuilt[split]):
                    # Smokes deliberately use a prefix of the full windows.
                    saved=arrays[split+'__'+part]
                    expected=actual[:len(saved)] if lock['config'].get('smoke') else actual
                    same(saved,expected,split+' saved '+part)
        x,y=arrays['validation__x'],arrays['validation__y'];base=arrays['validation__cosdir']
        ref=metrics(x,y,base)
        for family,selection in lock['selection'].items():
            check_score(x,y,base,selection['reference'],str(case)+'/'+family+'/reference')
            eligible=[]
            for row in selection['candidates']:
                check_score(x,y,arrays['validation__'+row['key']],row,str(case)+'/'+row['key'])
                expected=row['raw_accuracy']>ref['raw_accuracy'] and row['mse']<=ref['mse']+1e-8
                same(expected,row['eligible'],'eligibility')
                if expected:eligible.append(row)
            chosen=sorted(eligible,key=lambda r:(-r['raw_accuracy'],r['mse'],r['key']))[0]['key'] if eligible else None
            same(chosen,selection['chosen'],'validation chosen')
            same(lock['choices'][family],chosen or ('cosdir' if family=='competitor' else None),'lock chosen')
        tx,ty=arrays['test__x'],arrays['test__y'];tb=arrays['test__cosdir']
        for name,record in result['test'].items():check_score(tx,ty,arrays['test__'+name],record,str(case)+'/'+name)
        for family in ('posterior','top1_gap','mean_is_base','training_prior','greedy'):
            choice=lock['choices'][family]
            same(arrays['test__'+family+'_selected'],tb if choice is None else arrays['test__'+choice],family+' selected actual path')
        same(arrays['test__competitor_selected'],arrays['test__'+lock['choices']['competitor']],'competitor selected path')
        same((arrays['test__probabilities'].argmax(-1)==classes(tx,ty)).mean(),result['head_accuracy'],'head accuracy')
        same(result['choices'],lock['choices'],'result choices frozen')
        same(lock['test_labels_used_for_selection'],False,'no test labels in selection')
        if lock_path.stat().st_mtime_ns>(case/'checkpoint.pt').stat().st_mtime_ns or (case/'checkpoint.pt').stat().st_mtime_ns>(case/'predictions.npz').stat().st_mtime_ns:
            raise AssertionError('case artifacts saved before locking/checkpoint')
        checks+=1
        for row in result['training']:
            if not 1<=row['best_epoch']<=row['executed_epochs']<=lock['config']['epochs'] or not np.isfinite(row['earlystop_loss']):
                raise AssertionError('invalid training budget/checkpoint epoch')
            checks+=1
        for row in result['reload_checks']:
            if not row['max_abs_error']<=2e-5:raise AssertionError('reload tolerance')
            checks+=1
        shared=case.parent/('shared_seed'+str(lock['seed']))
        shared_result=json.loads((shared/'result.json').read_text())
        authorized=Path(shared_result['test_authorized_by_lock'])
        if not authorized.exists():raise AssertionError('shared test missing authorized lock')
        checks+=1
        if authorized.stat().st_mtime_ns>(shared/'predictions.npz').stat().st_mtime_ns:raise AssertionError('shared predictions precede lock')
        checks+=1
        metadata=lock['metadata'];b=Path(metadata['csv']).read_bytes()
        same(hashlib.sha256(b).hexdigest(),metadata['csv_sha256'],'data hash')
        if metadata['spec']['grid_mode']=='native':
            lower=-np.array(metadata['mean'])/np.array(metadata['std'])
            for name in result['test']:
                if np.any(arrays['test__'+name]<lower):raise AssertionError('negative physical prediction')
                checks+=1
        if args.checkpoint_audit:
            saved=torch.load(case/'checkpoint.pt',weights_only=True,map_location='cpu')
            same(saved['spec'],lock,'disk checkpoint lock metadata')
            dims=(metadata['lookback'],metadata['horizon'],len(metadata['columns']))
            anchor=backbone(lock['backbone'],*dims,lock['config']['reference_root']).eval()
            anchor.load_state_dict(saved['states']['cosdir'])
            probe=torch.tensor(tx[:11],dtype=torch.float32)
            head=None;errors={}
            with torch.no_grad():
                for method,state in saved['states'].items():
                    if method in ('mse_base','cosdir','tdalign'):
                        model=backbone(lock['backbone'],*dims,lock['config']['reference_root']).eval()
                        model.load_state_dict(state);actual=model(probe).numpy();expected=arrays['test__neural_raw_'+method][:11]
                    elif method=='raw_head':
                        model=DirectionHead(*dims,[1],hidden=32).eval();model.load_state_dict(state);head=model
                        actual=(model(probe,anchor(probe))[:,0]/lock['temperature']).softmax(-1).numpy();expected=arrays['test__probabilities'][:11]
                    else:
                        model=ResidualAdapter(*dims,hidden=lock['mean_width'] if method=='mean_estimator' else lock['capacity_width']).eval()
                        model.load_state_dict(state);actual=model(probe,anchor(probe)).numpy()
                        expected=(arrays['test__mean'] if method=='mean_estimator' else arrays['test__neural_raw_'+method])[:11]
                        if method=='mean_estimator' and metadata['spec']['grid_mode']=='native':
                            actual=np.maximum(actual,-np.array(metadata['mean'])/np.array(metadata['std']))
                    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5,err_msg=method+' independent CPU checkpoint inference')
                    checks+=1;errors[method]=float(np.max(np.abs(actual-expected)))
                # Independently choose temperature from CPU checkpoint logits
                # on the separately reconstructed calibration segment.
                cx,cy,_=rebuilt['calibration']
                if lock['config'].get('smoke'):cx,cy=cx[:12],cy[:12]
                cxt=torch.tensor(cx,dtype=torch.float32)
                logits=head(cxt,anchor(cxt))[:,0].numpy().astype(np.float64)
                labels=classes(cxt.numpy(),torch.tensor(cy,dtype=torch.float32).numpy())
                nll=[]
                for temperature in (.5,1.,2.,4.):
                    values=logits/temperature;mx=values.max(-1,keepdims=True)
                    lognormal=mx[...,0]+np.log(np.exp(values-mx).sum(-1))
                    nll.append(float(np.mean(lognormal-np.take_along_axis(values,labels[...,None],axis=-1)[...,0])))
                same((.5,1.,2.,4.)[int(np.argmin(nll))],lock['temperature'],'train-earlier calibration temperature')
            cpu_rows.append({'case':str(case.resolve()),'max_abs_errors':errors,'calibration_nll':nll})
            if str(shared) not in shared_audited:
                shared_audited.add(str(shared));shared_state=torch.load(shared/'checkpoint.pt',weights_only=True,map_location='cpu')
                same(shared_state['metadata'],metadata,'shared checkpoint metadata')
                for method,state in shared_state['states'].items():
                    if method=='oce_fixed_support':
                        same(state['lower'].numpy(),metadata['train_min'],'OCE train-only lower support')
                        same(state['upper'].numpy(),metadata['train_max'],'OCE train-only upper support')
                        model=OCETimeSeries(*dims,metadata['train_min'],metadata['train_max'],bins=45,sigma=.015)
                    else:model=CaReTSPhaseTransformer(*dims,variant=int(method[6]))
                    model.load_state_dict(state);model.eval()
                    with torch.no_grad():actual=model(probe).numpy()
                    if metadata['spec']['grid_mode']=='native':actual=np.maximum(actual,-np.array(metadata['mean'])/np.array(metadata['std']))
                    np.testing.assert_allclose(actual,arrays['test__'+method][:11],atol=2e-5,rtol=2e-5)
                    checks+=1
        case_rows.append({'case':str(case.resolve()),'methods_rescored':len(result['test']),'validation_families':len(lock['selection'])})
    if frozen.exists() and (run.parent/'source_manifest_v2.json').exists():
        manifest=json.loads((run.parent/'source_manifest_v2.json').read_text())
        for name,digest in manifest['files'].items():same(hashlib.sha256((frozen/name).read_bytes()).hexdigest(),digest,'frozen source '+name)
        for name,digest in manifest['configs'].items():same(hashlib.sha256((run.parent/'configs'/name).read_bytes()).hexdigest(),digest,'frozen config '+name)
    if args.expected_cases is not None and len(case_rows)!=args.expected_cases:
        raise AssertionError('Expected %d complete cases, audited %d'%(args.expected_cases,len(case_rows)))
    output={'run':str(run.resolve()),'checks':checks,'failures':0,'cases':case_rows,
            'independent_cpu_checkpoint_checks':cpu_rows,
            'observed_label_dtype_audit':dtype_label_rows,
            'scope':'Independent NumPy metric, complete saved validation selection, final selected-path fallback, lock order, data/source/config hashes; optional independent train-stat/split-window reconstruction, checkpoint CPU inference and temperature selection. No retraining or independent full DP solver.'}
    Path(args.output).write_text(json.dumps(output,indent=2));print(json.dumps({'checks':checks,'failures':0,'cases':len(case_rows)}))


if __name__=='__main__':main()
