"""Repair CaReTS observed-anchor precision without changing trained states.

This is a post-protocol numerical repair, triggered by the formula's known
last-observation anchor and float32 cancellation, not a tuned method variant.
Signed deviation is computed directly from learned float32 branches; it is
never recovered by subtracting the rounded float32 forecast from history.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from audit_runner_outputs import classes,metrics


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))


def bounded(z,meta):
    return np.maximum(z,-np.asarray(meta['mean'])/np.asarray(meta['std'])) if meta['spec']['grid_mode']=='native' else z


def select(x,y,paths,base):
    ref=metrics(x,y,base);rows=[]
    for key,z in paths.items():
        row={'key':key,**metrics(x,y,z)}
        row['eligible']=bool(row['raw_accuracy']>ref['raw_accuracy'] and row['mse']<=ref['mse']+1e-8)
        rows.append(row)
    eligible=[r for r in rows if r['eligible']]
    chosen=min(eligible,key=lambda r:(-r['raw_accuracy'],r['mse'],r['key']))['key'] if eligible else 'cosdir'
    return chosen,{'reference':ref,'candidates':rows,'chosen':chosen,'rule':'raw > same bounded CosDir and MSE <= CosDir + 1e-8; else CosDir'}


def branch_signed(model,x32):
    h=model.encoder(x32.transpose(1,2)).squeeze(-1)
    logits=model.classifier(h).reshape(len(x32),model.horizon,model.channels,2)
    probability=logits.softmax(-1)
    if model.variant==3:
        deviation=F.relu(model.regressor(h)).reshape_as(probability)
        signed=probability[...,0]*deviation[...,0]-probability[...,1]*deviation[...,1]
    else:
        deviation=None
        signed=model.regressor(torch.cat((h,probability.flatten(1)),dim=1)).reshape(len(x32),model.horizon,model.channels)
    return signed,probability,deviation


def infer(model,x,original_gpu,meta):
    traces={'signed32':[],'probability32':[],'raw32':[],'unbounded_anchor64':[]}
    if model.variant==3:traces['deviation32']=[]
    with torch.no_grad():
        for start in range(0,len(x),32):
            actual_x=x[start:start+32];x32=torch.tensor(actual_x,dtype=torch.float32)
            signed,p,d=branch_signed(model,x32)
            raw32=x32[:,-1:,:]+signed
            torch.testing.assert_close(model(x32),raw32,atol=0,rtol=0)
            signed_numpy=signed.numpy()
            if model.variant==3:
                independent=p.numpy()[...,0]*d.numpy()[...,0]-p.numpy()[...,1]*d.numpy()[...,1]
                np.testing.assert_allclose(independent,signed_numpy,atol=2e-7,rtol=2e-6)
                traces['deviation32'].append(d.numpy())
            traces['signed32'].append(signed_numpy);traces['probability32'].append(p.numpy());traces['raw32'].append(raw32.numpy())
            traces['unbounded_anchor64'].append(actual_x[:,-1:,:]+signed_numpy.astype(np.float64))
    traces={k:np.concatenate(v) for k,v in traces.items()}
    np.testing.assert_allclose(traces['raw32'],original_gpu,atol=2e-5,rtol=2e-5,
                               err_msg='full CPU original formula vs saved GPU output')
    corrected=bounded(traces['unbounded_anchor64'],meta)
    original=bounded(original_gpu.astype(np.float64),meta)
    return corrected,traces,{'cpu_original_formula_max_abs_error':float(np.max(np.abs(traces['raw32']-original_gpu))),
                            'anchor64_numpy_formula_max_abs_error':float(np.max(np.abs(traces['unbounded_anchor64']-(x[:,-1:,:]+traces['signed32'].astype(np.float64))))),
                            'max_actual_path_change':float(np.max(np.abs(corrected-original))),
                            'first_step_symbols_changed':int(np.count_nonzero(classes(x,corrected)[:,0]!=classes(x,original)[:,0])),
                            'later_step_symbols_changed':int(np.count_nonzero(classes(x,corrected)[:,1:]!=classes(x,original)[:,1:])),
                            'nonzero_signed_lost_by_old_addition':int(np.count_nonzero((traces['signed32']!=0)&(traces['raw32']==x[:,-1:,:].astype(np.float32))))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--dataset');args=parser.parse_args();run=Path(args.run).resolve();out=Path(args.output).resolve()
    frozen=run.parent/'source/v2';sys.path.insert(0,str(frozen))
    from trend_lab.trajectory_baselines import CaReTSPhaseTransformer
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    protocol=out/'repair_protocol.json'
    if not protocol.exists():write(protocol,{'client_date':'2026-09-30','declared_utc':datetime.now(timezone.utc).isoformat(),
              'reason':'CaReTS formula anchors signed deviation to actual last observation; float32 anchor addition can fabricate first-step raw direction and swallow a small nonzero deviation.',
              'formula':'signed32 is computed from unchanged learned branches; path64 = observed xlast64 + signed32.double; apply same already declared physical bounds.',
              'test_selection':'Original case model/method choices remain fixed. Competitor corrected uses the identical original validation selector on complete original competitor library plus repaired CaReTS3/4.',
              'scope':'post-protocol numerical audit repair, no retraining, no test tuning, not bitwise GPU reproduction',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    completed=[];summary=[]
    for shared in sorted(run.glob('*/shared_seed*')):
        if args.dataset and shared.parent.name!=args.dataset:continue
        if not (shared/'result.json').exists():continue
        seed=int(shared.name.split('seed')[1]);cases=[p for p in sorted(shared.parent.glob('*_seed'+str(seed)))
                    if not p.name.startswith('shared') and (p/'result.json').exists() and (p/'validation_lock.json').exists()]
        if not cases:continue
        original_lock=json.loads((cases[0]/'validation_lock.json').read_text());meta=original_lock['metadata']
        saved=torch.load(shared/'checkpoint.pt',map_location='cpu',weights_only=True)
        with np.load(shared/'predictions.npz') as a:original_shared={k:a[k].copy() for k in a.files}
        with np.load(cases[0]/'predictions.npz') as a:common={k:a[k].copy() for k in a.files}
        models={};dims=(meta['lookback'],meta['horizon'],len(meta['columns']))
        for variant in (3,4):
            name='carets%d_phase'%variant;model=CaReTSPhaseTransformer(*dims,variant=variant).eval()
            model.load_state_dict(saved['states'][name]);models[name]=model
        shared_out=out/shared.parent.name/shared.name;shared_out.mkdir(parents=True,exist_ok=True)
        val_paths={};all_trace={};shared_checks={};reload_checks={}
        for name,model in models.items():
            key=name+'_anchor64';corrected,trace,checks=infer(model,common['validation__x'],original_shared['validation__'+name],meta)
            val_paths[key]=corrected;shared_checks['validation__'+key]=checks
            all_trace.update({'validation__'+name+'__'+k:v for k,v in trace.items()})
            # Reload from actual disk into a separate model and compare the
            # learned branch deviation directly, bypassing float32 addition.
            reloaded=CaReTSPhaseTransformer(*dims,variant=model.variant).eval()
            disk=torch.load(shared/'checkpoint.pt',map_location='cpu',weights_only=True)
            reloaded.load_state_dict(disk['states'][name])
            probe=common['validation__x'][:11]
            with torch.no_grad():signed,_,_=branch_signed(reloaded,torch.tensor(probe,dtype=torch.float32))
            path=bounded(probe[:,-1:,:]+signed.numpy().astype(np.float64),meta)
            np.testing.assert_allclose(path,corrected[:11],atol=2e-5,rtol=2e-5)
            reload_checks[name]={'max_abs_error':float(np.max(np.abs(path-corrected[:11]))),'tolerance':2e-5}
        shared_lock={'client_date':'2026-09-30','metadata':meta,'original_shared_checkpoint':str(shared/'checkpoint.pt'),
                     'original_shared_checkpoint_sha256':hashlib.sha256((shared/'checkpoint.pt').read_bytes()).hexdigest(),
                     'protocol':str(protocol),'validation_checks':shared_checks,'reload_checks':reload_checks,
                     'method_choices_changed':False}
        write(shared_out/'repair_lock.json',shared_lock)
        selections={};case_data={}
        for case in cases:
            lock=json.loads((case/'validation_lock.json').read_text())
            with np.load(case/'predictions.npz') as a:arrays={k:a[k].copy() for k in a.files}
            for part in ('x','y','origins'):
                np.testing.assert_array_equal(arrays['validation__'+part],common['validation__'+part])
                np.testing.assert_array_equal(arrays['test__'+part],common['test__'+part])
            original_keys=[r['key'] for r in lock['selection']['competitor']['candidates']]
            library={key:arrays['validation__'+key] for key in original_keys};library.update(val_paths)
            chosen,table=select(arrays['validation__x'],arrays['validation__y'],library,arrays['validation__cosdir'])
            repair_lock={'original_case_lock':str(case/'validation_lock.json'),
                         'original_case_lock_sha256':hashlib.sha256((case/'validation_lock.json').read_bytes()).hexdigest(),
                         'original_choices':lock['choices'],'choices':{'competitor_corrected':chosen},'selection':table,
                         'original_competitor_keys':original_keys,'new_fixed_keys':list(val_paths),
                         'selection_data':'original validation only','protocol':str(protocol),
                         'test_labels_used_for_selection':False,'retraining':False}
            case_out=out/shared.parent.name/case.name;write(case_out/'repair_lock.json',repair_lock)
            selections[case.name]=(chosen,table,case_out,repair_lock);case_data[case.name]=arrays
        # Every repaired competitor selection for this available shared seed
        # is persisted before any repaired test path is generated or scored.
        assert all((value[2]/'repair_lock.json').exists() for value in selections.values())
        assert all((case/'validation_lock.json').exists() for case in cases)
        test_paths={}
        for name,model in models.items():
            key=name+'_anchor64';corrected,trace,checks=infer(model,common['test__x'],original_shared['test__'+name],meta)
            test_paths[key]=corrected;shared_checks['test__'+key]=checks
            all_trace.update({'test__'+name+'__'+k:v for k,v in trace.items()})
        np.savez_compressed(shared_out/'signed_trace.npz',**all_trace)
        write(shared_out/'result.json',{'checks':shared_checks,'reload_checks':reload_checks,
                 'test_authorized_by_repair_locks':[str(v[2]/'repair_lock.json') for v in selections.values()],
                 'seconds':time.perf_counter()-started})
        for name,(chosen,table,case_out,repair_lock) in selections.items():
            arrays=case_data[name];paths={}
            for split,corrected in (('validation',val_paths),('test',test_paths)):
                paths.update({split+'__'+k:v for k,v in corrected.items()})
                paths[split+'__competitor_corrected']=corrected[chosen] if chosen in corrected else arrays[split+'__'+chosen]
            np.savez_compressed(case_out/'predictions.npz',**paths)
            test_scores={key:metrics(arrays['test__x'],arrays['test__y'],paths['test__'+key])
                         for key in list(test_paths)+['competitor_corrected']}
            result={'test':test_scores,'choices':{'competitor_corrected':chosen},'selection':table,
                    'original_choices':repair_lock['original_choices'],'repair_status':'post-protocol numerical repair',
                    'shared_checks':shared_checks,'reload_checks':reload_checks,'seconds':time.perf_counter()-started,
                    'original_case':str(run/shared.parent.name/name),'shared_trace':str(shared_out/'signed_trace.npz')}
            write(case_out/'result.json',result);completed.append(str(case_out))
            summary.append({'domain':shared.parent.name,'case':name,'competitor_original':repair_lock['original_choices']['competitor'],
                            'competitor_corrected':chosen,
                            'carets3_test_raw_change_pp':float(100*(test_scores['carets3_phase_anchor64']['raw_accuracy']-metrics(arrays['test__x'],arrays['test__y'],arrays['test__carets3_phase'])['raw_accuracy']))})
            print('repaired %s %s: competitor=%s'%(shared.parent.name,name,chosen),flush=True)
    write(out/'completion.json',{'client_date':'2026-09-30','cases':len(completed),'completed':completed,
                'seconds':time.perf_counter()-started,'summary':summary,'all_matrix_cases_available':(run/'completion.json').exists()})


if __name__=='__main__':main()
