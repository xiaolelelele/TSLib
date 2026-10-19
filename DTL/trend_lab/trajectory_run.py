"""Frozen first-stage falsification pilot. Choices are validation-only.

This is supervised evaluation on TIME exports, not the TIME zero-shot track,
not an online vintage replay and not a conditional risk safety guarantee.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from .models import DirectionHead, ResidualAdapter
from .recent_methods import backbone, raw_labels, raw_differences, raw_project
from .baseline_losses import cosdir_loss, tdt_loss
from .losses import direction_surrogate_loss, fit_slope_scales
from .recent_run import fit_model, write_json
from .run import _seed, _params, _temperature, _residual_width
from .trajectory_data import load_trajectory_data
from .trajectory_baselines import OCETimeSeries, CaReTSPhaseTransformer
from .trajectory_decision import (decode_trajectory, greedy_trajectory,
                                  make_common_grids, raw_classes)


def predict(model, x, batch=64):
    # Adapters are callables around a frozen anchor; only modules have eval().
    if isinstance(model, torch.nn.Module):
        model.eval()
    with torch.no_grad():
        return torch.cat([model(x[o:o+batch]) for o in range(0,len(x),batch)])


def shared_prediction(model,tensor,history,batch=32):
    if isinstance(model,CaReTSPhaseTransformer):
        model.eval()
        real_history=torch.as_tensor(history,dtype=torch.float64,device=tensor.device)
        return predict(model.forward_with_true_anchor,real_history,batch)
    return predict(model,tensor,batch)


def score(x,y,z):
    x,y,z=[np.asarray(a,np.float64) for a in (x,y,z)]
    t,p=raw_classes(x,y),raw_classes(x,z)
    confusion=np.bincount((t*3+p).ravel(),minlength=9).reshape(3,3)
    correct=np.diag(confusion).astype(float)
    tp=confusion.sum(1);pp=confusion.sum(0)
    f1=np.divide(2*correct,tp+pp,out=np.zeros(3),where=(tp+pp)>0)
    precision=[float(correct[k]/pp[k]) if pp[k] else None for k in range(3)]
    recall=[float(correct[k]/tp[k]) if tp[k] else None for k in range(3)]
    return {'mse':float(np.mean((z-y)**2)),'mae':float(np.mean(np.abs(z-y))),
            'raw_accuracy':float(np.mean(t==p)),'macro_f1':float(f1.mean()),
            'precision':precision,'recall':recall,'confusion':confusion.tolist(),
            'true_zero_fraction':float(np.mean(t==1)),'predicted_zero_fraction':float(np.mean(p==1)),
            'nonzero_accuracy':float(np.mean((t==p)[t!=1])) if (t!=1).any() else None}


def choose(x,y,paths,b):
    ref=score(x,y,b);rows=[]
    for key,z in paths.items():
        row={'key':key,**score(x,y,z)}
        row['eligible']=row['raw_accuracy']>ref['raw_accuracy'] and row['mse']<=ref['mse']+1e-8
        rows.append(row)
    eligible=[r for r in rows if r['eligible']]
    selected=min(eligible,key=lambda r:(-r['raw_accuracy'],r['mse'],r['key']))['key'] if eligible else None
    return selected,{'reference':ref,'candidates':rows,'chosen':selected}


def bounded(path,meta):
    if meta['spec']['grid_mode']=='native':
        return np.maximum(np.asarray(path,np.float64),(-np.array(meta['mean'])/np.array(meta['std']))[None,None,:])
    return np.asarray(path,np.float64)


def rounded(path,meta):
    z=np.asarray(path,np.float64);mean=np.array(meta['mean']);std=np.array(meta['std'])
    return (np.rint(z*std+mean)-mean)/std


def neural_shared(splits,tensors,meta,cfg,out,seed):
    out.mkdir(parents=True,exist_ok=True)
    dims=(meta['lookback'],meta['horizon'],len(meta['columns']))
    models,states,training,paths={}, {}, [], {}
    for method in ('oce_fixed_support','carets3_phase','carets4_phase'):
        _seed(seed+81001+len(models))
        model=(OCETimeSeries(*dims,meta['train_min'],meta['train_max'],bins=45,sigma=.015)
               if method=='oce_fixed_support' else CaReTSPhaseTransformer(*dims,variant=int(method[6])))
        model=model.to(cfg['device'])
        local=dict(cfg)
        if method.startswith('carets'):local['batch']=32
        row,state=fit_model(method,model,model.loss,lambda x,y:F.mse_loss(model(x),y),tensors,local,seed+81020+len(models))
        models[method]=model;states[method]=state;training.append(row)
    write_json(out/'lock.json',{'metadata':meta,'config':cfg,'seed':seed,'training':training,
                                'adaptations':'OCE train-only affine; phase-preserving CaReTS3/4 with true float64 anchor; not full-paper replications'})
    for split in ('validation',):
        for name,model in models.items():
            paths[split+'__'+name]=shared_prediction(model,tensors[split][0],splits[split].x).cpu().numpy()
    torch.save({'states':states,'metadata':meta},out/'checkpoint.pt')
    reload_rows=[]
    saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
    for name in models:
        model=(OCETimeSeries(*dims,meta['train_min'],meta['train_max'],bins=45,sigma=.015)
               if name=='oce_fixed_support' else CaReTSPhaseTransformer(*dims,variant=int(name[6])))
        model.load_state_dict(saved['states'][name]);model.eval()
        with torch.no_grad():
            actual=(model.forward_with_true_anchor(torch.as_tensor(splits['validation'].x[:11],dtype=torch.float64)) if isinstance(model,CaReTSPhaseTransformer)
                    else model(torch.as_tensor(splits['validation'].x[:11],dtype=torch.float32))).numpy()
        expected=paths['validation__'+name][:11]
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
        reload_rows.append({'method':name,'max_abs_error':float(np.max(np.abs(actual-expected)))})
    np.savez_compressed(out/'predictions.npz',**paths)
    write_json(out/'result.json',{'training':training,'reload_checks':reload_rows,
               'validation':{name:score(splits['validation'].x,splits['validation'].y,bounded(paths['validation__'+name],meta)) for name in models}})
    del models
    if cfg['device'].startswith('cuda'):torch.cuda.empty_cache()
    return paths


def shared_test(splits,tensors,meta,cfg,out,paths,validation_lock):
    """Shared test outputs are created only after an actual case lock exists."""
    assert validation_lock.exists()
    if 'test__oce_fixed_support' in paths:
        return
    dims=(meta['lookback'],meta['horizon'],len(meta['columns']))
    saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
    checks=[]
    for name,state in saved['states'].items():
        model=(OCETimeSeries(*dims,meta['train_min'],meta['train_max'],bins=45,sigma=.015)
               if name=='oce_fixed_support' else CaReTSPhaseTransformer(*dims,variant=int(name[6])))
        model.load_state_dict(state);model=model.to(cfg['device']).eval()
        values=shared_prediction(model,tensors['test'][0],splits['test'].x).cpu().numpy()
        paths['test__'+name]=values
        model=model.cpu()
        with torch.no_grad():
            actual=(model.forward_with_true_anchor(torch.as_tensor(splits['test'].x[:11],dtype=torch.float64)) if isinstance(model,CaReTSPhaseTransformer)
                    else model(torch.as_tensor(splits['test'].x[:11],dtype=torch.float32))).numpy()
        np.testing.assert_allclose(actual,values[:11],atol=2e-5,rtol=2e-5)
        checks.append({'method':name,'max_abs_error':float(np.max(np.abs(actual-values[:11])))})
    np.savez_compressed(out/'predictions.npz',**paths)
    result=json.loads((out/'result.json').read_text())
    result.update(test={name:score(splits['test'].x,splits['test'].y,bounded(paths['test__'+name],meta)) for name in saved['states']},
                  test_reload_checks=checks,test_authorized_by_lock=str(validation_lock.resolve()))
    write_json(out/'result.json',result)


def decisions(x,b,mu,p,prior,meta,cfg,validation=True,choices=None):
    lower=(-np.array(meta['mean'])/np.array(meta['std'])) if meta['spec']['grid_mode']=='native' else None
    grids,grid_info=make_common_grids(x,b,mu,mode=meta['spec']['grid_mode'],step=.1,padding=.25,
              normalizer_mean=meta['mean'],normalizer_std=meta['std'],native_step=1.,native_origin=0.,
              lower=lower,max_states=257,coarsen=True)
    specs={};paths={};diagnostics={'grid':grid_info}
    family_specs={
        'posterior':(mu,p,'posterior',False),
        'top1_gap':(mu,p,'top1_gap',False),
        'mean_is_base':(b,p,'posterior',False),
        'training_prior':(mu,np.broadcast_to(prior,p.shape),'posterior',False),
        'greedy':(mu,p,'posterior',True)}
    for family,(estimate,prob,reward,greedy) in family_specs.items():
        strengths=cfg['lambda_grid'] if validation else sorted(set([float(choices[family].rsplit('_',1)[1])] if choices.get(family) else []))
        for strength in strengths:
            strength=float(strength)
            key=family+'_'+str(strength)
            solver=greedy_trajectory if greedy else decode_trajectory
            paths[key],diagnostics[key]=solver(x,b,estimate,prob,strength,grids,reward=reward)
            specs[key]={'family':family,'strength':strength}
    # Mean-only and raw mean are distinct controls. The finite grid is shared.
    paths['mean_only_grid'],diagnostics['mean_only_grid']=decode_trajectory(x,b,mu,p,0.,grids)
    paths['mean_raw']=mu
    return paths,diagnostics,specs


def run_case(splits,tensors,meta,cfg,out,seed,name,shared,shared_folder):
    out.mkdir(parents=True,exist_ok=True)
    dims=(meta['lookback'],meta['horizon'],len(meta['columns']))
    _seed(seed);initial=backbone(name,*dims,cfg['reference_root']).to(cfg['device'])
    initial_state=copy.deepcopy(initial.state_dict());models={};states={};training=[]
    for method in ('mse_base','cosdir','tdalign'):
        m=backbone(name,*dims,cfg['reference_root']).to(cfg['device']);m.load_state_dict(initial_state)
        loss=(lambda x,y:F.mse_loss(m(x),y)) if method=='mse_base' else ((lambda x,y:cosdir_loss(x,m(x),y,.5)) if method=='cosdir' else (lambda x,y:tdt_loss(x,m(x),y)))
        row,state=fit_model(method,m,loss,lambda x,y:F.mse_loss(m(x),y),tensors,cfg,seed+100003)
        models[method]=m;states[method]=state;training.append(row)
    del initial,initial_state
    anchor=models['cosdir'];anchor.eval().requires_grad_(False)
    _seed(seed+20001);head=DirectionHead(*dims,[1],hidden=32).to(cfg['device'])
    def logits(x):return head(x,anchor(x).detach())[:,0]
    def head_loss(x,y):return F.cross_entropy(logits(x).reshape(-1,3),raw_labels(x,y).reshape(-1))
    row,state=fit_model('raw_head',head,head_loss,head_loss,tensors,cfg,seed+20002)
    states['raw_head']=state;training.append(row)
    width=_residual_width(*dims,_params(head)['total'],32)
    _seed(seed+21001);mean_model=ResidualAdapter(*dims,hidden=width).to(cfg['device'])
    mean_loss=lambda x,y:F.mse_loss(mean_model(x,anchor(x).detach()),y)
    row,state=fit_model('mean_estimator',mean_model,mean_loss,mean_loss,tensors,cfg,seed+21002)
    states['mean_estimator']=state;training.append(row)
    total_target=_params(head)['total']+_params(mean_model)['total']
    big_width=_residual_width(*dims,total_target,32)
    _seed(seed+22001);big=ResidualAdapter(*dims,hidden=big_width).to(cfg['device'])
    big_initial=copy.deepcopy(big.state_dict())
    scale=fit_slope_scales(tensors['train'][0],tensors['train'][1],[1],floor=.02).to(cfg['device'])
    for method in ('residual_capacity','residual_direction_capacity'):
        m=ResidualAdapter(*dims,hidden=big_width).to(cfg['device']);m.load_state_dict(big_initial)
        def adapter_loss(x,y):
            z=m(x,anchor(x).detach());value=F.mse_loss(z,y)
            if method=='residual_direction_capacity':
                slopes=raw_differences(x.double(),z.double())[:,None]
                value=value+.1*direction_surrogate_loss(slopes,raw_labels(x,y)[:,None],torch.zeros_like(scale),scale)
            return value
        monitor=lambda x,y:F.mse_loss(m(x,anchor(x).detach()),y)
        row,state=fit_model(method,m,adapter_loss,monitor,tensors,cfg,seed+22002)
        models[method]=m;states[method]=state;training.append(row)
    del big,big_initial
    with torch.no_grad():
        temperature=_temperature(logits(tensors['calibration'][0]),raw_labels(*tensors['calibration']),[.5,1,2,4])
    tx=splits['train'].x;ty=splits['train'].y
    labels=raw_classes(tx,ty);prior=np.stack([(labels==i).sum((0,1)) for i in range(3)],-1)
    prior=prior/prior.sum(-1,keepdims=True)
    margin=.1*np.median(np.abs(np.diff(tx,axis=1)),axis=(0,1))
    arrays={};model_predictions={};stage_meta={}
    def neural_predictions(split):
        xx=tensors[split][0];npaths={}
        for method,m in models.items():
            values=predict((lambda v:m(v,anchor(v).detach())) if method.startswith('residual_') else m,xx).cpu().numpy()
            npaths[method]=bounded(values,meta)
            arrays[split+'__neural_raw_'+method]=values
        npaths.update({k.split('__',1)[1]:bounded(v,meta) for k,v in shared.items() if k.startswith(split+'__')})
        mu=predict(lambda v:mean_model(v,anchor(v).detach()),xx).cpu().numpy()
        with torch.no_grad():p=(logits(xx)/temperature).softmax(-1).cpu().numpy()
        npaths['mean_raw']=bounded(mu,meta)
        npaths['last_value']=np.repeat(splits[split].x[:,-1:,:],dims[1],axis=1)
        npaths['seasonal_naive']=splits[split].x[:,:dims[1],:]
        npaths['cosdir_rounded']=bounded(rounded(npaths['cosdir'],meta),meta)
        return npaths,bounded(mu,meta),p
    v=splits['validation'];vpaths,mu,p=neural_predictions('validation');b=vpaths['cosdir']
    decoded,diagnostic,dspec=decisions(v.x,b,mu,p,prior,meta,cfg)
    vpaths.update(decoded);choices={};selection={}
    for family in ('posterior','top1_gap','mean_is_base','training_prior','greedy'):
        selected,rows=choose(v.x,v.y,{k:z for k,z in decoded.items() if k.startswith(family+'_')},b)
        choices[family]=selected;selection[family]=rows
    alpha_paths={}
    for alpha in cfg['alpha_grid']:
        alpha=float(alpha)
        key='soft_alpha_'+str(alpha)
        if alpha==0:alpha_paths[key]=b
        else:
            z,diag=raw_project(torch.as_tensor(v.x,dtype=torch.float64,device=cfg['device']),torch.as_tensor(b,device=cfg['device']),
                  torch.as_tensor(p,dtype=torch.float64,device=cfg['device']),torch.as_tensor(margin,device=cfg['device']),alpha)
            if not diag['all_converged']:raise ValueError('soft projection did not converge')
            alpha_paths[key]=bounded(z.cpu().numpy(),meta);diagnostic[key]=diag
    soft,selection['soft_projection']=choose(v.x,v.y,alpha_paths,b);choices['soft_projection']=soft
    vpaths.update(alpha_paths)
    competitor_keys=['mse_base','cosdir','tdalign','residual_capacity','residual_direction_capacity',
                     'oce_fixed_support','carets3_phase','carets4_phase','last_value','seasonal_naive','cosdir_rounded','mean_raw','mean_only_grid']
    competitor,selection['competitor']=choose(v.x,v.y,{k:vpaths[k] for k in competitor_keys},b)
    choices['competitor']=competitor or 'cosdir'
    for key,z in vpaths.items():arrays['validation__'+key]=z
    arrays.update(validation__x=v.x,validation__y=v.y,validation__origins=v.origins,
                  validation__probabilities=p,validation__mean=mu)
    lock={'metadata':meta,'config':cfg,'seed':seed,'backbone':name,'anchor_method':'cosdir',
          'temperature':temperature,'prior':prior.tolist(),'margin':margin.tolist(),'mean_width':width,
          'capacity_width':big_width,'choices':choices,'selection':selection,'training':training,
          'test_labels_used_for_selection':False,'conditional_risk_guarantee':False}
    write_json(out/'validation_lock.json',lock)
    torch.save({'states':states,'spec':lock},out/'checkpoint.pt')
    # Choices persisted above. No test prediction/score in this case until here.
    shared_test(splits,tensors,meta,cfg,shared_folder,shared,out/'validation_lock.json')
    t=splits['test'];tpaths,tmu,tp=neural_predictions('test');tb=tpaths['cosdir']
    td,t_diagnostics,_=decisions(t.x,tb,tmu,tp,prior,meta,cfg,validation=False,choices=choices)
    tpaths.update(td)
    for family in ('posterior','top1_gap','mean_is_base','training_prior','greedy'):
        tpaths[family+'_selected']=tb if choices[family] is None else td[choices[family]]
    if soft is None or soft=='soft_alpha_0.0':tpaths['soft_selected']=tb
    else:
        alpha=float(soft.rsplit('_',1)[1])
        z,diag=raw_project(torch.as_tensor(t.x,dtype=torch.float64,device=cfg['device']),torch.as_tensor(tb,device=cfg['device']),
             torch.as_tensor(tp,dtype=torch.float64,device=cfg['device']),torch.as_tensor(margin,device=cfg['device']),alpha)
        if not diag['all_converged']:raise ValueError('test soft projection did not converge')
        tpaths['soft_selected']=bounded(z.cpu().numpy(),meta);t_diagnostics['soft_selected']=diag
    tpaths['competitor_selected']=tpaths[choices['competitor']]
    for key,z in tpaths.items():arrays['test__'+key]=z
    arrays.update(test__x=t.x,test__y=t.y,test__origins=t.origins,test__probabilities=tp,test__mean=tmu)
    # Independently reload all seven models on CPU using actual saved states.
    saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
    cpu_anchor=backbone(name,*dims,cfg['reference_root']);cpu_anchor.load_state_dict(saved['states']['cosdir']);cpu_anchor.eval()
    xx=torch.as_tensor(t.x[:11],dtype=torch.float32);reload_rows=[]
    with torch.no_grad():
        for method in ('mse_base','cosdir','tdalign','raw_head','mean_estimator','residual_capacity','residual_direction_capacity'):
            if method in ('mse_base','cosdir','tdalign'):
                m=backbone(name,*dims,cfg['reference_root']);expected=arrays['test__neural_raw_'+method][:11]
            elif method=='raw_head':m=DirectionHead(*dims,[1],hidden=32);expected=tp[:11]
            else:m=ResidualAdapter(*dims,hidden=width if method=='mean_estimator' else big_width);expected=(tmu if method=='mean_estimator' else arrays['test__neural_raw_'+method])[:11]
            m.load_state_dict(saved['states'][method]);m.eval()
            actual=((m(xx,cpu_anchor(xx))[:,0]/temperature).softmax(-1).numpy() if method=='raw_head'
                    else (m(xx,cpu_anchor(xx)).numpy() if method.startswith(('residual','mean')) else m(xx).numpy()))
            if method=='mean_estimator':actual=bounded(actual,meta)
            np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
            reload_rows.append({'method':method,'max_abs_error':float(np.max(np.abs(actual-expected)))})
    np.savez_compressed(out/'predictions.npz',**arrays)
    write_json(out/'decoder_diagnostics.json',{'validation':diagnostic,'test':t_diagnostics})
    result={'training':training,'choices':choices,'test':{k:score(t.x,t.y,z) for k,z in tpaths.items()},
            'reload_checks':reload_rows,'head_accuracy':float(np.mean(tp.argmax(-1)==raw_classes(t.x,t.y))),
            'p_plus_mean_parameters':total_target,'residual_capacity_parameters':_params(models['residual_capacity'])['total']}
    write_json(out/'result.json',result)
    print('DONE %s %s %s: posterior=%s'%(meta['spec']['name'],name,seed,choices['posterior']),flush=True)
    del models,head,mean_model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    manifest={'config':cfg,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
              'data_sha256':{s['name']:hashlib.sha256(Path(s['csv']).read_bytes()).hexdigest() for s in cfg['datasets']},
              'reference_sha256':{str(p.relative_to(cfg['reference_root'])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(cfg['reference_root']).rglob('*.py'))}}
    manifest_path=out/'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=manifest:raise ValueError('output contains a different code/config/data version; use a fresh output folder')
    elif any(out.glob('*/*/result.json')):
        raise ValueError('cannot mix existing unversioned results with this runner; use a fresh output folder')
    else:write_json(manifest_path,manifest)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
    if cfg['device'].startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.045,device=torch.device(cfg['device']))
    started=time.perf_counter();completed=[]
    for spec in cfg['datasets']:
        splits,meta,times=load_trajectory_data(spec)
        if cfg.get('smoke'):
            from .data import Split
            splits={k:Split(v.x[:48 if k=='train' else 12],v.y[:48 if k=='train' else 12],v.origins[:48 if k=='train' else 12]) for k,v in splits.items()}
        folder=out/spec['name'];write_json(folder/'dataset.json',meta)
        tensors={k:(torch.as_tensor(v.x,dtype=torch.float32,device=cfg['device']),torch.as_tensor(v.y,dtype=torch.float32,device=cfg['device'])) for k,v in splits.items()}
        for seed in cfg['seeds']:
            shared_folder=folder/('shared_seed'+str(seed))
            if (shared_folder/'result.json').exists():
                with np.load(shared_folder/'predictions.npz') as ar:shared={k:ar[k].copy() for k in ar.files}
            else:shared=neural_shared(splits,tensors,meta,cfg,shared_folder,seed)
            for name in cfg['backbones']:
                case=folder/(name+'_seed'+str(seed))
                if not (case/'result.json').exists():run_case(splits,tensors,meta,cfg,case,seed,name,shared,shared_folder)
                completed.append(str(case));write_json(out/'progress.json',{'completed':completed,'seconds':time.perf_counter()-started})
                if cfg['device'].startswith('cuda'):torch.cuda.empty_cache()
        del tensors
    write_json(out/'completion.json',{'cases':len(completed),'completed':completed,'seconds':time.perf_counter()-started,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if cfg['device'].startswith('cuda') else None})


if __name__=='__main__':main()
