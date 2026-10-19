import argparse, csv, importlib.util, json, pathlib, sys
import numpy as np
import torch
root=pathlib.Path('/home/lgc/TSLib/DTL')
sys.path.insert(0,str(root/'research'))
from analyze_repaired_mechanisms import analyze
parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='repair_electricity_dlinear');parser.add_argument('--artifacts',default='/tmp/repaired_mechanisms_electricity_20260929');args=parser.parse_args()
runroot=root/'trend_lab_runs/repair_posterior_20260929/results_v1'/args.dataset
a,b,c,decisions=analyze(runroot)
result=json.loads((runroot/'results.json').read_text())
with np.load(runroot/'predictions.npz') as archive:
 arrays={k:archive[k] for k in archive.files}
checkpoints=torch.load(runroot/'checkpoint.pt',map_location='cpu',weights_only=True)
x,y=arrays['test__history'].astype('float64'),arrays['test__truth'].astype('float64')
scales=result['config']['scales'];eps=np.array(result['epsilon']);count=0;max_error=0.
for row in a:
 seed,method=row['seed'],row['method'];z=arrays[f'seed{seed}__test__{method}'].astype('float64');base=arrays[f'seed{seed}__test__base'].astype('float64')
 def raw(t):return np.sign(t-np.concatenate((x[:,-1:,:],t[:,:-1,:]),axis=1))
 tr,br,zr=raw(y),raw(base),raw(z)
 # A 3x3 joint correctness table across booleans is unnecessary: count each
 # condition directly and verify all-point denominators and directional changes.
 N=y.size;fix=int(np.count_nonzero((br!=tr)&(zr==tr)));harm=int(np.count_nonzero((br==tr)&(zr!=tr)))
 for key,value in [('fixed_points_pct',100*fix/N),('harmed_points_pct',100*harm/N),('raw_gain_pp',100*(fix-harm)/N)]:
  err=abs(value-row[key]);max_error=max(max_error,err);assert err<1e-11,(key,err)
 probs=arrays[f'seed{seed}__test__direction_head__probs'].astype('float64')
 joined=np.concatenate((x,z),axis=1); truth_joined=np.concatenate((x,y),axis=1)
 score=np.zeros(z.shape,dtype='float64');acc=np.zeros(z.shape,dtype='float64')
 for si,k in enumerate(scales):
  delta=(joined[:,x.shape[1]:]-joined[:,x.shape[1]-k:joined.shape[1]-k])/k
  truthdelta=(truth_joined[:,x.shape[1]:]-truth_joined[:,x.shape[1]-k:truth_joined.shape[1]-k])/k
  # Explicit class masks replace take_along_axis; separate scale loop.
  cls=np.where(delta>eps[si],2,np.where(delta< -eps[si],0,1));tru=np.where(truthdelta>eps[si],2,np.where(truthdelta< -eps[si],0,1))
  for label in range(3):score+=probs[:,si,:,:,label]*(cls==label)/len(scales)
  acc+=(cls==tru)/len(scales)
 for key,value in [('predicted_class_utility',score.mean()),('empirical_class_accuracy',acc.mean())]:
  err=abs(value-row[key]);max_error=max(max_error,err);assert err<1e-12,(key,err)
 channel=[rr for rr in b if rr['seed']==seed and rr['method']==method]
 horizon=[rr for rr in c if rr['seed']==seed and rr['method']==method]
 assert abs(np.mean([r['raw_gain_pp'] for r in channel])-row['raw_gain_pp'])<1e-11
 assert abs(np.mean([r['raw_gain_pp'] for r in horizon])-row['raw_gain_pp'])<1e-11
 for rr in channel:
  ci=result['config']['data']['value_columns'].index(rr['column']);assert abs(rr['mse_delta']-np.mean((z[:,:,ci]-y[:,:,ci])**2-(base[:,:,ci]-y[:,:,ci])**2))<1e-12
 for rr in horizon:
  hi=rr['horizon']-1;assert abs(rr['mse_delta']-np.mean((z[:,hi]-y[:,hi])**2-(base[:,hi]-y[:,hi])**2))<1e-12
 count+=1
for d in decisions:
 seed=d['seed'];cp=checkpoints[str(seed)];choice=d['posterior_validation_choice']
 if choice=='base':saved=[0]*len(y);library=['base']
 else:saved=result['runs'][seed]['test_convergence'][choice]['chosen_candidate_indices'];library=cp['entries'][choice]['library']
 expected={name:int(np.count_nonzero(np.asarray(saved)==i)) for i,name in enumerate(library)}
 assert expected==d['test_candidate_counts'];assert sum(expected.values())==len(y)
 assert all(0<=i<len(library) for i in saved)
with (pathlib.Path(args.artifacts)/'per_seed.csv').open() as f:old=list(csv.DictReader(f))
assert len(old)==len(a)
for before,after in zip(old,a):
 for key,value in after.items():
  if isinstance(value,(str,)): assert before[key]==value
  else: assert abs(float(before[key])-value)<1e-11
print(json.dumps({'run':str(runroot),'checked_method_seed_rows':count,'channels':len(b),'horizons':len(c),'decision_seed_rows':len(decisions),'max_numeric_error':max_error,'existing_per_seed_csv_matches':True,'all_checks_passed':True}))
# Exact rational script reviewed independently of its saved output.
p=root/'research/analysis/direction_risk_identifiability_20260929/probe_direction_risk_identifiability.py'
spec=importlib.util.spec_from_file_location('risk_probe',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
computed=module.analyze();saved=json.loads((p.parent/'result.json').read_text());assert computed==saved
print(json.dumps({'rational_saved_result_exact_match':True,'regrets':[r['mse_change_exact'] for r in computed['rows']]}))
