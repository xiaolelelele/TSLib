import sys, json, pathlib, numpy as np, torch
sys.path.insert(0, '/home/lgc/TSLib/DTL/research/analysis/review_20260929/source')
from trend_lab.operators import multiscale_slopes, direction_labels
from trend_lab.decoding import project_trajectory
from trend_lab.metrics import evaluate_predictions
from trend_lab.run import run

torch.set_num_threads(1)
x=torch.zeros(1,2,1); eps=torch.ones(1,1); slopes=torch.tensor([[[[.75]]]])
r=slopes/eps[None,:,None,:]
logits=torch.stack((-r,1-r.abs(),r),dim=-1)
print('residual_direction toy', {'r': .75,'logits':logits.flatten().tolist(),'aux_argmax':logits.argmax(-1).item(),'evaluation_label':direction_labels(slopes,eps).item()})
roots=[pathlib.Path('/home/lgc/TSLib/DTL/trend_lab_runs/etth1_pilot_20260927T074135Z/pilot'), pathlib.Path('/home/lgc/TSLib/DTL/trend_lab_runs/etth2_pilot_20260927T074525Z/pilot')]
for root in roots:
 result=json.loads((root/'results.json').read_text()); a=np.load(root/'predictions.npz'); eps=torch.tensor(result['epsilon'],dtype=torch.float32); ks=result['scales']
 for seed in result['seeds']:
  x=torch.from_numpy(a[f'seed{seed}_history']); y=torch.from_numpy(a[f'seed{seed}_truth']); b=torch.from_numpy(a[f'seed{seed}_base']); p=torch.from_numpy(a[f'seed{seed}_direction_head_probs'])
  rd=torch.from_numpy(a[f'seed{seed}_residual_direction']); ratios=multiscale_slopes(x,rd,ks)/eps[None,:,None,:]
  auxiliary_class=torch.stack((-ratios,1-ratios.abs(),ratios),-1).argmax(-1)
  eval_class=direction_labels(multiscale_slopes(x,rd,ks),eps)
  row={'dataset':root.parent.name,'seed':seed,'eps_min':eps.min().item(),'rd_class_disagreement':float((auxiliary_class!=eval_class).float().mean())}
  for method, ids in [('projection',list(range(len(ks)))),('projection_single',[0])]:
   alpha=result['runs'][seed]['methods'][method]['selected_alpha']
   old=torch.from_numpy(a[f'seed{seed}_{method}'])
   new=project_trajectory(x,b,p[:,ids],[ks[i] for i in ids],eps[ids],strength=alpha,steps=400,max_rms=.5)
   me=evaluate_predictions(x,y,new,ks,eps)
   om=result['runs'][seed]['methods'][method]['metrics']
   row[method]={'alpha':alpha,'maxabs_40_400':float((new-old).abs().max()),'rms_40_400':float((new-old).square().mean().sqrt()),'mse_40':om['mse'],'mse_400':me['mse'],'sign_40':om['raw_sign_accuracy'],'sign_400':me['raw_sign_accuracy'],'threeclass_40':np.mean([v['accuracy'] for v in om['scales'].values()]),'threeclass_400':np.mean([v['accuracy'] for v in me['scales'].values()])}
  print(json.dumps(row))
