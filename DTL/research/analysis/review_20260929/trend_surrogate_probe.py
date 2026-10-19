import sys,json,pathlib,numpy as np,torch,math
sys.path.insert(0,'/home/lgc/TSLib/DTL/research/analysis/review_20260929/source')
from trend_lab.operators import multiscale_slopes,direction_labels
from trend_lab.run import _nll
F=torch.nn.functional
torch.set_num_threads(1)
q=torch.tensor([0.,.4,.6]); r=torch.linspace(0,3,30001); logits=torch.stack((-r,1-r.abs(),r),-1); objective=-(q*logits.log_softmax(-1)).sum(-1); ri=float(r[objective.argmin()]); print('CE population optimum q=[0,.4,.6]', {'r':ri,'evaluation_label':'flat','bayes_0_1_label':'up','loss_at_opt':float(objective.min())})
for name in ['etth1_pilot_20260927T074135Z','etth2_pilot_20260927T074525Z']:
 root=pathlib.Path('/home/lgc/TSLib/DTL/trend_lab_runs')/name/'pilot'; result=json.loads((root/'results.json').read_text()); conf=json.loads((root/'config.json').read_text()); a=np.load(root/'predictions.npz'); eps=torch.tensor(result['epsilon'],dtype=torch.float32); ks=result['scales']; print(name, 'epsilon',eps.tolist(), 'columns',conf['data']['value_columns'])
 for seed in result['seeds']:
  x=torch.from_numpy(a[f'seed{seed}_history']); y=torch.from_numpy(a[f'seed{seed}_truth']); z=torch.from_numpy(a[f'seed{seed}_residual_direction']); r=multiscale_slopes(x,z,ks)/eps[None,:,None,:]; lab=direction_labels(multiscale_slopes(x,y,ks),eps); logits=torch.stack((-r,1-r.abs(),r),-1)
  ce=F.cross_entropy(logits.reshape(-1,3),lab.reshape(-1),reduction='none').reshape(lab.shape); mse=float(F.mse_loss(z,y)); w=conf['training'].get('direction_weight',.1); print({'seed':seed,'mse':mse,'weighted_direction_loss':w*float(ce.mean()),'ratio':w*float(ce.mean())/mse, 'unweighted_ce_by_channel':ce.mean((0,1,2)).tolist()})
