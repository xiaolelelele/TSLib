import sys,pathlib,json,numpy as np,torch
sys.path.insert(0,'/home/lgc/TSLib/DTL/research/analysis/review_20260929/source')
from trend_lab.operators import multiscale_slopes,direction_labels
torch.set_num_threads(1)
roots=[pathlib.Path('/home/lgc/TSLib/DTL/trend_lab_runs')/d/'pilot' for d in ['etth1_pilot_20260927T074135Z','etth2_pilot_20260927T074525Z']]
roots += [pathlib.Path('/home/lgc/TSLib/DTL/trend_lab_fusion_runs/triple_mean_revalidation_20260928T092723Z/results')/d/'data__dlinear' for d in ['Electricity','Traffic']]
for root in roots:
 result=json.loads((root/'results.json').read_text()); eps=torch.tensor(result['epsilon'],dtype=torch.float32); ks=result['scales']; arc=np.load(root/'predictions.npz')
 for seed in result['seeds']:
  x=torch.from_numpy(arc[f'seed{seed}_history']); y=torch.from_numpy(arc[f'seed{seed}_truth']); labels32=direction_labels(multiscale_slopes(x,y,ks),eps); labels64=direction_labels(multiscale_slopes(x.double(),y.double(),ks),eps.double()); probs=torch.from_numpy(arc[f'seed{seed}_direction_head_probs']); h=probs.argmax(-1)
  record={'root':str(root),'seed':seed,'truth_label_disagreement_percent':float((labels32!=labels64).float().mean())*100, 'head_accuracy_float32_minus_float64_pp':float(((h==labels32).float()-(h==labels64).float()).mean())*100,'path_accuracy_float32_minus_float64_pp':{}}
  for m in ['base','projection','projection_single','residual','residual_direction','difference']:
   z=torch.from_numpy(arc[f'seed{seed}_{m}']); pred32=direction_labels(multiscale_slopes(x,z,ks),eps);pred64=direction_labels(multiscale_slopes(x.double(),z.double(),ks),eps.double()); record['path_accuracy_float32_minus_float64_pp'][m]=float(((pred32==labels32).float()-(pred64==labels64).float()).mean())*100
  print(json.dumps(record))
