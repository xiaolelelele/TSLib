import sys,pathlib,json,numpy as np,torch
sys.path.insert(0,'/home/lgc/TSLib/DTL/research/analysis/review_20260929/source')
from trend_lab.decoding import project_trajectory
from trend_lab.metrics import evaluate_predictions
torch.set_num_threads(1)
root=pathlib.Path('/home/lgc/TSLib/DTL/trend_lab_runs/etth1_pilot_20260927T074135Z/pilot'); r=json.loads((root/'results.json').read_text()); arc=np.load(root/'predictions.npz'); eps=torch.tensor(r['epsilon'],dtype=torch.float32); ks=r['scales']; seed=2; x=torch.from_numpy(arc[f'seed{seed}_history']);y=torch.from_numpy(arc[f'seed{seed}_truth']);b=torch.from_numpy(arc[f'seed{seed}_base']);p=torch.from_numpy(arc[f'seed{seed}_direction_head_probs']); paths={}
for steps in [40,400,1000]:
 z=project_trajectory(x,b,p[:,:1],[1],eps[:1],strength=10.,steps=steps,max_rms=.5); paths[steps]=z; m=evaluate_predictions(x,y,z,ks,eps);print(json.dumps({'dataset':'ETTh1','seed':2,'method':'projection_single','device':'cpu','steps':steps,'mse':m['mse'],'raw_sign_accuracy':m['raw_sign_accuracy'],'mean_threeclass_accuracy':np.mean([q['accuracy'] for q in m['scales'].values()])}))
for a,b in [(40,400),(400,1000)]:
 d=paths[a]-paths[b];print(json.dumps({'steps_pair':[a,b],'maxabs':float(d.abs().max()),'rms':float(d.square().mean().sqrt())}))
saved=torch.from_numpy(arc['seed2_projection_single']);d=saved-paths[40];print(json.dumps({'comparison':'saved CUDA40 vs CPU40','maxabs':float(d.abs().max()),'rms':float(d.square().mean().sqrt())}))
