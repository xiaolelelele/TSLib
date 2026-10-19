import sys,json,pathlib,torch,numpy as np
sys.path.insert(0,'/home/lgc/TSLib/DTL/research/analysis/review_20260929/source')
from trend_lab.data import load_data
from trend_lab.run import load_checkpoint,predict_from_checkpoint,_nll
from trend_lab.operators import multiscale_slopes,direction_labels
F=torch.nn.functional
torch.set_num_threads(1)
for dataset,backbone in [('Electricity','linear'),('Electricity','dlinear'),('Traffic','dlinear')]:
 root=pathlib.Path('/home/lgc/TSLib/DTL/trend_lab_fusion_runs/triple_mean_revalidation_20260928T092723Z/results')/dataset/('data__'+backbone)
 cfg=json.loads((root/'config.json').read_text()); metrics=json.loads((root/'results.json').read_text()); bundle=load_data(cfg['data']); cp=load_checkpoint(root/'checkpoint.pt',seed=0)
 x=torch.from_numpy(bundle.calibration.x); y=torch.from_numpy(bundle.calibration.y); z=predict_from_checkpoint(cp,x,'residual_direction').detach().requires_grad_(True); eps=cp['epsilon']; ks=cfg['scales']; sl=multiscale_slopes(x,z,ks); r=sl/eps[None,:,None,:]; lab=direction_labels(multiscale_slopes(x,y,ks),eps); logits=torch.stack((-r,1-r.abs(),r),-1); ce=.1*_nll(logits,lab); mse=F.mse_loss(z,y); gmse=torch.autograd.grad(mse,z,retain_graph=True)[0]; gce=torch.autograd.grad(ce,z)[0]
 print(json.dumps({'dataset':dataset,'backbone':backbone,'seed':0,'n_cal':len(x),'epsilon_floor_cells':int((eps<=1.0001e-6).sum()),'mse':mse.item(),'weighted_ce':ce.item(),'total':(mse+ce).item(),'saved_cal_loss':metrics['runs'][0]['methods']['residual_direction']['calibration_monitor_loss'],'direction_to_mse_loss_ratio':(ce/mse).item(),'direction_to_mse_gradient_norm_ratio':(gce.norm()/gmse.norm()).item()}))
 for split in ['train','calibration','test']:
  ss=getattr(bundle,split); xx=torch.from_numpy(ss.x); yy=torch.from_numpy(ss.y); l32=direction_labels(multiscale_slopes(xx,yy,ks),eps); l64=direction_labels(multiscale_slopes(xx.double(),yy.double(),ks),eps.double()); print('label_numeric_check',dataset,backbone,split, float((l32!=l64).float().mean()), (l32!=l64).sum((0,2)).tolist())
