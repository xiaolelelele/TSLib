import json,pathlib,sys,torch
sys.path.insert(0,'/home/lgc/TSLib/DTL')
from trend_lab.data import load_data
from trend_lab.run import load_checkpoint,predict_from_checkpoint
from trend_lab.losses import fit_slope_scales,residual_direction_diagnostics
from trend_lab.operators import multiscale_slopes,multiscale_direction_labels
F=torch.nn.functional
torch.set_num_threads(1)
for dataset,backbone in [('Electricity','linear'),('Electricity','dlinear'),('Traffic','dlinear')]:
 root=pathlib.Path('/home/lgc/TSLib/DTL/trend_lab_fusion_runs/triple_mean_revalidation_20260928T092723Z/results')/dataset/('data__'+backbone)
 cfg=json.loads((root/'config.json').read_text());bundle=load_data(cfg['data']);cp=load_checkpoint(root/'checkpoint.pt',seed=0);ks=cfg['scales'];eps=cp['epsilon'];x=torch.from_numpy(bundle.calibration.x);y=torch.from_numpy(bundle.calibration.y);p=predict_from_checkpoint(cp,x,'residual_direction')
 scale=fit_slope_scales(torch.from_numpy(bundle.train.x),torch.from_numpy(bundle.train.y),ks)
 fixed=residual_direction_diagnostics(x,y,p,ks,eps,scale,weight=.1)
 print(json.dumps({'dataset':dataset,'backbone':backbone,'seed':0,'purpose':'loss-scale diagnostic on unchanged historical checkpoint, not retrained effectiveness evidence','slope_scale':scale.tolist(),'fixed_objective':fixed}))
