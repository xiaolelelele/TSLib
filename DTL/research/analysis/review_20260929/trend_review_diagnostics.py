import json,csv
from pathlib import Path
import numpy as np
root=Path('/home/lgc/TSLib/DTL/trend_lab_runs/fixed_threeway_additional_validation_electricity_traffic_20260928T102453Z')
results=json.loads((root/'results.json').read_text());output={}
for group,method in [('Traffic_dlinear','learned_selected'),('Traffic_linear','learned_selected'),('Electricity_dlinear','learned_threeway_deployed'),('Electricity_linear','learned_threeway_deployed')]:
 g=results['groups'][group]; z=np.load(root/g['artifacts']['predictions_npz']);gs=[]; bs=[];ys=[]; qs=[];ms=[];deltas=[]
 for seed in range(5):
  p=f'seed{seed}__test__';x=z[p+'split__history'].astype(np.float64);y=z[p+'split__truth'].astype(np.float64);b=z[p+'base__path'].astype(np.float64);q=z[p+method+'__path'].astype(np.float64)
  dy=np.diff(np.concatenate([x[:,-1:],y],axis=1),axis=1);db=np.diff(np.concatenate([x[:,-1:],b],axis=1),axis=1);dq=np.diff(np.concatenate([x[:,-1:],q],axis=1),axis=1)
  bc=np.sign(db)==np.sign(dy);qc=np.sign(dq)==np.sign(dy)
  gs.append(qc.astype(float)-bc);bs.append(db);ys.append(dy);qs.append(dq);ms.append(np.mean((q-y)**2)/np.mean((b-y)**2)-1);deltas.append((np.mean(qc)-np.mean(bc))*100)
 gain=np.stack(gs);bd=np.stack(bs);yd=np.stack(ys);qd=np.stack(qs)
 raw=[]
 with open(g['source']['data_file']) as f:
  for row in csv.DictReader(f):raw.append([float(row[str(c)]) for c in range(8)])
 raw=np.array(raw); origins=z['seed0__test__split__origins'];raw_y=np.stack([raw[o:o+25] for o in origins]);true_raw_sign=np.sign(np.diff(raw_y,axis=1))
 r={'method':method,'direction_gain_pp':100*gain.mean(),'seed_gain_pp':deltas,'per_channel_gain_pp':(100*gain.mean(axis=(0,1,2))).tolist(),'per_horizon_gain_pp':(100*gain.mean(axis=(0,1,3))).tolist(),'first_step_gain_pp':100*gain[:,:,0,:].mean(),'remaining_steps_gain_pp':100*gain[:,:,1:,:].mean(),'first_step_fraction_of_net_gain':float(gain[:,:,0,:].sum()/gain.sum()),'true_raw_csv_sign_mismatches':int(np.count_nonzero(true_raw_sign!=np.sign(yd[0]))),'gain_by_base_delta_threshold':{},'gain_by_true_delta_threshold':{},'corrected_pp':float(100*np.mean(gain==1)),'harmed_pp':float(100*np.mean(gain==-1))}
 for t in [1e-7,1e-5,1e-3,1e-2,1e-1]:
  for key,arr in [('gain_by_base_delta_threshold',bd),('gain_by_true_delta_threshold',yd)]:
   mask=np.abs(arr)<=t
   r[key][str(t)]={'point_fraction':float(mask.mean()),'net_contribution_pp':float(100*np.sum(gain*mask)/gain.size),'share_of_positive_corrections':float(np.sum((gain==1)*mask)/np.sum(gain==1)),'share_of_negative_corrections':float(np.sum((gain==-1)*mask)/np.sum(gain==-1))}
 output[group]=r
Path('/tmp/trend_review_diagnostics.json').write_text(json.dumps(output,indent=2))
for k,v in output.items():
 print(k,json.dumps(v))
