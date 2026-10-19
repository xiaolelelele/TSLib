import csv,json,pathlib
from collections import defaultdict
import numpy as np
p=pathlib.Path('/tmp/repaired_mechanisms_timeblocks_review_20260929')
read=lambda name:list(csv.DictReader((p/name).open()))
blocks=read('per_time_block.csv');seeds=read('per_seed.csv');summary=read('time_block_summary.csv')
groups=defaultdict(list)
for r in blocks:groups[(r['group'],r['seed'],r['method'])].append(r)
max_error=0.
for row in seeds:
 key=row['group'],row['seed'],row['method'];bb=sorted(groups[key],key=lambda r:int(r['time_block']));assert [int(r['time_block']) for r in bb]==[1,2,3,4]
 n=np.array([int(r['windows']) for r in bb]);val=np.array([float(r['raw_gain_pp']) for r in bb]);err=abs(np.average(val,weights=n)-float(row['raw_gain_pp']));max_error=max(max_error,err);assert err<1e-11
 for left,right in zip(bb,bb[1:]):assert int(left['origin_end'])<int(right['origin_start']);assert int(left['target_end_inclusive'])<int(right['target_start'])
 root=pathlib.Path('/home/lgc/TSLib/DTL/trend_lab_runs/repair_posterior_20260929/results_v1')/row['group']
 with np.load(root/'predictions.npz') as a:
  y=a['test__truth'].astype(float);z=a['seed%s__test__%s'%(row['seed'],row['method'])].astype(float);b=a['seed%s__test__base'%row['seed']].astype(float)
  delta=float(np.mean((z-y)**2-(b-y)**2));reconstructed=float(np.average([float(r['mse_delta']) for r in bb],weights=n));assert abs(delta-reconstructed)<1e-12
  start=0
  for r in bb:
   end=start+int(r['windows']);assert int(r['origin_start'])==a['test__origins'][start];assert int(r['origin_end'])==a['test__origins'][end-1];start=end
  assert start==len(y)
for row in summary:
 rr=[r for r in blocks if all(r[k]==row[k] for k in ['group','method','time_block'])]
 assert len(rr)==int(row['seed_replicates'])==3
 for metric in ['raw_gain_pp','mse_delta','base_mse','method_mse']:
  assert abs(float(row[metric])-np.mean([float(r[metric]) for r in rr]))<1e-12
 assert abs(float(row['mean_seed_mse_ratio'])-np.mean([float(r['mse_ratio']) for r in rr]))<1e-12
 assert abs(float(row['ratio_of_seed_mean_mse'])-np.mean([float(r['method_mse']) for r in rr])/np.mean([float(r['base_mse']) for r in rr]))<1e-12
print(json.dumps({'per_seed_block_rows':len(blocks),'block_summary_rows':len(summary),'seed_method_groups':len(groups),'max_raw_reaggregation_error':max_error,'all_checks_passed':True}))
