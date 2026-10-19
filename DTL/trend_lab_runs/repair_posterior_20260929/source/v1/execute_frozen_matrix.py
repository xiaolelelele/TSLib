"""Run only the predeclared matrix on the reserved GPU, from frozen source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

GROUPS = ['repair_electricity_dlinear', 'repair_traffic_dlinear', 'repair_traffic_linear',
          'additional_traffic_dlinear', 'additional_electricity_dlinear']
GPU_UUID = 'GPU-e054b9b5-123d-0aa7-4bed-6102d26e243a'

def now(): return datetime.now(timezone.utc).isoformat()
def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def query(args): return subprocess.check_output(['nvidia-smi']+args,text=True).strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--version',required=True)
    parser.add_argument('--phase',choices=['smoke','full'],required=True)
    args=parser.parse_args();root=args.root.resolve();source=root/'source'/args.version
    manifest=json.loads((source/'freeze_manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        if sha(source/name)!=expected:raise RuntimeError('Frozen source hash mismatch: '+name)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise RuntimeError('Only the explicitly reserved GPU3 UUID is allowed')
    gpu=query(['--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'])
    processes=query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
    if any(line.split(',')[0].strip()==GPU_UUID for line in processes.splitlines()):
        raise RuntimeError('Reserved GPU has a compute process; refusing to interfere')
    gpu_row=next(line for line in gpu.splitlines() if GPU_UUID in line)
    if int(gpu_row.split(',')[3].strip())>128:raise RuntimeError('Reserved GPU memory is not idle: '+gpu_row)
    import torch
    import numpy as np
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                 'torch':torch.__version__,'numpy':np.__version__,'torch_cuda':torch.version.cuda,
                 'cudnn':torch.backends.cudnn.version(),'cuda_visible_devices':GPU_UUID,
                 'gpu_snapshot':gpu,'compute_process_snapshot':processes,'created_utc':now()}
    write(root/f'environment_{args.version}_{args.phase}.json',environment)
    if args.phase=='full':
        smoke=root/f'cuda_smoke_{args.version}'/'repair_traffic_dlinear'/'results.json'
        passed=json.loads(smoke.read_text())
        if not passed.get('complete') or not passed.get('smoke') or not all(r.get('checkpoint_reload_verified') for r in passed['runs']):
            raise RuntimeError('CUDA smoke incomplete or checkpoint reload failed')
    groups=['repair_traffic_dlinear'] if args.phase=='smoke' else GROUPS
    status_path=root/f'execution_{args.version}_{args.phase}.json'
    if status_path.exists():raise FileExistsError('Execution status already exists: '+str(status_path))
    status={'phase':args.phase,'version':args.version,'started_utc':now(),'complete':False,'success':False,
            'source_dir':str(source),'freeze_manifest_sha256':sha(source/'freeze_manifest.json'),
            'gpu_uuid':GPU_UUID,'groups':groups,'finished_groups':[],'commands':[]}
    write(status_path,status)
    for group in groups:
        config=source/'configs'/(group+'.json')
        output=root/(f'cuda_smoke_{args.version}' if args.phase=='smoke' else f'results_{args.version}')/group
        log=root/'logs'/f'{args.version}_{args.phase}_{group}.log'
        command=[sys.executable,'-u','-m','trend_lab.run','--config',str(config),'--output',str(output)]
        if args.phase=='smoke':command.append('--smoke')
        env=os.environ.copy();env['PYTHONPATH']=str(source);env['PYTHONUNBUFFERED']='1'
        env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
        status['current_group']=group;status['commands'].append({'group':group,'argv':command,'output':str(output),'log':str(log),'start_utc':now()});write(status_path,status)
        print('START',args.phase,group,now(),flush=True)
        with log.open('x') as stream:
            completed=subprocess.run(command,cwd=str(source),env=env,stdout=stream,stderr=subprocess.STDOUT)
        status['commands'][-1]['exit_code']=completed.returncode;status['commands'][-1]['end_utc']=now()
        if completed.returncode:
            status['failed_group']=group;status['ended_utc']=now();write(status_path,status)
            raise RuntimeError('Training failed; inspect '+str(log))
        result=json.loads((output/'results.json').read_text())
        if not result.get('complete') or not all(row.get('checkpoint_reload_verified') for row in result['runs']):
            raise RuntimeError('Completed process lacks verified results: '+group)
        status['finished_groups'].append(group);write(status_path,status)
        print('DONE',args.phase,group,now(),flush=True)
    status.update(complete=True,success=True,current_group=None,ended_utc=now());write(status_path,status)
    print('COMPLETE',args.phase,args.version,now(),flush=True)

if __name__=='__main__':main()
