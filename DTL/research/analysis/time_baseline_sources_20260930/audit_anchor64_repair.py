"""Independent trace/formula, scores, selector and chronology audit.

No import of repair_shared_anchor64 or its score/selection helper.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def score(x,y,z):
    x,y,z=[np.asarray(v,dtype=np.float64) for v in (x,y,z)]
    truth=np.sign(np.diff(np.concatenate((x[:,-1:],y),axis=1),axis=1)).astype(int)+1
    predicted=np.sign(np.diff(np.concatenate((x[:,-1:],z),axis=1),axis=1)).astype(int)+1
    matrix=np.bincount((3*truth+predicted).ravel(),minlength=9).reshape(3,3)
    correct=np.diag(matrix);actual=matrix.sum(1);positive=matrix.sum(0)
    return {'mse':float(np.mean(np.square(y-z))),'mae':float(np.mean(np.abs(y-z))),
            'raw_accuracy':float(np.mean(truth==predicted)),
            'macro_f1':float(np.mean(np.divide(2*correct,actual+positive,out=np.zeros(3),where=(actual+positive)>0))),
            'precision':[float(correct[k]/positive[k]) if positive[k] else None for k in range(3)],
            'recall':[float(correct[k]/actual[k]) if actual[k] else None for k in range(3)],
            'confusion':matrix.tolist(),'true_zero_fraction':float(np.mean(truth==1)),
            'predicted_zero_fraction':float(np.mean(predicted==1)),
            'nonzero_accuracy':float(np.mean((truth==predicted)[truth!=1])) if np.any(truth!=1) else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--repair',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--expected-cases',type=int)
    args=parser.parse_args();run=Path(args.run);repair=Path(args.repair);checks=0;cases=[]
    def equal(actual,expected,label):
        nonlocal checks
        checks+=1
        if actual is None or expected is None:
            if actual is not None or expected is not None:raise AssertionError(label)
        elif isinstance(actual,(dict,str,bool)):
            if actual!=expected:raise AssertionError(label)
        elif isinstance(actual,list) and any(v is None for v in actual):
            for a,e in zip(actual,expected):equal(a,e,label)
        else:np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12,err_msg=label)
    def metrics_equal(x,y,z,saved,label):
        for key,value in score(x,y,z).items():equal(value,saved[key],label+'/'+key)
    for lock_path in sorted(repair.glob('*/*_seed*/repair_lock.json')):
        if 'shared' in lock_path.parent.name or not (lock_path.parent/'result.json').exists():continue
        case=lock_path.parent;domain=case.parent.name;lock=json.loads(lock_path.read_text())
        source=run/domain/case.name;original=json.loads((source/'validation_lock.json').read_text());meta=original['metadata']
        result=json.loads((case/'result.json').read_text())
        with np.load(source/'predictions.npz') as a:baseline={k:a[k] for k in a.files}
        with np.load(case/'predictions.npz') as a:paths={k:a[k] for k in a.files}
        shared=repair/domain/('shared_seed'+str(original['seed']))
        with np.load(shared/'signed_trace.npz') as a:trace={k:a[k] for k in a.files}
        source_shared=run/domain/('shared_seed'+str(original['seed']))
        shared_lock=json.loads((shared/'repair_lock.json').read_text())
        equal(hashlib.sha256((source/'validation_lock.json').read_bytes()).hexdigest(),lock['original_case_lock_sha256'],'original lock checksum')
        equal(hashlib.sha256((source_shared/'checkpoint.pt').read_bytes()).hexdigest(),shared_lock['original_shared_checkpoint_sha256'],'original checkpoint checksum')
        equal(original['choices'],lock['original_choices'],'original choices retained')
        equal(lock['original_choices'],result['original_choices'],'result original choices')
        equal(lock['test_labels_used_for_selection'],False,'no test labels for selection')
        if lock_path.stat().st_mtime_ns>(shared/'signed_trace.npz').stat().st_mtime_ns or lock_path.stat().st_mtime_ns>(case/'predictions.npz').stat().st_mtime_ns:
            raise AssertionError('corrected test artifact precedes corrected case lock')
        checks+=1
        for split in ('validation','test'):
            x,y=baseline[split+'__x'],baseline[split+'__y']
            for variant in (3,4):
                stem='carets%d_phase'%variant;key=stem+'_anchor64';signed=trace[split+'__'+stem+'__signed32']
                if variant==3:
                    p=trace[split+'__'+stem+'__probability32'];d=trace[split+'__'+stem+'__deviation32']
                    expected_signed=p[...,0]*d[...,0]-p[...,1]*d[...,1]
                    np.testing.assert_allclose(expected_signed,signed,atol=2e-7,rtol=2e-6);checks+=1
                    if np.any(d<0):raise AssertionError('negative nonnegative-deviation branch')
                    checks+=1
                expected=x[:,-1:]+signed.astype(np.float64)
                equal(expected,trace[split+'__'+stem+'__unbounded_anchor64'],'original observed anchor + learned branch signed')
                if meta['spec']['grid_mode']=='native':expected=np.maximum(expected,-np.array(meta['mean'])/np.array(meta['std']))
                equal(expected,paths[split+'__'+key],key+' corrected output path')
                if split=='test':metrics_equal(x,y,expected,result['test'][key],key+' test metrics')
        x,y=baseline['validation__x'],baseline['validation__y'];ref=baseline['validation__cosdir']
        metrics_equal(x,y,ref,lock['selection']['reference'],'selector reference')
        eligible=[]
        for row in lock['selection']['candidates']:
            key=row['key'];path=paths['validation__'+key] if key in ('carets3_phase_anchor64','carets4_phase_anchor64') else baseline['validation__'+key]
            independent=score(x,y,path);metrics_equal(x,y,path,row,key+' valmetrics')
            good=independent['raw_accuracy']>score(x,y,ref)['raw_accuracy'] and independent['mse']<=score(x,y,ref)['mse']+1e-8
            equal(good,row['eligible'],key+' eligible')
            if good:eligible.append(row)
        chosen=sorted(eligible,key=lambda r:(-r['raw_accuracy'],r['mse'],r['key']))[0]['key'] if eligible else 'cosdir'
        equal(chosen,lock['choices']['competitor_corrected'],'corrected chosen val')
        equal(chosen,result['choices']['competitor_corrected'],'chosen test unchanged')
        for split in ('validation','test'):
            expected=paths[split+'__'+chosen] if chosen.endswith('_anchor64') else baseline[split+'__'+chosen]
            equal(expected,paths[split+'__competitor_corrected'],'actual competitor corrected '+split)
        metrics_equal(baseline['test__x'],baseline['test__y'],paths['test__competitor_corrected'],result['test']['competitor_corrected'],'competitor corrected test metrics')
        original_keys=set(r['key'] for r in original['selection']['competitor']['candidates'])
        if lock.get('rule_version')==2:
            expected_keys=(original_keys-{'carets3_phase','carets4_phase'})|{'carets3_phase_anchor64','carets4_phase_anchor64'}
            equal(len(original_keys),13,'original fixed competitor budget')
            equal(len(lock['selection']['candidates']),13,'v2 fixed competitor budget')
            equal(lock['competitor_cardinality'],13,'v2 lock declared cardinality')
            equal(result['competitor_cardinality'],13,'v2 result declared cardinality')
            equal(chosen in ('carets3_phase','carets4_phase'),False,'known-bug original CaReTS paths forbidden')
            equal(set(lock['corrected_competitor_keys'])==expected_keys,True,'explicit replacement keys')
            archived_case=Path(lock['v1_archive_case'])
            equal(hashlib.sha256((archived_case/'repair_lock.json').read_bytes()).hexdigest(),lock['v1_case_lock_sha256'],'historical V1 case lock checksum')
            equal(hashlib.sha256((shared/'signed_trace.npz').read_bytes()).hexdigest(),shared_lock['v1_trace_sha256'],'V2 trace is an exact V1 copy')
        else:
            expected_keys=original_keys|{'carets3_phase_anchor64','carets4_phase_anchor64'}
        equal({r['key']:True for r in lock['selection']['candidates']},{k:True for k in expected_keys},'complete fixed competitor library')
        cases.append({'case':str(case.resolve()),'chosen':chosen,'original_chosen':original['choices']['competitor']})
    if args.expected_cases is not None and len(cases)!=args.expected_cases:
        raise AssertionError('Expected %d complete cases, audited %d'%(args.expected_cases,len(cases)))
    report={'checks':checks,'failures':0,'cases':cases,'scope':'Independent NumPy formula reconstruction from saved signed/probability/deviation traces, val/test metrics, original checksum/choices, same validation eligibility/selection/fallback, complete competitor pool and repair lock chronology. V2 explicitly checks replacement cardinality=13 and prohibits original known-bug CaReTS paths. Learned branch CPU inference independently crosschecked by checkpoint reload in repair.'}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps({'checks':checks,'failures':0,'cases':len(cases)}))


if __name__=='__main__':main()
