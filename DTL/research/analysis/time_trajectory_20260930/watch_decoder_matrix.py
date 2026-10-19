"""Audit completed cases once as they arrive, without rerunning training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from audit_decoder_matrix import Audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--expected-cases',type=int,default=18);parser.add_argument('--timeout-seconds',type=int,default=7200)
    args=parser.parse_args();root=Path(args.results);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    audit=Audit();seen=set();hashes={};start=time.perf_counter();active=0.
    def save(final=False):
        result={'results_root':str(root.resolve()),'audit_source':'independent objectives/grids + dense O(H*M^2) transitions, no solver import',
                'cases':len(seen),'expected_cases':args.expected_cases,'complete':final,
                'checks':len(audit.checks),'failures':sum(not row['pass'] for row in audit.checks),
                'sample_rule':'origin indices 0 and floor(N/2), every channel, selected controls plus validation lambda=0; all saved objectives recomputed',
                'independent_dense_channel_paths':audit.dense_cells,'active_audit_seconds':active,
                'wall_seconds_including_wait':time.perf_counter()-start,'case_rows':audit.case_rows,
                'details':audit.checks,'artifact_sha256':hashes}
        name='decoder_matrix_audit.json' if final else 'decoder_matrix_partial_audit.json'
        temporary=out/(name+'.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(out/name)
        print(json.dumps({k:result[k] for k in ['cases','complete','checks','failures','independent_dense_channel_paths','active_audit_seconds']}),flush=True)
        return result
    while time.perf_counter()-start<args.timeout_seconds:
        cases=sorted(path.parent for path in root.glob('*/*/result.json') if (path.parent/'decoder_diagnostics.json').exists())
        changed=False
        for case in cases:
            if str(case) in seen:continue
            # result.json is written last; wait a short interval if it is being
            # created during this read rather than treating partial JSON as data.
            try:json.loads((case/'result.json').read_text())
            except json.JSONDecodeError:continue
            before=time.perf_counter();audit.case(case);active+=time.perf_counter()-before;seen.add(str(case));changed=True
            for name in ['predictions.npz','decoder_diagnostics.json','validation_lock.json']:
                path=case/name;hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        complete=(len(seen)==args.expected_cases and (root/'completion.json').exists())
        if changed or complete:
            result=save(complete)
            if result['failures']:raise AssertionError('decoder audit failures; see partial report')
        if complete:return
        time.sleep(30)
    save(False)
    raise TimeoutError('matrix not complete within read-only watcher limit')


if __name__=='__main__':main()
