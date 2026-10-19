"""Replace the two erroneous CaReTS inference paths in a fixed 13-method pool.

V1 repair outputs are preserved as historical evidence. This rule correction
uses their fixed learned-branch paths, with no retraining/new neural inference
or expanded validation budget. V2 locks precede rereading held-test paths.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from audit_runner_outputs import metrics


REPLACEMENTS={'carets3_phase':'carets3_phase_anchor64',
              'carets4_phase':'carets4_phase_anchor64'}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,
        default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def select(x,y,library,reference):
    ref=metrics(x,y,reference);rows=[]
    for key,path in library.items():
        row={'key':key,**metrics(x,y,path)}
        row['eligible']=bool(row['raw_accuracy']>ref['raw_accuracy'] and
                             row['mse']<=ref['mse']+1e-8)
        rows.append(row)
    eligible=[r for r in rows if r['eligible']]
    chosen=min(eligible,key=lambda r:(-r['raw_accuracy'],r['mse'],r['key']))['key'] if eligible else 'cosdir'
    return chosen,{'reference':ref,'candidates':rows,'chosen':chosen,
                  'rule':'raw > same bounded CosDir and MSE <= CosDir + 1e-8; else CosDir'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--archive',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run).resolve();archive=Path(args.archive).resolve()
    out=Path(args.output).resolve();started=time.perf_counter();completed=[];summary=[]
    if out==archive:raise ValueError('V1 archive and canonical V2 must differ')
    protocol=out/'repair_protocol.json'
    if not protocol.exists():
        write(protocol,{'client_date':'2026-09-30','declared_utc':datetime.now(timezone.utc).isoformat(),
            'version':2,'repair_scope':'post-protocol numerical repair and competitor rule correction',
            'reason':'V1 wrongly kept both known-bug original CaReTS paths and corrected paths in a 15-method pool. Replace those two slots instead; known-bug paths are historical only and the original 13-method validation budget is preserved.',
            'formula':'Unchanged learned signed32 + actual observed xlast64, with the same declared physical bounds.',
            'selection':'Original 11 remaining competitor methods plus two fixed corrected CaReTS methods. Identical original validation eligibility/tie/fallback rule; original main-method locks remain unchanged.',
            'test_selection':'No test performance is used to choose the repair or pool. Lock on saved validation paths before rereading and rescoring held-test paths.',
            'replacements':REPLACEMENTS,'competitor_cardinality':13,
            'retraining':False,'new_neural_inference':False,'v1_archive':str(archive),
            'v1_protocol_sha256':digest(archive/'repair_protocol.json'),
            'script_sha256':digest(Path(__file__))})
    for archived_shared in sorted(archive.glob('*/shared_seed*')):
        if not (archived_shared/'result.json').exists():continue
        domain=archived_shared.parent.name;seed=archived_shared.name.split('seed')[1]
        source_cases=[p for p in sorted((run/domain).glob('*_seed'+seed))
            if not p.name.startswith('shared') and (p/'result.json').exists() and
            (archive/domain/p.name/'result.json').exists()]
        if not source_cases:continue
        shared_out=out/domain/archived_shared.name
        archived_shared_lock=json.loads((archived_shared/'repair_lock.json').read_text())
        write(shared_out/'repair_lock.json',{**archived_shared_lock,
            'protocol':str(protocol),'v1_archive_shared':str(archived_shared),
            'v1_trace_sha256':digest(archived_shared/'signed_trace.npz')})
        selections={}
        for case in source_cases:
            original=json.loads((case/'validation_lock.json').read_text())
            archived_case=archive/domain/case.name
            v1lock=json.loads((archived_case/'repair_lock.json').read_text())
            original_keys=[r['key'] for r in original['selection']['competitor']['candidates']]
            if len(original_keys)!=13 or not set(REPLACEMENTS).issubset(original_keys):
                raise AssertionError('Unexpected original competitor pool')
            with np.load(case/'predictions.npz') as baseline,np.load(archived_case/'predictions.npz') as fixed:
                x,y=baseline['validation__x'],baseline['validation__y']
                library={REPLACEMENTS.get(key,key):
                    (fixed['validation__'+REPLACEMENTS[key]] if key in REPLACEMENTS else baseline['validation__'+key])
                    for key in original_keys}
                chosen,table=select(x,y,library,baseline['validation__cosdir'])
                valfixed={key:fixed['validation__'+key] for key in REPLACEMENTS.values()}
                valfixed['competitor_corrected']=library[chosen]
            if len(library)!=13 or set(REPLACEMENTS)&set(library):
                raise AssertionError('V2 pool must replace, never retain, known-bug methods')
            case_out=out/domain/case.name
            lock={'original_case_lock':str(case/'validation_lock.json'),
                'original_case_lock_sha256':digest(case/'validation_lock.json'),
                'original_choices':original['choices'],
                'choices':{'competitor_corrected':chosen},'selection':table,
                'original_competitor_keys':original_keys,
                'corrected_competitor_keys':list(library),'replaced_fixed_keys':REPLACEMENTS,
                'competitor_cardinality':13,'rule_version':2,
                'selection_data':'saved original validation only','protocol':str(protocol),
                'test_labels_used_for_selection':False,'retraining':False,
                'new_neural_inference':False,'v1_archive_case':str(archived_case),
                'v1_archive_choice':v1lock['choices']['competitor_corrected'],
                'v1_case_lock_sha256':digest(archived_case/'repair_lock.json')}
            write(case_out/'repair_lock.json',lock)
            selections[case.name]=(case,archived_case,case_out,lock,valfixed)
        # All v2 case locks exist before the archived held-test traces/paths
        # are copied, reread, or rescored for this dataset/seed.
        assert all((v[2]/'repair_lock.json').exists() for v in selections.values())
        shutil.copyfile(archived_shared/'signed_trace.npz',shared_out/'signed_trace.npz')
        shared_result=json.loads((archived_shared/'result.json').read_text())
        write(shared_out/'result.json',{**shared_result,
            'test_authorized_by_repair_locks':[str(v[2]/'repair_lock.json') for v in selections.values()],
            'trace_reused_from_v1':True,'protocol':str(protocol)})
        for case,archived_case,case_out,lock,valfixed in selections.values():
            chosen=lock['choices']['competitor_corrected'];paths={'validation__'+k:v for k,v in valfixed.items()}
            with np.load(case/'predictions.npz') as baseline,np.load(archived_case/'predictions.npz') as fixed:
                paths.update({'test__'+k:fixed['test__'+k] for k in REPLACEMENTS.values()})
                paths['test__competitor_corrected']=(fixed['test__'+chosen] if chosen in REPLACEMENTS.values()
                    else baseline['test__'+chosen])
                test_scores={k:metrics(baseline['test__x'],baseline['test__y'],paths['test__'+k])
                             for k in list(REPLACEMENTS.values())+['competitor_corrected']}
            np.savez_compressed(case_out/'predictions.npz',**paths)
            result={'test':test_scores,'choices':lock['choices'],'selection':lock['selection'],
                'original_choices':lock['original_choices'],
                'repair_status':'post-protocol numerical repair; v2 replaces two CaReTS slots',
                'rule_version':2,'competitor_cardinality':13,
                'shared_checks':shared_result['checks'],'reload_checks':shared_result['reload_checks'],
                'seconds':time.perf_counter()-started,'original_case':str(case),
                'shared_trace':str(shared_out/'signed_trace.npz'),
                'v1_archive_case':str(archived_case),'v1_archive_choice':lock['v1_archive_choice']}
            write(case_out/'result.json',result);completed.append(str(case_out))
            summary.append({'domain':domain,'case':case.name,
                'competitor_original':lock['original_choices']['competitor'],
                'competitor_v1':lock['v1_archive_choice'],'competitor_corrected':chosen})
            print('v2 %s %s: competitor=%s'%(domain,case.name,chosen),flush=True)
    write(out/'completion.json',{'client_date':'2026-09-30','cases':len(completed),'completed':completed,
        'seconds':time.perf_counter()-started,'summary':summary,
        'all_matrix_cases_available':(run/'completion.json').exists(),
        'rule_version':2,'competitor_cardinality':13})


if __name__=='__main__':main()
