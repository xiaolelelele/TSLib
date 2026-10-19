"""Post hoc descriptive diagnostics of locked paths, with no new selection.

Full posterior equals p_second + top-gap reward - bottom-gap penalty. This
script validates that algebra on finite fixed-lambda examples and then uses
only saved probabilities/paths/labels from the completed TIME pilot cases.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


LAMBDAS=[0.,.001,.01,.1,1.]


def raw_classes(anchor,path):
    prior=np.concatenate((anchor[:,None,:],path[:,:-1,:]),axis=1)
    return np.where(path<prior,0,np.where(path>prior,2,1))


def components(p,classes):
    ordered=np.sort(p,axis=-1)
    first,second,third=ordered[...,2],ordered[...,1],ordered[...,0]
    top=np.argmax(p,axis=-1);lowest=np.argmin(p,axis=-1)
    gap12=first-second;gap23=second-third
    probability=np.take_along_axis(p,classes[...,None],axis=-1)[...,0]
    reward=gap12*(classes==top)
    penalty=gap23*(classes==lowest)
    return probability,reward,penalty,second


def small_tests():
    """All finite paths and fixed lambdas; does not call any decoder."""
    rng=np.random.default_rng(260930121)
    posterior_sets=[np.array([[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]]),
                    np.array([[.5,.5,0.],[0.,.5,.5],[.5,0.,.5]]),
                    np.full((3,3),1/3),rng.dirichlet([.7,.7,.7],size=3)]
    mean=np.array([.1,-.3,.2]);base=np.array([.2,.1,.3])
    candidates=np.array(list(itertools.product([-1.,0.,1.],repeat=3))+[tuple(base)])
    labels=raw_classes(np.zeros((len(candidates),1)),candidates[:,:,None])[:,:,0]
    checks=[]
    def verify(name,a,b,tol=1e-12):
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        checks.append({'name':name,'pass':bool(error<=tol),'max_absolute_error':error})
    for which,p in enumerate(posterior_sets):
        repeated=np.broadcast_to(p,(len(candidates),)+p.shape)
        assigned,topreward,bottompenalty,second=components(repeated,labels)
        verify('case%d all-path pointwise probability identity'%which,assigned,second+topreward-bottompenalty)
        squared=np.mean((candidates-mean)**2,axis=1)
        for strength in LAMBDAS:
            full=squared-strength*assigned.mean(1)
            augmented=squared-strength*topreward.mean(1)-strength*second.mean(1)+strength*bottompenalty.mean(1)
            verify('case%d lambda%g all path objective identity'%(which,strength),full,augmented)
            verify('case%d lambda%g exhaustive minima identity'%(which,strength),full.min(),augmented.min())
            if strength>0:
                gap=squared-strength*topreward.mean(1)
                full_index=int(np.argmin(full));gap_index=int(np.argmin(gap))
                verify('case%d lambda%g full lowest penalty cannot exceed topgap optimum'%(which,strength),
                       float(bottompenalty[full_index].mean()<=bottompenalty[gap_index].mean()+1e-12),1.,0.)
        if np.all(np.sort(p,axis=-1)[:,1]==np.sort(p,axis=-1)[:,0]):
            # Equal secondary probabilities make bottom penalty identically zero.
            for strength in LAMBDAS:
                full=squared-strength*assigned.mean(1)
                gap=squared-strength*topreward.mean(1)
                verify('case%d lambda%g same objective up to common constant'%(which,strength),
                       full-full[0],gap-gap[0])
    # Binary class actions with zero excluded also have a path-constant difference.
    p=np.array([[.2,0.,.8],[.7,0.,.3],[.1,0.,.9]])
    nonzero=np.all(labels!=1,axis=1)
    assigned,topreward,bottompenalty,second=components(np.broadcast_to(p,(nonzero.sum(),)+p.shape),labels[nonzero])
    verify('binary action paths do not choose zero lowest',bottompenalty,np.zeros_like(bottompenalty))
    for strength in LAMBDAS:
        full=np.mean((candidates[nonzero]-mean)**2,axis=1)-strength*assigned.mean(1)
        gap=np.mean((candidates[nonzero]-mean)**2,axis=1)-strength*topreward.mean(1)
        verify('binary lambda%g same objective up to common constant'%strength,full-full[0],gap-gap[0])
    return {'seed':260930121,'candidate_paths_per_ternary_case':len(candidates),'fixed_lambda_grid':LAMBDAS,
            'checks':len(checks),'failures':sum(not row['pass'] for row in checks),'details':checks}


def path_statistics(p,labels,mu,path,anchor,locked_choice):
    classes=raw_classes(anchor,path)
    assigned,topreward,penalty,second=components(p,classes)
    sorted_p=np.sort(p,axis=-1);gap=sorted_p[...,1]-sorted_p[...,0];lowest=np.argmin(p,axis=-1)
    strength=None if locked_choice is None else float(locked_choice.rsplit('_',1)[1])
    squared=float(np.mean((path-mu)**2))
    out={'locked_choice':locked_choice,'locked_lambda':strength,
         'lowest_class_hit_rate':float(np.mean(classes==lowest)),
         'strict_lowest_class_hit_rate':float(np.mean((classes==lowest)&(gap>0))),
         'predicted_lowest_identity_fraction':[float(np.mean((classes==lowest)&(lowest==k))) for k in range(3)],
         'bottom_gap_penalty_mean_unscaled':float(np.mean(penalty)),
         'bottom_gap_penalty_scaled_by_locked_lambda':None if strength is None else float(strength*np.mean(penalty)),
         'estimated_direction_utility':float(np.mean(assigned)),
         'top_gap_reward_mean':float(np.mean(topreward)),
         'mean_squared_distance_to_mu':squared,'observed_raw_accuracy':float(np.mean(classes==labels))}
    if strength is not None:
        full=squared-strength*float(np.mean(assigned))
        top=squared-strength*float(np.mean(topreward))
        reconstructed=top-strength*float(np.mean(second))+strength*float(np.mean(penalty))
        out.update(posterior_objective_at_locked_lambda=full,top_gap_objective_at_locked_lambda=top,
                   algebra_reconstructed_posterior_objective=reconstructed,
                   algebra_max_error=abs(full-reconstructed))
    else:out['algebra_max_error']=None
    return out


def analyze_case(case):
    lock=json.loads((case/'validation_lock.json').read_text());meta=lock['metadata']
    with np.load(case/'predictions.npz') as archive:
        p=np.asarray(archive['test__probabilities'],np.float64)
        anchor=archive['test__x'][:,-1,:];truth=raw_classes(anchor,archive['test__y']);mu=archive['test__mean']
        full=archive['test__posterior_selected'];topgap=archive['test__top1_gap_selected']
        ordered=np.sort(p,axis=-1);gap=ordered[...,1]-ordered[...,0];lowest=np.argmin(p,axis=-1)
        first=path_statistics(p,truth,mu,full,anchor,lock['choices']['posterior'])
        second=path_statistics(p,truth,mu,topgap,anchor,lock['choices']['top1_gap'])
        same_lambda=first['locked_lambda'] is not None and second['locked_lambda'] is not None and first['locked_lambda']==second['locked_lambda']
        # Check only algebra of already saved paths at fixed scalar weights;
        # these are not additional decoded candidates and do not reselect.
        checks=[]
        for family,path in [('posterior_selected',full),('top1_gap_selected',topgap)]:
            classes=raw_classes(anchor,path)
            assigned,reward,penalty,psecond=components(p,classes)
            checks.append({'name':family+' pointwise decomposition','max_error':float(np.max(np.abs(assigned-(psecond+reward-penalty))))})
            squared=float(np.mean((path-mu)**2))
            for strength in LAMBDAS:
                original=squared-strength*float(np.mean(assigned))
                alternative=squared-strength*float(np.mean(reward))-strength*float(np.mean(psecond))+strength*float(np.mean(penalty))
                checks.append({'name':family+' fixed_lambda_'+str(strength)+' objective decomposition','max_error':abs(original-alternative)})
        penalty_order=None
        if same_lambda and first['locked_lambda']>0:
            full_penalty=components(p,raw_classes(anchor,full))[2].mean((1,2))
            top_penalty=components(p,raw_classes(anchor,topgap))[2].mean((1,2))
            penalty_order={'max_full_minus_topgap_per_origin_penalty':float(np.max(full_penalty-top_penalty)),
                           'violations_above_1e_10':int(np.sum(full_penalty>top_penalty+1e-10))}
            checks.append({'name':'same_positive_lambda full lowest penalty ordering all origins',
                           'max_error':float(max(0.,np.max(full_penalty-top_penalty)))})
        return {'case':str(case),'dataset':meta['spec']['name'],'backbone':lock['backbone'],'seed':lock['seed'],
                'test_shape':list(mu.shape),'mean_second_minus_third_probability':float(np.mean(gap)),
                'second_minus_third_quantiles':{str(q):float(np.quantile(gap,q)) for q in [.1,.5,.9]},
                'exact_lowest_probability_tie_fraction':float(np.mean(gap==0)),
                'true_label_lowest_class_rate':float(np.mean(truth==lowest)),
                'true_label_strict_lowest_class_rate':float(np.mean((truth==lowest)&(gap>0))),
                'lowest_identity_fraction':[float(np.mean(lowest==k)) for k in range(3)],
                'true_label_and_lowest_identity_fraction':[float(np.mean((truth==lowest)&(lowest==k))) for k in range(3)],
                'posterior_implied_true_lowest_rate':float(np.mean(ordered[...,0])),
                'posterior_selected':first,'top1_gap_selected':second,
                'same_selected_lambda':same_lambda,
                'path_equal_coordinate_fraction':float(np.mean(full==topgap)),
                'path_equal_raw_class_fraction':float(np.mean(raw_classes(anchor,full)==raw_classes(anchor,topgap))),
                'same_positive_lambda_penalty_order':penalty_order,
                'full_minus_top1_observed_raw_accuracy':first['observed_raw_accuracy']-second['observed_raw_accuracy'],
                'full_minus_top1_observed_standardized_mse':float(np.mean((full-archive['test__y'])**2)-np.mean((topgap-archive['test__y'])**2)),
                'same_lambda_explanation_permitted':same_lambda,
                'interpretation':'observational locked-path diagnostic; different chosen lambdas do not support a same-lambda causal comparison',
                'algebra_checks':checks,'algebra_failures':sum(row['max_error']>1e-12 for row in checks)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--expected-cases',type=int,default=18);args=parser.parse_args()
    root=Path(args.results);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    tests=small_tests();(out/'posterior_top1_gap_identity_audit.json').write_text(json.dumps(tests,indent=2,allow_nan=False)+'\n')
    cases=sorted(p.parent for p in root.glob('*/*/result.json') if (p.parent/'decoder_diagnostics.json').exists())
    rows=[analyze_case(case) for case in cases]
    complete=len(rows)==args.expected_cases and (root/'completion.json').exists()
    group_rows=[]
    for dataset,backbone in sorted(set((row['dataset'],row['backbone']) for row in rows)):
        selected=[row for row in rows if row['dataset']==dataset and row['backbone']==backbone]
        record={'dataset':dataset,'backbone':backbone,'fit_seeds':len(selected),
                'same_selected_lambda_seeds':sum(row['same_selected_lambda'] for row in selected),
                'seed_mean_second_minus_third_probability':float(np.mean([row['mean_second_minus_third_probability'] for row in selected])),
                'seed_mean_true_label_lowest_rate':float(np.mean([row['true_label_lowest_class_rate'] for row in selected])),
                'seed_mean_posterior_implied_lowest_rate':float(np.mean([row['posterior_implied_true_lowest_rate'] for row in selected]))}
        for method in ['posterior_selected','top1_gap_selected']:
            for metric in ['lowest_class_hit_rate','bottom_gap_penalty_mean_unscaled']:
                record[method+'__seed_mean_'+metric]=float(np.mean([row[method][metric] for row in selected]))
        group_rows.append(record)
    result={'cases':len(rows),'expected_cases':args.expected_cases,'complete':complete,
            'analysis_status':'post hoc descriptive mechanism analysis after partial primary outcomes, no re-selection or new decoding',
            'identity':'p(c)=p2+(p1-p2)*I(c=top)-(p2-p3)*I(c=lowest)',
            'tie_rule':'argmax/argmin first class index in down,zero,up; lowest-hit tie is arbitrary, but penalty is zero when p2=p3',
            'same_positive_lambda_property':'Full minimizer has no larger mean lowest penalty than topgap minimizer on the same library; this does not order unweighted hits or actual raw/MSE',
            'aggregate_rule':'per-dataset/backbone averages over already fitted seeds; these are not independent test samples',
            'small_fixed_lambda_tests':tests,'path_algebra_checks':sum(len(row['algebra_checks']) for row in rows),
            'path_algebra_failures':sum(row['algebra_failures'] for row in rows),'group_rows':group_rows,'case_rows':rows,
            'artifact_sha256':{str((case/name).relative_to(root)):hashlib.sha256((case/name).read_bytes()).hexdigest()
                               for case in cases for name in ['predictions.npz','validation_lock.json']}}
    basename='posterior_top1_gap_mechanism' if complete else 'posterior_top1_gap_mechanism_partial'
    (out/(basename+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 完整后验与top1-gap的机制分解（2026-09-30）','',
           '这是观察过部分主实验结果后声明的机制描述，使用已保存的p、真值与两条锁定输出路径；没有重新选择lambda、没有再解码测试候选。',
           '', '**已完成%d/%d case；小lambda测试%d项、保存路径代数%d项，失败%d。**'%(len(rows),args.expected_cases,tests['checks'],result['path_algebra_checks'],tests['failures']+result['path_algebra_failures']),
           '', '将每步概率排序为p1≥p2≥p3，完整概率奖励可写成 `p2+(p1−p2)1[top]−(p2−p3)1[lowest]`。p2是相同路径间的常数；完整后验相对top1-gap额外利用的部分，恰好是对最低概率类的惩罚。p2=p3或可比较路径从不使用最低类时，二者可等价。这个基础恒等式解释了必要性的边界，不是新定理。',
           '', '同p/mu/G、同lambda>0的两个精确解还满足：full平均最低类加权penalty不高于top1-gap。把两个最优性不等式相减即可得到；未加权类命中率及真实raw/MSE却没有这一顺序保证。小例子逐固定lambda和保存的同lambda输出逐origin核对这条基本最优性性质。',
           '', '`lowest_class_hit_rate`是最终数值路径诱导的类别是否命中该步最低概率类；`penalty`是其p2−p3加权均值。概率最低类有tie时类命中率依赖固定tie规则，惩罚却为零，另列strict-lowest指标。真值落到预测最低类的比例与平均p3只作校准/排序诊断；不能据此认定逐实例概率可靠。',
           '', '| 数据/骨干 | seeds | 同lambda seeds | 平均p2−p3 | 真值最低类率 | full最低类率 | top1最低类率 | full未缩放penalty | top1未缩放penalty |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in group_rows:
        lines.append('| %s/%s | %d | %d | %.5f | %.5f | %.5f | %.5f | %.5f | %.5f |'%(row['dataset'],row['backbone'],row['fit_seeds'],row['same_selected_lambda_seeds'],row['seed_mean_second_minus_third_probability'],row['seed_mean_true_label_lowest_rate'],row['posterior_selected__seed_mean_lowest_class_hit_rate'],row['top1_gap_selected__seed_mean_lowest_class_hit_rate'],row['posterior_selected__seed_mean_bottom_gap_penalty_mean_unscaled'],row['top1_gap_selected__seed_mean_bottom_gap_penalty_mean_unscaled']))
    lines+=['','同lambda只表示两方法使用相同训练p/mu/网格和锁定权重，算法选择规则仍来自同一验证段；不把观察关联称随机化因果效应。不同lambda的seed不能把两条输出的差异单独归因于最低类惩罚；None回退没有部署lambda，亦不计为同lambda。低类惩罚越小不自动意味着真值raw更准或MSE更好，仍需主实验与竞争基准。',
            '', '完整case级选择、缩放penalty、原路径objective代数与hash见[%s.json](%s.json)。'%(basename,basename)]
    (out/(basename+'_note.md')).write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['cases','complete','path_algebra_checks','path_algebra_failures']}))
    if tests['failures'] or result['path_algebra_failures']:raise AssertionError('posterior/top1-gap algebra failed')


if __name__=='__main__':main()
