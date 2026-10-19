"""Independent exhaustive re-evaluation; does not import the decision solver."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent.parent
REPO = RESEARCH.parent.parent
SOURCE = ROOT.parent / 'trajectory_mechanisms_20260930' / 'results.json'
checks = []


def check(name, actual, expected, tolerance=1e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
    ok = a.shape == b.shape and error <= tolerance
    checks.append({'name': name, 'pass': bool(ok), 'max_absolute_error': error})


def direction(path, anchor):
    out, previous = [], anchor
    for value in path:
        out.append(0 if value < previous else (2 if value > previous else 1))
        previous = value
    return out


def evaluate(path, support, mass, anchor=0.):
    classes = direction(path, anchor)
    risk, utility = 0., 0.
    for target, probability in zip(support, mass):
        target_classes = direction(target, anchor)
        risk += probability * sum((prediction-truth)**2 for prediction,truth in zip(path,target)) / len(path)
        utility += probability * sum(pred == truth for pred,truth in zip(classes,target_classes)) / len(path)
    return float(risk), float(utility)


def posterior(support, mass, anchor=0.):
    p = np.zeros((len(support[0]),3))
    for target, weight in zip(support,mass):
        for h,c in enumerate(direction(target,anchor)):
            p[h,c] += weight
    return p


def decision_objective(path, mean, p, strength, reward):
    cls = direction(path,0.)
    squared = sum((value-m)**2 for value,m in zip(path,mean)) / len(path)
    utility = 0.
    for h,c in enumerate(cls):
        if reward == 'posterior':
            utility += p[h,c]
        else:
            top = int(np.argmax(p[h]))
            if c == top:
                utility += 1. if reward == 'top1' else np.sort(p[h])[-1]-np.sort(p[h])[-2]
    return float(squared-strength*utility/len(path))


def main():
    r=json.loads(SOURCE.read_text())
    known=r['known_joint_dp_vs_enumeration']
    support,mass=known['support_paths'],known['weights']
    p=posterior(support,mass)
    check('known posterior from genuine joint support',p,known['direction_posterior_down_zero_up'])
    check('known weights sum one',sum(mass),1.)
    candidates=list(itertools.product(known['shared_grid'],repeat=len(known['mean'])))+[known['mean']]
    base_risk,_=evaluate(known['mean'],support,mass)
    for i,row in enumerate(known['decisions']):
        risk,utility=evaluate(row['path'],support,mass)
        check('known[%d] expected MSE'%i,risk,row['expected_mse'])
        check('known[%d] expected raw accuracy'%i,utility,row['expected_raw_accuracy'])
        check('known[%d] risk difference'%i,risk-base_risk,row['estimated_delta_risk'])
        actual=decision_objective(row['path'],known['mean'],p,row['lambda'],row['reward'])
        optimum=min(decision_objective(path,known['mean'],p,row['lambda'],row['reward']) for path in candidates)
        check('known[%d] grid union base exhaustive optimality'%i,actual,optimum)
    u=r['unsupported_pareto_point']
    grid=sorted(set(value for path in u['support_paths'] for value in path))
    candidates=list(itertools.product(grid,repeat=len(u['mean'])))+[u['mean']]
    base_risk,_=evaluate(u['mean'],u['support_paths'],u['weights'])
    points=[(evaluate(path,u['support_paths'],u['weights'])[0]-base_risk,
             evaluate(path,u['support_paths'],u['weights'])[1]) for path in candidates]
    for i,row in enumerate(u['pareto_points']):
        risk,utility=evaluate(row['path'],u['support_paths'],u['weights'])
        check('Pareto[%d] expected MSE'%i,risk,row['expected_mse'])
        check('Pareto[%d] expected accuracy'%i,utility,row['expected_raw_accuracy'])
        check('Pareto[%d] conditional risk difference'%i,risk-base_risk,row['squared_distance_to_mean'])
        dominated=any(rr <= risk-base_risk+1e-12 and uu >= utility-1e-12 and
                      (rr < risk-base_risk-1e-12 or uu > utility+1e-12) for rr,uu in points)
        check('Pareto[%d] not dominated'%i,dominated,False,tolerance=0.)
    witness=u['nonsupported_budget_optimum'];rr=witness['squared_distance_to_mean'];uu=witness['expected_raw_accuracy']
    budget_optimum=max(utility for risk,utility in points if risk <= u['budget_tau']+1e-12)
    check('nonsupported constrained budget optimum',uu,budget_optimum)
    lower,upper=0.,np.inf
    for risk,utility in points:
        delta_u=uu-utility;delta_r=rr-risk
        if delta_u>1e-12:lower=max(lower,delta_r/delta_u)
        elif delta_u<-1e-12:upper=min(upper,delta_r/delta_u)
    check('no lambda can support witness',lower>upper+1e-12,True,tolerance=0.)
    check('uniform no-signal posterior',r['no_direction_signal']['posterior'],np.full((3,3),1/3))
    for i,row in enumerate(r['no_direction_signal']['decisions']):
        support=list(itertools.product([-1.,0.,1.],repeat=3));mass=[1/27]*27
        risk,utility=evaluate(row['path'],support,mass)
        check('no signal[%d] utility'%i,utility,1/3)
        check('no signal[%d] actual MSE'%i,risk,row['expected_mse'])
    for i,row in enumerate(r['same_p_different_mean']):
        p=posterior(row['support_paths'],row['weights'])
        check('same p case[%d]'%i,p,[[.2,0.,.8]])
        risk,_=evaluate([.1],row['support_paths'],row['weights']);base,_=evaluate([0.],row['support_paths'],row['weights'])
        check('same p actual delta[%d]'%i,risk-base,row['candidate_minus_base_expected_mse'])
    check('same posterior risk signs reversed',r['same_p_different_mean'][0]['candidate_minus_base_expected_mse']>0>
          r['same_p_different_mean'][1]['candidate_minus_base_expected_mse'],True,tolerance=0.)
    quant=r['true_zero_quantization_negative_control']
    for label in ['mean_prediction','ordinary_rounding_no_direction_model']:
        row=quant[label];risk,utility=evaluate(row['path'],quant['support_paths'],quant['weights'])
        check('rounding '+label+' MSE',risk,row['expected_mse'])
        check('rounding '+label+' accuracy',utility,row['expected_raw_accuracy'])
    check('rounding direction change pp',100*(quant['ordinary_rounding_no_direction_model']['expected_raw_accuracy']-
          quant['mean_prediction']['expected_raw_accuracy']),60.)
    check('rounding relative MSE change',quant['ordinary_rounding_no_direction_model']['expected_mse']/quant['mean_prediction']['expected_mse']-1,.25)
    clean=r['artificial_cleaning_zero_control']
    check('cleaning original classes',direction(clean['original_path'],0.),clean['original_classes'])
    check('cleaning artificial classes',direction(clean['forward_replacement_path'],0.),clean['cleaned_classes'])
    singular=r['bayes_mean_zero_budget']
    check('declared Bayes mean zero',singular['mean'],[0.],tolerance=0.)
    base_risk,_=evaluate([0.],singular['support_paths'],singular['weights'])
    for i,row in enumerate(singular['decisions']):
        risk,utility=evaluate(row['path'],singular['support_paths'],singular['weights'])
        check('Bayes[%d] expected MSE'%i,risk,row['expected_mse'])
        check('Bayes[%d] expected utility'%i,utility,row['expected_raw_accuracy'])
        check('Bayes[%d] exact additional risk'%i,risk-base_risk,row['path'][0]**2)
    source_files=[SOURCE,REPO/'DTL/trend_lab/trajectory_decision.py',REPO/'DTL/trend_lab/tests/test_trajectory_decision.py',
                  SOURCE.parent/'probe.py']
    report={'audit_implementation':'independent Python loops and exhaustive enumeration, no decision solver import',
            'checks':len(checks),'failures':sum(not c['pass'] for c in checks),
            'reported_probe_assertions':r['independent_exact_checks'],
            'non_supported_lambda_interval':{'required_lower_bound':float(lower),'required_upper_bound':float(upper)},
            'source_sha256':{str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
            'details':checks}
    (ROOT/'independent_mechanism_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['checks','failures','reported_probe_assertions','non_supported_lambda_interval']}))
    if report['failures']:raise AssertionError('independent mechanism audit failed')


if __name__=='__main__':main()
