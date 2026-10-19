"""CPU-only exact finite-distribution probe; no real dataset or novelty claim.

The protocol (including distributions and candidate grids) is written before
any results. A grid oracle is an oracle only over its finite candidate set.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trend_lab.decoding import project_trajectory, score_posterior_candidates


def random_walk(increments, weights):
    paths, masses = [], []
    for i, j in itertools.product(range(len(increments)), repeat=2):
        paths.append([increments[i], increments[i] + increments[j]])
        masses.append(weights[i] * weights[j])
    return paths, masses


def protocol():
    symmetric, sym_weights = random_walk([-.4, .4], [.5, .5])
    skewed, skew_weights = random_walk([-.8, .2], [.2, .8])
    biased, biased_weights = random_walk([-.3, .2], [.3, .7])
    return {
        "version": 1,
        "purpose": "Separate finite-step optimization, top1-gap surrogate, posterior accuracy, and true MSE risk; no novelty claim.",
        "device": "cpu", "dtype": "float64", "history": [0., 0.], "horizon": 2,
        "scales": [1, 2], "epsilon_per_scale": [.05, .05],
        "primary_direction": "mean multiscale three-class expected accuracy",
        "secondary_direction": "raw first-difference sign expected accuracy including exact zero; not used to select the primary winner",
        "paired_scales": [[1], [1, 2]],
        "max_rms_from_base": .5,
        "alpha_grid": [.1, .3, 1., 3., 10., 100.],
        "legacy_steps": 40, "converged_max_steps": 10000, "convergence_tol": 1e-10,
        "margin_factors": [0.5, 1.01, 1.5, 3.],
        "margin_candidate_rule": "all length-2 down/flat/up increment patterns times epsilon_scale1*factor, cumulative from last history; RMS clip to the same base ball",
        "candidate_set": "base + paired legacy/converged soft paths for all alpha + fixed margin paths; no ground-truth mean added unless already base",
        "posterior_selection": "maximize mean p(class(candidate)); exact ties minimize displacement from base then candidate index",
        "oracle_grid": {"absolute_min": -1., "absolute_max": 1., "step": .025,
                        "rule": "all absolute length-2 paths on grid, discard outside RMS ball, add every practical candidate"},
        "oracle_rule": "expected-accuracy maximizer over finite grid+practical set; risk-limited oracle also requires exact MSE regret <=1e-12",
        "risk_definition": "E[MSE(z,Y)-MSE(base,Y)] = mean((z-mu)^2-(base-mu)^2); exact known mu, not closeness-to-base",
        "scenarios": {
            "symmetric_random_walk": {"paths": symmetric, "weights": sym_weights, "base_bias": [0., 0.]},
            "skewed_zero_mean_increments": {"paths": skewed, "weights": skew_weights, "base_bias": [0., 0.]},
            "correlated_two_step_conflicting_modes": {
                "paths": [[.4, .8], [.4, -.8], [-.8, -.4]], "weights": [.35, .34, .31], "base_bias": [0., 0.],
                "mode_conflict": "P(step1 up)=.69, P(step2 up)=.66, P(two-step slope down)=.65; these three modes cannot hold in one path."},
            "biased_base": {"paths": biased, "weights": biased_weights, "base_bias": [-.25, .30]},
        },
        "limitations": ["Known exact posteriors and conditional mean are unavailable in general real forecasting.",
                        "Grid oracle is not a continuous/global Bayes optimum.",
                        "The practical candidate set changes with the chosen scale set; exact grid uses a common fixed grid plus their union.",
                        "No train/test estimate, learned classifier, or hyperparameter generalization is tested.",
                        "Only an additive expected direction objective is tested; joint probabilities matter for non-additive path utilities."],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def labels(paths, scales, epsilon):
    """Independent NumPy label construction, histories are the fixed zeros."""
    path = np.asarray(paths, dtype=np.float64)
    joined = np.concatenate([np.zeros((len(path), 2)), path], axis=1)
    end = np.arange(2, 4)
    slopes = np.stack([(joined[:, end] - joined[:, end-k]) / k for k in scales], axis=1)
    eps = np.asarray(epsilon)[None, :, None]
    return np.where(slopes > eps, 2, np.where(slopes < -eps, 0, 1))


def posterior(paths, weights, scales, epsilon):
    cls = labels(paths, scales, epsilon)
    return np.stack([(weights[:, None, None] * (cls == c)).sum(axis=0) for c in range(3)], axis=-1)


def evaluate(paths, outcomes, weights, base, mu, scales, epsilon):
    candidate_labels = labels(paths, scales, epsilon)
    truth = labels(outcomes, scales, epsilon)
    # Direct exact integration over full correlated paths, separate from helper.
    accuracy = ((candidate_labels[:, None] == truth[None]).mean(axis=(2, 3)) * weights[None]).sum(axis=1)
    raw_truth = labels(outcomes, [1], [0.])
    raw_candidates = labels(paths, [1], [0.])
    raw_accuracy = ((raw_candidates[:, None] == raw_truth[None]).mean(axis=(2, 3)) * weights[None]).sum(axis=1)
    mse = (((paths[:, None] - outcomes[None])**2).mean(axis=2) * weights[None]).sum(axis=1)
    base_mse = (((base[None] - outcomes)**2).mean(axis=1) * weights).sum()
    regret = ((paths-mu)**2 - (base-mu)**2).mean(axis=1)
    np.testing.assert_allclose(mse-base_mse, regret, atol=1e-12, rtol=1e-12)
    return {"expected_3class_accuracy": accuracy, "expected_raw_sign_accuracy": raw_accuracy,
            "expected_mse": mse, "exact_mse_regret_vs_base": regret,
            "rms_from_base": np.sqrt(((paths-base)**2).mean(axis=1))}


def winner(scores, paths, base, mask=None):
    indices = np.arange(len(paths)) if mask is None else np.flatnonzero(mask)
    # Exact ties only: probabilities have their literal finite-precision values.
    return min(indices.tolist(), key=lambda i: (-float(scores[i]), float(((paths[i]-base)**2).mean()), i))


def build_candidates(history, base, posterior_all, scales, cfg):
    names, paths, solver = ['base'], [base.copy()], {}
    tbase = torch.tensor(base).reshape(1, 2, 1)
    epsilon = torch.tensor(cfg['epsilon_per_scale'][:len(scales)], dtype=torch.float64).reshape(-1, 1)
    probs = torch.tensor(posterior_all[:len(scales)], dtype=torch.float64).reshape(1, len(scales), 2, 1, 3)
    for alpha in cfg['alpha_grid']:
        for kind, steps, tol in [('legacy40', cfg['legacy_steps'], None),
                                 ('converged', cfg['converged_max_steps'], cfg['convergence_tol'])]:
            name = f'{kind}_alpha_{alpha:g}'
            path, diag = project_trajectory(history, tbase, probs, scales, epsilon,
                                            strength=alpha, steps=steps,
                                            max_rms=cfg['max_rms_from_base'], convergence_tol=tol,
                                            return_diagnostics=True)
            names.append(name); paths.append(path.numpy().reshape(2)); solver[name] = diag
            if tol is not None and not diag['all_converged']:
                raise RuntimeError(f'Converged reference did not converge: {name}: {diag}')
    for factor in cfg['margin_factors']:
        for pattern in itertools.product([-1, 0, 1], repeat=2):
            path = np.cumsum(np.array(pattern) * cfg['epsilon_per_scale'][0] * factor)
            delta = path-base; rms = np.sqrt((delta**2).mean())
            path = base + delta * min(1., cfg['max_rms_from_base']/rms) if rms else base.copy()
            names.append(f'margin_{factor:g}_{pattern[0]}_{pattern[1]}'); paths.append(path)
    return names, np.stack(paths), solver


def run(cfg):
    history = torch.zeros(1, 2, 1, dtype=torch.float64)
    output = {}
    for scenario, spec in cfg['scenarios'].items():
        outcomes = np.asarray(spec['paths'], dtype=np.float64); weights = np.asarray(spec['weights'], dtype=np.float64)
        np.testing.assert_allclose(weights.sum(), 1., atol=1e-15)
        mu = (weights[:, None]*outcomes).sum(axis=0); base = mu+np.array(spec['base_bias'])
        # Eliminate only roundoff in analytically zero conditional means; record it.
        mu[np.abs(mu) < 1e-15] = 0.; base = mu+np.array(spec['base_bias'])
        p_all = posterior(outcomes, weights, cfg['scales'], cfg['epsilon_per_scale'])
        candidate_sets = {str(scales): build_candidates(history, base, p_all, scales, cfg) for scales in cfg['paired_scales']}
        grid_cfg=cfg['oracle_grid'];axis=np.arange(grid_cfg['absolute_min'], grid_cfg['absolute_max']+grid_cfg['step']/2, grid_cfg['step'])
        axis=np.round(axis, 12)
        grid=np.array(list(itertools.product(axis, repeat=2)))
        grid=grid[np.sqrt(((grid-base)**2).mean(axis=1)) <= cfg['max_rms_from_base']+1e-12]
        oracle_paths=np.concatenate([grid]+[entry[1] for entry in candidate_sets.values()])
        result={"conditional_mean":mu.tolist(),"base":base.tolist(),"posterior_multiscale":p_all.tolist(),
                "marginal_modes":p_all.argmax(axis=-1).tolist(),"paired_scales":{}}
        for scales in cfg['paired_scales']:
            key=str(scales); names,paths,solver=candidate_sets[key]; eps=cfg['epsilon_per_scale'][:len(scales)]
            metrics=evaluate(paths,outcomes,weights,base,mu,scales,eps)
            probabilities=torch.tensor(p_all[:len(scales)]).reshape(1,len(scales),2,1,3)
            helper_scores=score_posterior_candidates(history,torch.tensor(paths).reshape(-1,1,2,1),probabilities,scales,torch.tensor(eps,dtype=torch.float64).reshape(-1,1)).numpy()[:,0]
            np.testing.assert_allclose(helper_scores,metrics['expected_3class_accuracy'],atol=1e-12,rtol=1e-12)
            assert np.all(metrics['rms_from_base'] <= cfg['max_rms_from_base']+1e-12)
            best=winner(helper_scores,paths,base)
            oracle=evaluate(oracle_paths,outcomes,weights,base,mu,scales,eps)
            unrestricted=winner(oracle['expected_3class_accuracy'],oracle_paths,base)
            risk_limited=winner(oracle['expected_3class_accuracy'],oracle_paths,base,oracle['exact_mse_regret_vs_base']<=1e-12)
            rows=[]
            for i,name in enumerate(names):
                row={'name':name,'path':paths[i].tolist(),**{metric:float(values[i]) for metric,values in metrics.items()}}
                if name in solver:row['solver']=solver[name]
                rows.append(row)
            row={'finite_candidates':len(paths),'oracle_candidates_including_duplicates':len(oracle_paths),
                 'selected_full_posterior_candidate':rows[best], 'candidates':rows,
                 'oracle_expected_accuracy':{'path':oracle_paths[unrestricted].tolist(),**{metric:float(values[unrestricted]) for metric,values in oracle.items()}},
                 'oracle_with_true_mse_constraint':{'path':oracle_paths[risk_limited].tolist(),**{metric:float(values[risk_limited]) for metric,values in oracle.items()}},
                 'independent_marginal_mode_accuracy_upper_bound':float(p_all[:len(scales)].max(axis=-1).mean())}
            assert row['oracle_expected_accuracy']['expected_3class_accuracy']+1e-12 >= rows[best]['expected_3class_accuracy']
            result['paired_scales'][key]=row
        output[scenario]=result
    return output


def markdown(results, output):
    lines=['# Exact finite-distribution posterior decision probe', '',
           'CPU-only synthetic evidence. Primary accuracy is three-class and paired single/multiscale; raw sign is a separate metric. No novelty claim or real-data generalization claim.', '',
           '| Scenario | Scales | Decision | Expected 3-class % | Expected raw-sign % | Exact MSE regret |',
           '|---|---|---|---:|---:|---:|']
    for scenario,result in results.items():
        for scales,row in result['paired_scales'].items():
            selected=[('base',row['candidates'][0])]
            selected += [(name,next(r for r in row['candidates'] if r['name']==name)) for name in ('legacy40_alpha_10','converged_alpha_10')]
            selected += [('full posterior (finite candidates)',row['selected_full_posterior_candidate']),
                         ('finite-grid oracle',row['oracle_expected_accuracy']),('finite-grid oracle + true MSE constraint',row['oracle_with_true_mse_constraint'])]
            for name,r in selected:
                lines.append(f"| {scenario} | {scales} | {name} | {100*r['expected_3class_accuracy']:.3f} | {100*r['expected_raw_sign_accuracy']:.3f} | {r['exact_mse_regret_vs_base']:+.8f} |")
    lines += ['', 'The oracle includes the fixed grid and every practical candidate, so its score bounds this finite candidate comparison only. It is not a global continuous optimum. The exact-risk oracle uses the known conditional mean and is not a deployable method with only class probabilities.', '',
              'A conditional-mean base has minimum expected MSE. Any changed path has nonnegative MSE regret there; directional improvement need not preserve MSE. The biased-base case separates this structural limit from correctable baseline error.', '',
              'Legacy40 and converged use the same top1-gap objective and RMS cap; posterior selection changes the decision objective to the probability of the candidate’s actual class. A changed objective can improve direction while increasing MSE. Results.json retains every alpha, candidate, convergence residual, and paired scale result.', '']
    (output/'evidence.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True,type=Path);args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Refusing to overwrite an existing experiment')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cfg=protocol();cfg['frozen_utc']=datetime.now(timezone.utc).isoformat()
    dependencies=[Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'trend_lab'/'decoding.py']
    cfg['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    write_json(args.output_dir/'protocol.json',cfg)
    # No outcome scoring precedes the saved protocol. No real data are read.
    torch.set_num_threads(1)
    results=run(cfg);write_json(args.output_dir/'results.json',results);markdown(results,args.output_dir)
    print(json.dumps({'output_dir':str(args.output_dir.resolve()),'scenarios':len(results),'paired_scale_settings':8,'cpu_only':True}))


if __name__=='__main__':main()
