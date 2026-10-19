"""Post-result interpretation audit; reads frozen artifacts and never rewrites them.

This is not a preregistered experiment or a learned-method evaluation. It
re-scores stored paths using exact finite outcomes to separate alpha choice,
candidate coverage, and objective ranking. Run with the TFB Python environment.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def labels(paths, scales, epsilon):
    paths = np.asarray(paths, dtype=np.float64).reshape(-1, 2)
    joined = np.concatenate((np.zeros((len(paths), 2)), paths), axis=1)
    end = np.arange(2, 4)
    slopes = np.stack([(joined[:, end] - joined[:, end - k]) / k for k in scales], axis=1)
    eps = np.asarray(epsilon)[None, :, None]
    return np.where(slopes > eps, 2, np.where(slopes < -eps, 0, 1)), slopes


def metrics(paths, spec, base, mu, scales, epsilon):
    paths = np.asarray(paths, dtype=np.float64)
    outcomes, weights = np.asarray(spec['paths']), np.asarray(spec['weights'])
    cls, _ = labels(paths, scales, epsilon)
    truth, _ = labels(outcomes, scales, epsilon)
    raw, _ = labels(paths, [1], [0.])
    raw_truth, _ = labels(outcomes, [1], [0.])
    return {
        'expected_3class_accuracy': ((cls[:, None] == truth[None]).mean((2, 3)) * weights[None]).sum(1),
        'expected_raw_sign_accuracy': ((raw[:, None] == raw_truth[None]).mean((2, 3)) * weights[None]).sum(1),
        'exact_mse_regret_vs_base': (((paths - mu) ** 2 - (base - mu) ** 2).mean(1)),
        'rms_from_base': np.sqrt(((paths - base) ** 2).mean(1)),
    }


def compact(row):
    return {k: row[k] for k in ('name', 'path', 'expected_3class_accuracy',
            'expected_raw_sign_accuracy', 'exact_mse_regret_vs_base', 'rms_from_base')}


def winner(rows):
    # The post-result audit treats differences below 1e-12 as numerical ties.
    # This is not a retroactive change to the frozen protocol's literal ties.
    best = max(r['expected_3class_accuracy'] for r in rows)
    tied = [r for r in rows if abs(r['expected_3class_accuracy'] - best) <= 1e-12]
    return min(tied, key=lambda r: (r['rms_from_base'] ** 2, rows.index(r)))


def top_gap_objective(paths, base, posterior, scales, epsilon, alpha):
    _, d = labels(paths, scales, epsilon)
    classes = posterior.argmax(-1)
    sorted_p = np.sort(posterior, axis=-1)
    weights = sorted_p[..., -1] - sorted_p[..., -2]
    eps = np.asarray(epsilon)[None, :, None]
    penalty = np.where(classes[None] == 2, np.maximum(1.5 * eps - d, 0) ** 2 / 2,
                       np.where(classes[None] == 0, np.maximum(1.5 * eps + d, 0) ** 2 / 2,
                                np.maximum(np.abs(d) - eps, 0) ** 2 / 2))
    return 0.5 * ((paths - base) ** 2).sum(1) + alpha / len(scales) * (penalty * weights[None]).sum((1, 2))


def main():
    manifest = json.loads((ROOT / 'artifact_sha256.json').read_text())
    for filename, expected in manifest.items():
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        assert actual == expected, (filename, actual, expected)
    cfg = json.loads((ROOT / 'protocol.json').read_text())
    results = json.loads((ROOT / 'results.json').read_text())
    output = {'stage': 'post-result interpretation audit, not preregistered',
              'input_manifest_verified': manifest,
              'tie_convention_for_audit_summary': 'accuracy within 1e-12; then lower displacement, then listed order',
              'scenarios': {}}
    for name, scene in results.items():
        base, mu = np.asarray(scene['base']), np.asarray(scene['conditional_mean'])
        spec = cfg['scenarios'][name]
        scene_out = {}
        # The union has no fixed-margin paths and resembles the real runner's
        # library composition; alpha grid here remains the synthetic probe's.
        union = [compact(scene['paired_scales']['[1]']['candidates'][0])]
        for scale_key, cell in scene['paired_scales'].items():
            for row in cell['candidates']:
                if row['name'].startswith('converged_'):
                    item = compact(row)
                    item['name'] = scale_key + '::' + row['name']
                    union.append(item)
        for scale_key, cell in scene['paired_scales'].items():
            scales = json.loads(scale_key)
            epsilon = cfg['epsilon_per_scale'][:len(scales)]
            rows = cell['candidates']
            paths = np.array([r['path'] for r in rows])
            recalculated = metrics(paths, spec, base, mu, scales, epsilon)
            for key, values in recalculated.items():
                np.testing.assert_allclose(values, [r[key] for r in rows], atol=1e-12, rtol=1e-12)
            converged = [r for r in rows if r['name'] == 'base' or r['name'].startswith('converged_')]
            all_soft = [r for r in rows if r['name'] == 'base' or 'alpha_' in r['name']]
            best_conv, best_soft = winner(converged), winner(all_soft)
            full = cell['selected_full_posterior_candidate']
            best_all = winner(rows)
            assert abs(full['expected_3class_accuracy'] - best_all['expected_3class_accuracy']) <= 1e-12
            alpha_rows, common_objective_rows = [], []
            posterior = np.asarray(scene['posterior_multiscale'])[:len(scales)]
            for alpha in cfg['alpha_grid']:
                pair = [next(r for r in rows if r['name'] == '%s_alpha_%g' % (kind, alpha))
                        for kind in ('legacy40', 'converged')]
                alpha_rows.append({'alpha': alpha, 'legacy40': compact(pair[0]), 'converged': compact(pair[1])})
                objective = top_gap_objective(paths, base, posterior, scales, epsilon, alpha)
                chosen_index = int(np.argmin(objective))
                chosen = compact(rows[chosen_index])
                chosen.update({'alpha': alpha, 'top_gap_objective': float(objective[chosen_index])})
                common_objective_rows.append(chosen)
            common_best = winner(common_objective_rows)
            union_scores = metrics(np.asarray([r['path'] for r in union]), spec, base, mu, scales, epsilon)
            union_rows = []
            for i, r in enumerate(union):
                union_rows.append({'name': r['name'], 'path': r['path'],
                                   **{key: float(values[i]) for key, values in union_scores.items()}})
            alpha10 = next(r for r in rows if r['name'] == 'converged_alpha_10')
            scene_out[scale_key] = {
                'candidate_counts': {'base_plus_converged': len(converged), 'base_plus_all_soft': len(all_soft),
                                     'full_library': len(rows), 'cross_scale_converged_union': len(union_rows)},
                'all_alpha_pairs': alpha_rows,
                'best_converged_plus_base': compact(best_conv),
                'posterior_on_unchanged_soft_only_library': compact(best_soft),
                'frozen_full_posterior_selection': compact(full),
                'same_49_candidates_top_gap_minimizer_by_alpha': common_objective_rows,
                'best_same_library_top_gap_minimizer': compact(common_best),
                'posterior_on_cross_scale_converged_union': compact(winner(union_rows)),
                'cross_scale_union_rows_rescored_for_this_fixed_metric': union_rows,
                'gap_decomposition_3class_pp': {
                    'full_minus_fixed_alpha10': 100 * (full['expected_3class_accuracy'] - alpha10['expected_3class_accuracy']),
                    'best_converged_minus_fixed_alpha10': 100 * (best_conv['expected_3class_accuracy'] - alpha10['expected_3class_accuracy']),
                    'full_minus_best_converged': 100 * (full['expected_3class_accuracy'] - best_conv['expected_3class_accuracy']),
                },
                'best_raw_sign_among_converged_plus_base': max(r['expected_raw_sign_accuracy'] for r in converged),
                'full_minus_best_raw_sign_pp': 100 * (full['expected_raw_sign_accuracy'] - max(r['expected_raw_sign_accuracy'] for r in converged)),
                'independent_marginal_mode_upper_bound': cell['independent_marginal_mode_accuracy_upper_bound'],
                'finite_grid_oracle_expected_accuracy': cell['oracle_expected_accuracy'],
                'finite_grid_oracle_with_true_mse_constraint': cell['oracle_with_true_mse_constraint'],
            }
        output['scenarios'][name] = scene_out
    destination = ROOT / 'interpretation_audit.json'
    destination.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(str(destination))
    for name, cells in output['scenarios'].items():
        for scales, cell in cells.items():
            keys = ('best_converged_plus_base', 'frozen_full_posterior_selection',
                    'best_same_library_top_gap_minimizer', 'posterior_on_cross_scale_converged_union')
            print(name, scales, [(k, cell[k]['name'], cell[k]['expected_3class_accuracy']) for k in keys])
    for filename, expected in manifest.items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected


if __name__ == '__main__':
    main()
