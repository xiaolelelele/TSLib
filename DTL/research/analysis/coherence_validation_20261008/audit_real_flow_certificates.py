#!/usr/bin/env python3
"""Replay saved sparse witnesses independently of the LP constraint builder.

Also compare the fixed validation sub-sample's p/mu with the root agent's
independently implemented simple repair. No target or test arrays are loaded.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    result = json.loads((HERE / 'real_flow_certificates.json').read_text())
    checks, root_hashes, maxima = [], {}, {}

    def check(name, truth, details=None):
        checks.append(dict(name=name, passed=bool(truth), details=details))

    for row in result['rows']:
        if row['status'] != 'CERTIFIED_FEASIBLE_NUMERICAL':
            continue
        path = Path(row['certificate_path'])
        label = f"{row['case']}/o{row['origin_array_index']}/c{row['channel_index']}/{row['variant']}"
        check(label + '/sha', sha(path) == row['certificate_sha256'])
        with np.load(path) as a:
            grid, pos, value = a['grid'], a['edge_index'], a['edge_value']
            v, p, mu, x, scale = a['v'], a['probabilities'], a['mean_raw'], int(a['anchor']), float(a['train_std'])
        H, M = len(mu), len(grid)
        check(label + '/native_integer_support', np.all(grid >= 0) and np.all(grid == np.rint(grid)))
        check(label + '/anchor', x == row['anchor'] and grid[np.argmax(v[0])] == x
              and abs(v[0].sum() - 1) < 1e-12 and np.count_nonzero(v[0]) == 1)
        check(label + '/positive_sparse_edges', np.all(value >= 0))
        recovered = np.zeros((H, M, M)); recovered[tuple(pos.T)] = value
        flow_error, direction_error, mean_error, marginal_error = 0., 0., 0., 0.
        for t in range(H):
            outgoing = np.zeros(M); incoming = np.zeros(M); pc = np.zeros(3)
            mask = pos[:, 0] == t
            for (_, i, j), mass in zip(pos[mask], value[mask]):
                outgoing[i] += mass; incoming[j] += mass
                pc[0 if grid[j] < grid[i] else 2 if grid[j] > grid[i] else 1] += mass
            flow_error = max(flow_error, np.abs(outgoing - v[t]).max())
            marginal_error = max(marginal_error, np.abs(incoming - v[t + 1]).max())
            direction_error = max(direction_error, np.abs(pc - p[t]).max())
            mean_error = max(mean_error, abs(float(incoming @ grid) - mu[t]))
        check(label + '/flow', flow_error <= 1e-8, float(flow_error))
        check(label + '/marginal', marginal_error <= 1e-8, float(marginal_error))
        check(label + '/directions', direction_error <= 1e-8, float(direction_error))
        check(label + '/mean', mean_error <= max(1., scale) * 1e-8, float(mean_error))
        for k, val in [('flow_error', flow_error), ('direction_error', direction_error),
                       ('mean_raw_error', mean_error), ('marginal_error', marginal_error)]:
            maxima[k] = max(maxima.get(k, 0), float(val))
        root = HERE / 'cases' / row['case'] / 'validation.npz'
        if not root.exists():
            raise FileNotFoundError('Root simple-repair output is required for the cross-implementation check')
        root_hashes[str(root)] = sha(root)
        with np.load(root) as a:
            stage = 'p_normalized' if row['variant'] == 'original' else 'p_masked'
            np64 = a[stage][row['origin_array_index'], :, row['channel_index'], :]
            key = 'mean_original' if row['variant'] == 'original' else 'mean_fixed'
            normalized_mu = a[key][row['origin_array_index'], :, row['channel_index']]
        data_json = Path('/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930/results_v2') / row['domain'] / 'dataset.json'
        meta = json.loads(data_json.read_text()); c = row['channel_index']
        root_mean = normalized_mu * meta['std'][c] + meta['mean'][c]
        check(label + '/root_probability_agrees', np.array_equal(p, np64), float(np.abs(p - np64).max()))
        check(label + '/root_mean_agrees', np.allclose(mu, root_mean, atol=1e-9, rtol=0), float(np.abs(mu - root_mean).max()))
    check('214_saved_certificates', len([r for r in result['rows'] if r['status'] == 'CERTIFIED_FEASIBLE_NUMERICAL']) == 214)
    out = dict(scope='Independent saved sparse-edge replay and root simple-repair validation-only agreement; shares no LP constraint builder.',
               source_result_sha256=sha(HERE / 'real_flow_certificates.json'), root_output_hashes=root_hashes,
               check_count=len(checks), failure_count=sum(not c['passed'] for c in checks),
               maxima=maxima, checks=checks)
    (HERE / 'audit_real_flow_certificates.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: out[k] for k in ['check_count', 'failure_count', 'maxima']}, indent=2))
    if out['failure_count']:
        raise AssertionError('Certificate replay audit failed')


if __name__ == '__main__':
    main()
