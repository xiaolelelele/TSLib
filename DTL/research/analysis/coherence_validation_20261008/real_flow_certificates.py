#!/usr/bin/env python3
"""Validation-only fixed sample: numerical finite-support count-flow witnesses.

Reads no future target arrays and no test arrays. A failed finite-support search
is UNKNOWN. Native incompatibility is asserted only by analytic necessary
conditions, with documented numerical comparison tolerance.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from theory_probe import flow_lp, one_step_unbounded

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/lgc/TSLib/DTL/trend_lab_runs/time_trajectory_20260930/results_v2')
TOL = 1e-9
TOTAL_SECONDS = 300.0
PER_LP_SECONDS = 5.0
FACTORS = [1, 2, 4, 8]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def repair(x, p, mean):
    p, mean = p.copy(), mean.copy()
    if x == 0:
        p[0, 0] = 0
        p[0] /= p[0].sum()
    bounds = one_step_unbounded(x, p[0], mean[0], tol=TOL)
    if mean[0] < bounds['lower'] - TOL:
        mean[0] = bounds['lower']
    if bounds['upper'] is not None and mean[0] > bounds['upper'] + TOL:
        mean[0] = bounds['upper']
    return p, mean


def necessary_violations(x, p, mean):
    violations = []

    def add(name, step, gap):
        if gap > TOL:
            violations.append(dict(condition=name, step=step, gap=float(gap)))

    first = one_step_unbounded(x, p[0], float(mean[0]), tol=TOL)
    if x == 0:
        add('true_zero_anchor_cannot_go_down', 1, p[0, 0])
    add('firststep_exact_mean_lower', 1, first['lower'] - mean[0])
    if first['upper'] is not None:
        add('firststep_no_up_exact_mean_upper', 1, mean[0] - first['upper'])
    previous = float(x)
    for t in range(len(mean)):
        add('mean_nonnegative', t + 1, -mean[t])
        add('current_mean_at_least_up_mass', t + 1, p[t, 2] - mean[t])
        add('previous_mean_at_least_down_mass', t + 1, p[t, 0] - previous)
        if p[t, 0] == 0:
            add('no_down_mean_increases_at_least_up_mass', t + 1, previous + p[t, 2] - mean[t])
        if p[t, 2] == 0:
            add('no_up_mean_decreases_at_least_down_mass', t + 1, mean[t] - previous + p[t, 0])
        previous = float(mean[t])
    if x == 0 and len(mean) >= 2:
        add('zero_initial_down_step2_no_more_than_up_step1', 2, p[1, 0] - p[0, 2])
    return violations


def support_sequence(x, mean, b):
    visible = np.r_[float(x), mean, b]
    if not np.all(np.isfinite(visible)):
        raise ValueError('Visible forecasts must be finite')
    vmax, vmin = float(visible.max()), float(visible.min())
    span = max(1.0, vmax - vmin)
    states = {0, 1, x, max(0, x - 1), x + 1}
    for v in np.r_[mean, b]:
        states.update([max(0, int(np.floor(v))), max(0, int(np.ceil(v)))])
    sequence = []
    for factor in FACTORS:
        tail = max(1, int(np.ceil(vmax + factor * span)))
        states.update([tail - 1, tail, tail + 1])
        sequence.append((factor, np.asarray(sorted(states), dtype=np.int64)))
    return sequence, dict(visible_min=vmin, visible_max=vmax, span=span)


def verify(grid, x, p, mean, edge, scale):
    H, M, _ = edge.shape
    raw_min = float(edge.min())
    edge = np.maximum(edge, 0.0)
    previous = np.zeros(M); previous[np.flatnonzero(grid == x)[0]] = 1.0
    v = [previous.copy()]
    continuity, probability_error, mean_error, normalization = 0.0, 0.0, 0.0, 0.0
    sign = np.sign(grid[None, :] - grid[:, None])
    for t in range(H):
        source, destination = edge[t].sum(axis=1), edge[t].sum(axis=0)
        continuity = max(continuity, float(np.max(np.abs(source - previous))))
        recovered = np.array([edge[t][sign == c].sum() for c in [-1, 0, 1]])
        probability_error = max(probability_error, float(np.max(np.abs(recovered - p[t]))))
        mean_error = max(mean_error, abs(float(destination @ grid) - float(mean[t])))
        normalization = max(normalization, abs(float(destination.sum()) - 1.0))
        previous = destination
        v.append(destination.copy())
    diagnostics = dict(raw_min_edge=raw_min, max_flow_continuity_error=continuity,
                       max_direction_probability_error=probability_error,
                       max_mean_raw_units_error=mean_error,
                       max_mean_scaled_error=mean_error / scale,
                       max_normalization_error=normalization)
    accepted = (raw_min >= -1e-10 and max(continuity, probability_error, normalization) <= 1e-8
                and mean_error <= max(1.0, scale) * 1e-8)
    return bool(accepted), diagnostics, edge, np.asarray(v)


def main():
    start = time.monotonic()
    protocol = HERE / 'real_flow_protocol.md'
    if not protocol.exists():
        raise FileNotFoundError('Prewritten protocol is required')
    run_hashes = {p.name: sha(p) for p in [protocol, Path(__file__), HERE / 'theory_probe.py']}
    cert_dir = HERE / 'real_flow_certificates'
    cert_dir.mkdir(exist_ok=True)
    rows, paired, source_hashes = [], [], {}
    for domain in ['carpark', 'finland']:
        mp = ROOT / domain / 'dataset.json'
        metadata = json.loads(mp.read_text())
        mm, ss = np.asarray(metadata['mean']), np.asarray(metadata['std'])
        source_hashes[str(mp)] = sha(mp)
        for bone in ['dlinear', 'itransformer']:
            for seed in [2024, 2025, 2026]:
                case = f'{domain}/{bone}_seed{seed}'
                file = ROOT / case / 'predictions.npz'
                source_hashes[str(file)] = sha(file)
                # Lazy npz member access: no test member or validation target is loaded.
                with np.load(file, allow_pickle=False) as a:
                    source_x = a['validation__x'][:, -1, :]
                    source_p = a['validation__probabilities']
                    source_mu = a['validation__mean']
                    source_b = a['validation__cosdir']
                    source_origins = a['validation__origins']
                N, H, C = source_mu.shape
                if H != 16:
                    raise AssertionError('Protocol requires H16')
                for origin_index in [0, N // 2]:
                    for c in range(C):
                        raw_x = float(source_x[origin_index, c] * ss[c] + mm[c])
                        x = int(np.rint(raw_x))
                        if x < 0 or abs(raw_x - x) > TOL:
                            raise AssertionError('Invalid integer anchor reconstruction')
                        p = source_p[origin_index, :, c].astype(np.float64)
                        p /= p.sum(axis=1, keepdims=True)
                        mean = source_mu[origin_index, :, c] * ss[c] + mm[c]
                        b = source_b[origin_index, :, c] * ss[c] + mm[c]
                        fixed_p, fixed_mean = repair(x, p, mean)
                        pair_status = {}
                        for label, pp, mu in [('original', p, mean), ('simple_repair', fixed_p, fixed_mean)]:
                            row = dict(case=case, domain=domain, bone=bone, seed=seed,
                                       origin_array_index=origin_index, origin_row=int(source_origins[origin_index]),
                                       channel_index=c, channel=metadata['columns'][c], variant=label,
                                       anchor=x, anchor_reconstruction_error=abs(raw_x - x),
                                       firststep_probability_l1_change=float(np.abs(pp[0] - p[0]).sum()),
                                       firststep_mean_raw_change=float(mu[0] - mean[0]), stages=[])
                            violations = necessary_violations(x, pp, mu)
                            row['native_necessary_violations'] = violations
                            if violations:
                                row['status'] = 'NECESSARY_INCOMPATIBLE'
                            else:
                                row['status'] = 'UNKNOWN_TRUNCATION_OR_NUMERICAL'
                                supports, summary = support_sequence(x, mu, b)
                                row['support_rule_values'] = summary
                                for factor, grid in supports:
                                    remaining = TOTAL_SECONDS - (time.monotonic() - start)
                                    if remaining < .1:
                                        row['status'] = 'UNKNOWN_BUDGET'
                                        break
                                    time_limit = min(PER_LP_SECONDS, remaining)
                                    before = time.monotonic()
                                    fit, scaled_residual = flow_lp(
                                        grid.astype(np.float64) / ss[c], x / ss[c], pp, mu / ss[c],
                                        options=dict(time_limit=time_limit, primal_feasibility_tolerance=1e-9,
                                                     dual_feasibility_tolerance=1e-9))
                                    stage = dict(tail_factor=factor, state_count=int(len(grid)),
                                                 support_max=int(grid[-1]), seconds=time.monotonic() - before,
                                                 time_limit=time_limit, highs_status=int(fit.status),
                                                 highs_message=str(fit.message), numerical_success=bool(fit.success),
                                                 solver_scaled_eq_residual=scaled_residual)
                                    if fit.success:
                                        accepted, diagnostics, edge, v = verify(
                                            grid, x, pp, mu, fit.x.reshape(H, len(grid), len(grid)), float(ss[c]))
                                        stage['verification'] = diagnostics
                                        stage['certificate_accepted'] = accepted
                                        if accepted:
                                            positive = np.argwhere(edge > 0)
                                            values = edge[tuple(positive.T)]
                                            name = f'{domain}_{bone}_seed{seed}_o{origin_index}_c{c}_{label}.npz'
                                            dest = cert_dir / name
                                            np.savez_compressed(dest, grid=grid, edge_index=positive.astype(np.int32),
                                                                edge_value=values, v=v, anchor=np.array(x),
                                                                probabilities=pp, mean_raw=mu,
                                                                train_std=np.array(ss[c]))
                                            row['status'] = 'CERTIFIED_FEASIBLE_NUMERICAL'
                                            row['certificate_path'] = str(dest)
                                            row['certificate_sha256'] = sha(dest)
                                            row['certificate_verification'] = diagnostics
                                            row['successful_tail_factor'] = factor
                                            row['successful_state_count'] = len(grid)
                                            row['stages'].append(stage)
                                            break
                                    row['stages'].append(stage)
                            rows.append(row)
                            pair_status[label] = row['status']
                        paired.append(dict(case=case, origin_array_index=origin_index, channel_index=c,
                                           original=pair_status['original'], simple_repair=pair_status['simple_repair']))
                print(json.dumps(dict(case=case, elapsed_seconds=time.monotonic() - start,
                                      status_counts=dict(Counter(z['status'] for z in rows if z['case'] == case))),
                                 ensure_ascii=False), flush=True)
    if len(paired) != 108 or len(rows) != 216:
        raise AssertionError(f'Expected 108 paired curves, got {len(paired)} / {len(rows)}')
    summary = {}
    for domain in ['all', 'carpark', 'finland']:
        selected = rows if domain == 'all' else [r for r in rows if r['domain'] == domain]
        summary[domain] = {label: dict(Counter(r['status'] for r in selected if r['variant'] == label))
                           for label in ['original', 'simple_repair']}
    output = dict(scope='Fixed 108 H16 validation-visible forecast curves; no future target, no test arrays, no new training/tuning.',
                  protocol_sha256=run_hashes['real_flow_protocol.md'], code_hashes=run_hashes,
                  source_hashes=source_hashes, elapsed_seconds=time.monotonic() - start,
                  numpy=np.__version__, scipy=scipy.__version__, sample_pair_count=len(paired),
                  evaluated_pair_variant_count=len(rows), total_budget_seconds=TOTAL_SECONDS,
                  per_lp_max_time_limit=PER_LP_SECONDS, necessary_compare_tolerance_raw=TOL,
                  summary=summary,
                  paired_transitions=dict(Counter(f"{q['original']} -> {q['simple_repair']}" for q in paired)),
                  rows=rows, paired_rows=paired,
                  limitations=['Floating-point numerical certificates, not formal exact-arithmetic proofs.',
                               'Finite-support failure is UNKNOWN, never native incompatibility.',
                               'Fixed validation subsample; no test performance or full validation prevalence estimate.'])
    dest = HERE / 'real_flow_certificates.json'
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(summary=summary, elapsed_seconds=output['elapsed_seconds'],
                          paired_transitions=output['paired_transitions']), indent=2), flush=True)


if __name__ == '__main__':
    main()
