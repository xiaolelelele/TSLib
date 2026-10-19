#!/usr/bin/env python3
"""Data-free, independent count-direction compatibility checks.

No experiment/model/decoder code is imported.  The finite-support flow LP and
the independently enumerated path LP use scipy HiGHS, without cvxpy.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def one_step_unbounded(x: int, p, mean: float, tol=1e-10):
    """Exact existence test on {0,1,...}, with finite first moment.

    p is ordered [down, zero increment, up]; zero increment is not level zero.
    The tolerance is solely a numerical comparison tolerance for this probe.
    """
    p = np.asarray(p, dtype=np.float64)
    if x < 0 or x != int(x) or p.shape != (3,) or np.any(p < 0):
        raise ValueError("Require a nonnegative integer anchor and three probabilities")
    if not np.isclose(p.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Probabilities must sum to one")
    if not np.isfinite(mean):
        raise ValueError("Require a finite mean")
    boundary_ok = not (x == 0 and p[0] > 0)
    lower = float(p[1] * x + p[2] * (x + 1))
    upper = None if p[2] > 0 else float(p[0] * (x - 1) + p[1] * x)
    compatible = boundary_ok and mean >= lower - tol
    if upper is not None:
        compatible = compatible and mean <= upper + tol
    return dict(compatible=bool(compatible), boundary_ok=bool(boundary_ok),
                lower=lower, upper=upper)


def flow_lp(grid, anchor, probabilities, means, edge_cost=None, options=None):
    """Finite exact flow model with H M^2 nonnegative edge variables."""
    grid = np.asarray(grid, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    means = np.asarray(means, dtype=float)
    H, M = len(means), len(grid)
    if grid.ndim != 1 or len(np.unique(grid)) != M or p.shape != (H, 3):
        raise ValueError("Bad dimensions")
    hits = np.flatnonzero(grid == anchor)
    if len(hits) != 1:
        raise ValueError("The known anchor must be a grid state")
    ai = int(hits[0])
    rows, cols, vals, rhs = [], [], [], []

    def add(coefficients, target):
        r = len(rhs)
        for k, v in coefficients:
            if v:
                rows.append(r); cols.append(k); vals.append(v)
        rhs.append(float(target))

    def k(t, i, j):
        return (t * M + i) * M + j

    for t in range(H):
        for i in range(M):
            coefficients = [(k(t, i, j), 1.0) for j in range(M)]
            if t:
                coefficients += [(k(t - 1, j, i), -1.0) for j in range(M)]
            add(coefficients, float(i == ai) if t == 0 else 0.0)
        for c in range(3):
            add([(k(t, i, j), 1.0) for i in range(M) for j in range(M)
                 if int(np.sign(grid[j] - grid[i])) + 1 == c], p[t, c])
        add([(k(t, i, j), grid[j]) for i in range(M) for j in range(M)], means[t])
    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), H * M * M)).tocsr()
    objective = np.zeros(H * M * M) if edge_cost is None else np.asarray(edge_cost).reshape(-1)
    fit = linprog(objective, A_eq=A, b_eq=np.asarray(rhs), bounds=(0, None),
                  method="highs", options=options)
    residual = None if not fit.success else float(np.max(np.abs(A @ fit.x - rhs)))
    return fit, residual


def enumerated_path_lp(grid, anchor, probabilities, means, edge_cost=None):
    """Independent path-distribution model; enumerates all M^H trajectories."""
    grid = np.asarray(grid, dtype=float)
    p, means = np.asarray(probabilities), np.asarray(means)
    H, M = len(means), len(grid)
    ai = int(np.flatnonzero(grid == anchor)[0])
    paths = list(itertools.product(range(M), repeat=H))
    A = np.zeros((1 + H * 4, len(paths)))
    objective = np.zeros(len(paths))
    A[0] = 1.0
    rhs = [1.0]
    for t in range(H):
        rhs.extend(p[t]); rhs.append(float(means[t]))
    for n, path in enumerate(paths):
        last = ai
        for t, j in enumerate(path):
            c = int(np.sign(grid[j] - grid[last])) + 1
            A[1 + t * 4 + c, n] = 1
            A[1 + t * 4 + 3, n] = grid[j]
            if edge_cost is not None:
                objective[n] += edge_cost[t, last, j]
            last = j
    fit = linprog(objective, A_eq=A, b_eq=np.asarray(rhs), bounds=(0, None),
                  method="highs")
    return fit


def markov_summaries(grid, anchor, transition):
    grid = np.asarray(grid, dtype=float)
    H, M, _ = transition.shape
    v = np.zeros(M); v[np.flatnonzero(grid == anchor)[0]] = 1
    probabilities, means, edges = [], [], []
    for t in range(H):
        e = v[:, None] * transition[t]
        v = e.sum(axis=0)
        p = [e[np.sign(grid[None, :] - grid[:, None]) == c].sum()
             for c in [-1, 0, 1]]
        probabilities.append(p); means.append(float(v @ grid)); edges.append(e)
    return np.asarray(probabilities), np.asarray(means), np.asarray(edges)


def main():
    checks = []

    def check(name, truth, detail=None):
        checks.append(dict(name=name, passed=bool(truth), detail=detail))
        if not truth:
            raise AssertionError(f"{name}: {detail}")

    named = []

    def case(name, anchor, p, mu, upper_states, expected, explanation):
        statuses = []
        for last in upper_states:
            grid = np.arange(last + 1)
            fit, residual = flow_lp(grid, anchor, p, mu)
            enum = enumerated_path_lp(grid, anchor, p, mu) if last <= 4 else None
            statuses.append(dict(max_state=last, feasible=bool(fit.success),
                                 highs_status=int(fit.status), max_eq_residual=residual))
            check(f"{name}: U={last} feasibility", fit.success == expected[last])
            if enum is not None:
                check(f"{name}: U={last} independent path equality", fit.success == enum.success)
        named.append(dict(name=name, anchor=anchor, probabilities=p, means=mu,
                          finite_support_results=statuses, explanation=explanation))

    case("boundary_down_from_zero", 0, [[.1, .9, 0]], [0], [4, 16],
         {4: False, 16: False}, "Y1>=0 rules out a down event from true anchor zero on any support.")
    case("first_step_mean_below_minimum", 2, [[.2, .3, .5]], [2.0], [4, 16],
         {4: False, 16: False}, "The exact unbounded lower bound is .3*2+.5*3=2.1.")
    case("first_step_lower_endpoint", 2, [[.2, .3, .5]], [2.1], [4],
         {4: True}, "Put down mass at zero, zero-increment mass at 2, up mass at 3.")
    case("no_up_finite_upper_violation", 3, [[.4, .6, 0]], [2.8], [4, 16],
         {4: False, 16: False}, "With no up mass, mu<=.4*2+.6*3=2.6 even without physical U.")
    case("no_up_upper_endpoint", 3, [[.4, .6, 0]], [2.6], [4],
         {4: True}, "The class-specific maximum is attained by down mass at 2 and zero mass at 3.")
    case("one_step_conditions_pass_but_multistep_impossible", 0,
         [[0, .75, .25], [.5, .5, 0]], [1.0, .5], [4, 16],
         {4: False, 16: False},
         "P(Y1>0)=.25 from the true zero anchor; down at t2 has probability .5, which is impossible. "
         "Treating the mean mu1=1 as a deterministic anchor wrongly accepts t2 (.5<=mu2<=.5).")
    case("mean_as_anchor_wrongly_rejects_legal_chain", 0,
         [[0, .5, .5], [.5, .5, 0]], [1.0, 0.0], [4], {4: True},
         "Half the paths are 0->0->0; half are 0->2->0. A false t2 deterministic-mean bound demands mu2>=.5.")
    case("finite_truncation_false_incompatibility", 0, [[0, .9, .1]], [1.2], [4, 16],
         {4: False, 16: True},
         "The unbounded problem is feasible: .9 at 0 and .1 at 12. U=4 rules out the legal tail.")
    check("counterexample passes first-step exact test", one_step_unbounded(0, [0, .75, .25], 1)["compatible"])
    check("counterexample also passes incorrect second-step mean-anchor test",
          one_step_unbounded(1, [.5, .5, 0], .5)["compatible"])
    check("false truncation has unbounded-compatible mean", one_step_unbounded(0, [0, .9, .1], 1.2)["compatible"])

    # Exhaustive rational-probability / mean checks against a sufficiently large
    # finite realization.  Support is chosen from the constructive proof, not
    # interpreted as a physical upper bound.
    first_step_count = 0
    for x in range(4):
        for down_units in range(5):
            for zero_units in range(5 - down_units):
                up_units = 4 - down_units - zero_units
                p = np.array([down_units, zero_units, up_units]) / 4.0
                for mu in np.arange(0, 5.01, .5):
                    exact = one_step_unbounded(x, p, float(mu))
                    last = max(x + 1, int(math.ceil(mu / p[2])) + x + 2) if p[2] else x + 1
                    fit, residual = flow_lp(np.arange(last + 1), x, p[None, :], [mu])
                    check(f"first-step exact vs constructive LP {first_step_count}",
                          fit.success == exact["compatible"], dict(x=x, p=p.tolist(), mu=float(mu), last=last))
                    first_step_count += 1

    rng = np.random.default_rng(20261008)
    legal_counts = 0
    max_residual = 0.0
    max_objective_difference = 0.0
    for H in [1, 2, 3]:
        for M in [2, 3, 4, 5]:
            grid = np.arange(M)
            for repeat in range(3):
                anchor = int(rng.integers(M))
                trans = rng.dirichlet(np.ones(M), size=(H, M))
                p, mu, edges = markov_summaries(grid, anchor, trans)
                costs = rng.normal(size=(H, M, M))
                flow, residual = flow_lp(grid, anchor, p, mu, costs)
                path = enumerated_path_lp(grid, anchor, p, mu, costs)
                check(f"legal Markov flow {legal_counts}", flow.success)
                check(f"legal Markov enumerated path {legal_counts}", path.success)
                diff = abs(float(flow.fun) - float(path.fun))
                check(f"Markov objective equality {legal_counts}", diff < 1e-8, diff)
                check(f"Markov moment/class residual {legal_counts}", residual < 1e-8, residual)
                max_residual = max(max_residual, residual)
                max_objective_difference = max(max_objective_difference, diff)
                # Weak native moment requirements are also checked directly.
                prev = float(anchor)
                for t in range(H):
                    check(f"weak current up lower {legal_counts}/{t}", mu[t] >= p[t, 2] - 1e-10)
                    check(f"weak previous down lower {legal_counts}/{t}", prev >= p[t, 0] - 1e-10)
                    prev = float(mu[t])
                legal_counts += 1

    arbitrary_count = 0
    for H in [1, 2, 3]:
        for M in [2, 3, 4, 5]:
            grid = np.arange(M)
            for repeat in range(3):
                anchor = int(rng.integers(M))
                p = rng.dirichlet(np.ones(3), size=H)
                mu = rng.uniform(0, M - 1, H)
                flow, _ = flow_lp(grid, anchor, p, mu)
                path = enumerated_path_lp(grid, anchor, p, mu)
                check(f"arbitrary flow/path existence {arbitrary_count}", flow.success == path.success)
                arbitrary_count += 1

    # Explicit equality / directional edge cases for weak moment bounds.
    for name, anchor, p, mu in [
        ("no down requires increase", 2, [[0, .75, .25]], [2.25]),
        ("no up requires decrease", 2, [[.25, .75, 0]], [1.75]),
        ("all zero means same state", 2, [[0, 1, 0]], [2]),
    ]:
        fit, _ = flow_lp(np.arange(5), anchor, p, mu)
        check(name, fit.success)

    output = dict(
        scope="Data-free mathematical correctness probe; no training, no real test data, no novelty claim.",
        environment=dict(numpy=np.__version__, scipy=scipy.__version__, solver="scipy.optimize.linprog(method=highs)"),
        first_step_exact_condition=dict(
            boundary="x=0 implies p_down=0",
            lower="mu >= p_zero*x + p_up*(x+1)",
            upper="none if p_up>0; otherwise mu <= p_down*(x-1)+p_zero*x=x-p_down",
            caveat="Integer nonnegative x; finite mean; normalized nonnegative p; native q=1, no physical U assumed."),
        multistep_necessary_only=[
            "mu_t >= p_up_t", "mu_(t-1) >= p_down_t",
            "if p_down_t=0: mu_t >= mu_(t-1)+p_up_t",
            "if p_up_t=0: mu_t <= mu_(t-1)-p_down_t",
            "if true x0=0: p_down_2 <= p_up_1",
        ],
        named_cases=named, first_step_formula_vs_lp_cases=first_step_count,
        legal_markov_cases=legal_counts, arbitrary_flow_vs_path_cases=arbitrary_count,
        max_legal_eq_residual=max_residual, max_flow_path_objective_difference=max_objective_difference,
        check_count=len(checks), failure_count=sum(not c["passed"] for c in checks), checks=checks)
    dest = Path(__file__).with_name("theory_results.json")
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ["check_count", "failure_count", "first_step_formula_vs_lp_cases",
                     "legal_markov_cases", "arbitrary_flow_vs_path_cases", "max_legal_eq_residual",
                     "max_flow_path_objective_difference"]}, indent=2))


if __name__ == "__main__":
    main()
