"""Exact known-distribution mechanism probes, not real-data efficacy evidence."""
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from DTL.trend_lab.trajectory_decision import decode_trajectory, raw_classes

ROOT = Path(__file__).resolve().parent


def information(paths, weights, anchor=0., exact_decimal=False):
    paths, weights = np.asarray(paths, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    mean = weights @ paths
    if exact_decimal:
        # Named rational distributions have exact zero means; compute those
        # from their declared decimal masses, not by rounding decoded outputs.
        mean = np.array([float(sum((Fraction(str(float(w))) * Fraction(str(float(y)))
                                   for w, y in zip(weights, paths[:, h])), Fraction(0)))
                         for h in range(paths.shape[1])])
    previous = np.column_stack((np.full(len(paths), anchor), paths[:, :-1]))
    classes = np.where(paths < previous, 0, np.where(paths > previous, 2, 1))
    p = np.column_stack([(weights[:, None] * (classes == cls)).sum(0) for cls in range(3)])
    return mean, p


def exact_measure(path, paths, weights, anchor=0.):
    path, paths = np.asarray(path), np.asarray(paths)
    risk = float((weights[:, None] * (paths - path) ** 2).sum() / len(path))
    pred = np.where(path < np.r_[anchor, path[:-1]], 0, np.where(path > np.r_[anchor, path[:-1]], 2, 1))
    previous = np.column_stack((np.full(len(paths), anchor), paths[:, :-1]))
    truth = np.where(paths < previous, 0, np.where(paths > previous, 2, 1))
    utility = float((weights[:, None] * (truth == pred)).sum() / len(path))
    return {"path": path.tolist(), "expected_mse": risk, "expected_raw_accuracy": utility}


def candidate_points(paths, weights, grid, anchor=0., base=None, exact_decimal=False):
    mean, p = information(paths, weights, anchor, exact_decimal=exact_decimal)
    all_paths = list(itertools.product(grid, repeat=paths.shape[1]))
    if base is not None:
        all_paths.append(tuple(base))
    rows = [exact_measure(z, paths, weights, anchor) for z in all_paths]
    for row in rows:
        row["squared_distance_to_mean"] = float(np.mean((np.asarray(row["path"]) - mean) ** 2))
    return mean, p, rows


def pareto(rows):
    # Collapse numerically equivalent (risk,utility) points by exact candidates.
    out = []
    best = -np.inf
    for row in sorted(rows, key=lambda r: (r["squared_distance_to_mean"], -r["expected_raw_accuracy"], r["path"])):
        if row["expected_raw_accuracy"] > best + 1e-12:
            out.append(row)
            best = row["expected_raw_accuracy"]
    return out


def supported_interval(row, all_rows):
    lower, upper = 0., np.inf
    risk, utility = row["squared_distance_to_mean"], row["expected_raw_accuracy"]
    for other in all_rows:
        du = utility - other["expected_raw_accuracy"]
        dr = risk - other["squared_distance_to_mean"]
        if abs(du) < 1e-12:
            if dr > 1e-12:
                return None
        elif du > 0:
            lower = max(lower, dr / du)
        else:
            upper = min(upper, dr / du)
    if upper < lower - 1e-12:
        return None
    return [float(lower), None if np.isinf(upper) else float(upper)]


def main():
    rng = np.random.default_rng(20260930)
    lambdas = [0., .001, .01, .1, 1., 10.]
    paths = np.array([[1., 1., 2.], [-3., -3., -2.], [0., 0., 0.], [1., 0., -1.], [2., 1., 0.]])
    weights = np.array([.35, .30, .20, .10, .05])
    grid = np.arange(-3., 3.)
    mean, p, all_rows = candidate_points(paths, weights, grid, base=None, exact_decimal=True)
    x = np.zeros((1, 1, 1))
    base = mean[None, :, None]
    rows = []
    independent_checks = 0
    for strength in lambdas:
        for reward in ["posterior", "top1", "top1_gap"]:
            decoded, diagnostic = decode_trajectory(x, base, base, p[None, :, None, :], strength, grid, reward=reward)
            row = exact_measure(decoded[0, :, 0], paths, weights)
            row.update({"lambda": strength, "reward": reward, "selected_base": diagnostic["selected_base"][0],
                        "estimated_delta_risk": diagnostic["estimated_risk_delta"][0]})
            rows.append(row)
            if reward == "posterior":
                base_row = exact_measure(mean, paths, weights)
                best = min(r["expected_mse"] - strength * r["expected_raw_accuracy"] for r in all_rows + [base_row])
                actual = row["expected_mse"] - strength * row["expected_raw_accuracy"]
                assert abs(best - actual) < 1e-12
                independent_checks += 1
    mechanism = {"description": "Known joint distribution; p and mean computed exactly, no independent-step sampling",
                 "support_paths": paths.tolist(), "weights": weights.tolist(), "mean": mean.tolist(),
                 "direction_posterior_down_zero_up": p.tolist(), "shared_grid": grid.tolist(), "decisions": rows}

    # Find a reproducible valid joint distribution with a nonsupported Pareto point.
    finite_support = np.array(list(itertools.product([-2., -1., 0., 1., 2.], repeat=2)))
    unsupported = None
    for trial in range(100):
        w = rng.dirichlet(np.full(len(finite_support), .7))
        mu, probability, points = candidate_points(finite_support, w, [-2., -1., 0., 1., 2.], base=None)
        points.append(dict(exact_measure(mu, finite_support, w), squared_distance_to_mean=0.))
        frontier = pareto(points)
        missing = [r for r in frontier if supported_interval(r, points) is None]
        if missing:
            point = missing[0]
            tau = point["squared_distance_to_mean"]
            constrained = max(r["expected_raw_accuracy"] for r in points if r["squared_distance_to_mean"] <= tau + 1e-12)
            assert abs(constrained - point["expected_raw_accuracy"]) < 1e-12
            unsupported = {"trial": trial, "support_paths": finite_support.tolist(), "weights": w.tolist(),
                           "mean": mu.tolist(), "posterior": probability.tolist(), "pareto_points": frontier,
                           "nonsupported_budget_optimum": point, "budget_tau": tau,
                           "lambda_support_interval": None,
                           "meaning": "No nonnegative lambda attains this budget-optimal Pareto point; lambda scanning is not an exact risk-budget solver"}
            independent_checks += 1
            break
    assert unsupported is not None

    # Direction-uninformative negative control: iid uniform {-1,0,1} has each
    # marginal adjacent direction class exactly 1/3, including the first step.
    no_signal_paths = np.array(list(itertools.product([-1., 0., 1.], repeat=3)))
    no_signal_weights = np.full(len(no_signal_paths), 1 / len(no_signal_paths))
    no_signal_mean, no_signal_p = information(no_signal_paths, no_signal_weights)
    np.testing.assert_allclose(no_signal_p, 1 / 3, rtol=0, atol=1e-15)
    no_signal_results = []
    for strength in lambdas:
        b = no_signal_mean[None, :, None]
        z, diagnostic = decode_trajectory(x, b, b, no_signal_p[None, :, None, :], strength, np.array([-1., 0., 1.]))
        no_signal_results.append(dict(exact_measure(z[0, :, 0], no_signal_paths, no_signal_weights), lambda_value=strength))
        assert abs(no_signal_results[-1]["expected_raw_accuracy"] - 1 / 3) < 1e-12
        independent_checks += 1

    # Same probabilities, genuinely different mean and amendment MSE effects.
    same_p_cases = []
    for up in [.25, 1.]:
        y, w = np.array([[-1.], [up]]), np.array([.2, .8])
        mu, probability = information(y, w, exact_decimal=True)
        b, candidate = np.array([0.]), np.array([.1])
        delta = exact_measure(candidate, y, w)["expected_mse"] - exact_measure(b, y, w)["expected_mse"]
        same_p_cases.append({"support_paths": y.tolist(), "weights": w.tolist(), "mean": mu.tolist(),
                             "posterior": probability.tolist(), "candidate_minus_base_expected_mse": delta})
        assert abs(delta - (.01 - .2 * mu[0])) < 1e-12
        independent_checks += 1
    assert same_p_cases[0]["candidate_minus_base_expected_mse"] > 0 > same_p_cases[1]["candidate_minus_base_expected_mse"]

    # True zero atoms and quantization-only controls: raw integer observation
    # support does not require an integer conditional mean forecast.
    zero_support, zero_w = np.array([[0.], [1.]]), np.array([.8, .2])
    zero_mean, zero_p = information(zero_support, zero_w, exact_decimal=True)
    unrounded = exact_measure(zero_mean, zero_support, zero_w)
    rounded = exact_measure(np.round(zero_mean), zero_support, zero_w)
    assert rounded["expected_raw_accuracy"] > unrounded["expected_raw_accuracy"]
    assert rounded["expected_mse"] > unrounded["expected_mse"]
    quantization = {"support_paths": zero_support.tolist(), "weights": zero_w.tolist(),
                    "mean_prediction": unrounded, "ordinary_rounding_no_direction_model": rounded,
                    "meaning": "Rounding alone produces +60 pp direction and +25% MSE; do not attribute such a gain to learned direction information"}
    independent_checks += 1

    # Repeating a prior valid value can artificially create flat transitions.
    original = np.array([[-1., 0., 1.]])
    cleaned = np.array([[-1., -1., 1.]])
    original_cls = raw_classes(x, original[:, :, None])[0, :, 0]
    cleaned_cls = raw_classes(x, cleaned[:, :, None])[0, :, 0]
    assert (original_cls == 1).sum() == 0 and (cleaned_cls == 1).sum() == 1
    cleaning = {"original_path": original[0].tolist(), "forward_replacement_path": cleaned[0].tolist(),
                "original_classes": original_cls.tolist(), "cleaned_classes": cleaned_cls.tolist(),
                "meaning": "Illustrative cleaning artifact only; not evidence about any acquired TIME dataset"}
    independent_checks += 1

    # Bayes mean b=0 and up-probability .6: infinitesimal positive prediction
    # trades a fixed utility gain for vanishing positive MSE cost, but never
    # reaches an optimum at continuous zero.  A fixed grid makes it attainable.
    singular_y, singular_w = np.array([[1.], [-3.], [0.]]), np.array([.6, .2, .2])
    singular_mean, singular_p = information(singular_y, singular_w, exact_decimal=True)
    singular_rows = [exact_measure(np.array([value]), singular_y, singular_w) for value in [0., .1, .01, .001]]
    risk0 = singular_rows[0]["expected_mse"]
    for row in singular_rows[1:]:
        value = row["path"][0]
        assert abs(row["expected_mse"] - risk0 - value ** 2) < 1e-12
        independent_checks += 1
    singular = {"support_paths": singular_y.tolist(), "weights": singular_w.tolist(),
                "mean": singular_mean.tolist(), "posterior": singular_p.tolist(), "decisions": singular_rows,
                "meaning": "Every positive modification raises Bayes MSE. Zero budget cannot strictly improve direction. Continuous scalarized raw-sign optimum may not be attained"}

    result = {"seed": 20260930, "independent_exact_checks": independent_checks,
              "interpretation": "Standard decision identities/DP mechanism validation; no new theorem, no learned-model or real-data efficacy claim",
              "known_joint_dp_vs_enumeration": mechanism, "unsupported_pareto_point": unsupported,
              "no_direction_signal": {"posterior": no_signal_p.tolist(), "decisions": no_signal_results},
              "same_p_different_mean": same_p_cases, "true_zero_quantization_negative_control": quantization,
              "artificial_cleaning_zero_control": cleaning, "bayes_mean_zero_budget": singular}
    (ROOT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    note = """# 已知分布上的轨迹决策机制验证（2026-09-30）

这些探针使用完整列明的短H联合分布，精确计算条件均值、相邻方向边际概率及期望MSE；没有独立采样各步来伪造联合路径。它们只验证标准决策机制和有限网格求解，不能作为新定理或实际TIME数据有效性的证据。

- 同一个共同候选集下，后验目标的DP与穷举一致；另有单元测试60组随机小矩阵，每组独立枚举核对。
- `b=E[Y|I]`时零风险预算不能容许任何非零修改而保持条件MSE。示例up/down/zero为0.6/0.2/0.2，均值为零；正预测越靠近零，其方向收益不变、额外MSE趋于零，但预测零改变类别。这不是连续目标存在最优解的证明。
- 完全相同的down/up概率0.2/0.8，改变幅度分布便使base=0到candidate=0.1的期望MSE差从+0.01变成−0.11。方向置信度不能单独识别风险。
- iid uniform{−1,0,1}负对照中，每步方向边际均为1/3；任何方向路径都不能得到严格期望方向增益。
- 原生整数例Y=0/1概率0.8/0.2中，无方向模型的普通rounding将raw方向从20%改为80%，但MSE从0.16增至0.20。整数观测不要求条件均值预测为整数，rounding必须是单列对照。
- 一条[-1,0,1]轨迹把第二值替为前值会凭空产生一个zero；这里只说明清洗平台的潜在机制，没有推断真实TIME文件被何种清洗处理。
- 一组明确的合法H2联合分布产生非加权目标支持的Pareto点；在该点的风险预算下它是约束效用最优，却不能由任何非负lambda实现。因此lambda扫描不能宣称完整Pareto前沿或精确逐实例风险预算解。

所有数值、分布、路径、预算及后验见[results.json](results.json)，可由[probe.py](probe.py)复算。所有探针都是标准性质或负对照，后续论文价值仍需真实学习概率、强基准、共同候选集和新数据上的证据支持。
"""
    (ROOT / "note.md").write_text(note, encoding="utf-8")
    print(json.dumps({"independent_exact_checks": independent_checks, "unsupported_trial": unsupported["trial"],
                      "known_joint_rows": len(rows), "output": str(ROOT / "results.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
