"""Summarize frozen fusion results without choosing methods from test scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


FAMILIES = [
    "base_plus_difference", "base_plus_residual", "base_plus_residual_direction",
    "learned_selected_plus_flat", "learned_selected_plus_prior",
    "learned_selected_plus_difference", "threeway_mean", "fusion_selected",
]
ORDER = [
    "base", "difference", "residual", "residual_direction", "learned_single",
    "learned_multi", "prior", "flat", "learned_selected", "standalone_selected",
    *FAMILIES,
]


def display_dataset(name):
    if "exchange" in name.lower():
        return "Exchange Rate"
    return name


def joint(candidate, reference):
    # One direction point is orders of magnitude larger than this tolerance.
    # Avoid treating a seed-mean summation roundoff as a strict improvement.
    return bool(candidate["raw_sign_accuracy"] > reference["raw_sign_accuracy"] + 1e-12
                and candidate["standardized_mse"] <= reference["standardized_mse"] + 1e-8)


def differences(candidate, reference):
    sign_delta = candidate["raw_sign_accuracy"] - reference["raw_sign_accuracy"]
    three_delta = candidate["mean_3class_accuracy"] - reference["mean_3class_accuracy"]
    return {
        "raw_sign_delta_pp": 0.0 if abs(sign_delta) < 1e-12 else 100 * sign_delta,
        "three_class_delta_pp": 0.0 if abs(three_delta) < 1e-12 else 100 * three_delta,
        "mse_delta_pct": 100 * (candidate["standardized_mse"] / reference["standardized_mse"] - 1),
        "standardized_mse_delta": candidate["standardized_mse"] - reference["standardized_mse"],
        "joint_improvement": joint(candidate, reference),
    }


def selection(run, method):
    val = run["validation_selection"]
    if method == "fusion_selected":
        return val[method]
    if method == "standalone_selected":
        return val[method]
    if method in val["pairwise_fusions"]:
        sel = dict(val["pairwise_fusions"][method])
        w = sel["selected_weight"]
        sel["is_nontrivial_blend"] = bool(w is not None and 0 < w < 1
            and sel["members_different_path"] and not sel["fallback_to_base"])
        return sel
    if method == "threeway_mean":
        sel = dict(val["fixed_threeway_mean"])
        sel["is_nontrivial_blend"] = bool(not sel.get("fallback_to_base", not sel["eligible"])
            and sel.get("at_least_two_distinct_members", sel.get("is_nontrivial_blend",
                        sel.get("members_pairwise_different", False))))
        return sel
    if method == "learned_selected":
        return val[method]
    return val["atom_selection"].get(method, {})


def save_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def summarize(paths):
    groups, sources = {}, []
    for path in paths:
        obj = json.loads(path.read_text())
        manifest = json.loads((path.parent / "output_manifest.json").read_text())
        if manifest.get("complete") is not True or set(obj["groups"]) != set(manifest["completed_groups"]):
            raise ValueError("refusing incomplete or mismatched input: " + str(path))
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for name, group in obj["groups"].items():
            if name in groups:
                raise ValueError("duplicate group: " + name)
            groups[name] = group
    rows, seeds, identities, member_rows = [], [], [], []
    for name, group in groups.items():
        runs = list(group["seeds"].values())
        if sorted(run["seed"] for run in runs) != [0, 1, 2]:
            raise ValueError("expected three seeds")
        methods = [method for method in ORDER if method in runs[0]["test_metrics"]]
        if set(methods) != set(runs[0]["test_metrics"]):
            raise ValueError("unrecognized result method")
        common = {"group": name, "cohort": group["cohort"],
                  "dataset": display_dataset(group["dataset"]), "backbone": group["backbone"]}
        scalar_keys = ["raw_sign_accuracy", "mean_3class_accuracy", "standardized_mse",
                       "original_units_mse", "mean_macro_f1"]
        means = {method: {key: float(np.mean([run["test_metrics"][method]["metrics"][key]
                                             for run in runs])) for key in scalar_keys} for method in methods}
        for method in methods:
            records = [run["test_metrics"][method]["metrics"] for run in runs]
            selections = [selection(run, method) for run in runs]
            row = {**common, "method": method, **means[method],
                   "raw_sign_seed_std_pp": 100 * float(np.std([rec["raw_sign_accuracy"] for rec in records], ddof=1)),
                   "mse_seed_std": float(np.std([rec["standardized_mse"] for rec in records], ddof=1)),
                   "nontrivial_blend_seeds": sum(bool(s.get("is_nontrivial_blend", False)) for s in selections),
                   "nontrivial_with_learned_seeds": sum(bool(s.get("is_nontrivial_blend", False)
                       and s.get("includes_actual_learned_component", False)) for s in selections)}
            for ref in ["base", "standalone_selected"]:
                row.update({key + "_vs_" + ref: value for key, value in differences(means[method], means[ref]).items()})
                row["joint_seeds_vs_" + ref] = sum(joint(rec, run["test_metrics"][ref]["metrics"])
                    for rec, run in zip(records, runs))
                ci = group["time_block_bootstrap_after_seed_mean"].get(method + "__vs__" + ref)
                if ci:
                    row["raw_sign_delta_ci95_pp_vs_" + ref] = [100 * value for value in ci["metrics"]["raw_sign_accuracy"]["ci95"]]
                    row["mse_delta_ci95_absolute_vs_" + ref] = ci["metrics"]["standardized_mse"]["ci95"]
            row["joint_improvement_vs_both"] = row["joint_improvement_vs_base"] and row["joint_improvement_vs_standalone_selected"]
            both = [joint(rec, run["test_metrics"]["base"]["metrics"])
                    and joint(rec, run["test_metrics"]["standalone_selected"]["metrics"])
                    for rec, run in zip(records, runs)]
            row["joint_seeds_vs_both"] = sum(both)
            row["nontrivial_joint_seeds_vs_both"] = sum(flag and bool(sel.get("is_nontrivial_blend", False))
                                                       for flag, sel in zip(both, selections))
            rows.append(row)
            for run, rec, sel in zip(runs, records, selections):
                seedrow = {**common, "seed": run["seed"], "method": method, **{key: rec[key] for key in scalar_keys}}
                for ref in ["base", "standalone_selected"]:
                    seedrow.update({key + "_vs_" + ref: value for key, value in differences(
                        rec, run["test_metrics"][ref]["metrics"]).items()})
                seeds.append(seedrow)
                identities.append({**common, "seed": run["seed"], "method": method,
                                   "test_candidate_id": run["test_metrics"][method]["candidate_id"], **sel})
        for run in runs:
            for family in FAMILIES:
                sel = selection(run, family)
                scheme = sel.get("selected_scheme", family) if family == "fusion_selected" else family
                if scheme == "base":
                    members = ["base"]
                elif scheme == "threeway_mean":
                    members = ["learned_selected", "flat", "difference"]
                else:
                    chosen = run["validation_selection"]["pairwise_fusions"][scheme]
                    members = [chosen["member_a"], chosen["member_b"]]
                for member in members:
                    member_rows.append({**common, "seed": run["seed"], "family": family,
                        "selected_scheme": scheme, "member": member,
                        **differences(run["test_metrics"][family]["metrics"], run["test_metrics"][member]["metrics"])})
    counts = []
    for cohort in sorted({row["cohort"] for row in rows}):
        for method in ORDER:
            subset = [row for row in rows if row["cohort"] == cohort and row["method"] == method]
            if not subset:
                continue
            counts.append({"cohort": cohort, "method": method, "settings": len(subset),
                **{key: sum(row[key] for row in subset) for key in [
                    "joint_improvement_vs_base", "joint_improvement_vs_standalone_selected",
                    "joint_seeds_vs_base", "joint_seeds_vs_standalone_selected",
                    "joint_improvement_vs_both", "joint_seeds_vs_both", "nontrivial_joint_seeds_vs_both",
                    "nontrivial_blend_seeds", "nontrivial_with_learned_seeds"]}})
    return {"sources": sources, "generator": {"path": str(Path(__file__).resolve()),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, "definitions": {
        "mse_delta_pct": "100*(mean seed MSE(method)/mean seed MSE(reference)-1)",
        "joint": "raw sign strictly improves and standardized MSE <= reference + 1e-8; accuracy deltas <=1e-12 are numerical ties (far below one point)",
        "counts": "descriptive counts, not an average of accuracy across methods or independent datasets",
        "no_test_selection": "All deployed choices originate in validation_selection; tables retain all fixed methods."},
        "group_method_rows": rows, "cohort_counts": counts}, seeds, identities, member_rows, groups


def write_markdown(output, report, groups):
    lines = ["# 固定融合方案：逐方法全量结果", "", "方向单位为百分点（pp）；MSE 是相对同一设置 base 的变化百分比。每行先在三个 seed 内取均值；不跨方法平均。严格联合达标另按 seed 计数。", "",
             "`standalone_selected`、`fusion_selected` 都只使用验证集选择。ETTh 是已使用测试段的探索性复核；Weather/Exchange Rate 是额外数据验证。", ""]
    for name, group in groups.items():
        lines += [f"## {display_dataset(group['dataset'])} / {group['backbone']} ({group['cohort']})", "",
                  "| 方法 | sign Δpp | MSE Δ% | 三分类 Δpp | 联合 seeds /3 | 相对选中单方案 sign Δpp / MSE Δ% | 非端点融合 seeds /3 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for row in report["group_method_rows"]:
            if row["group"] != name:
                continue
            lines.append(f"| {row['method']} | {row['raw_sign_delta_pp_vs_base']:+.3f} | {row['mse_delta_pct_vs_base']:+.3f} | {row['three_class_delta_pp_vs_base']:+.3f} | {row['joint_seeds_vs_base']} | {row['raw_sign_delta_pp_vs_standalone_selected']:+.3f} / {row['mse_delta_pct_vs_standalone_selected']:+.3f} | {row['nontrivial_blend_seeds']} |")
        lines += ["", "验证集选择（按 seed 0/1/2；权重是第二成员的系数）：", "", "| seed | 单方案 | 融合家族 | w | 真正混合 | 有实际学习方向校正 |", "|---|---|---|---:|---|---|"]
        for run in group["seeds"].values():
            val = run["validation_selection"]
            fusion = val["fusion_selected"]
            scheme = fusion["selected_scheme"]
            w = val["pairwise_fusions"].get(scheme, {}).get("selected_weight", "—")
            lines.append(f"| {run['seed']} | {val['standalone_selected']['selected_member']} | {scheme} | {w} | {fusion['is_nontrivial_blend']} | {fusion['includes_actual_learned_component']} |")
        lines.append("")
    lines += ["## 工件", "", "- `summary.json`：同表数值、逐方案达标次数、来源哈希。", "- `method_summary.csv` / `per_seed.csv`：均值和逐 seed 指标。", "- `selections.csv`：alpha、权重、退回 base 与端点身份。", "- `fusion_vs_members.csv`：每个融合及最终选择相对实际成员。", "- `direction_diagnostics.csv`（若启用诊断）：各变量/步长、真实精确零与非零变化条件下的方向准确率、改对/改错比例；不参与选参。", "- 源结果保留所有验证网格、互补性、混淆矩阵、按尺度指标及时间块 bootstrap。", ""]
    (output / "evidence.md").write_text("\n".join(lines), encoding="utf-8")


def plot(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [row for row in report["group_method_rows"] if row["cohort"] == "additional_validation"]
    group_names = list(dict.fromkeys(row["group"] for row in rows))
    if len(group_names) != 4:
        return
    styles = {
        "difference": ("Difference loss", "#777777", "o"),
        "learned_selected": ("Learned correction", "#167eab", "^"),
        "prior": ("Train prior", "#ab7a00", "s"),
        "flat": ("Flat correction", "#70953d", "D"),
        "standalone_selected": ("Validation-selected single", "#7253a1", "P"),
        "fusion_selected": ("Validation-selected fusion", "#c9414b", "X"),
        "threeway_mean": ("Fixed mean of three", "#d07c20", "*"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, group_names):
        subset = [row for row in rows if row["group"] == name and row["method"] in styles]
        for row in subset:
            label, color, marker = styles[row["method"]]
            ax.scatter(row["mse_delta_pct_vs_base"], row["raw_sign_delta_pp_vs_base"],
                       s=70, marker=marker, color=color, edgecolors="white", linewidths=.5, label=label, zorder=3)
        ax.axhline(0, color="#999999", linewidth=.8)
        ax.axvline(0, color="#999999", linewidth=.8)
        ax.grid(alpha=.15)
        ax.set_title(subset[0]["dataset"] + " / " + subset[0]["backbone"])
        ax.set_xlabel("MSE change vs base (%)  [lower is better]")
        ax.set_ylabel("Raw sign accuracy change (pp)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.10), ncol=4, fontsize=9)
    fig.suptitle("Additional-data validation: direction accuracy and MSE\nThree-seed means; upper-left of the origin improves both")
    fig.savefig(output / "fusion_tradeoffs.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "fusion_tradeoffs.pdf", bbox_inches="tight")
    plt.close(fig)


def direction_diagnostics(output, result_paths):
    """Report exact-zero and channel/horizon effects; never change selection."""
    rows = []
    for result_path in result_paths:
        groups = json.loads(result_path.read_text())["groups"]
        for name, group in groups.items():
            config = json.loads((Path(group["cell_dir"]) / "config.json").read_text())
            columns = config["data"]["value_columns"]
            archive_path = result_path.parent / "cell_artifacts" / name / "predictions.npz"
            with np.load(archive_path, allow_pickle=False) as archive:
                for run in group["seeds"].values():
                    prefix = f"{name}__seed{run['seed']}__test__split__"
                    history = archive[prefix + "history"].astype(np.float64)
                    truth = archive[prefix + "truth"].astype(np.float64)
                    anchor = history[:, -1:, :]
                    true_sign = np.sign(np.diff(np.concatenate([anchor, truth], axis=1), axis=1))
                    base = archive[run["test_metrics"]["base"]["prediction_key"]].astype(np.float64)
                    base_correct = np.sign(np.diff(np.concatenate([anchor, base], axis=1), axis=1)) == true_sign
                    for method, item in run["test_metrics"].items():
                        prediction = archive[item["prediction_key"]].astype(np.float64)
                        pred_sign = np.sign(np.diff(np.concatenate([anchor, prediction], axis=1), axis=1))
                        correct = pred_sign == true_sign
                        squared_error = (prediction - truth) ** 2
                        strata = [("overall", "all", (slice(None), slice(None), slice(None)))]
                        strata += [("channel", column, (slice(None), slice(None), channel))
                                   for channel, column in enumerate(columns)]
                        strata += [("horizon", horizon + 1, (slice(None), horizon, slice(None)))
                                   for horizon in range(truth.shape[1])]
                        for dimension, label, idx in strata:
                            zero = true_sign[idx] == 0
                            accuracy = correct[idx]
                            base_accuracy = base_correct[idx]
                            rows.append({"group": name, "cohort": group["cohort"], "seed": run["seed"],
                                "method": method, "dimension": dimension, "label": label,
                                "points": int(zero.size), "true_exact_zero_fraction": float(zero.mean()),
                                "predicted_exact_zero_fraction": float((pred_sign[idx] == 0).mean()),
                                "raw_sign_accuracy": float(accuracy.mean()),
                                "accuracy_when_truth_nonzero": float(accuracy[~zero].mean()) if (~zero).any() else None,
                                "accuracy_when_truth_zero": float(accuracy[zero].mean()) if zero.any() else None,
                                "standardized_mse": float(squared_error[idx].mean()),
                                "base_wrong_to_right_fraction": float((accuracy & ~base_accuracy).mean()),
                                "base_right_to_wrong_fraction": float((~accuracy & base_accuracy).mean())})
    save_csv(output / "direction_diagnostics.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    report, seeds, identities, members, groups = summarize(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for filename, values in [("method_summary.csv", report["group_method_rows"]), ("per_seed.csv", seeds),
                             ("selections.csv", identities), ("fusion_vs_members.csv", members),
                             ("cohort_counts.csv", report["cohort_counts"])]:
        save_csv(args.output_dir / filename, values)
    write_markdown(args.output_dir, report, groups)
    if args.plot:
        plot(args.output_dir, report)
    if args.diagnostics:
        direction_diagnostics(args.output_dir, args.results)
    print(json.dumps({"groups": len(groups), "method_rows": len(report["group_method_rows"]),
                      "per_seed_rows": len(seeds), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
