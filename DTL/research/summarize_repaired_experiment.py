"""Summarize only schema-v2 runs that passed the independent NumPy audit.

CSV metrics use the independently recomputed audit values. Convergence values
are saved solver diagnostics, explicitly not an independent optimizer proof.
All destinations must be new; no historical experiment artifact is modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative_percent(value, baseline):
    return 100 * (value / baseline - 1) if baseline else None


def _convergence_row(group, seed, stage, family, key, diagnostics, indices=None, total=None):
    grad = np.asarray(diagnostics.get("projected_gradient_rms", []), dtype=float)
    steps = np.asarray(diagnostics.get("iterations_per_path", []), dtype=int)
    cap = np.asarray(diagnostics.get("cap_binding_per_path", []), dtype=bool)
    tol = diagnostics.get("convergence_tol")
    if indices is not None:
        grad, steps, cap = grad[indices], steps[indices], cap[indices]
    n = len(grad)
    return {"group": group, "seed": seed, "stage": stage, "family": family, "candidate": key,
            "paths": n if total is None else total, "projected_paths": n,
            "max_projected_gradient_rms": float(grad.max()) if n else None,
            "paths_with_tol": n if tol is not None else 0,
            "not_converged_paths": int((grad > tol).sum()) if tol is not None else 0,
            "at_max_steps_paths": int((steps >= diagnostics.get("max_steps", 0)).sum()) if n else 0,
            "at_400_steps_paths": int((steps >= 400).sum()), "cap_binding_paths": int(cap.sum()),
            "max_iterations": int(steps.max()) if n else None,
            "max_perturbation_rms": (float(np.asarray(diagnostics["perturbation_rms"])[indices].max())
                                     if indices is not None and n else
                                     float(max(diagnostics.get("perturbation_rms", [0]))) if n else None)}


def convergence_rows(group, run, n_test, library):
    seed, choices, result = run["seed"], run["validation"]["choices"], []
    for stage, diagnostics in (("validation_all_alpha", run["validation"]["convergence"]),
                               ("test_evaluated_candidates", run["test_convergence"])):
        for key, diag in diagnostics.items():
            if "projected_gradient_rms" in diag:
                result.append(_convergence_row(group, seed, stage, key.split("_alpha_")[0], key, diag))

    def resolve(key):
        seen = set()
        while key in choices:
            if key in seen:
                raise ValueError("cyclic validation choices")
            seen.add(key)
            key = choices[key]
        return key

    for method in choices:
        key = resolve(method)
        if "_alpha_" in key:
            result.append(_convergence_row(group, seed, "test_deployed", method, key,
                                           run["test_convergence"][key]))
        elif key.startswith("posterior_beta_"):
            selected = np.asarray(run["test_convergence"][key]["chosen_candidate_indices"])
            for i, candidate in enumerate(library):
                indices = np.flatnonzero(selected == i)
                if not len(indices):
                    continue
                diag = run["test_convergence"].get(candidate, {})
                result.append(_convergence_row(group, seed, "test_deployed", method, candidate, diag,
                                               indices if candidate != "base" else None, total=len(indices)))
        else:
            result.append(_convergence_row(group, seed, "test_deployed", method, key, {}, total=n_test))
        # Base and neural paths count in total paths, with no projection criterion.
    return result


def aggregate_convergence(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["group"], row["stage"], row["family"])].append(row)
    result = []
    for (group, stage, family), items in buckets.items():
        total = {key: sum(row[key] for row in items) for key in (
            "paths", "projected_paths", "paths_with_tol", "not_converged_paths",
            "at_max_steps_paths", "at_400_steps_paths", "cap_binding_paths")}
        row = {"group": group, "stage": stage, "family": family, **total}
        for key in ("max_projected_gradient_rms", "max_iterations", "max_perturbation_rms"):
            values = [r[key] for r in items if r[key] is not None]
            row[key] = max(values) if values else None
        for name, count, denominator in (
                ("not_converged_fraction", "not_converged_paths", "paths_with_tol"),
                ("at_max_steps_fraction", "at_max_steps_paths", "projected_paths"),
                ("at_400_steps_fraction", "at_400_steps_paths", "projected_paths"),
                ("cap_binding_fraction", "cap_binding_paths", "projected_paths")):
            row[name] = total[count] / total[denominator] if total[denominator] else None
        result.append(row)
    return result


def summarize(audit):
    per_seed, paired, convergence, scales_rows, training_rows = [], [], [], [], []
    for group_audit in audit["groups"]:
        directory = Path(group_audit["run_directory"])
        for name, digest in group_audit["artifact_sha256"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError("artifact changed after audit: " + str(directory / name))
        result = json.loads((directory / "results.json").read_text())
        group = result["config"].get("group", directory.name)
        with np.load(directory / "predictions.npz", allow_pickle=False) as arrays:
            truth = arrays["test__truth"].astype(np.float64)
            last = arrays["test__history"][:, -1:].astype(np.float64)
            true_sign = np.sign(np.diff(np.concatenate((last, truth), axis=1), axis=1))
            for run, reviewed in zip(result["runs"], group_audit["runs"]):
                if run["seed"] != reviewed["seed"]:
                    raise ValueError("seed ordering mismatch")
                seed, metrics = run["seed"], reviewed["test_metrics_recomputed"]
                base = metrics["base"]
                base_path = arrays["seed%s__test__base" % seed].astype(np.float64)
                base_sign = np.sign(np.diff(np.concatenate((last, base_path), axis=1), axis=1))
                base_raw_mse = float(np.mean(((base_path-truth) * np.asarray(result["normalization"]["std"]))**2))
                tables = [("deployed", metrics, "test"),
                          ("raw_model", reviewed.get("test_model_metrics_recomputed", {}), "test_model")]
                for kind, table, namespace in tables:
                    for method, metric in table.items():
                        path_key = "seed%s__%s__%s" % (seed, namespace, method)
                        path = arrays[path_key].astype(np.float64) if path_key in arrays else base_path
                        pred_sign = np.sign(np.diff(np.concatenate((last, path), axis=1), axis=1))
                        raw_mse = np.mean(((path-truth) * np.asarray(result["normalization"]["std"]))**2)
                        row = {"group": group, "cohort": result["config"].get("cohort", ""),
                               "kind": kind, "seed": seed, "method": method,
                               "validation_choice": run["validation"]["choices"].get(method, method),
                               "mse": metric["mse"], "base_mse": base["mse"],
                               "delta_mse_percent": relative_percent(metric["mse"], base["mse"]),
                               "raw_sign_accuracy": metric["raw_sign_accuracy"],
                               "delta_raw_sign_pp": 100*(metric["raw_sign_accuracy"]-base["raw_sign_accuracy"]),
                               "mean_3class_accuracy": metric["mean_3class_accuracy"],
                               "delta_3class_pp": 100*(metric["mean_3class_accuracy"]-base["mean_3class_accuracy"]),
                               "original_units_mse": float(raw_mse), "base_original_units_mse": base_raw_mse,
                               "original_units_mse_increased": bool(raw_mse > base_raw_mse),
                               "strict_joint_success": bool(metric["raw_sign_accuracy"] > base["raw_sign_accuracy"]
                                                            and metric["mse"] <= base["mse"]+1e-8),
                               "modification_rms": float(np.sqrt(np.mean((path-base_path)**2))),
                               "raw_wrong_to_right": int(((base_sign != true_sign) & (pred_sign == true_sign)).sum()),
                               "raw_right_to_wrong": int(((base_sign == true_sign) & (pred_sign != true_sign)).sum()),
                               "total_raw_entries": int(true_sign.size)}
                        per_seed.append(row)
                        for scale, values in metric["scales"].items():
                            scales_rows.append({"group": group, "kind": kind, "seed": seed, "method": method,
                                                "scale": scale, "accuracy": values["accuracy"],
                                                "macro_f1": values["macro_f1"],
                                                "delta_accuracy_pp": 100*(values["accuracy"]-base["scales"][scale]["accuracy"])})
                for reference in ("base", "learned_selected"):
                    method = "posterior_candidates"
                    a, b = metrics[method], metrics[reference]
                    paired.append({"group": group, "seed": seed, "method": method, "reference": reference,
                                   "mse": a["mse"], "reference_mse": b["mse"],
                                   "delta_mse_percent": relative_percent(a["mse"], b["mse"]),
                                   "delta_raw_sign_pp": 100*(a["raw_sign_accuracy"]-b["raw_sign_accuracy"]),
                                   "delta_3class_pp": 100*(a["mean_3class_accuracy"]-b["mean_3class_accuracy"]),
                                   "strict_joint_success": bool(a["raw_sign_accuracy"] > b["raw_sign_accuracy"]
                                                                and a["mse"] <= b["mse"]+1e-8)})
                convergence.extend(convergence_rows(group, run, len(truth), reviewed["posterior_library"]))
                for method, values in run["training"].items():
                    row = {"group": group, "seed": seed, "method": method, "best_epoch": values["best_epoch"],
                           "earlystop_loss": values["earlystop_loss"], "fit_seconds": values["fit_seconds"]}
                    row.update({"parameters_"+key: val for key, val in values["parameters"].items()})
                    row.update(values.get("calibration_loss_diagnostics", {}))
                    training_rows.append(row)
    grouped = defaultdict(list)
    for row in per_seed:
        grouped[(row["group"], row["kind"], row["method"])].append(row)
    summary = []
    for (group, kind, method), rows in grouped.items():
        mse, base_mse = np.mean([r["mse"] for r in rows]), np.mean([r["base_mse"] for r in rows])
        raw, base_raw = np.mean([r["original_units_mse"] for r in rows]), np.mean([r["base_original_units_mse"] for r in rows])
        gain = float(np.mean([r["delta_raw_sign_pp"] for r in rows]))
        successes = sum(r["strict_joint_success"] for r in rows)
        summary.append({"group": group, "kind": kind, "method": method, "seeds": len(rows),
                        "mse": float(mse), "delta_mse_percent": relative_percent(mse, base_mse),
                        "delta_raw_sign_pp": gain,
                        "delta_3class_pp": float(np.mean([r["delta_3class_pp"] for r in rows])),
                        "delta_original_units_mse_percent": relative_percent(raw, base_raw),
                        "original_units_increase_seeds": sum(r["original_units_mse_increased"] for r in rows),
                        "strict_joint_success_seeds": successes,
                        "base_fallback_seeds": sum(r["validation_choice"] == "base" for r in rows) if kind == "deployed" and method != "base" else 0,
                        "local_signal_3of3_and_0.2pp": len(rows) == 3 and successes == 3 and gain >= .2})
    paired_summary = []
    for group, reference in dict.fromkeys((r["group"], r["reference"]) for r in paired):
        rows = [r for r in paired if r["group"] == group and r["reference"] == reference]
        paired_summary.append({"group": group, "reference": reference, "seeds": len(rows),
                               "delta_mse_percent": relative_percent(np.mean([r["mse"] for r in rows]), np.mean([r["reference_mse"] for r in rows])),
                               "delta_raw_sign_pp": float(np.mean([r["delta_raw_sign_pp"] for r in rows])),
                               "delta_3class_pp": float(np.mean([r["delta_3class_pp"] for r in rows])),
                               "strict_joint_success_seeds": sum(r["strict_joint_success"] for r in rows)})
    return {"per_seed": per_seed, "method_summary": summary, "posterior_pairs_per_seed": paired,
            "posterior_pairs_summary": paired_summary, "per_scale": scales_rows,
            "convergence_per_seed": convergence, "convergence_summary": aggregate_convergence(convergence),
            "training_diagnostics": training_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text())
    if not audit["passed"]:
        raise ValueError("refusing to summarize a failed audit")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    tables = summarize(audit)
    output.mkdir(parents=True)
    for name, rows in tables.items():
        write_csv(output / (name + ".csv"), rows)
    (output / "summary.json").write_text(json.dumps(tables, indent=2, allow_nan=False) + "\n")
    report = ["# 修复实验独立审计汇总", "", "数值指标来自 NumPy 重算；模型参数、训练及求解诊断来自保存报告。平均 MSE 百分比采用 seed 均值之比。raw sign 保留真实零方向，与带死区三分类分别报告。", "",
              "| 组 | 方法 | Δraw (pp) | ΔMSE (%) | Δ三分类 (pp) | 联合成功 seeds | 原单位 MSE 增加 seeds |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in tables["method_summary"]:
        if row["kind"] == "deployed" and row["method"] not in ("base", "direction_head"):
            report.append("| {group} | {method} | {delta_raw_sign_pp:+.4f} | {delta_mse_percent:+.4f} | {delta_3class_pp:+.4f} | {strict_joint_success_seeds}/{seeds} | {original_units_increase_seeds}/{seeds} |".format(**row))
    report += ["", "## 完整后验与同库统一选择的配对", "", "| 组 | 对照 | Δraw (pp) | ΔMSE (%) | Δ三分类 (pp) | 联合成功 seeds |", "|---|---|---:|---:|---:|---:|"]
    for row in tables["posterior_pairs_summary"]:
        report.append("| {group} | {reference} | {delta_raw_sign_pp:+.4f} | {delta_mse_percent:+.4f} | {delta_3class_pp:+.4f} | {strict_joint_success_seeds}/{seeds} |".format(**row))
    report += ["", "## 求解诊断", "", "convergence_summary.csv 按组、family、阶段汇总最大 projected-gradient RMS、未达容差比例、达到最大步数/400 步比例与 cap binding。validation_all_alpha 包含所有非零 alpha；test_evaluated_candidates 是实际求解的候选；test_deployed 跟随验证锁定选择及 posterior 每条路径 index。base 和神经模型只计总路径，不计投影收敛；无容差 smoke 不计未收敛。各比例分母为已求解投影路径（未收敛比例进一步要求有容差），不是将 base 计成收敛。", "", "## 解释限制", "", "这是固定测试集上的探索性复验和额外通道检查。3 个随机 seed 不是 3 个独立数据域；联合成功不代表显著性或部署风险保证。raw_model 行保留未选择的各权重基线，仅描述表现，不能按测试挑模型。CaReTS 尚未完整复现。旧结果与新结果同时存在多项协议变化，不能做单因素因果归因。", ""]
    (output / "evidence.md").write_text("\n".join(report))
    (output / "provenance.json").write_text(json.dumps({"audit_path": str(Path(args.audit).resolve()),
        "audit_sha256": hashlib.sha256(Path(args.audit).read_bytes()).hexdigest(),
        "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+"\n")
    print(json.dumps({"output": str(output), "groups": len(audit["groups"]), "seed_rows": len(tables["per_seed"])}))


if __name__ == "__main__":
    main()
