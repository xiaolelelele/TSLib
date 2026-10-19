#!/usr/bin/env python3
"""Report every frozen fixed-mean method; never select a method on test scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


COMBOS = ["learned_threeway", "nohead_threeway", "base_difference_mean", "flat_difference_mean"]
ATOMS = ["base", "difference", "residual", "residual_direction", "learned_single",
         "learned_multi", "prior", "flat", "learned_selected", "standalone_selected"]
METHODS = ATOMS + [name + "_" + mode for name in COMBOS for mode in ("raw", "deployed")]
SCALARS = ["raw_sign_accuracy", "standardized_mse", "original_units_mse",
           "mean_3class_accuracy", "mean_macro_f1"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def difference(a, b):
    direction = a["raw_sign_accuracy"] - b["raw_sign_accuracy"]
    if abs(direction) < 1e-12:
        direction = 0.0
    return {"raw_sign_delta_pp": 100 * direction,
            "mse_delta_pct": 100 * (a["standardized_mse"] / b["standardized_mse"] - 1),
            "mse_delta_absolute": a["standardized_mse"] - b["standardized_mse"],
            "original_mse_delta_pct": 100 * (a["original_units_mse"] / b["original_units_mse"] - 1),
            "three_class_delta_pp": 100 * (a["mean_3class_accuracy"] - b["mean_3class_accuracy"]),
            "joint": bool(direction > 0 and a["standardized_mse"] <= b["standardized_mse"] + 1e-8)}


def references(method):
    refs = ["base", "standalone_selected"]
    if method.startswith("learned_threeway_"):
        mode = method.rsplit("_", 1)[1]
        refs += [name + "_" + mode for name in COMBOS[1:]]
    return [ref for ref in refs if ref != method]


def build_tables(cells):
    method_rows, seed_rows, pair_rows, selections = [], [], [], []
    for cell in cells:
        runs = sorted(cell["runs"], key=lambda row: row["seed"])
        n = len(runs)
        if sorted(row["seed"] for row in runs) != list(range(n)) or n not in (3, 5):
            raise ValueError("unexpected fitted seeds: " + cell["group"])
        if cell["cohort"] == "additional_validation" and n != 5:
            raise ValueError("fresh verification requires all five seeds")
        for run in runs:
            if set(run["metrics"]) != set(METHODS):
                raise ValueError("method coverage mismatch: " + cell["group"])
        common = {key: cell[key] for key in ("group", "dataset", "backbone", "cohort")}
        common["n_seeds"] = n
        means = {method: {key: float(np.mean([run["metrics"][method][key] for run in runs]))
                          for key in SCALARS} for method in METHODS}
        for method in METHODS:
            method_row = {**common, "method": method, **means[method]}
            method_row.update({key + "_seed_std": float(np.std(
                [run["metrics"][method][key] for run in runs], ddof=1)) for key in SCALARS})
            method_row["accuracy_by_scale"] = {scale: float(np.mean([
                run["metrics"][method]["accuracy_by_scale"][scale] for run in runs]))
                for scale in ("1", "2", "4", "8")}
            method_row["macro_f1_by_scale"] = {scale: float(np.mean([
                run["metrics"][method]["macro_f1_by_scale"][scale] for run in runs]))
                for scale in ("1", "2", "4", "8")}
            for reference in references(method):
                paired = [difference(run["metrics"][method], run["metrics"][reference]) for run in runs]
                delta = difference(means[method], means[reference])
                pair = {**common, "method": method, "reference": reference, **delta,
                        "joint_seeds": sum(row["joint"] for row in paired),
                        "paired_sign_delta_seed_std_pp": float(np.std([row["raw_sign_delta_pp"] for row in paired], ddof=1)),
                        "paired_mse_delta_seed_std_absolute": float(np.std([row["mse_delta_absolute"] for row in paired], ddof=1))}
                ci = cell["intervals"].get((method, reference))
                if ci:
                    pair["sign_ci95_pp"] = [100 * value for value in ci["raw_sign_accuracy"]["ci95"]]
                    pair["mse_ci95_absolute"] = ci["standardized_mse"]["ci95"]
                pair["stable_screen_4of5"] = bool(n == 5 and delta["joint"] and pair["joint_seeds"] >= 4)
                pair_rows.append(pair)
                method_row.update({key + "_vs_" + reference: value for key, value in pair.items()
                                   if key not in common and key not in ("method", "reference")})
            both = [difference(run["metrics"][method], run["metrics"]["base"])["joint"]
                    and difference(run["metrics"][method], run["metrics"]["standalone_selected"])["joint"]
                    for run in runs]
            method_row["joint_seeds_vs_both"] = sum(both)
            method_row["mean_joint_vs_both"] = bool(difference(means[method], means["base"])["joint"]
                and difference(means[method], means["standalone_selected"])["joint"])
            method_row["stable_vs_both_4of5_each"] = bool(n == 5 and method_row["mean_joint_vs_both"]
                and sum(difference(run["metrics"][method], run["metrics"]["base"])["joint"] for run in runs) >= 4
                and sum(difference(run["metrics"][method], run["metrics"]["standalone_selected"])["joint"] for run in runs) >= 4)
            method_rows.append(method_row)
            for run in runs:
                seed_row = {**common, "seed": run["seed"], "method": method, **run["metrics"][method]}
                for reference in references(method):
                    seed_row.update({key + "_vs_" + reference: value for key, value in difference(
                        run["metrics"][method], run["metrics"][reference]).items()})
                seed_rows.append(seed_row)
                if method in run["identities"]:
                    selections.append({**common, "seed": run["seed"], "method": method,
                                       **run["identities"][method]})
    counts = []
    for cohort in sorted({cell["cohort"] for cell in cells}):
        for method in METHODS:
            subset = [row for row in method_rows if row["cohort"] == cohort and row["method"] == method]
            counts.append({"cohort": cohort, "method": method, "settings": len(subset),
                           "seed_fits": sum(row["n_seeds"] for row in subset),
                           "mean_joint_vs_both_settings": sum(row["mean_joint_vs_both"] for row in subset),
                           "joint_seeds_vs_both": sum(row["joint_seeds_vs_both"] for row in subset),
                           "stable_vs_both_settings": sum(row["stable_vs_both_4of5_each"] for row in subset),
                           **{key + "_vs_" + ref: sum(row.get(key + "_vs_" + ref, 0) for row in subset)
                              for key in ("joint", "joint_seeds") for ref in ("base", "standalone_selected")}})
    return method_rows, seed_rows, pair_rows, selections, counts


def display_pair(row):
    return f'{row["raw_sign_delta_pp"]:+.3f} / {row["mse_delta_pct"]:+.3f}'


def markdown_report(output, cells, method_rows, pair_rows):
    lines = ["# 固定三元均值：完整复核结果", "",
             "每个数据×骨干分别在拟合 seeds 内取均值；不跨方法或数据平均方向准确率。方向变化单位为百分点，MSE为同一标准化空间中seed均值的相对变化百分比。raw为直接固定均值，deployed为按validation门槛回退base后的流程。", "",
             "新增Electricity/Traffic只取预先固定的前8个数值列，各5 seeds；既有四域各3 seeds的去方向头复核为探索性结果。", ""]
    lookup = {(r["group"], r["method"], r["reference"]): r for r in pair_rows}
    for cell in cells:
        group = cell["group"]
        n = len(cell["runs"])
        lines.extend([f'## {cell["dataset"]} / {cell["backbone"]} ({cell["cohort"]}, n={n})', "",
                      "| 方法 | raw sign % | 标准化MSE | 相对base Δpp / ΔMSE% | 联合优于base | 相对选中单方案 Δpp / ΔMSE% | 联合优于单方案 |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in (r for r in method_rows if r["group"] == group):
            a = lookup.get((group, row["method"], "base"))
            b = lookup.get((group, row["method"], "standalone_selected"))
            lines.append(f'| {row["method"]} | {100 * row["raw_sign_accuracy"]:.4f} | {row["standardized_mse"]:.8f} | '
                         f'{display_pair(a) if a else "0 / 0"} | {str(a["joint_seeds"]) if a else "0"}/{n} | '
                         f'{display_pair(b) if b else "0 / 0"} | {str(b["joint_seeds"]) if b else "0"}/{n} |')
        lines += ["", "### 去方向头比较", "", "| 比较 | 方向Δpp / MSEΔ% | 联合改善seed | 方向95%区间pp | MSE差值95%区间 |", "|---|---:|---:|---:|---:|"]
        for mode in ("raw", "deployed"):
            row = lookup[(group, "learned_threeway_" + mode, "nohead_threeway_" + mode)]
            lines.append(f'| {mode}: learned − nohead | {display_pair(row)} | {row["joint_seeds"]}/{n} | '
                         f'{row.get("sign_ci95_pp")} | {row.get("mse_ci95_absolute")} |')
        lines.append("")
    lines += ["## 解释边界", "", "联合判据：raw sign严格提高且标准化MSE不高于参照+1e-8。汇总中小于1e-12的方向差视为浮点并列，远小于一个方向观测点。验证选法未改变。", "",
              "区间为2000次、8窗口非环形移动块bootstrap，先将每个窗口的配对差值在拟合seed内平均。区间条件于当前数据和模型；未作多重检验校正。五次拟合并非五个独立数据集。原单位MSE、三分类和逐尺度指标完整保留在CSV/JSON中。", "",
              "CSV/JSON配对比较表另列主方案对两个固定二元控制的描述性差值；未用测试分数选择部署方法。原子alpha、单方案选择与deployed回退均来自validation。"]
    (output / "evidence.md").write_text("\n".join(lines) + "\n")


def figures(output, method_rows, pair_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fresh = sorted({r["group"] for r in method_rows if r["cohort"] == "additional_validation"})
    if not fresh:
        return
    look = {(r["group"], r["method"], r["reference"]): r for r in pair_rows}
    colors = ["#1664bd", "#cb4b16", "#20804b", "#7e57a6"]
    labels = ["Learned + flat + diff", "Base + flat + diff", "Base + diff", "Flat + diff"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    for ax, group in zip(axes.flat, fresh):
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.axvline(0, color="#777777", linewidth=.8)
        for family, color, label in zip(COMBOS, colors, labels):
            row = look[(group, family + "_deployed", "base")]
            ax.scatter(row["mse_delta_pct"], row["raw_sign_delta_pp"], color=color, s=66, label=label, marker="s")
            raw = look[(group, family + "_raw", "base")]
            ax.scatter(raw["mse_delta_pct"], raw["raw_sign_delta_pp"], facecolor="none", edgecolor=color, s=66, marker="o")
            ax.plot([raw["mse_delta_pct"], row["mse_delta_pct"]], [raw["raw_sign_delta_pp"], row["raw_sign_delta_pp"]], color=color, alpha=.6, linewidth=.8)
        ax.set_title(group.replace("_", " "))
        ax.set_xlabel("MSE change vs base (%)")
        ax.set_ylabel("Raw direction change vs base (pp)")
        ax.grid(alpha=.17)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Fixed means on first 8 channels, 5 fitted seeds\nSquares: validation-gated deployment; open circles: raw means", fontsize=12)
    fig.savefig(output / "fixed_mean_tradeoffs.png", dpi=180)
    fig.savefig(output / "fixed_mean_tradeoffs.pdf")
    plt.close(fig)


def load_cells(paths):
    cells, sources, seen = [], [], set()
    for root in paths:
        root = root.resolve()
        document_path = root / "results.json"
        document = json.loads(document_path.read_text())
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") is not True:
            raise ValueError("refusing an incomplete evaluation: " + str(root))
        audit_path = root / "independent_audit.json"
        audit = json.loads(audit_path.read_text())
        if audit.get("passed") is not True:
            raise ValueError("refusing evaluation without a passing independent audit: " + str(root))
        sources.append({"root": str(root), "results_sha256": sha256(document_path),
                        "manifest_sha256": sha256(manifest_path), "audit_sha256": sha256(audit_path)})
        for name, group in document["groups"].items():
            if name in seen:
                raise ValueError("duplicate group: " + name)
            seen.add(name)
            if "seeds" not in group:
                cell_path = Path(group["artifacts"]["results_json"])
                if not cell_path.is_absolute():
                    cell_path = root / cell_path
                group = json.loads(cell_path.read_text())
            dataset = group["dataset"]
            if "exchange" in dataset.lower():
                dataset = "Exchange Rate"
            runs = []
            for seed, record in group["seeds"].items():
                metrics = {method: record["test"]["methods"][method]["metrics"] for method in ATOMS}
                identities = {method: {key: value for key, value in record["test"]["methods"][method].items()
                                       if key != "metrics"} for method in ATOMS}
                for atom, choice in record["validation"].get("atom_selection", {}).items():
                    identities[atom]["validation_selection"] = choice
                for selector in ("learned_selected", "standalone_selected"):
                    identities[selector]["validation_selection"] = record["validation"][selector]
                for combo in COMBOS:
                    for mode in ("raw", "deployed"):
                        method = combo + "_" + mode
                        row = record["test"]["combinations"][combo][mode]
                        metrics[method] = row["metrics"]
                        identities[method] = {key: value for key, value in row.items() if key != "metrics"}
                        identities[method]["validation_combination"] = record["validation"]["combinations"][combo]
                runs.append({"seed": int(seed), "metrics": metrics, "identities": identities})
            intervals = {}
            for pair in group["time_block_bootstrap_after_seed_mean"]["comparisons"].values():
                key = (pair["candidate_id"], pair["reference_id"])
                if key in intervals:
                    raise ValueError("duplicate paired interval")
                intervals[key] = {metric: values["bootstrap_ci"] for metric, values in pair["metrics"].items()}
            cells.append({"group": name, "dataset": dataset, "backbone": group["backbone"],
                          "cohort": group["cohort"], "runs": runs, "intervals": intervals})
    cells.sort(key=lambda cell: (cell["cohort"] != "additional_validation", cell["dataset"], cell["backbone"]))
    return cells, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("refusing to overwrite existing report")
    cells, sources = load_cells(args.root)
    methods, seeds, pairs, choices, counts = build_tables(cells)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in [("method_summary.csv", methods), ("per_seed.csv", seeds),
                           ("comparisons.csv", pairs), ("selections.csv", choices), ("cohort_counts.csv", counts)]:
        csv_write(args.output_dir / filename, rows)
    json_write(args.output_dir / "summary.json", {"sources": sources, "methods": methods,
               "comparisons": pairs, "cohort_counts": counts, "definitions": {
                   "joint": "strict raw-sign improvement and standardized MSE <= reference + 1e-8",
                   "mse_percent": "ratio of seed mean MSE, not mean of seed ratios",
                   "numerical_ties": "absolute accuracy difference <1e-12 is a numerical tie",
                   "scope": "New first-eight-channel Electricity/Traffic pilots, 5 seeds; old four domains exploratory, 3 seeds.",
                   "stable_screen": "mean joint and >=4/5 seed joint, evaluated per comparison",
                   "no_test_selection": True}})
    markdown_report(args.output_dir, cells, methods, pairs)
    figures(args.output_dir, methods, pairs)
    shutil.copy2(__file__, args.output_dir / Path(__file__).name)
    json_write(args.output_dir / "reproduction.json", {"python": "/home/lgc/.conda/envs/TFB/bin/python",
               "script": str(Path(__file__).resolve()), "roots": [str(root.resolve()) for root in args.root],
               "output_dir": str(args.output_dir.resolve()), "environment": {"MPLCONFIGDIR": "/tmp/fixed_threeway_mpl"}})
    json_write(args.output_dir / "artifact_sha256.json", {path.name: sha256(path)
               for path in sorted(args.output_dir.iterdir()) if path.is_file() and path.name != "artifact_sha256.json"})
    print(json.dumps({"groups": len(cells), "method_rows": len(methods), "seed_rows": len(seeds),
                      "comparisons": len(pairs), "output": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
