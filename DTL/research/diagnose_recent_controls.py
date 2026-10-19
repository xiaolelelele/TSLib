"""Declared post hoc fixed controls; all inputs at forecast time or training-only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


def signs(x, y):
    full = np.concatenate((x[:, -1:].astype(float), y.astype(float)), 1)
    return np.sign(full[:, 1:]-full[:, :-1])


def scores(x, y, z):
    t, p = signs(x, y), signs(x, z)
    nz = t!=0
    return {"raw_accuracy": float((t==p).mean()), "mse": float(((z.astype(float)-y)**2).mean()),
            "raw_nonzero_accuracy": float((t==p)[nz].mean()), "raw_zero_accuracy": float((t==p)[~nz].mean()),
            "zero_fraction": float((~nz).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root/"source/v2"))
    from DTL.trend_lab.recent_methods import raw_project
    torch.set_num_threads(2)
    rows, heads, diagnostics = [], [], {}
    for filename in sorted((root/"results_v2").glob("*/*/*_seed*/result.json")):
        case = filename.parent
        if not (case/"validation_lock.json").exists():
            continue
        result, lock = json.loads(filename.read_text()), json.loads((case/"validation_lock.json").read_text())
        dataset, year = case.parents[1].name, int(case.parent.name)
        context = {"dataset": dataset, "year": year, "backbone": result["backbone"], "seed": result["seed"]}
        key = f"{dataset}_{year}_{result['backbone']}_seed{result['seed']}"
        with np.load(case/"predictions.npz") as ar:
            x, y, b, p = [ar["test__"+s] for s in ["x", "y", "base", "probabilities"]]
            tx, ty = ar["train__x"], ar["train__y"]
            truth = signs(x, y).astype(int)+1
            label = p.argmax(-1)
            nll = -np.take_along_axis(np.log(np.maximum(p, 1e-12)), truth[..., None], axis=-1).mean()
            heads.append({**context, "argmax_raw_accuracy": float((label==truth).mean()), "nll": float(nll),
                          "base_raw_accuracy": result["test"]["base"]["raw_accuracy"],
                          "selected_path_raw_accuracy": result["test"]["raw_selected"]["raw_accuracy"],
                          "head_flat_prediction_fraction": float((label==1).mean()),
                          "head_zero_accuracy": float((label==truth)[truth==1].mean()),
                          "head_nonzero_accuracy": float((label==truth)[truth!=1].mean())})
            frequency = np.stack([(signs(tx, ty)==cls).mean((0, 1)) for cls in [-1, 0, 1]], -1).astype(np.float32)
            prior = np.broadcast_to(frequency[None, None], p.shape).copy()
            flat = np.zeros_like(p)
            flat[..., 1] = 1
            history_last = np.sign(x[:, -1].astype(float)-x[:, -2].astype(float)).astype(int)+1
            recent = np.eye(3, dtype=np.float32)[history_last][:, None].repeat(24, axis=1)
            sorted_p = np.sort(p, axis=-1)
            gap = sorted_p[..., -1]-sorted_p[..., -2]
            confidence_flat = np.repeat(((1-gap)/3)[..., None], 3, axis=-1)
            confidence_flat[..., 1] = (1+2*gap)/3
            controls = {"training_prior": prior, "always_flat": flat, "last_direction": recent,
                        "confidence_only_flat": confidence_flat}
            predictions = {}
            for method, q in controls.items():
                path, diagnostic = raw_project(torch.as_tensor(x), torch.as_tensor(b), torch.as_tensor(q),
                                                torch.tensor(lock["margin"], dtype=torch.float32), 10.)
                if not diagnostic["all_converged"]:
                    raise ValueError("control did not reach fixed convergence tolerance")
                z = path.numpy()
                predictions[method] = z
                diagnostic["max_rms"] = max(diagnostic["rms"])
                diagnostics[key+"__"+method] = diagnostic
                measure = scores(x, y, z)
                ref = result["test"]["base"]
                observed, original, modified = signs(x, y), signs(x, b), signs(x, z)
                rows.append({**context, "method": method, **measure,
                             "raw_gain_pp": 100*(measure["raw_accuracy"]-ref["raw_accuracy"]),
                             "mse_change_pct": 100*(measure["mse"]/ref["mse"]-1),
                             "fix_rate": float(((modified==observed)&(original!=observed)).mean()),
                             "harm_rate": float(((modified!=observed)&(original==observed)).mean())})
            np.savez_compressed(out/(key+".npz"), **predictions)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"per_seed.csv", index=False)
    frame.groupby(["dataset", "year", "backbone", "method"]).mean(numeric_only=True).reset_index().to_csv(out/"summary.csv", index=False)
    pd.DataFrame(heads).to_csv(out/"classification_to_path.csv", index=False)
    (out/"solver_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(heads), "controls": len(rows), "output": str(out)}))


if __name__ == "__main__":
    main()
