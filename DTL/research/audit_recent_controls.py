"""Independent checks of all post hoc controls and original-unit range bounds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_recent_rolling import Audit, score, signs, check_solver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--analysis", required=True)
    args = parser.parse_args()
    root, analysis = Path(args.root), Path(args.analysis)
    audit = Audit()
    rows = pd.read_csv(analysis/"controls/per_seed.csv")
    ranges = pd.read_csv(analysis/"summary/physical_ranges_posthoc.csv")
    diag = json.loads((analysis/"controls/solver_diagnostics.json").read_text())
    cases = 0
    for file in sorted((root/"results_v2").glob("*/*/*_seed*/result.json")):
        case = file.parent
        if not (case/"validation_lock.json").exists():
            continue
        result, lock = json.loads(file.read_text()), json.loads((case/"validation_lock.json").read_text())
        dataset, year, name, seed = case.parents[1].name, int(case.parent.name), result["backbone"], result["seed"]
        key = f"{dataset}_{year}_{name}_seed{seed}"
        with np.load(case/"predictions.npz") as ar, np.load(analysis/"controls"/(key+".npz")) as controls:
            x, y, base, p = [ar["test__"+s] for s in ["x", "y", "base", "probabilities"]]
            training_labels = signs(ar["train__x"], ar["train__y"])
            prior = np.stack([(training_labels==i).mean((0, 1)) for i in [-1, 0, 1]], -1).astype(np.float32)
            last = np.sign(x[:, -1].astype(float)-x[:, -2].astype(float)).astype(int)+1
            probabilities = {"training_prior": np.broadcast_to(prior[None, None], p.shape),
                             "last_direction": np.repeat(np.eye(3, dtype=np.float32)[last][:, None], 24, 1)}
            flat = np.zeros_like(p)
            flat[..., 1] = 1
            probabilities["always_flat"] = flat
            sorted_p = np.sort(p, -1)
            gap = sorted_p[..., -1]-sorted_p[..., -2]
            q = np.repeat(((1-gap)/3)[..., None], 3, -1)
            q[..., 1] = (1+2*gap)/3
            probabilities["confidence_only_flat"] = q
            for method, probability in probabilities.items():
                z = controls[method]
                selected = rows[(rows.dataset==dataset)&(rows.year==year)&(rows.backbone==name)&(rows.seed==seed)&(rows.method==method)].iloc[0]
                actual = score(x, y, z)
                for metric in ["raw_accuracy", "mse", "raw_nonzero_accuracy", "raw_zero_accuracy"]:
                    audit.close(actual[metric], selected[metric], key+" "+method+" "+metric)
                check_solver(audit, {"test__x": x, "test__base": base, "test__alpha_10.0": z, "test__probabilities": probability},
                             {"test__alpha_10.0": diag[key+"__"+method]}, np.asarray(lock["margin"], dtype=np.float32))
            meta = result["metadata"]
            mu, sd = np.array(meta["mean"]), np.array(meta["std"])
            low, high = np.full(len(mu), -np.inf), np.full(len(mu), np.inf)
            if dataset == "smard_de":
                low[:] = 0
            else:
                for i, col in enumerate(meta["columns"]):
                    if col.endswith("solar_radiation"):
                        low[i] = 0
                    elif col.endswith("relative_humidity"):
                        low[i], high[i] = 0, 100
            data = pd.read_csv(meta["csv"]).iloc[:, 1:].to_numpy(float)
            target = np.stack([data[t+1:t+25] for t in ar["test__origins"]])
            audit.check(np.all((target>=low)&(target<=high)), key+" raw truth satisfies domain ranges")
            bound_low, bound_high = ((low-mu)/sd).astype(np.float32), ((high-mu)/sd).astype(np.float32)
            with np.load(analysis/"summary"/(key+"_ranges.npz")) as clipped:
                for method in ["base", "raw_selected", "paired_gate"]:
                    label = "base_clipped" if method=="base" else method+"_clipped"
                    z = ar["test__"+method]
                    expected = np.minimum(np.maximum(z, bound_low), bound_high)
                    audit.check(np.array_equal(expected, clipped[label]), key+" physical clipping "+method)
                    audit.check(np.all((expected.astype(float)-y)**2 <= (z.astype(float)-y)**2+1e-12), key+" pointwise MSE monotonicity "+method)
                    selected = ranges[(ranges.dataset==dataset)&(ranges.year==year)&(ranges.backbone==name)&(ranges.seed==seed)&(ranges.method==label)].iloc[0]
                    actual = score(x, y, expected)
                    for metric in ["raw_accuracy", "mse", "raw_nonzero_accuracy", "raw_zero_accuracy"]:
                        audit.close(actual[metric], selected[metric], key+" clipping score "+method+" "+metric)
        cases += 1
    result = {"cases": cases, "checks": audit.checks, "failures": audit.failures, "failed_checks": len(audit.failures),
              "control_paths": len(rows), "range_paths": len(ranges)}
    (analysis/"controls_independent_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
