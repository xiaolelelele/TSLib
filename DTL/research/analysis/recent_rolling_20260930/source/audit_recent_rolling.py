"""Independent NumPy audit; does not import the runner or its scoring helpers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class Audit:
    def __init__(self):
        self.checks, self.failures = 0, []
    def check(self, value, label):
        self.checks += 1
        if not bool(value):
            self.failures.append(label)
    def close(self, a, b, label, tolerance=1e-10):
        self.check(np.allclose(a, b, rtol=tolerance, atol=tolerance, equal_nan=True), label)


def signs(x, path):
    extended = np.concatenate((np.asarray(x[:, -1:], dtype=float), np.asarray(path, dtype=float)), axis=1)
    return np.sign(extended[:, 1:] - extended[:, :-1])


def score(x, y, z):
    observed, predicted = signs(x, y), signs(x, z)
    nonzero = observed != 0
    return {"mse": float(np.mean((z.astype(float)-y.astype(float))**2)),
            "raw_accuracy": float(np.mean(observed == predicted)),
            "raw_nonzero_accuracy": float(np.mean((observed == predicted)[nonzero])) if nonzero.any() else None,
            "raw_zero_accuracy": float(np.mean((observed == predicted)[~nonzero])) if (~nonzero).any() else None,
            "true_zero_fraction": float(np.mean(~nonzero)),
            "anchored_binary_accuracy": float(np.mean((z < x[:, -1:]) == (y < x[:, -1:])))}


def paired(x, y, b, z):
    truth = signs(x, y)
    gain = ((signs(x, z) == truth).astype(float)-(signs(x, b) == truth)).mean((1, 2))
    # Independent squared-risk identity, including the base residual and amplitude.
    delta = z.astype(float)-b.astype(float)
    risk = (delta**2 + 2*delta*(b.astype(float)-y.astype(float))).mean((1, 2))
    return np.c_[gain, risk]


def features(x, b, z, p):
    x, b, z, p = [a.astype(float) for a in (x, b, z, p)]
    correction = z-b
    slope = np.diff(np.concatenate((x[:, -1:], b), axis=1), axis=1)
    historical_slope = np.diff(x, axis=1)
    ref_class, alt_class = signs(x, b).astype(int)+1, signs(x, z).astype(int)+1
    prob_gain = (np.take_along_axis(p, alt_class[..., None], -1)-np.take_along_axis(p, ref_class[..., None], -1))[..., 0]
    def mean(a): return a.mean((1, 2))
    return np.c_[mean(prob_gain), mean(-(p*np.log(np.maximum(p, 1e-12))).sum(-1)), mean(p.max(-1)),
                 mean(correction**2), mean(abs(correction)), abs(correction).max((1, 2)), mean(abs(slope)),
                 mean(abs(historical_slope)), mean(historical_slope**2), mean(abs(historical_slope[:, -24:])),
                 mean(abs(b-x[:, -1:])), mean(x[:, -24:].var(1, keepdims=True)), mean((p.argmax(-1)==1).astype(float)),
                 mean((ref_class!=alt_class).astype(float)), mean(correction*(b-x[:, -1:])),
                 mean(correction*(b-x[:, -24:].mean(1, keepdims=True))), mean(correction*b)]


def check_selection(audit, ar, lock, field, keys):
    x, y, b = ar["validation__x"], ar["validation__y"], ar["validation__base"]
    baseline = score(x, y, b)
    rows = []
    for key in keys:
        z = b if key == "alpha_0.0" else ar["validation__"+key]
        measured = score(x, y, z)
        item = next(v for v in lock[field]["candidates"] if v["key"] == key)
        for metric, value in measured.items():
            if value is None:
                audit.check(item[metric] is None, field+" "+key+" "+metric)
            else:
                audit.close(value, item[metric], field+" "+key+" "+metric)
        allowed = measured["raw_accuracy"] > baseline["raw_accuracy"] and measured["mse"] <= baseline["mse"]+1e-8
        audit.check(allowed == item["eligible"], field+" eligibility")
        if allowed:
            rows.append((key, measured))
    selected = sorted(rows, key=lambda r: (-r[1]["raw_accuracy"], r[1]["mse"], r[0]))[0][0] if rows else None
    audit.check(selected == lock[field]["chosen"], field+" locked choice")


def check_solver(audit, ar, diagnostics, margin):
    for key, diagnostic in diagnostics.items():
        split, name = key.split("__")
        alpha = float(name.split("_")[1])
        x, b, z, p = [ar[split+"__"+k] for k in ("x", "base", name, "probabilities")]
        cls = p.argmax(-1)
        sorted_p = np.sort(p, axis=-1)
        w = sorted_p[..., -1]-sorted_p[..., -2]
        d = np.diff(np.concatenate((x[:, -1:], z), axis=1), axis=1)
        r = np.where(cls==0, np.maximum(d+margin, 0), np.where(cls==2, -np.maximum(margin-d, 0), d)).astype(np.float32)
        weighted = w*r
        transpose = weighted.copy()
        transpose[:, :-1] -= weighted[:, 1:]
        grad = z-b+alpha*transpose
        eta = 1/(1+4*alpha)
        step_delta = z-eta*grad-b
        norm = np.sqrt(np.mean(step_delta**2, axis=(1, 2), keepdims=True))
        next_z = b+step_delta*np.minimum(1, .5/np.maximum(norm, np.finfo(np.float32).tiny))
        pg = np.sqrt(np.mean((z-next_z)**2, axis=(1, 2)))/eta
        rms = np.sqrt(np.mean((z-b)**2, axis=(1, 2)))
        audit.check(np.max(pg) <= 1.1e-5, key+" KKT residual")
        audit.check(np.max(rms) <= .500002, key+" cap")
        audit.check(diagnostic["all_converged"] and diagnostic["max_iterations"] < 600, key+" convergence status")
        audit.close(rms, diagnostic["rms"], key+" saved RMS", 2e-6)
        objective = .5*np.sum((z-b)**2+alpha*w*r**2, axis=(1, 2))
        audit.close(objective, diagnostic["objective"], key+" objective", 2e-6)


def audit_case(audit, case):
    result = json.loads((case/"result.json").read_text())
    lock = json.loads((case/"validation_lock.json").read_text())
    metadata = lock["metadata"]
    audit.check(not lock["test_labels_used_for_selection"], str(case)+" selection label flag")
    audit.check((case/"validation_lock.json").stat().st_mtime <= (case/"result.json").stat().st_mtime, "lock before result")
    audit.check(result["checkpoint_reload"], "runner checkpoint reload")
    frame = pd.read_csv(metadata["csv"])
    values = frame.iloc[:, 1:].to_numpy(float)
    mean, std = np.nanmean(values[:metadata["cuts"][1]], 0), np.nanstd(values[:metadata["cuts"][1]], 0)
    audit.close(mean, metadata["mean"], "train-only mean")
    audit.close(std, metadata["std"], "train-only std")
    normalized = ((values-mean)/std).astype(np.float32)
    finite = np.isfinite(normalized).all(1)
    prefix = np.r_[0, np.cumsum(~finite)]
    with np.load(case/"predictions.npz") as ar:
        for key in ar.files:
            audit.check(np.isfinite(ar[key]).all(), str(case)+" finite "+key)
        for split, meta in metadata["splits"].items():
            origins = ar[split+"__origins"]
            low, high = meta["target_bounds"]
            audit.check(np.all((origins+1 >= low)&(origins+25<=high)), split+" target boundary")
            audit.check(set(origins%24)=={0, 6, 12, 18}, split+" all four phases")
            audit.check(np.all(prefix[origins+25]==prefix[origins-95]), split+" valid full window")
            expected_origins = np.arange(0, len(values), 6)
            expected_origins = expected_origins[(expected_origins>=95)&(expected_origins+1>=low)&(expected_origins+25<=high)]
            expected_origins = expected_origins[prefix[expected_origins+25]==prefix[expected_origins-95]]
            audit.check(np.array_equal(origins, expected_origins), split+" exact eligible origins")
            expected_x = np.stack([normalized[t-95:t+1] for t in origins])
            expected_y = np.stack([normalized[t+1:t+25] for t in origins])
            audit.check(np.array_equal(expected_x, ar[split+"__x"]), split+" original history")
            audit.check(np.array_equal(expected_y, ar[split+"__y"]), split+" original truth")
        for method, saved in result["test"].items():
            actual = score(ar["test__x"], ar["test__y"], ar["test__"+method])
            for metric, value in actual.items():
                audit.check(saved[metric] is None, method+" "+metric) if value is None else audit.close(value, saved[metric], method+" "+metric)
        check_selection(audit, ar, lock, "alpha", ["alpha_0.0", "alpha_0.1", "alpha_1.0", "alpha_10.0"])
        check_selection(audit, ar, lock, "gate_selection", list(lock["threshold_specs"]))
        fx = features(ar["riskfit__x"], ar["riskfit__base"], ar["riskfit__alpha_1.0"], ar["riskfit__probabilities"])
        target = paired(ar["riskfit__x"], ar["riskfit__y"], ar["riskfit__base"], ar["riskfit__alpha_1.0"])
        audit.close(fx, ar["riskfit__features"], "observable riskfit features")
        audit.close(target, ar["riskfit__targets"], "paired risk identity")
        spec = lock["gate"]
        mu, sd = fx.mean(0), np.maximum(fx.std(0), 1e-8)
        audit.close(mu, spec["mean"], "gate fit-only mean")
        audit.close(sd, spec["std"], "gate fit-only std")
        design = np.c_[np.ones(len(fx)), (fx-mu)/sd]
        penalty = np.diag(np.r_[0., np.full(fx.shape[1], np.sqrt(10.))])
        coef = np.linalg.lstsq(np.vstack((design, penalty)), np.vstack((target, np.zeros((fx.shape[1]+1, 2)))), rcond=None)[0]
        audit.close(coef, spec["coef"], "independent ridge least squares", 1e-8)
        for split in ("validation", "test"):
            f = features(ar[split+"__x"], ar[split+"__base"], ar[split+"__alpha_1.0"], ar[split+"__probabilities"])
            prediction = np.c_[np.ones(len(f)), (f-mu)/sd] @ coef
            audit.close(prediction, ar[split+"__gate_prediction"], split+" gate inference", 1e-8)
            if split == "validation":
                for key, (gain, risk) in lock["threshold_specs"].items():
                    mask = (prediction[:, 0]>gain)&(prediction[:, 1]<=risk)
                    audit.check(np.array_equal(np.where(mask[:, None, None], ar["validation__alpha_1.0"], ar["validation__base"]), ar["validation__"+key]), key+" gate candidate")
        chosen = lock["gate_selection"]["chosen"]
        if chosen is None:
            mask = np.zeros(len(ar["test__x"]), bool)
        else:
            gain, risk = lock["threshold_specs"][chosen]
            pred = ar["test__gate_prediction"]
            mask = (pred[:, 0]>gain)&(pred[:, 1]<=risk)
        audit.check(np.array_equal(mask, ar["test__gate_mask"]), "test gate mask")
        audit.check(np.array_equal(ar["test__paired_gate"], np.where(mask[:, None, None], ar["test__alpha_1.0"], ar["test__base"])), "test gate output")
        alpha = lock["alpha"]["chosen"]
        audit.check(np.array_equal(ar["test__raw_selected"], ar["test__base"] if alpha is None else ar["test__"+alpha]), "test alpha output")
        for method in ("raw_selected", "paired_gate", "alpha_1.0"):
            audit.close(paired(ar["test__x"], ar["test__y"], ar["test__base"], ar["test__"+method]), ar["test__"+method+"__paired_targets"], method+" test paired identity")
        diagnostic = json.loads((case/"solver_diagnostics.json").read_text())
        check_solver(audit, ar, diagnostic, np.asarray(lock["margin"], dtype=np.float32))
        logits = ar["calibration__logits"].astype(float)
        labels = signs(ar["calibration__x"], ar["calibration__y"]).astype(int)+1
        nll = {}
        for temperature in [.5, 1, 2, 4]:
            scaled = logits/temperature
            shifted = scaled-scaled.max(-1, keepdims=True)
            logp = shifted-np.log(np.exp(shifted).sum(-1, keepdims=True))
            nll[temperature] = -np.take_along_axis(logp, labels[..., None], -1).mean()
        audit.check(min(nll, key=nll.get)==lock["temperature"], "calibration-only temperature choice")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    audit = Audit()
    manifest = json.loads((root/"input_sha256_v2.json").read_text())
    for filename, expected in manifest.items():
        path = Path(filename)
        audit.check(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected, "frozen input "+filename)
    datasets = json.loads((root/"configs/matrix_v2.json").read_text())["datasets"]
    raw_root = Path(datasets[0]["csv"]).parent
    raw_records = json.loads((raw_root/"download_manifest.json").read_text())
    for row in raw_records:
        audit.check(hashlib.sha256(Path(row["file"]).read_bytes()).hexdigest()==row["sha256"], "raw download "+row["file"])
    cases = sorted((root/"results_v2").glob("*/*/*_seed*/result.json"))
    cases = [f.parent for f in cases if (f.parent/"validation_lock.json").exists()]
    if not args.partial:
        audit.check(len(cases)==24, "full matrix 24 cases")
        audit.check((root/"results_v2/completion.json").is_file(), "full completion")
    for case in cases:
        audit_case(audit, case)
    output = {"cases": len(cases), "checks": audit.checks, "failures": audit.failures,
              "failed_checks": len(audit.failures), "raw_files": len(raw_records), "frozen_files": len(manifest)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if audit.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
