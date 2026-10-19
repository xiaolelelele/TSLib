"""Frozen two-domain/two-year research matrix, separate from historical runners."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .baseline_losses import cosdir_loss, tdt_loss
from .models import DirectionHead, ResidualAdapter
from .recent_data import load_recent
from .recent_methods import (CaReTS, backbone, raw_labels, raw_project, symbols,
                             paired_targets, gate_features, fit_gate, predict_gate)
from .run import _seed, _params, _temperature, _residual_width


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def metrics(x, y, path):
    x, y, path = [np.asarray(a, dtype=np.float64) for a in (x, y, path)]
    truth, predicted = symbols(x, y), symbols(x, path)
    nonzero = truth != 0
    return {"mse": float(np.mean((path - y)**2)), "raw_accuracy": float(np.mean(truth == predicted)),
            "raw_nonzero_accuracy": float(np.mean((truth == predicted)[nonzero])) if nonzero.any() else None,
            "raw_zero_accuracy": float(np.mean((truth == predicted)[~nonzero])) if (~nonzero).any() else None,
            "true_zero_fraction": float(np.mean(~nonzero)),
            "anchored_binary_accuracy": float(np.mean((path < x[:, -1:, :]) == (y < x[:, -1:, :])))}


def fit_model(name, model, loss, monitor_loss, splits, cfg, shuffle_seed):
    _seed(shuffle_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, best_epoch, stale = float("inf"), 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tx, ty = splits["train"]
    ex, ey = splits["earlystop"]
    start = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        model.train()
        for ids in torch.randperm(len(tx)).split(cfg["batch"]):
            optimizer.zero_grad(set_to_none=True)
            value = loss(tx[ids], ty[ids])
            if not torch.isfinite(value):
                raise ValueError(f"nonfinite {name} training loss")
            value.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f"nonfinite {name} gradient")
            optimizer.step()
            if isinstance(model, CaReTS):
                with torch.no_grad():
                    model.log_variances.clamp_(-10, 10)
        model.eval()
        total = 0.
        with torch.no_grad():
            for offset in range(0, len(ex), cfg["batch"]):
                xx, yy = ex[offset:offset+cfg["batch"]], ey[offset:offset+cfg["batch"]]
                total += float(monitor_loss(xx, yy)) * len(xx)
        score = total / len(ex)
        if not np.isfinite(score):
            raise ValueError(f"nonfinite {name} earlystop loss")
        if score < best - 1e-8:
            best, best_epoch, stale = score, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    row = {"method": name, "best_epoch": best_epoch, "executed_epochs": epoch+1,
           "earlystop_loss": best, "seconds": time.perf_counter()-start, **_params(model)}
    print(f"fit {name}: epoch {best_epoch}/{epoch+1}", flush=True)
    return row, best_state


def predict(model, x, batch=64):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[o:o+batch]) for o in range(0, len(x), batch)])


def select_paths(x, y, paths, base):
    reference = metrics(x, y, base)
    rows, eligible = [], []
    for key, path in paths.items():
        row = {"key": key, **metrics(x, y, path)}
        row["eligible"] = row["raw_accuracy"] > reference["raw_accuracy"] and row["mse"] <= reference["mse"] + 1e-8
        rows.append(row)
        if row["eligible"]:
            eligible.append(row)
    chosen = min(eligible, key=lambda r: (-r["raw_accuracy"], r["mse"], r["key"]))["key"] if eligible else None
    return chosen, {"base": reference, "candidates": rows, "chosen": chosen}


def fit_carets(splits, tensors, metadata, cfg, out, seed):
    arrays, rows, states = {}, [], {}
    dimensions = (metadata["lookback"], metadata["horizon"], len(metadata["columns"]))
    models = {}
    for variant in (3, 4):
        name = f"carets{variant}_cnn"
        _seed(seed + 5000 + variant)
        model = CaReTS(*dimensions, variant).to(cfg["device"])
        row, state = fit_model(name, model, model.loss, lambda x, y: F.mse_loss(model(x), y), tensors, cfg, seed+7000+variant)
        rows.append(row)
        states[name] = state
        models[name] = model
    # The two architectures and training rules are fixed; no test architecture selection.
    write_json(out / "lock.json", {"dataset": metadata, "seed": seed, "training": rows,
                                   "definition": "paper-formula adaptations; both variants reported"})
    for split_name in ("validation", "test"):
        split = splits[split_name]
        arrays[split_name + "__x"] = split.x
        arrays[split_name + "__y"] = split.y
        arrays[split_name + "__origins"] = split.origins
        for name, model in models.items():
            arrays[split_name + "__" + name] = predict(model, tensors[split_name][0]).cpu().numpy()
    torch.save({"metadata": metadata, "seed": seed, "states": states}, out / "checkpoint.pt")
    result = {"training": rows, "test": {name: metrics(splits["test"].x, splits["test"].y, arrays["test__"+name]) for name in models}}
    np.savez_compressed(out / "predictions.npz", **arrays)
    write_json(out / "result.json", result)
    return {name: arrays["test__"+name] for name in models}


def run_case(splits, tensors, metadata, cfg, out, seed, name, carets_predictions):
    out.mkdir(parents=True, exist_ok=True)
    dimensions = (metadata["lookback"], metadata["horizon"], len(metadata["columns"]))
    device = torch.device(cfg["device"])
    arrays, training, states, models = {}, [], {}, {}
    for split_name, split in splits.items():
        arrays[split_name + "__x"], arrays[split_name + "__y"] = split.x, split.y
        arrays[split_name + "__origins"] = split.origins
    _seed(seed)
    initial = backbone(name, *dimensions, cfg["reference_root"]).to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    for method in ("base", "cosdir", "tdalign"):
        model = backbone(name, *dimensions, cfg["reference_root"]).to(device)
        model.load_state_dict(initial_state)
        if method == "base":
            loss = lambda x, y: F.mse_loss(model(x), y)
        elif method == "cosdir":
            loss = lambda x, y: cosdir_loss(x, model(x), y, weight=.5)
        else:
            loss = lambda x, y: tdt_loss(x, model(x), y, variant="official")
        row, state = fit_model(method, model, loss, lambda x, y: F.mse_loss(model(x), y), tensors, cfg, seed+100003)
        training.append(row)
        states[method], models[method] = state, model
    del initial, initial_state
    base = models["base"]
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)
    _seed(seed+20001)
    head = DirectionHead(*dimensions, [1], hidden=32).to(device)
    def head_logits(x):
        return head(x, base(x).detach())[:, 0]
    def head_loss(x, y):
        return F.cross_entropy(head_logits(x).reshape(-1, 3), raw_labels(x, y).reshape(-1))
    row, state = fit_model("raw_head", head, head_loss, head_loss, tensors, cfg, seed+20002)
    training.append(row)
    states["raw_head"] = state
    _seed(seed+21001)
    width = _residual_width(*dimensions, _params(head)["total"], 32)
    residual = ResidualAdapter(*dimensions, hidden=width).to(device)
    residual_loss = lambda x, y: F.mse_loss(residual(x, base(x).detach()), y)
    row, state = fit_model("residual", residual, residual_loss, residual_loss, tensors, cfg, seed+21002)
    training.append(row)
    states["residual"] = state
    cal_x, cal_y = tensors["calibration"]
    with torch.no_grad():
        logits = head_logits(cal_x)
        temperature = _temperature(logits, raw_labels(cal_x, cal_y), [.5, 1, 2, 4])
    arrays["calibration__logits"] = logits.cpu().numpy()
    margin = .1 * np.median(np.abs(np.diff(splits["train"].x.astype(np.float64), axis=1)), axis=(0, 1))
    margin_tensor = torch.as_tensor(margin, dtype=torch.float32, device=device)
    diagnostics, probabilities, candidate_paths = {}, {}, {}
    for split_name in ("riskfit", "validation"):
        xx, yy = tensors[split_name]
        predicted_base = predict(base, xx)
        arrays[split_name + "__base"] = predicted_base.cpu().numpy()
        with torch.no_grad():
            probs = (head_logits(xx) / temperature).softmax(-1)
        probabilities[split_name] = probs
        arrays[split_name + "__probabilities"] = probs.cpu().numpy()
        paths = {"alpha_0.0": predicted_base.cpu().numpy()}
        for alpha in ([1.] if split_name == "riskfit" else [.1, 1., 10.]):
            path, diagnostic = raw_project(xx, predicted_base, probs, margin_tensor, alpha)
            if not diagnostic["all_converged"]:
                raise ValueError("raw projection did not reach frozen tolerance")
            key = "alpha_"+str(alpha)
            paths[key] = path.cpu().numpy()
            arrays[split_name + "__" + key] = paths[key]
            diagnostics[split_name+"__"+key] = diagnostic
        candidate_paths[split_name] = paths
    validation = splits["validation"]
    selected_alpha, alpha_lock = select_paths(validation.x, validation.y, candidate_paths["validation"], arrays["validation__base"])
    riskfit = splits["riskfit"]
    risk_features = gate_features(riskfit.x, arrays["riskfit__base"], arrays["riskfit__alpha_1.0"], arrays["riskfit__probabilities"])
    targets = paired_targets(riskfit.x, riskfit.y, arrays["riskfit__base"], arrays["riskfit__alpha_1.0"])
    gate = fit_gate(risk_features, targets, ridge=10.)
    arrays["riskfit__features"], arrays["riskfit__targets"] = risk_features, targets
    val_features = gate_features(validation.x, arrays["validation__base"], arrays["validation__alpha_1.0"], arrays["validation__probabilities"])
    gate_prediction = predict_gate(gate, val_features)
    arrays["validation__gate_prediction"] = gate_prediction
    gate_paths, threshold_specs = {}, {}
    for gain in (0., .01, .03):
        for risk in (0., -.01):
            key = f"gain_{gain}_risk_{risk}"
            mask = (gate_prediction[:, 0] > gain) & (gate_prediction[:, 1] <= risk)
            threshold_specs[key] = [gain, risk]
            gate_paths[key] = np.where(mask[:, None, None], arrays["validation__alpha_1.0"], arrays["validation__base"])
            arrays["validation__"+key] = gate_paths[key]
    selected_gate, gate_lock = select_paths(validation.x, validation.y, gate_paths, arrays["validation__base"])
    lock = {"metadata": metadata, "seed": seed, "backbone": name, "temperature": temperature,
            "margin": margin.tolist(), "residual_width": width, "alpha": alpha_lock,
            "gate_selection": gate_lock, "gate": gate, "threshold_specs": threshold_specs,
            "training": training, "config": cfg, "test_labels_used_for_selection": False}
    write_json(out / "validation_lock.json", lock)
    checkpoint = {"spec": lock, "states": states}
    torch.save(checkpoint, out / "checkpoint.pt")
    # No test predictions or test-derived statistics occur above the persisted lock.
    xx, yy = tensors["test"]
    test = splits["test"]
    for method, model in models.items():
        arrays["test__"+method] = predict(model, xx).cpu().numpy()
    with torch.no_grad():
        test_probabilities = (head_logits(xx) / temperature).softmax(-1)
    arrays["test__probabilities"] = test_probabilities.cpu().numpy()
    test_base_tensor = torch.as_tensor(arrays["test__base"], device=device)
    for alpha in (.1, 1., 10.):
        path, diagnostic = raw_project(xx, test_base_tensor, test_probabilities, margin_tensor, alpha)
        if not diagnostic["all_converged"]:
            raise ValueError("test raw projection did not converge")
        key = "alpha_"+str(alpha)
        arrays["test__"+key] = path.cpu().numpy()
        diagnostics["test__"+key] = diagnostic
    arrays["test__raw_selected"] = arrays["test__base"] if selected_alpha is None else arrays["test__"+selected_alpha]
    test_features = gate_features(test.x, arrays["test__base"], arrays["test__alpha_1.0"], arrays["test__probabilities"])
    test_gate_prediction = predict_gate(gate, test_features)
    if selected_gate is None:
        mask = np.zeros(len(test.x), dtype=bool)
    else:
        gain, risk = threshold_specs[selected_gate]
        mask = (test_gate_prediction[:, 0] > gain) & (test_gate_prediction[:, 1] <= risk)
    arrays["test__gate_prediction"], arrays["test__gate_mask"] = test_gate_prediction, mask
    arrays["test__paired_gate"] = np.where(mask[:, None, None], arrays["test__alpha_1.0"], arrays["test__base"])
    arrays["test__residual"] = predict(lambda_residual(residual, base), xx).cpu().numpy()
    arrays["test__seasonal_naive"] = test.x[:, -24:, :].copy()
    arrays["test__last_value"] = np.repeat(test.x[:, -1:, :], 24, axis=1)
    for method, path in carets_predictions.items():
        arrays["test__"+method] = path
    reported = ["base", "cosdir", "tdalign", "residual", "seasonal_naive", "last_value",
                "carets3_cnn", "carets4_cnn", "raw_selected", "paired_gate", "alpha_1.0"]
    scores = {method: metrics(test.x, test.y, arrays["test__"+method]) for method in reported}
    for method in ("raw_selected", "paired_gate", "alpha_1.0"):
        arrays["test__"+method+"__paired_targets"] = paired_targets(test.x, test.y, arrays["test__base"], arrays["test__"+method])
    checkpoint_reload_check(checkpoint, out, dimensions, cfg, tensors["test"][0][:11], arrays, base, residual)
    np.savez_compressed(out / "predictions.npz", **arrays)
    write_json(out / "solver_diagnostics.json", diagnostics)
    write_json(out / "result.json", {"backbone": name, "seed": seed, "metadata": metadata, "training": training,
                                     "test": scores, "test_gate_fraction": float(mask.mean()), "checkpoint_reload": True})
    print(f"complete {out.name}: selected {selected_alpha}; gate {selected_gate}, fraction {mask.mean():.3f}", flush=True)


class lambda_residual(torch.nn.Module):
    def __init__(self, residual, base):
        super().__init__()
        self.residual, self.base = residual, base
    def forward(self, x):
        return self.residual(x, self.base(x))


def checkpoint_reload_check(checkpoint, out, dimensions, cfg, x, arrays, base, residual):
    restored = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        for method in ("base", "cosdir", "tdalign"):
            model = backbone(restored["spec"]["backbone"], *dimensions, cfg["reference_root"])
            model.load_state_dict(restored["states"][method])
            model.eval()
            np.testing.assert_allclose(model(x.cpu()).numpy(), arrays["test__"+method][:len(x)], atol=2e-5, rtol=2e-5)
        head = DirectionHead(*dimensions, [1], hidden=32)
        head.load_state_dict(restored["states"]["raw_head"])
        head.eval()
        reference = backbone(restored["spec"]["backbone"], *dimensions, cfg["reference_root"])
        reference.load_state_dict(restored["states"]["base"])
        reference.eval()
        probability = (head(x.cpu(), reference(x.cpu()))[:, 0] / restored["spec"]["temperature"]).softmax(-1).numpy()
        np.testing.assert_allclose(probability, arrays["test__probabilities"][:len(x)], atol=2e-5, rtol=2e-5)
        adapter = ResidualAdapter(*dimensions, hidden=restored["spec"]["residual_width"])
        adapter.load_state_dict(restored["states"]["residual"])
        adapter.eval()
        np.testing.assert_allclose(adapter(x.cpu(), reference(x.cpu())).numpy(), arrays["test__residual"][:len(x)], atol=2e-5, rtol=2e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if cfg["device"].startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.045, device=torch.device(cfg["device"]))
    start = time.perf_counter()
    complete = []
    for dataset in cfg["datasets"]:
        for year in cfg["years"]:
            splits, metadata = load_recent(dataset["csv"], year)
            tensors = {name: (torch.as_tensor(s.x, device=cfg["device"]), torch.as_tensor(s.y, device=cfg["device"])) for name, s in splits.items()}
            write_json(out / dataset["name"] / str(year) / "dataset.json", metadata)
            for seed in cfg["seeds"]:
                carets_dir = out / dataset["name"] / str(year) / f"carets_seed{seed}"
                if (carets_dir / "result.json").exists():
                    with np.load(carets_dir / "predictions.npz") as ar:
                        carets_predictions = {f"carets{v}_cnn": ar[f"test__carets{v}_cnn"].copy() for v in (3, 4)}
                else:
                    carets_dir.mkdir(parents=True, exist_ok=True)
                    carets_predictions = fit_carets(splits, tensors, metadata, cfg, carets_dir, seed)
                for name in cfg["backbones"]:
                    case_dir = out / dataset["name"] / str(year) / f"{name}_seed{seed}"
                    if not (case_dir / "result.json").exists():
                        print(f"START {dataset['name']}/{year}/{name}/{seed}", flush=True)
                        run_case(splits, tensors, metadata, cfg, case_dir, seed, name, carets_predictions)
                    complete.append(str(case_dir))
                write_json(out / "progress.json", {"completed": complete, "seconds": time.perf_counter()-start})
            del tensors
    write_json(out / "completion.json", {"completed": complete, "cases": len(complete), "seconds": time.perf_counter()-start,
                                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cfg["device"].startswith("cuda") else None})


if __name__ == "__main__":
    main()
