"""Version 2 experiment: disjoint early stopping/calibration and audited controls.

Historical runners/checkpoints remain readable. All deployment choices are made
on validation before this module scores any test trajectory. This is a research
pilot, not a claim of distribution-free risk control or a new forecasting model.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_data, split_earlystop_calibration, Split
from .models import build_base, DirectionHead, ResidualAdapter
from .operators import fit_thresholds, multiscale_slopes, multiscale_direction_labels
from .losses import fit_slope_scales, direction_surrogate_loss, residual_direction_diagnostics
from .baseline_losses import cosdir_loss, tdt_loss
from .decoding import project_trajectory, score_posterior_candidates
from .metrics import evaluate_predictions
from .run import _train, _seed, _sync, _params, _residual_width, _temperature


def _write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _stats(x, y, path, scales, eps):
    """Small validation score, with the same float64 convention as final scoring."""
    x, y, path = x.double(), y.double(), path.double()
    truth = torch.diff(torch.cat((x[:, -1:], y), 1), dim=1).sign()
    pred = torch.diff(torch.cat((x[:, -1:], path), 1), dim=1).sign()
    labels = multiscale_direction_labels(x, y, scales, eps)
    guessed = multiscale_direction_labels(x, path, scales, eps)
    return {"mse": float((path-y).square().mean()),
            "raw_sign_accuracy": float((truth == pred).double().mean()),
            "mean_3class_accuracy": float((labels == guessed).double().mean())}


def choose(rows, base):
    """Strict raw-direction improvement and non-increasing standardized MSE."""
    feasible = [r for r in rows if r["metrics"]["raw_sign_accuracy"] > base["raw_sign_accuracy"]
                and r["metrics"]["mse"] <= base["mse"] + 1e-8]
    return min(feasible, key=lambda r: (-r["metrics"]["raw_sign_accuracy"],
                                      r["metrics"]["mse"], r["order"])) if feasible else None


def _posterior_select(x, base, candidates, probabilities, scales, eps, beta):
    paths = torch.stack(candidates)
    score = score_posterior_candidates(x, paths, probabilities, scales, eps)
    # This is a proximity cost, NOT an estimate or bound on true MSE risk.
    distance = (paths.double() - base.double()[None]).square().mean((2, 3))
    index = (score - float(beta) * distance).argmax(0)
    return paths[index, torch.arange(len(base), device=base.device)], index


def run(config, output, smoke=False):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("refusing to overwrite " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    cfg["protocol_version"], cfg["smoke"] = 2, bool(smoke)
    training = cfg.setdefault("training", {})
    if smoke:
        training.update(epochs=1, patience=1)
        cfg["seeds"] = cfg.get("smoke_seeds", [cfg.get("seeds", [0])[0]])
        cfg["alpha_grid"] = [0., .3]
        cfg["projection_steps"] = 8
        cfg["projection_tolerance"] = None
        cfg["cosdir_weights"] = [.1]
        cfg["direction_weights"] = [.1]
    _write(out / "config.json", cfg)
    bundle = load_data(cfg["data"])
    early, calibration, split_info = split_earlystop_calibration(
        bundle, earlystop_fraction=cfg.get("earlystop_fraction", .5))
    splits = {"train": bundle.train, "earlystop": early, "calibration": calibration,
              "validation": bundle.validation, "test": bundle.test}
    if smoke:
        splits = {name: Split(s.x[:64 if name == "train" else 8],
                            s.y[:64 if name == "train" else 8],
                            s.origins[:64 if name == "train" else 8]) for name, s in splits.items()}
    device = torch.device(cfg.get("device", "cpu"))
    tensors = {name: (torch.as_tensor(s.x, device=device), torch.as_tensor(s.y, device=device))
               for name, s in splits.items()}
    tx, ty = tensors["train"]
    ex, ey = tensors["earlystop"]
    cx, cy = tensors["calibration"]
    vx, vy = tensors["validation"]
    scales = list(cfg.get("scales", [1, 2, 4, 8]))
    if (not scales or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in scales)
            or scales[0] != 1 or len(set(scales)) != len(scales) or max(scales) >= tx.shape[1]):
        raise ValueError("scales must be unique positive integers, start at 1, and be smaller than lookback")
    eps = fit_thresholds(tx.cpu(), ty.cpu(), scales,
                         quantile=cfg.get("epsilon_quantile", .2), precision="float64").float().to(device)
    scale = fit_slope_scales(tx.cpu(), ty.cpu(), scales,
                            floor=cfg.get("slope_scale_floor", .02)).float().to(device)
    lookback, horizon, channels = tx.shape[1], ty.shape[1], tx.shape[2]
    hidden = int(cfg.get("hidden", 32))
    model_args = (lookback, horizon, channels)
    alphas = list(map(float, cfg.get("alpha_grid", [0, .1, .3, 1, 3, 10])))
    betas = list(map(float, cfg.get("posterior_betas", [0, .1, 1, 10])))
    if not alphas or alphas[0] != 0 or len(set(alphas)) != len(alphas):
        raise ValueError("alpha grid must start at 0 and contain unique values")
    if any(a < 0 or not np.isfinite(a) for a in alphas+betas):
        raise ValueError("alpha/beta must be finite nonnegative")
    projection_args = {"steps": int(cfg.get("projection_steps", 400)),
                       "max_rms": float(cfg.get("max_rms", .5)),
                       "convergence_tol": cfg.get("projection_tolerance", 1e-5)}
    train_labels = multiscale_direction_labels(tx, ty, scales, eps)
    counts = torch.stack([(train_labels == c).sum((0, 2)) for c in range(3)], -1)
    prior = counts.float() / counts.sum(-1, keepdim=True)
    result = {"schema_version": 2, "smoke": smoke, "config": cfg,
              "data": bundle.metadata, "earlystop_calibration_split": split_info,
              "split_windows": {name: len(s.x) for name, s in splits.items()},
              "normalization": {"mean": bundle.mean.tolist(), "std": bundle.std.tolist()},
              "epsilon": eps.cpu().tolist(), "slope_scale": scale.cpu().tolist(),
              "prior_counts": counts.cpu().tolist(), "runs": [],
              "selection_rule": "validation raw sign strictly improves; standardized MSE <= base + 1e-8",
              "notes": ["earlystop/calibration targets do not overlap; not statistical independence",
                        "posterior candidate cost is squared distance to base, not true MSE risk",
                        "direction surrogate is a stable boundary-aligned control; no Fisher-consistency claim",
                        "same epoch/patience/initialization policy; hyperparameter grids differ and are explicit"]}
    arrays, checkpoints = {}, {}
    for name in ("earlystop", "calibration", "validation", "test"):
        arrays[name+"__history"] = splits[name].x
        arrays[name+"__truth"] = splits[name].y
        arrays[name+"__origins"] = splits[name].origins
    for seed in cfg.get("seeds", [0, 1, 2]):
        print("[repair-v2] seed %s: training %s" % (seed, cfg["backbone"]), flush=True)
        training_rows, models, model_specs = {}, {}, {}
        def fit(name, model, loss, shuffle_seed, spec):
            _seed(shuffle_seed)
            _sync(device)
            start = time.perf_counter()
            epoch, monitor = _train(model, loss, tx, ty, ex, ey,
                int(training.get("epochs", 40)), int(training.get("batch_size", 64)),
                float(training.get("learning_rate", .001)), int(training.get("patience", 8)))
            _sync(device)
            model.eval()
            models[name], model_specs[name] = model, spec
            training_rows[name] = {"best_epoch": epoch, "earlystop_loss": monitor,
                                   "fit_seconds": time.perf_counter()-start, "parameters": _params(model)}
            print("[repair-v2] seed %s fitted %s epoch=%s" % (seed, name, epoch), flush=True)
        _seed(seed)
        base = build_base(cfg["backbone"], *model_args, hidden).to(device)
        initial = copy.deepcopy(base.state_dict())
        fit("base", base, lambda x, y: F.mse_loss(base(x), y), seed+100003, {"type": "base"})
        base.requires_grad_(False)
        def new_base():
            m = build_base(cfg["backbone"], *model_args, hidden).to(device)
            m.load_state_dict(initial)
            return m
        difference = new_base()
        def difference_loss(x, y):
            z = difference(x)
            return F.mse_loss(z, y) + F.mse_loss(torch.diff(torch.cat((x[:, -1:], z), 1), dim=1),
                                                torch.diff(torch.cat((x[:, -1:], y), 1), dim=1))
        fit("difference", difference, difference_loss, seed+100003, {"type": "base"})
        td = new_base()
        fit("tdalign_official", td, lambda x, y: tdt_loss(x, td(x), y, variant="official"),
            seed+100003, {"type": "base"})
        cos_names = []
        for weight in cfg.get("cosdir_weights", [.1, 1.]):
            name = "cosdir_weight_%g" % weight
            m = new_base()
            fit(name, m, lambda x, y, m=m, w=weight: cosdir_loss(x, m(x), y, weight=w),
                seed+100003, {"type": "base", "weight": float(weight)})
            cos_names.append(name)
        temps, val_prob = {}, {}
        for name, ks, ids in (("direction_head", scales, list(range(len(scales)))),
                              ("direction_head_single", [1], [0])):
            _seed(seed+20001)
            head = DirectionHead(*model_args, ks, hidden).to(device)
            def hloss(x, y, h=head, ks=ks, ids=ids):
                logits = h(x, base(x).detach())
                labels = multiscale_direction_labels(x, y, ks, eps[ids])
                return F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1))
            fit(name, head, hloss, seed+20002, {"type": "head", "scales": ks})
            with torch.no_grad():
                logits = head(cx, base(cx))
                labels = multiscale_direction_labels(cx, cy, ks, eps[ids])
                temps[name] = _temperature(logits, labels, cfg.get("temperature_grid", [.5, 1., 2., 4.]))
                val_prob[name] = torch.softmax(head(vx, base(vx))/temps[name], -1)
                val_prob[name+"_T1"] = torch.softmax(head(vx, base(vx)), -1)
        width = _residual_width(lookback, horizon, channels, _params(models["direction_head"])["total"], hidden)
        _seed(seed+21001)
        residual_initial = copy.deepcopy(ResidualAdapter(*model_args, width).state_dict())
        rd_names = []
        for weight in [0.] + cfg.get("direction_weights", [.01, .1]):
            name = "residual" if weight == 0 else "residual_direction_weight_%g" % weight
            m = ResidualAdapter(*model_args, width).to(device)
            m.load_state_dict(residual_initial)
            def rloss(x, y, m=m, w=weight):
                z = m(x, base(x).detach())
                mse = F.mse_loss(z, y)
                if w == 0: return mse
                labels = multiscale_direction_labels(x, y, scales, eps)
                return mse + w*direction_surrogate_loss(
                    multiscale_slopes(x.double(), z.double(), scales), labels, eps, scale)
            fit(name, m, rloss, seed+21001, {"type": "residual", "hidden": width, "weight": float(weight)})
            if weight:
                with torch.no_grad(): path = m(cx, base(cx))
                training_rows[name]["calibration_loss_diagnostics"] = residual_direction_diagnostics(
                    cx, cy, path, scales, eps, scale, weight=weight)
                rd_names.append(name)
        with torch.no_grad():
            val_paths = {name: m(vx, base(vx)) if model_specs[name]["type"] == "residual" else m(vx)
                         for name, m in models.items() if model_specs[name]["type"] != "head"}
        base_stats = _stats(vx, vy, val_paths["base"], scales, eps)
        entries = {name: {"kind": "model", "model": name} for name in val_paths}
        choices, val_grids, convergence = {}, {}, {}
        def choose_paths(name, keys):
            rows = [{"key": key, "order": i, "metrics": _stats(vx, vy, val_paths[key], scales, eps)}
                    for i, key in enumerate(keys)]
            selected = choose(rows, base_stats)
            key = selected["key"] if selected else "base"
            choices[name] = key
            val_grids[name] = rows
            val_paths[name] = val_paths[key]
            entries[name] = {"kind": "alias", "target": key}
        choose_paths("cosdir", cos_names)
        choose_paths("residual_direction", rd_names)
        family_keys = {}
        families = [("projection", "direction_head", scales, list(range(len(scales))), None),
                    ("projection_single", "direction_head", [1], [0], None),
                    ("projection_single_trained", "direction_head_single", [1], [0], None),
                    ("projection_T1", "direction_head", scales, list(range(len(scales))), 1.),
                    ("flat", "flat", scales, list(range(len(scales))), None),
                    ("prior", "prior", scales, list(range(len(scales))), None)]
        for name, head_name, ks, ids, fixed_temp in families:
            keys = ["base"]
            if head_name in ("prior", "flat"):
                probs = prior[None, :, None].expand(len(vx), -1, horizon, -1, -1).clone()
                if head_name == "flat":
                    probs.zero_(); probs[..., 1] = 1
            else:
                probs = val_prob[head_name + ("_T1" if fixed_temp else "")]
                if head_name == "direction_head": probs = probs[:, ids]
            for alpha in alphas[1:]:
                key = "%s_alpha_%g" % (name, alpha)
                path, diagnostics = project_trajectory(vx, val_paths["base"], probs, ks, eps[ids],
                    strength=alpha, return_diagnostics=True, **projection_args)
                val_paths[key] = path
                convergence[key] = diagnostics
                entries[key] = {"kind": "projection", "head": head_name, "scales": ks,
                                "indices": ids, "temperature": fixed_temp,
                                "alpha": float(alpha)}
                keys.append(key)
            family_keys[name] = keys
            choose_paths(name, keys[1:])
        choose_paths("learned_selected", ["projection", "projection_single"])
        # Same candidate library for every beta; base is first so ties do not modify it.
        library = ["base"] + family_keys["projection"][1:] + family_keys["projection_single"][1:]
        posterior_keys = []
        for beta in betas:
            key = "posterior_beta_%g" % beta
            path, index = _posterior_select(vx, val_paths["base"], [val_paths[k] for k in library],
                                            val_prob["direction_head"], scales, eps, beta)
            val_paths[key] = path
            entries[key] = {"kind": "posterior", "library": library, "beta": float(beta)}
            arrays["seed%s__validation__%s__indices" % (seed, key)] = index.cpu().numpy()
            posterior_keys.append(key)
        choose_paths("posterior_candidates", posterior_keys)
        primary = ["base", "difference", "tdalign_official", "cosdir", "residual", "residual_direction",
                   "projection", "projection_single", "learned_selected", "projection_single_trained", "projection_T1",
                   "flat", "prior", "posterior_candidates"]
        choose_paths("standalone_selected", primary[1:])
        primary.append("standalone_selected")
        # Lock choices on disk BEFORE retrieving/scoring test tensors.
        validation = {"base": base_stats, "choices": choices, "grids": val_grids,
                      "convergence": convergence, "temperature": temps,
                      "all_model_metrics": {k: _stats(vx, vy, p, scales, eps) for k, p in val_paths.items()}}
        _write(out / ("seed%s_validation_lock.json" % seed), validation)
        cp = {"schema_version": 2, "config": cfg,
              "model_config": {"backbone": cfg["backbone"], "lookback": lookback, "horizon": horizon,
                               "channels": channels, "hidden": hidden, "scales": scales},
              "models": {k: {n: v.detach().cpu().clone() for n, v in m.state_dict().items()} for k, m in models.items()},
              "model_specs": model_specs, "entries": entries, "temperature": temps,
              "epsilon": eps.cpu(), "slope_scale": scale.cpu(), "prior": prior.cpu(),
              "projection": projection_args,
              "normalization": {"mean": torch.tensor(bundle.mean), "std": torch.tensor(bundle.std)}}
        checkpoints[str(seed)] = cp
        qx, qy = tensors["test"]
        predictor = Predictor(cp, device)
        raw_model_names = [name for name, spec in model_specs.items() if spec["type"] != "head"]
        test_paths, test_probs, test_diag = predictor.all(qx, primary + raw_model_names)
        test_metrics = {}
        def test_metric(path):
            metric = evaluate_predictions(qx, qy, path, scales, eps)
            metric["original_units_mse"] = float(((path.double()-qy.double()) *
                torch.tensor(bundle.std, device=device).double()).square().mean())
            return metric
        for name in primary:
            path = test_paths[name]
            test_metrics[name] = test_metric(path)
            arrays["seed%s__test__%s" % (seed, name)] = path.cpu().numpy()
        raw_model_metrics = {}
        for name in raw_model_names:
            raw_model_metrics[name] = test_metrics[name] if name in test_metrics else test_metric(test_paths[name])
            arrays["seed%s__test_model__%s" % (seed, name)] = test_paths[name].cpu().numpy()
        test_metrics["direction_head"] = evaluate_predictions(qx, qy, test_paths["base"], scales, eps,
                                                              probabilities=test_probs["direction_head"])
        for name, p in val_paths.items(): arrays["seed%s__validation__%s" % (seed, name)] = p.cpu().numpy()
        for name, p in val_prob.items(): arrays["seed%s__validation__%s__probs" % (seed, name)] = p.cpu().numpy()
        for name, p in test_probs.items(): arrays["seed%s__test__%s__probs" % (seed, name)] = p.cpu().numpy()
        result["runs"].append({"seed": seed, "training": training_rows, "validation": validation,
                                "methods": test_metrics, "test_models": raw_model_metrics,
                                "test_convergence": test_diag})
        torch.save(checkpoints, out / "checkpoint.pt")
        reloaded = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)[str(seed)]
        reload_paths, _, _ = Predictor(reloaded, device).all(qx, primary + raw_model_names)
        for name in primary + raw_model_names:
            torch.testing.assert_close(reload_paths[name], test_paths[name], rtol=1e-5, atol=1e-6)
        result["runs"][-1]["checkpoint_reload_verified"] = True
        _write(out / "results.json", result)
        np.savez_compressed(out / "predictions.npz", **arrays)
        print("[repair-v2] seed %s complete; checkpoint reload verified" % seed, flush=True)
    result["complete"] = True
    _write(out / "results.json", result)
    return result


class Predictor:
    """Rebuild one v2 checkpoint; inference receives history only."""
    def __init__(self, cp, device="cpu"):
        self.cp, self.device = cp, torch.device(device)
        spec = cp["model_config"]
        args = (spec["lookback"], spec["horizon"], spec["channels"])
        self.models = {}
        for name, detail in cp["model_specs"].items():
            if detail["type"] == "base":
                model = build_base(spec["backbone"], *args, spec["hidden"])
            elif detail["type"] == "head":
                model = DirectionHead(*args, detail["scales"], spec["hidden"])
            else:
                model = ResidualAdapter(*args, detail["hidden"])
            model.load_state_dict(cp["models"][name])
            self.models[name] = model.to(self.device).eval()

    def all(self, history, methods):
        cp, device = self.cp, self.device
        x = torch.as_tensor(history, dtype=torch.float32, device=device)
        eps = cp["epsilon"].to(device)
        paths, probs, diagnostics = {}, {}, {}
        with torch.no_grad():
            paths["base"] = self.models["base"](x)
            for name in ("direction_head", "direction_head_single"):
                logits = self.models[name](x, paths["base"])
                probs[name] = torch.softmax(logits/cp["temperature"][name], -1)
                probs[name+"_T1"] = torch.softmax(logits, -1)
        def get(name):
            if name in paths: return paths[name]
            entry = cp["entries"][name]
            kind = entry["kind"]
            if kind == "alias": path = get(entry["target"])
            elif kind == "model":
                model = self.models[entry["model"]]
                with torch.no_grad():
                    path = model(x, paths["base"]) if cp["model_specs"][entry["model"]]["type"] == "residual" else model(x)
            elif kind == "projection":
                head = entry["head"]
                if head in ("prior", "flat"):
                    p = cp["prior"].to(device)[None, :, None].expand(len(x), -1, paths["base"].shape[1], -1, -1).clone()
                    if head == "flat": p.zero_(); p[..., 1] = 1
                else:
                    p = probs[head + ("_T1" if entry["temperature"] else "")]
                    if head == "direction_head": p = p[:, entry["indices"]]
                path, diag = project_trajectory(x, paths["base"], p, entry["scales"], eps[entry["indices"]],
                    strength=entry["alpha"], return_diagnostics=True, **cp["projection"])
                diagnostics[name] = diag
            elif kind == "posterior":
                path, index = _posterior_select(x, paths["base"], [get(k) for k in entry["library"]],
                    probs["direction_head"], cp["model_config"]["scales"], eps, entry["beta"])
                diagnostics[name] = {"chosen_candidate_indices": index.cpu().tolist()}
            else: raise ValueError("unknown entry kind " + kind)
            paths[name] = path
            return path
        for name in methods: get(name)
        return paths, probs, diagnostics
