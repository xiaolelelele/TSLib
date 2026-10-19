"""Fixed post-hoc native-precision controls; never train or choose candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

METHODS = ("mean_raw", "residual_capacity", "residual_direction_capacity", "mse_base", "cosdir", "tdalign")
DOMAINS = ("carpark", "finland")


def classes(history, path):
    previous = np.concatenate((history[:, -1:, :], path[:, :-1, :]), axis=1)
    return np.where(path < previous, 0, np.where(path > previous, 2, 1))


def metrics(history, truth, path):
    true, pred = classes(history, truth), classes(history, path)
    confusion = np.bincount((3*true+pred).ravel(), minlength=9).reshape(3, 3)
    correct, true_count, pred_count = np.diag(confusion), confusion.sum(1), confusion.sum(0)
    f1 = np.divide(2*correct, true_count+pred_count, out=np.zeros(3), where=(true_count+pred_count)>0)
    return {"mse": float(np.mean((path-truth)**2)), "mae": float(np.mean(np.abs(path-truth))),
            "raw_accuracy": float(np.mean(true == pred)), "macro_f1": float(f1.mean()),
            "confusion": confusion.tolist(),
            "precision": [float(correct[i]/pred_count[i]) if pred_count[i] else None for i in range(3)],
            "recall": [float(correct[i]/true_count[i]) if true_count[i] else None for i in range(3)],
            "true_zero_fraction": float(np.mean(true == 1)), "predicted_zero_fraction": float(np.mean(pred == 1)),
            "predicted_change_rate": float(np.mean(pred != 1))}


def native_round(path, mean, std):
    physical = np.asarray(path, dtype=np.float64)*std+mean
    return (np.maximum(np.rint(physical), 0)-mean)/std


def run_case(source, destination, frozen):
    from trend_lab.trajectory_decision import make_common_grids, decode_trajectory
    lock_path, prediction_path = source / "validation_lock.json", source / "predictions.npz"
    lock = json.loads(lock_path.read_text())
    meta = lock["metadata"]
    if meta["spec"]["grid_mode"] != "native" or not meta["native_integer_training"]:
        raise ValueError("control requires predeclared native integer training support")
    mean, std = np.array(meta["mean"]), np.array(meta["std"])
    stored, report = {}, {}
    with np.load(prediction_path) as archive:
        for split in ("validation", "test"):
            x, y, origins = archive[split+"__x"], archive[split+"__y"], archive[split+"__origins"]
            stored.update({split+"__x": x, split+"__y": y, split+"__origins": origins})
            paths = {}
            for method in METHODS:
                paths[method+"_native_round"] = native_round(archive[split+"__"+method], mean, std)
            # Additional point-estimate projection on exactly the primary grid.
            b, mu, p = archive[split+"__cosdir"], archive[split+"__mean"], archive[split+"__probabilities"]
            grids, grid_info = make_common_grids(x, b, mu, mode="native", step=.1, padding=.25,
                normalizer_mean=mean, normalizer_std=std, native_step=1., native_origin=0.,
                lower=-mean/std, max_states=257, coarsen=True)
            projected_details = {}
            for method in ("residual_capacity", "residual_direction_capacity"):
                key = method+"_shared_grid"
                paths[key], diagnostic = decode_trajectory(x, b, archive[split+"__"+method], p, 0., grids)
                projected_details[key] = {"strength": 0., "selected_base_samples": int(np.sum(diagnostic["selected_base"])),
                                          "p_affects_objective": False, "complete_base_alternative": True,
                                          "max_objective_recompute_error": diagnostic["max_dp_objective_recompute_error"]}
            # Confirm the shared-grid construction matches the primary control.
            replay, _ = decode_trajectory(x, b, mu, p, 0., grids)
            if not np.array_equal(replay, archive[split+"__mean_only_grid"]):
                raise ValueError("shared grid does not reproduce primary mean_only_grid")
            rows = {}
            for method, path in paths.items():
                stored[split+"__"+method] = path
                rows[method] = {**metrics(x, y, path),
                               "per_channel": {name: metrics(x[:, :, i:i+1], y[:, :, i:i+1], path[:, :, i:i+1])
                                               for i, name in enumerate(meta["columns"])}}
            report[split] = {"windows": len(x), "metrics": rows, "shared_grid": grid_info,
                             "projection_diagnostics": projected_details,
                             "reference_metrics": {method: metrics(x, y, archive[split+"__"+method]) for method in METHODS},
                             "primary_posterior_selected_metrics": metrics(x, y, archive[split+"__posterior_selected"]) if split == "test" else None}
    destination.mkdir(parents=True, exist_ok=True)
    # Write new analysis outputs, never the source case or its validation lock.
    np.savez_compressed(destination / "predictions.npz", **stored)
    result = {"posthoc_diagnostic": True, "no_training": True, "no_selection": True,
              "source_case": str(source.resolve()), "metadata": meta,
              "source_predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
              "source_validation_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
              "source_primary_result_sha256": hashlib.sha256((source/"result.json").read_bytes()).hexdigest(),
              "controls_predictions_sha256": hashlib.sha256((destination/"predictions.npz").read_bytes()).hexdigest(),
              "rule": "physical q1 r0 np.rint ties-to-even then nonnegative then same train normalization",
              "grid_rule": "primary x/b/mu common lattice; residual output as point center, lambda0, complete b fallback; p unused in objective",
              "source_decoder_sha256": hashlib.sha256((frozen/"trend_lab/trajectory_decision.py").read_bytes()).hexdigest(),
              "weak_surrogate_note": "eps0 [-d,0,d]/scale has flat probability <=1/3; not an arbitrary three-class posterior baseline",
              "stages": report}
    (destination / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-source", required=True)
    parser.add_argument("--watch-seconds", type=int, default=0)
    args = parser.parse_args()
    source, output, frozen = Path(args.results), Path(args.output), Path(args.frozen_source)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(frozen))
    started = time.monotonic()
    completed = []
    while True:
        for domain in DOMAINS:
            for primary_result in sorted((source/domain).glob("*_seed*/result.json")):
                if primary_result.parent.name.startswith("shared_"):
                    continue
                case = primary_result.parent
                target = output/domain/case.name
                if (target/"result.json").exists():
                    if str(target) not in completed:
                        completed.append(str(target))
                    continue
                try:
                    # Only finalized primary cases are eligible.
                    json.loads(primary_result.read_text())
                    run_case(case, target, frozen)
                except json.JSONDecodeError:
                    continue
                completed.append(str(target))
                print("CONTROL_DONE", domain, case.name, len(completed), flush=True)
                (output/"progress.json").write_text(json.dumps({"cases": len(completed), "completed": completed,
                                                                "posthoc": True, "no_selection": True}, indent=2))
        if len(completed) == 12:
            (output/"completion.json").write_text(json.dumps({"cases": 12, "completed": completed,
                                                              "seconds": time.monotonic()-started,
                                                              "posthoc": True, "no_selection": True}, indent=2))
            print("COMPLETE", len(completed), flush=True)
            return
        if time.monotonic()-started >= args.watch_seconds:
            print("PARTIAL", len(completed), flush=True)
            return
        time.sleep(min(30, args.watch_seconds-int(time.monotonic()-started)))


if __name__ == "__main__":
    main()
