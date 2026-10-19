"""JSON-safe forecasting, direction, probability, and event metrics."""

import numpy as np
import torch

from .operators import multiscale_slopes, turn_targets


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _mean_or_none(values):
    return float(np.mean(values)) if np.size(values) else None


def _classification(y_true, y_pred):
    true, pred = y_true.reshape(-1), y_pred.reshape(-1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    support = matrix.sum(axis=1)
    tp = np.diag(matrix)
    precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(3, dtype=float), where=matrix.sum(axis=0) != 0)
    recall = np.divide(tp, support, out=np.zeros(3, dtype=float), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3, dtype=float), where=precision + recall != 0)
    present = support > 0
    return {
        "accuracy": float(np.mean(true == pred)) if len(true) else None,
        "macro_f1": float(f1.mean()) if len(true) else None,
        "balanced_accuracy": float(recall[present].mean()) if present.any() else None,
        "confusion_matrix": matrix.tolist(),
        "nonflat_accuracy": float(np.mean(pred[true != 1] == true[true != 1])) if np.any(true != 1) else None,
        "nonflat_coverage": float(np.mean(true != 1)) if len(true) else None,
        "down_accuracy": float(np.mean(pred[true == 0] == 0)) if np.any(true == 0) else None,
        "up_accuracy": float(np.mean(pred[true == 2] == 2)) if np.any(true == 2) else None,
    }


def _events(history, future, epsilon):
    """Extract event indices using the training-label flat-hold convention."""
    turns = turn_targets(torch.as_tensor(history), torch.as_tensor(future), torch.as_tensor(epsilon))
    locations = torch.nonzero(turns, as_tuple=False).cpu().numpy()
    return [(int(n), int(c), int(h)) for n, h, c in locations]


def _event_scores(history, truth, pred, epsilon):
    eps = np.asarray(epsilon, dtype=float)
    if eps.ndim == 2:
        eps = eps[0]
    true_events = _events(history, truth, eps)
    pred_events = _events(history, pred, eps)
    by_key_true, by_key_pred = {}, {}
    for n, c, t in true_events:
        by_key_true.setdefault((n, c), []).append(t)
    for n, c, t in pred_events:
        by_key_pred.setdefault((n, c), []).append(t)
    matched, errors = 0, []
    for key in set(by_key_true) | set(by_key_pred):
        ts, ps = sorted(by_key_true.get(key, [])), sorted(by_key_pred.get(key, []))
        i = j = 0
        while i < len(ts) and j < len(ps):
            if abs(ts[i] - ps[j]) <= 1:
                matched += 1
                errors.append(abs(ts[i] - ps[j]))
                i += 1
                j += 1
            elif ts[i] < ps[j]:
                i += 1
            else:
                j += 1
    if not true_events and not pred_events:
        precision = recall = f1 = None
    else:
        precision = matched / len(pred_events) if pred_events else None
        recall = matched / len(true_events) if true_events else None
        f1 = 2 * matched / (len(true_events) + len(pred_events))
    return {"precision": precision, "recall": recall, "f1": f1, "mean_absolute_time_error": _mean_or_none(errors),
            "true_count": int(len(true_events)), "pred_count": int(len(pred_events))}


def _probability_scores(probabilities, labels):
    p = _array(probabilities)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite and lie in [0,1]")
    totals = p.sum(axis=-1, keepdims=True)
    if not np.allclose(totals, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("class probabilities must sum to 1")
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=-1, keepdims=True)
    truth = labels.reshape(-1)
    flat = p.reshape(-1, 3)
    pred = flat.argmax(axis=-1)
    onehot = np.eye(3)[truth]
    confidence = flat.max(axis=-1)
    correct = pred == truth
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        mask = (confidence >= lo) & (confidence < lo + 0.1 if lo < 0.9 else confidence <= 1)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {"accuracy": float(correct.mean()), "macro_f1": _classification(truth, pred)["macro_f1"],
            "brier": float(np.mean(np.sum((flat - onehot) ** 2, axis=-1))),
            "ece": float(ece), "nll": float(-np.log(flat[np.arange(len(truth)), truth]).mean())}


def evaluate_predictions(history, truth, pred, scales, epsilon, probabilities=None):
    x, y, z = _array(history), _array(truth), _array(pred)
    if x.ndim != 3 or y.shape != z.shape or y.ndim != 3 or x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError("history, truth, and pred must have compatible [N,T,C] shapes")
    if not all(np.isfinite(a).all() for a in (x, y, z)):
        raise ValueError("history, truth, and pred must be finite")
    if len(y) == 0 or y.shape[1] == 0:
        raise ValueError("metrics require at least one forecast window and horizon step")
    err = z - y
    result = {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err))),
              "mse_by_horizon": np.mean(err ** 2, axis=(0, 2)).tolist(),
              "mae_by_horizon": np.mean(np.abs(err), axis=(0, 2)).tolist(), "scales": {}}
    eps = _array(epsilon)
    if not np.isfinite(eps).all() or np.any(eps < 0):
        raise ValueError("epsilon must be finite and nonnegative")
    true_slopes = multiscale_slopes(torch.as_tensor(x), torch.as_tensor(y), scales)
    pred_slopes = multiscale_slopes(torch.as_tensor(x), torch.as_tensor(z), scales)
    # Convert torch operator results back to NumPy for JSON-native metrics.
    true_slopes, pred_slopes = _array(true_slopes), _array(pred_slopes)
    if eps.shape != (len(scales), x.shape[2]):
        raise ValueError("epsilon must have shape [len(scales), channels]")
    def labels(s, e):
        return np.where(s > e[None, :, None, :], 2, np.where(s < -e[None, :, None, :], 0, 1))
    true_labels, pred_labels = labels(true_slopes, eps), labels(pred_slopes, eps)
    for i, scale in enumerate(scales):
        result["scales"][str(scale)] = _classification(true_labels[:, i], pred_labels[:, i])
    raw_true = np.sign(np.diff(np.concatenate((x[:, -1:, :], y), axis=1), axis=1))
    raw_pred = np.sign(np.diff(np.concatenate((x[:, -1:, :], z), axis=1), axis=1))
    result["raw_sign_accuracy"] = float(np.mean(raw_true == raw_pred)) if raw_true.size else None
    result["events"] = _event_scores(x, y, z, eps)
    event_points = np.zeros((len(y), y.shape[1], y.shape[2]), dtype=bool)
    for n, c, h in _events(x, y, eps[0]):
        event_points[n, max(0, h - 1):min(y.shape[1], h + 2), c] = True
    errors = pred_labels[:, 0] != true_labels[:, 0]
    result["regimes"] = {
        "turn_neighborhood_direction_error_rate": float(errors[event_points].mean()) if event_points.any() else None,
        "turn_neighborhood_count": int(event_points.sum()),
        "non_turn_direction_error_rate": float(errors[~event_points].mean()) if (~event_points).any() else None,
        "non_turn_count": int((~event_points).sum()),
    }
    if probabilities is not None:
        probs = _array(probabilities)
        expected = (*true_labels.shape, 3)
        if probs.shape != expected:
            raise ValueError(f"probabilities must have shape {expected}")
        flat_labels = true_labels.reshape(-1)
        per_scale = {}
        for i, scale in enumerate(scales):
            per_scale[str(scale)] = _probability_scores(probs[:, i], true_labels[:, i])
            per_scale[str(scale)]["head_path_conflict_rate"] = float(np.mean(probs[:, i].argmax(axis=-1) != pred_labels[:, i]))
        result["probability"] = _probability_scores(probs, flat_labels)
        result["probability"]["by_scale"] = per_scale
        result["probability"]["head_path_conflict_rate"] = float(np.mean(probs.argmax(axis=-1) != pred_labels))
    return result


def paired_block_interval(differences, block_size=8, repeats=500, seed=0):
    values = _array(differences).reshape(-1)
    if block_size < 1 or repeats < 1:
        raise ValueError("block_size and repeats must be positive")
    if not values.size:
        return {"mean": None, "ci95": [None, None]}
    if not np.isfinite(values).all():
        raise ValueError("differences must be finite")
    rng = np.random.default_rng(seed)
    block_size = min(int(block_size), len(values))
    blocks_needed = int(np.ceil(len(values) / block_size))
    means = np.empty(repeats, dtype=float)
    starts = np.arange(len(values) - block_size + 1)
    for i in range(repeats):
        indices = np.concatenate([np.arange(start, start + block_size) for start in rng.choice(starts, blocks_needed)])[:len(values)]
        means[i] = values[indices].mean()
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist()}
