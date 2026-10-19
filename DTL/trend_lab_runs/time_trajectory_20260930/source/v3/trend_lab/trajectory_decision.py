"""Finite-candidate direction decisions; standard DP, not a risk guarantee.

All quantities are NumPy float64 in one fixed training-normalized coordinate
system.  Classes are down/zero/up.  Each sample/channel has a common sorted grid
at every future step.  The feasible set is the grid Cartesian product together
with the *complete* unmodified base path: no per-step or per-channel stitching
of the base is allowed.  Native-grid construction accepts explicit measurement
metadata, and never estimates precision from evaluation targets.
"""
from time import perf_counter

import numpy as np


def _inputs(x, base, mean, probabilities):
    x, b, mu, p = [np.asarray(v, dtype=np.float64) for v in (x, base, mean, probabilities)]
    if x.ndim != 3 or b.ndim != 3 or mu.shape != b.shape or p.shape != b.shape + (3,):
        raise ValueError("expected x[N,L,C], base/mean[N,H,C], probabilities[N,H,C,3]")
    if min(x.shape) < 1 or min(b.shape) < 1 or x.shape[0] != b.shape[0] or x.shape[2] != b.shape[2]:
        raise ValueError("positive, matching batch/channel and sequence dimensions required")
    if not all(np.isfinite(v).all() for v in (x, b, mu, p)):
        raise ValueError("decision inputs must be finite")
    if np.any(p < 0) or not np.allclose(p.sum(-1), 1., rtol=0., atol=1e-6):
        raise ValueError("probabilities must be nonnegative and sum to one")
    return x, b, mu, p


def raw_classes(x, path):
    """Use strict float64 comparisons, with no tolerance or output rounding."""
    x, z = np.asarray(x, dtype=np.float64), np.asarray(path, dtype=np.float64)
    if x.ndim != 3 or z.ndim != 3 or len(x) != len(z) or x.shape[2] != z.shape[2]:
        raise ValueError("invalid raw class dimensions")
    previous = np.concatenate((x[:, -1:, :], z[:, :-1, :]), axis=1)
    return np.where(z < previous, 0, np.where(z > previous, 2, 1))


def _rewards(p, reward):
    if reward == "posterior":
        return p
    if reward not in ("top1", "top1_gap"):
        raise ValueError("reward must be posterior, top1, or top1_gap")
    out = np.zeros_like(p)
    top = p.argmax(-1)  # Class-order tie is deterministic: down, zero, up.
    weight = np.ones(p.shape[:-1]) if reward == "top1" else np.sort(p, axis=-1)[..., -1] - np.sort(p, axis=-1)[..., -2]
    np.put_along_axis(out, top[..., None], weight[..., None], axis=-1)
    return out


def path_objective(x, path, mean, probabilities, strength, reward="posterior"):
    """Return per-sample/channel averaged squared error, reward and objective.

    Squared error relative to mean is estimated conditional risk up to an
    additive constant, not observed MSE and not an unconditional safety bound.
    """
    x, z, mu, p = _inputs(x, path, mean, probabilities)
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    cls = raw_classes(x, z)
    rewards = _rewards(p, reward)
    utility = np.take_along_axis(rewards, cls[..., None], axis=-1)[..., 0].mean(1)
    mse = np.mean((z - mu) ** 2, axis=1)
    return {"squared_error_to_mean": mse, "utility": utility,
            "objective": mse - strength * utility}


def _prefix_suffix(cost):
    """Running minima and earliest-index argmin, including each position."""
    size = len(cost)
    index = np.arange(size, dtype=np.int64)
    prefix = np.minimum.accumulate(cost)
    record = cost < np.r_[np.inf, prefix[:-1]]
    prefix_arg = np.maximum.accumulate(np.where(record, index, 0))
    suffix = np.minimum.accumulate(cost[::-1])[::-1]
    suffix_arg = np.minimum.accumulate(np.where(cost == suffix, index, size)[::-1])[::-1]
    return prefix, prefix_arg, suffix, suffix_arg


def _grid_dp(anchor, mu, reward, grid, strength):
    horizon, size = len(mu), len(grid)
    initial_classes = np.where(grid < anchor, 0, np.where(grid > anchor, 2, 1))
    cost = (grid - mu[0]) ** 2 - strength * reward[0, initial_classes]
    back = np.empty((horizon, size), dtype=np.int64)
    back[0] = -1
    index = np.arange(size, dtype=np.int64)
    for h in range(1, horizon):
        prefix, prefix_arg, suffix, suffix_arg = _prefix_suffix(cost)
        # Row 0/down uses a strictly larger predecessor; row 2/up smaller.
        transitions = np.full((3, size), np.inf)
        predecessor = np.full((3, size), size, dtype=np.int64)
        transitions[0, :-1] = suffix[1:] - strength * reward[h, 0]
        predecessor[0, :-1] = suffix_arg[1:]
        transitions[1] = cost - strength * reward[h, 1]
        predecessor[1] = index
        transitions[2, 1:] = prefix[:-1] - strength * reward[h, 2]
        predecessor[2, 1:] = prefix_arg[:-1]
        best = transitions.min(0)
        # Resolve objective ties by earliest predecessor index, not class order.
        back[h] = np.where(transitions == best[None, :], predecessor, size).min(0)
        cost = best + (grid - mu[h]) ** 2
    position = int(cost.argmin())  # Earliest final state on exact ties.
    objective = float(cost[position] / horizon)
    path = np.empty(horizon, dtype=np.float64)
    for h in range(horizon - 1, -1, -1):
        path[h] = grid[position]
        position = int(back[h, position])
    return path, objective



def _grid_dp_batch(anchors, means, rewards, grids, strength):
    """Vectorized scalar-equivalent DP for a chunk of same-sized grids."""
    batch, horizon, _ = rewards.shape
    size = grids.shape[1]
    row = np.arange(batch, dtype=np.int64)
    index = np.broadcast_to(np.arange(size, dtype=np.int64), (batch, size))
    initial_classes = np.where(grids < anchors[:, None], 0, np.where(grids > anchors[:, None], 2, 1))
    cost = (grids - means[:, :1]) ** 2 - strength * np.take_along_axis(rewards[:, 0], initial_classes, axis=1)
    back = np.empty((horizon, batch, size), dtype=np.int64)
    back[0] = -1
    for h in range(1, horizon):
        prefix = np.minimum.accumulate(cost, axis=1)
        prior_prefix = np.concatenate((np.full((batch, 1), np.inf), prefix[:, :-1]), axis=1)
        prefix_arg = np.maximum.accumulate(np.where(cost < prior_prefix, index, 0), axis=1)
        suffix = np.minimum.accumulate(cost[:, ::-1], axis=1)[:, ::-1]
        suffix_arg = np.minimum.accumulate(np.where(cost == suffix, index, size)[:, ::-1], axis=1)[:, ::-1]
        transitions = np.full((3, batch, size), np.inf)
        predecessors = np.full((3, batch, size), size, dtype=np.int64)
        transitions[0, :, :-1] = suffix[:, 1:] - strength * rewards[:, h, 0, None]
        predecessors[0, :, :-1] = suffix_arg[:, 1:]
        transitions[1] = cost - strength * rewards[:, h, 1, None]
        predecessors[1] = index
        transitions[2, :, 1:] = prefix[:, :-1] - strength * rewards[:, h, 2, None]
        predecessors[2, :, 1:] = prefix_arg[:, :-1]
        best = transitions.min(0)
        back[h] = np.where(transitions == best[None], predecessors, size).min(0)
        cost = best + (grids - means[:, h, None]) ** 2
    positions = cost.argmin(1)
    objectives = cost[row, positions] / horizon
    paths = np.empty((batch, horizon), dtype=np.float64)
    for h in range(horizon - 1, -1, -1):
        paths[:, h] = grids[row, positions]
        positions = back[h, row, positions]
    return paths, objectives



def _grid_greedy_batch(anchors, means, rewards, grids, strength):
    """Myopic decisions using the same library and immediate-stage objective."""
    batch, horizon, _ = rewards.shape
    rows = np.arange(batch)
    previous = anchors.copy()
    paths = np.empty((batch, horizon), dtype=np.float64)
    objective = np.zeros(batch)
    for h in range(horizon):
        classes = np.where(grids < previous[:, None], 0, np.where(grids > previous[:, None], 2, 1))
        costs = (grids - means[:, h, None]) ** 2 - strength * np.take_along_axis(rewards[:, h], classes, axis=1)
        positions = costs.argmin(1)
        previous = grids[rows, positions]
        paths[:, h] = previous
        objective += costs[rows, positions]
    return paths, objective / horizon


def _coerce_grids(grids, batch, channels):
    if isinstance(grids, np.ndarray) and grids.ndim == 1:
        grids = [[grids] * channels for _ in range(batch)]
    if len(grids) != batch or any(len(row) != channels for row in grids):
        raise ValueError("grids must have shape [N][C] with one 1D shared grid per cell")
    out = []
    for row in grids:
        converted = []
        for grid in row:
            grid = np.asarray(grid, dtype=np.float64)
            if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
                raise ValueError("each grid must be nonempty, finite, strictly increasing")
            converted.append(grid)
        out.append(converted)
    return out


def _decode_with_solver(x, base, mean, probabilities, strength, grids,
                        allow_base=True, reward="posterior", solver="dp"):
    """Apply a specified independent-channel solver using O(sum H*M) work.

    The grid solver is channel separable; then the whole multi-channel base is evaluated as
    one additional candidate.  An exact objective tie retains the whole base.
    This solves a scalarized objective, not arbitrary constrained risk budgets
    or the full Pareto frontier.  State storage for traceback is O(sum H*M).
    """
    started = perf_counter()
    x, b, mu, p = _inputs(x, base, mean, probabilities)
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    n, horizon, channels = b.shape
    grids = _coerce_grids(grids, n, channels)
    r = _rewards(p, reward)
    # Batch channels with the same grid size.  Every scalar DP recurrence is
    # unchanged; only the independent sample/channel dimension is vectorized.
    flat_mean = mu.transpose(0, 2, 1).reshape(n * channels, horizon)
    flat_reward = r.transpose(0, 2, 1, 3).reshape(n * channels, horizon, 3)
    flat_anchor = x[:, -1, :].reshape(-1)
    flat_path = np.empty((n * channels, horizon), dtype=np.float64)
    flat_objective = np.empty(n * channels)
    states = np.array([[len(grid) for grid in row] for row in grids], dtype=np.int64)
    grouped = {}
    for sample in range(n):
        for channel in range(channels):
            grouped.setdefault(len(grids[sample][channel]), []).append(sample * channels + channel)
    for size, group in grouped.items():
        for start in range(0, len(group), 128):
            cells = np.asarray(group[start:start + 128], dtype=np.int64)
            stacked = np.stack([grids[int(cell) // channels][int(cell) % channels] for cell in cells])
            algorithm = _grid_dp_batch if solver == "dp" else _grid_greedy_batch
            flat_path[cells], flat_objective[cells] = algorithm(
                flat_anchor[cells], flat_mean[cells], flat_reward[cells], stacked, strength)
    decoded = flat_path.reshape(n, channels, horizon).transpose(0, 2, 1)
    dp_objective = flat_objective.reshape(n, channels)
    # Evaluate again from the final path, avoiding a base-vs-DP tie affected by
    # the different arithmetic grouping in accumulated DP costs.
    dp_score = path_objective(x, decoded, mu, p, strength, reward)
    base_score = path_objective(x, b, mu, p, strength, reward)
    selected_base = (base_score["objective"].mean(1) <= dp_score["objective"].mean(1)) if allow_base else np.zeros(n, dtype=bool)
    decoded[selected_base] = b[selected_base]
    score = path_objective(x, decoded, mu, p, strength, reward)
    diagnostic = {"strength": float(strength), "reward": reward, "solver": solver,
                  "globally_optimal_finite_grid": solver == "dp",
                  "allow_base": bool(allow_base), "selected_base": selected_base.tolist(),
                  "objective": score["objective"].mean(1).tolist(),
                  "objective_by_channel": score["objective"].tolist(),
                  "base_objective": base_score["objective"].mean(1).tolist(),
                  "grid_objective": dp_score["objective"].mean(1).tolist(),
                  "estimated_risk_delta": (score["squared_error_to_mean"] - base_score["squared_error_to_mean"]).mean(1).tolist(),
                  "estimated_utility_delta": (score["utility"] - base_score["utility"]).mean(1).tolist(),
                  "max_dp_objective_recompute_error": float(np.max(np.abs(dp_objective - dp_score["objective"]))),
                  "grid_state_counts": states.tolist(), "state_updates": int(states.sum() * horizon),
                  "elapsed_seconds": perf_counter() - started}
    return decoded, diagnostic



def decode_trajectory(x, base, mean, probabilities, strength, grids,
                      allow_base=True, reward="posterior"):
    """Finite-grid global DP optimum, compared with the complete base path.

    Runtime is O(sum H*M), with chunked O(H*M) traceback state.  It solves the
    scalarized objective only; it does not guarantee a chosen risk budget.
    """
    return _decode_with_solver(x, base, mean, probabilities, strength, grids,
                               allow_base, reward, solver="dp")


def greedy_trajectory(x, base, mean, probabilities, strength, grids,
                      allow_base=True, reward="posterior"):
    """Immediate-stage greedy control on the identical finite candidate grid.

    The whole resulting greedy path is compared with the whole base using the
    same complete-path objective.  This control does not optimize G^H globally.
    """
    return _decode_with_solver(x, base, mean, probabilities, strength, grids,
                               allow_base, reward, solver="greedy")


def _channels(value, channels, name, positive=False):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(channels, float(array))
    if array.shape != (channels,) or not np.isfinite(array).all() or (positive and np.any(array <= 0)):
        raise ValueError("invalid " + name)
    return array


def make_common_grids(x, base, mean, mode="continuous", step=.1, padding=.5,
                      origin=0., normalizer_mean=0., normalizer_std=1.,
                      native_step=1., native_origin=0., lower=None, upper=None,
                      include_anchor=None, max_states=257, coarsen=True):
    """Build grids from current observations/predictions and fixed metadata.

    Continuous mode: step/origin/padding and optional bounds are in normalized
    coordinates; the exact current anchor is included by default so first-step
    zero is reachable.  Native mode: native_step/native_origin are in physical
    units, then transformed by the supplied train-only normalizer; anchor is
    excluded by default to preserve this lattice.  For exact first-step zero,
    pass x normalized in float64 from the raw measurement using this very
    normalizer.  Adding an off-lattice anchor is an explicit relaxed-support
    option, reported by the returned metadata.  With coarsen=True a fixed state
    budget deterministically subsamples the covering lattice using observable
    span, keeping endpoints and a reachable historical anchor.  Native values
    stay on the declared lattice; its sparse subset is explicitly reported and
    is not the full native support.  coarsen=False raises on excessive states.
    """
    x, b, mu = [np.asarray(v, dtype=np.float64) for v in (x, base, mean)]
    if x.ndim != 3 or b.ndim != 3 or b.shape != mu.shape or len(x) != len(b) or x.shape[2] != b.shape[2]:
        raise ValueError("invalid grid input dimensions")
    if min(x.shape) < 1 or min(b.shape) < 1 or not all(np.isfinite(v).all() for v in (x, b, mu)):
        raise ValueError("finite nonempty grid inputs required")
    n, _, channels = b.shape
    if mode not in ("continuous", "native") or not isinstance(max_states, (int, np.integer)) or max_states < 1:
        raise ValueError("invalid grid mode or max_states")
    pad = _channels(padding, channels, "padding")
    if np.any(pad < 0):
        raise ValueError("padding cannot be negative")
    if mode == "continuous":
        delta = _channels(step, channels, "step", positive=True)
        normalized_origin = _channels(origin, channels, "origin")
        raw_origin = raw_step = norm_mean = norm_std = None
    else:
        raw_step = _channels(native_step, channels, "native_step", positive=True)
        raw_origin = _channels(native_origin, channels, "native_origin")
        norm_mean = _channels(normalizer_mean, channels, "normalizer_mean")
        norm_std = _channels(normalizer_std, channels, "normalizer_std", positive=True)
        delta = raw_step / norm_std
        normalized_origin = (raw_origin - norm_mean) / norm_std
    include_anchor = (mode == "continuous") if include_anchor is None else bool(include_anchor)
    low = None if lower is None else _channels(lower, channels, "lower")
    high = None if upper is None else _channels(upper, channels, "upper")
    if low is not None and high is not None and np.any(low > high):
        raise ValueError("lower bound exceeds upper bound")
    grids, sizes, anchors, relaxed, full_sizes, sparse = [], [], [], [], [], []
    for sample in range(n):
        row, size_row, anchor_row, relaxed_row, full_row, sparse_row = [], [], [], [], [], []
        for channel in range(channels):
            anchor = x[sample, -1, channel]
            visible = np.r_[anchor, b[sample, :, channel], mu[sample, :, channel]]
            lo, hi = visible.min() - pad[channel], visible.max() + pad[channel]
            if low is not None:
                lo = max(lo, low[channel])
            if high is not None:
                hi = min(hi, high[channel])
            if lo > hi:
                raise ValueError("observable range has no overlap with declared bounds")
            first = int(np.floor((lo - normalized_origin[channel]) / delta[channel]))
            last = int(np.ceil((hi - normalized_origin[channel]) / delta[channel]))
            # Fixed-budget subset is chosen before values are materialized, so
            # a tiny native step cannot create an unbounded temporary array.
            full_size = last - first + 1
            anchor_index = int(np.rint((anchor - normalized_origin[channel]) / delta[channel]))
            if mode == "native":
                lattice_anchor = (raw_origin[channel] + anchor_index * raw_step[channel] - norm_mean[channel]) / norm_std[channel]
            else:
                lattice_anchor = normalized_origin[channel] + anchor_index * delta[channel]
            keep_anchor_index = bool(first <= anchor_index <= last and lattice_anchor == anchor)
            reserve = int(include_anchor or keep_anchor_index)
            is_sparse = full_size + (int(include_anchor) if not keep_anchor_index else 0) > max_states
            if is_sparse:
                if not coarsen:
                    raise ValueError("grid exceeds max_states and coarsen=False")
                slots = max_states - reserve
                if slots < 2:
                    raise ValueError("state budget cannot retain both endpoints and anchor")
                # Integer arithmetic defines an evenly spaced subset of the
                # requested lattice; the first and last points are retained.
                indices = np.array([first + (i * (last - first)) // (slots - 1)
                                    for i in range(slots)], dtype=np.float64)
                if keep_anchor_index:
                    indices = np.unique(np.r_[indices, anchor_index])
            else:
                indices = np.arange(first, last + 1, dtype=np.float64)
            if mode == "native":
                grid = (raw_origin[channel] + indices * raw_step[channel] - norm_mean[channel]) / norm_std[channel]
            else:
                grid = normalized_origin[channel] + indices * delta[channel]
            if low is not None:
                grid = grid[grid >= low[channel]]
            if high is not None:
                grid = grid[grid <= high[channel]]
            anchor_in_bounds = (low is None or anchor >= low[channel]) and (high is None or anchor <= high[channel])
            anchor_present_before = bool(np.any(grid == anchor))
            if include_anchor and anchor_in_bounds:
                grid = np.unique(np.r_[grid, anchor])
            if not len(grid) or len(grid) > max_states:
                raise ValueError("no supported grid points or too many anchor-augmented states")
            row.append(grid)
            size_row.append(len(grid))
            anchor_row.append(bool(np.any(grid == anchor)))
            relaxed_row.append(bool(mode == "native" and include_anchor and anchor_in_bounds and not anchor_present_before))
            full_row.append(full_size)
            sparse_row.append(bool(is_sparse))
        grids.append(row)
        sizes.append(size_row)
        anchors.append(anchor_row)
        relaxed.append(relaxed_row)
        full_sizes.append(full_row)
        sparse.append(sparse_row)
    return grids, {"mode": mode, "normalized_step": delta.tolist(),
                   "normalized_origin": normalized_origin.tolist(),
                   "padding": pad.tolist(), "include_anchor": include_anchor,
                   "grid_state_counts": sizes, "anchor_reachable": anchors,
                   "native_off_lattice_anchor_added": relaxed,
                   "max_states": int(max_states), "coarsen": bool(coarsen),
                   "full_covering_lattice_state_counts": full_sizes, "coarsened": sparse,
                   "sparse_native_lattice": sparse if mode == "native" else [[False] * channels for _ in range(n)],
                   "uses_only_current_observables_and_fixed_metadata": True}
