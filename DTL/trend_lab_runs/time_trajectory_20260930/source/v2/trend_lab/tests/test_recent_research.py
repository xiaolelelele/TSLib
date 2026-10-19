"""Numerical and temporal safeguards for the recent-data research protocol."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from DTL.trend_lab.recent_methods import (CaReTS, ReferenceITransformer, raw_project,
                                        raw_objective_gradient, fit_gate, predict_gate,
                                        paired_targets, gate_features)
from DTL.trend_lab.recent_data import load_recent
from DTL.trend_lab.recent_run import select_paths


class RecentResearchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(91)

    def test_raw_projection_gradient_against_autograd(self):
        x = torch.randn(3, 5, 2, dtype=torch.float64)
        b = torch.randn(3, 4, 2, dtype=torch.float64)
        z = torch.randn(3, 4, 2, dtype=torch.float64, requires_grad=True)
        p = torch.randn(3, 4, 2, 3, dtype=torch.float64).softmax(-1)
        margin = torch.tensor([.2, .3], dtype=torch.float64)
        top = p.topk(2, -1)
        classes = top.indices[..., 0]
        weights = top.values[..., 0] - top.values[..., 1]
        d = torch.diff(torch.cat((x[:, -1:], z), 1), dim=1)
        penalty = torch.where(classes == 0, torch.relu(d + margin).square(),
                              torch.where(classes == 2, torch.relu(margin - d).square(), d.square()))
        objective = .5 * (z-b).square().sum() + .65 * (weights * penalty).sum()
        expected, = torch.autograd.grad(objective, z)
        value, gradient = raw_objective_gradient(x, b, z.detach(), p, margin, 1.3)
        torch.testing.assert_close(gradient, expected)
        torch.testing.assert_close(value.sum(), objective)

    def test_projection_batch_invariance_cap_and_optimality(self):
        x, b = torch.randn(4, 10, 2), torch.randn(4, 6, 2)
        p = torch.randn(4, 6, 2, 3).softmax(-1)
        margin = torch.tensor([.2, .3])
        z, diagnostic = raw_project(x, b, p, margin, 10., max_rms=.15)
        self.assertTrue(diagnostic["all_converged"])
        self.assertLessEqual(max(diagnostic["rms"]), .150002)
        before, _ = raw_objective_gradient(x, b, b, p, margin, 10.)
        after, _ = raw_objective_gradient(x, b, z, p, margin, 10.)
        self.assertTrue(bool((after <= before + 1e-6).all()))
        for i in range(4):
            alone, one = raw_project(x[i:i+1], b[i:i+1], p[i:i+1], margin, 10., max_rms=.15)
            torch.testing.assert_close(z[i:i+1], alone, rtol=0, atol=0)
        unchanged, _ = raw_project(x, b, p, margin, 0)
        torch.testing.assert_close(unchanged, b)

    def test_carets3_mask_and_anchor_definition(self):
        model = CaReTS(5, 3, 2, 3)
        x = torch.randn(4, 5, 2)
        delta = torch.tensor([[[1., -2.], [0., 3.], [-1., .5]]]).expand(4, -1, -1)
        y = x[:, -1:] + delta
        path, logits, deviation = model.components(x)
        p = logits.softmax(-1)
        torch.testing.assert_close(path, x[:, -1:] + p[..., 0]*deviation[..., 0] - p[..., 1]*deviation[..., 1])
        losses = model.task_losses(x, y)
        labels = ((y - x[:, -1:]) < 0).long()
        selected = torch.where(labels == 0, deviation[..., 0], deviation[..., 1])
        torch.testing.assert_close(losses[1], ((selected - (y-x[:, -1:]).abs())**2).mean())
        model.loss(x, y).backward()
        self.assertIsNotNone(model.log_variances.grad)
        self.assertGreater(float(model.classifier[-1].weight.grad.abs().sum()), 0)

    def test_carets4_probability_conditioning_gradients(self):
        model = CaReTS(5, 3, 2, 4)
        x, y = torch.randn(4, 5, 2), torch.randn(4, 3, 2)
        self.assertEqual(model.task_losses(x, y).shape, (2,))
        # Regression alone must backpropagate into the probability branch.
        ((model(x)-y)**2).mean().backward()
        self.assertGreater(float(model.classifier[-1].weight.grad.abs().sum()), 0)

    def test_ridge_and_paired_targets_independent_identity(self):
        rng = np.random.default_rng(93)
        features = rng.normal(size=(200, 17))
        targets = rng.normal(size=(200, 2))
        spec = fit_gate(features, targets)
        design = np.column_stack((np.ones(200), (features-features.mean(0))/features.std(0)))
        augmented = np.vstack((design, np.diag(np.r_[0., np.repeat(np.sqrt(10.), 17)])))
        response = np.vstack((targets, np.zeros((18, 2))))
        expected = np.linalg.lstsq(augmented, response, rcond=None)[0]
        np.testing.assert_allclose(spec["coef"], expected, atol=1e-12)
        np.testing.assert_allclose(predict_gate(spec, features), design @ expected, atol=1e-12)
        x = np.zeros((2, 2, 1))
        base, z = np.zeros((2, 1, 1)), np.ones((2, 1, 1))*.1
        y = np.array([.2, .8]).reshape(2, 1, 1)
        paired = paired_targets(x, y, base, z)
        np.testing.assert_allclose(paired[:, 1], [.01-.04, .01-.16])
        p = np.tile([.2, 0, .8], (2, 1, 1, 1))
        # Features do not accept y, so the supervised risk target cannot leak.
        self.assertEqual(gate_features(x, base, z, p).shape, (2, 17))

    def test_dates_missing_history_and_train_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"data.csv"
            times = pd.date_range("2023-01-01", "2026-01-01", freq="h", inclusive="left", tz="UTC")
            value = np.sin(np.arange(len(times)) / 6)
            value[8800] = np.nan
            frame = pd.DataFrame({"date": times, "a": value, "b": value*3+4})
            frame.to_csv(path, index=False)
            splits, meta = load_recent(path, 2024)
            np.testing.assert_allclose(meta["mean"], np.nanmean(frame[["a", "b"]].to_numpy()[:8760], 0))
            for name, split in splits.items():
                start, end = meta["splits"][name]["target_bounds"]
                self.assertTrue((split.origins+1 >= start).all())
                self.assertTrue((split.origins+25 <= end).all())
                self.assertTrue(np.isfinite(split.x).all() and np.isfinite(split.y).all())
                self.assertTrue(set(split.origins % 24).issubset({0, 6, 12, 18}))
                self.assertFalse(((split.origins-95 <= 8800) & (split.origins+24 >= 8800)).any())
            old_mean = meta["mean"]
            frame.loc[frame.index >= 8760, ["a", "b"]] *= 100
            frame.to_csv(path, index=False)
            _, altered = load_recent(path, 2024)
            self.assertEqual(old_mean, altered["mean"])

    def test_gate_and_alpha_selection_reject_risk(self):
        x = np.zeros((3, 2, 1))
        y = np.repeat([[[.1], [.2]]], 3, 0)
        base = np.zeros_like(y)
        wrong = np.ones_like(y)*3
        good = y.copy()
        selected, lock = select_paths(x, y, {"risky": wrong, "good": good}, base)
        self.assertEqual(selected, "good")
        selected, _ = select_paths(x, y, {"risky": wrong}, base)
        self.assertIsNone(selected)

    def test_reference_itransformer_exact_evaluation(self):
        root = Path(__file__).resolve().parents[3]/"TSLib/Time-Series-Library"
        wrapped = ReferenceITransformer(96, 24, 8, root).eval()
        x = torch.randn(3, 96, 8)
        with torch.no_grad():
            torch.testing.assert_close(wrapped(x), wrapped.reference.forecast(x, None, None, None), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
