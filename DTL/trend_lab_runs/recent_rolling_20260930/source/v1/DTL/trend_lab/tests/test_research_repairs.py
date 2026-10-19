"""Regression checks for the independently audited prototype defects."""

import unittest

import numpy as np
import torch

from trend_lab.data import load_data, split_earlystop_calibration
from trend_lab.losses import (direction_surrogate_logits, direction_surrogate_loss,
                              fit_slope_scales, residual_direction_diagnostics)
from trend_lab.metrics import evaluate_predictions
from trend_lab.operators import (direction_labels, fit_thresholds,
                                 multiscale_direction_labels, multiscale_slopes)


class DirectionControlRepairTests(unittest.TestCase):
    def test_dead_zone_and_loss_scale_are_separate_and_robust(self):
        x = torch.zeros(3, 2, 2)
        y = torch.tensor([[[0.0, 0.1]], [[0.0, 0.2]], [[0.0, 100.0]]])
        scale = fit_slope_scales(x, y, [1])
        torch.testing.assert_close(scale, torch.tensor([[0.02, 0.2]], dtype=torch.float64))
        epsilon = fit_thresholds(x, y, [1], quantile=0.0, precision="float64")
        self.assertEqual(epsilon[0, 0].item(), 1e-6)
        self.assertEqual(scale[0, 0].item(), 0.02)
        with self.assertRaises(ValueError):
            fit_slope_scales(x, y, [1], floor=0)

    def test_surrogate_boundary_matches_evaluation_with_flat_ties(self):
        slopes = torch.tensor([[[[-1.001], [-1.0], [-0.75], [0.0], [0.75], [1.0], [1.001]]]],
                              dtype=torch.float64)
        epsilon = torch.tensor([[1.0]], dtype=torch.float64)
        scale = torch.tensor([[0.2]], dtype=torch.float64)
        labels = direction_labels(slopes, epsilon)
        self.assertEqual(labels.flatten().tolist(), [0, 1, 1, 1, 1, 1, 2])
        logits = direction_surrogate_logits(slopes, epsilon, scale)
        interior = [0, 2, 3, 4, 6]
        torch.testing.assert_close(logits.argmax(-1)[:, :, interior], labels[:, :, interior])
        # The negative boundary ties down/flat; use the shared label decoder.
        self.assertEqual(logits[0, 0, 1, 0, 0], logits[0, 0, 1, 0, 1])
        self.assertEqual(logits[0, 0, 5, 0, 1], logits[0, 0, 5, 0, 2])

    def test_tiny_epsilon_does_not_amplify_loss_or_slope_gradients(self):
        slopes = torch.tensor([[[[-1.0], [1.0], [0.001]]]], dtype=torch.float64,
                              requires_grad=True)
        labels = torch.tensor([[[[2], [0], [1]]]], dtype=torch.long)
        epsilon = torch.tensor([[1e-6]], dtype=torch.float64)
        scale = torch.tensor([[0.02]], dtype=torch.float64)
        losses = direction_surrogate_loss(slopes, labels, epsilon, scale, reduction="none")
        losses.sum().backward()
        self.assertTrue(torch.isfinite(losses).all())
        self.assertLess(losses.max().item(), 101.0)
        self.assertLessEqual(slopes.grad.abs().max().item(), 100.0 + 1e-9)
        # Sensitivity is bounded by 2/scale, not 2/epsilon=2,000,000.
        self.assertGreater(slopes.grad.abs().max().item(), 1.0)

    def test_gradient_diagnostics_do_not_modify_inputs_or_model_gradients(self):
        x = torch.tensor([[[0.0], [0.0]]], requires_grad=True)
        y = torch.tensor([[[0.1], [0.1]]])
        path = torch.tensor([[[-0.1], [-0.2]]], requires_grad=True)
        diagnostics = residual_direction_diagnostics(x, y, path, [1], [[1e-6]], [[0.02]])
        self.assertGreater(diagnostics["mse"], 0)
        self.assertGreater(diagnostics["weighted_direction_loss"], 0)
        self.assertTrue(np.isfinite(list(diagnostics.values())).all())
        self.assertLess(diagnostics["gradient_norm_ratio"], 100)
        self.assertIsNone(path.grad)
        self.assertIsNone(x.grad)
        exact = residual_direction_diagnostics(x, y, y, [1], [[1e-6]], [[0.02]])
        self.assertIsNone(exact["loss_ratio"])
        self.assertIsNone(exact["gradient_norm_ratio"])

    def test_shared_labels_subtract_endpoints_in_float64(self):
        x = torch.tensor([[[0.0], [-1.0]]], dtype=torch.float32)
        y = torch.tensor([[[1e-8]]], dtype=torch.float32)
        epsilon = torch.tensor([[1.0]], dtype=torch.float64)
        legacy = direction_labels(multiscale_slopes(x, y, [1]), epsilon)
        shared = multiscale_direction_labels(x, y, [1], epsilon)
        self.assertEqual(legacy.item(), 1)
        self.assertEqual(shared.item(), 2)
        torch.testing.assert_close(shared, multiscale_direction_labels(x.double(), y.double(), [1], epsilon))
        metric = evaluate_predictions(x, y, y, [1], epsilon)
        self.assertEqual(metric["scales"]["1"]["confusion_matrix"][2][2], 1)
        # Preserve the old default computation for frozen-config reproduction.
        default = fit_thresholds(x, y, [1])
        explicit_legacy = fit_thresholds(x, y, [1], precision="legacy")
        torch.testing.assert_close(default, explicit_legacy, rtol=0, atol=0)
        precise = fit_thresholds(x, y, [1], precision="float64")
        self.assertEqual(precise.dtype, torch.float64)
        self.assertGreater(precise.item(), default.item())

    def test_mixed_labels_do_not_imply_a_consistent_direction_decision(self):
        # Even with aligned pointwise boundaries, this CE family can predict a
        # flat path when up is the modal class. Retain this limitation explicitly.
        slopes = torch.linspace(0, 2, 20001, dtype=torch.float64).reshape(-1, 1, 1, 1)
        logits = direction_surrogate_logits(slopes, [[1.0]], [[1.0]])
        posterior = torch.tensor([0.15, 0.4, 0.45], dtype=torch.float64)
        expected_loss = -(posterior * logits.log_softmax(-1)).sum(-1).reshape(-1)
        optimum = slopes.reshape(-1)[expected_loss.argmin()]
        self.assertTrue(torch.isfinite(optimum))
        # Check the example rather than assert a general Fisher-consistency claim.
        self.assertLess(optimum.item(), 1.0)
        self.assertEqual(direction_labels(optimum.reshape(1, 1, 1, 1), [[1.0]]).item(), 1)


class IndependentCalibrationTargetTests(unittest.TestCase):
    def _bundle(self, length=400, stride=1):
        return load_data({"kind": "synthetic", "scenario": "random_walk", "length": length,
                          "channels": 1, "seed": 7, "noise": 0.0, "lookback": 8,
                          "horizon": 4, "train_stride": 2, "eval_stride": stride,
                          "split_ratios": [0.6, 0.15, 0.1, 0.15]})

    def test_target_intervals_disjoint_even_with_overlapping_windows(self):
        bundle = self._bundle(stride=1)
        original_origins = bundle.calibration.origins.copy()
        earlystop, calibration, metadata = split_earlystop_calibration(bundle)
        horizon = bundle.metadata["horizon"]
        self.assertLess(earlystop.origins.max() + horizon, calibration.origins.min() + 1)
        self.assertLess(earlystop.origins.max() + horizon, metadata["cut_index"])
        self.assertGreaterEqual(calibration.origins.min() + 1, metadata["cut_index"])
        self.assertEqual(len(metadata["discarded_origins"]), horizon - 1)
        np.testing.assert_array_equal(original_origins, bundle.calibration.origins)
        for split in (earlystop, calibration):
            ids = np.searchsorted(original_origins, split.origins)
            np.testing.assert_array_equal(split.x, bundle.calibration.x[ids])
            np.testing.assert_array_equal(split.y, bundle.calibration.y[ids])

    def test_no_fallback_to_reused_targets_when_segment_is_too_short(self):
        bundle = self._bundle(length=80)
        with self.assertRaisesRegex(ValueError, "too short"):
            split_earlystop_calibration(bundle, earlystop_fraction=0.01)
        for fraction in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                split_earlystop_calibration(bundle, earlystop_fraction=fraction)


if __name__ == "__main__":
    unittest.main()
