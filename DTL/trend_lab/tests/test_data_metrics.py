import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from trend_lab.data import _synthetic, load_data
from trend_lab.metrics import evaluate_predictions, paired_block_interval
from trend_lab.operators import direction_labels, fit_thresholds, multiscale_slopes, turn_targets


class OperatorTests(unittest.TestCase):
    def test_multiscale_anchor_flat_boundary_and_turn(self):
        x = torch.tensor([[[1.0], [2.0], [3.0]]])
        y = torch.tensor([[[4.0], [3.0]]])
        slopes = multiscale_slopes(x, y, [1, 2])
        torch.testing.assert_close(slopes[0, 0, :, 0], torch.tensor([1.0, -1.0]))
        torch.testing.assert_close(slopes[0, 1, :, 0], torch.tensor([1.0, 0.0]))
        labels = direction_labels(torch.tensor([[[[0.1], [-0.1001], [0.2]]]]), [[0.1]])
        self.assertEqual(labels.flatten().tolist(), [1, 0, 2])
        turns = turn_targets(x, y, torch.tensor([0.0]))
        self.assertEqual(turns.flatten().tolist(), [0.0, 1.0])

    def test_train_thresholds_are_per_scale_channel(self):
        x = torch.zeros(3, 4, 2)
        y = torch.arange(1, 4, dtype=torch.float32).view(3, 1, 1).expand(-1, 2, 2)
        eps = fit_thresholds(x, y, [1, 2])
        self.assertEqual(tuple(eps.shape), (2, 2))
        self.assertTrue(torch.all(eps >= 1e-6))


class DataAndMetricTests(unittest.TestCase):
    def test_chronological_splits_and_train_only_scaler(self):
        config = {"kind": "synthetic", "scenario": "random_walk", "length": 120, "channels": 2,
                  "seed": 5, "lookback": 8, "horizon": 4, "train_stride": 1,
                  "eval_stride": 4, "split_ratios": [0.6, 0.15, 0.1, 0.15], "noise": 0.1}
        bundle = load_data(config)
        self.assertLess(bundle.train.origins.max() + config["horizon"], bundle.metadata["split_bounds"][1])
        self.assertGreaterEqual(bundle.calibration.origins.min() + 1, bundle.metadata["split_bounds"][1])
        scales = [1, 2, 4, 8]
        epsilon = fit_thresholds(torch.from_numpy(bundle.train.x), torch.from_numpy(bundle.train.y), scales)
        changed = dict(config, seed=5)
        first = load_data(changed)
        self.assertTrue(np.array_equal(bundle.mean, first.mean))
        np.testing.assert_array_equal(epsilon, fit_thresholds(torch.from_numpy(first.train.x), torch.from_numpy(first.train.y), scales))

    def test_csv_order_validation_and_json_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.csv"
            values = list(range(40))
            def write_csv(vals):
                rows = "".join(f"{i},{value}\n" for i, value in enumerate(vals))
                path.write_text("time,v\n" + rows, encoding="utf-8")
            write_csv(values)
            config = {"kind": "csv", "scenario": None, "channels": 1,
                      "lookback": 3, "horizon": 2, "train_stride": 1, "eval_stride": 2,
                      "split_ratios": [0.6, 0.15, 0.1, 0.15], "csv_path": str(path),
                      "value_columns": ["v"], "time_column": "time"}
            first = load_data(config)
            scales = [1, 2]
            epsilon = fit_thresholds(torch.from_numpy(first.train.x), torch.from_numpy(first.train.y), scales)
            write_csv(values[:34] + [1000] * 6)
            second = load_data(config)
            np.testing.assert_array_equal(first.mean, second.mean)
            np.testing.assert_array_equal(epsilon, fit_thresholds(torch.from_numpy(second.train.x), torch.from_numpy(second.train.y), scales))
            path.write_text("time,v\n" + "".join(f"{(2 if i == 1 else 1 if i == 2 else i)},{value}\n" for i, value in enumerate(values)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                load_data(config)
            path.write_text("time,v\n0,0\ninf,1\n" + "".join(f"{i},{i}\n" for i in range(2, 40)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be finite"):
                load_data(config)
        x = np.array([[[0.0], [1.0]], [[1.0], [2.0]]], dtype=np.float32)
        y = np.array([[[2.0], [1.0]], [[3.0], [4.0]]], dtype=np.float32)
        p = y.copy()
        probs = np.full((2, 1, 2, 1, 3), 1 / 3, dtype=np.float32)
        result = evaluate_predictions(x, y, p, [1], np.array([[0.01]]), probs)
        self.assertEqual(result["mse"], 0.0)
        self.assertIn("1", result["scales"])
        json.dumps(result, allow_nan=False)
        self.assertEqual(paired_block_interval([1, 2, 3], block_size=2, repeats=20)["mean"], 2.0)

    def test_csv_fixture_normal_smoke_and_train_only_scaler(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.csv"
            values = np.arange(240, dtype=float) ** 1.2

            def write_values(series):
                rows = "".join(f"{i},{value}\n" for i, value in enumerate(series))
                path.write_text("time,value\n" + rows, encoding="utf-8")

            write_values(values)
            config = {"kind": "csv", "channels": 1, "lookback": 8, "horizon": 8,
                      "split_ratios": [0.6, 0.15, 0.1, 0.15], "train_stride": 1,
                      "eval_stride": 8, "csv_path": str(path), "value_columns": ["value"],
                      "time_column": "time"}
            full = load_data(config)
            self.assertEqual(full.metadata["length"], 240)
            self.assertFalse(full.metadata["smoke_truncated"])
            self.assertEqual(full.metadata["source_rows"], 240)

            smoke = load_data({**config, "smoke": True})
            self.assertEqual(smoke.metadata["length"], 180)
            self.assertEqual(smoke.metadata["smoke_row_limit"], 180)
            self.assertTrue(smoke.metadata["smoke_truncated"])
            np.testing.assert_allclose(smoke.mean, values[:108].mean(axis=0))
            np.testing.assert_allclose(smoke.std, values[:108].std(axis=0))

            changed_tail = values.copy()
            changed_tail[108:] += 1_000_000
            write_values(changed_tail)
            changed_smoke = load_data({**config, "smoke": True})
            np.testing.assert_array_equal(smoke.mean, changed_smoke.mean)
            np.testing.assert_array_equal(smoke.std, changed_smoke.std)

    def test_noisy_multiscale_default_period_is_length_independent(self):
        short = _synthetic({"scenario": "noisy_multiscale", "length": 160, "channels": 1,
                            "seed": 4, "noise": 0.0})
        long = _synthetic({"scenario": "noisy_multiscale", "length": 300, "channels": 1,
                           "seed": 4, "noise": 0.0})
        np.testing.assert_array_equal(short, long[:160])

    def test_event_first_step_flat_hold_and_unmatched_pair(self):
        history = np.array([[[0.0], [1.0], [1.0]]], dtype=np.float32)
        truth = np.array([[[0.0], [0.0], [-1.0], [-2.0]]], dtype=np.float32)
        pred = np.array([[[2.0], [3.0], [4.0], [3.0]]], dtype=np.float32)
        result = evaluate_predictions(history, truth, pred, [1], [[0.1]])
        self.assertEqual(result["events"]["true_count"], 1)
        self.assertEqual(result["events"]["pred_count"], 1)
        self.assertEqual(result["events"]["f1"], 0.0)
        self.assertEqual(result["regimes"]["turn_neighborhood_count"], 2)


if __name__ == "__main__":
    unittest.main()
