import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from trend_lab.run import run


class RunnerTest(unittest.TestCase):
    def _config(self, methods):
        return {
            "protocol_version": 1,
            "backbone": "linear", "hidden": 4, "scales": [1, 2],
            "methods": methods, "alpha_grid": [0.0, 0.3], "projection_steps": 3, "max_rms": 0.2,
            "seeds": [7], "training": {"epochs": 1, "patience": 1, "batch_size": 8,
                                          "learning_rate": 0.001, "direction_weight": 0.1},
            "data": {"kind": "synthetic", "scenario": "periodic_turn", "seed": 22,
                     "length": 180, "channels": 1, "lookback": 8, "horizon": 4,
                     "split_ratios": [0.6, 0.15, 0.1, 0.15], "train_stride": 1,
                     "eval_stride": 4, "noise": 0.01},
        }

    def _assert_artifacts(self, temp, methods):
        config = self._config(methods)
        out = Path(temp) / "run"
        result = run(config, out, smoke=True)
        self.assertTrue((out / "results.json").is_file())
        self.assertTrue((out / "predictions.npz").is_file())
        self.assertTrue(result["checkpoint_reload_verified"])
        with np.load(out / "predictions.npz") as archive:
            self.assertIn("origins", archive.files)
            self.assertIn("seed7_truth", archive.files)
            self.assertIn("seed7_history", archive.files)
            if "base" in methods:
                self.assertEqual(archive["seed7_base"].shape[1:], (4, 1))
            if "majority" in methods:
                self.assertEqual(archive["seed7_majority_probs"].shape[1:], (2, 4, 1, 3))
        checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["7"]["projection"], {"steps": 3, "max_rms": 0.2})
        self.assertEqual(checkpoint["7"]["selected_alpha"].keys(), {"projection", "projection_single"})
        self.assertTrue(json.loads((out / "results.json").read_text())["smoke"])
        with self.assertRaises(FileExistsError):
            run(config, out, smoke=True)

    def test_tiny_synthetic_all_methods(self):
        methods = ["base", "difference", "direction_head", "projection", "projection_single",
                   "residual", "residual_direction", "event", "event_nohazard",
                   "last_value", "last_slope", "majority", "last_direction"]
        with tempfile.TemporaryDirectory() as temp:
            self._assert_artifacts(temp, methods)

    def test_tiny_synthetic_method_subset_and_checkpoint_api(self):
        with tempfile.TemporaryDirectory() as temp:
            self._assert_artifacts(temp, ["projection_single", "last_direction"])


if __name__ == "__main__":
    unittest.main()
