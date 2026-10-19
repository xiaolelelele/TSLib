"""Small CPU end-to-end checks for the versioned repaired research runner."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from trend_lab import repaired
from trend_lab.data import load_data
from trend_lab.run import load_checkpoint, predict_from_checkpoint, run


class RepairedRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name) / "clean"
        cls.config = {
            "protocol_version": 2, "device": "cpu", "backbone": "linear", "hidden": 4,
            "scales": [1, 2], "seeds": [0], "earlystop_fraction": 0.5,
            "alpha_grid": [0.0, 0.3], "posterior_betas": [0.0, 0.1],
            "projection_steps": 8, "projection_tolerance": None, "max_rms": 0.2,
            "temperature_grid": [0.5, 1.0, 2.0], "cosdir_weights": [0.1, 1.0],
            "direction_weights": [0.01, 0.1], "slope_scale_floor": 0.02,
            "training": {"epochs": 1, "patience": 1, "batch_size": 8, "learning_rate": 0.001},
            "data": {"kind": "synthetic", "scenario": "periodic_turn", "seed": 22,
                     "length": 320, "channels": 2, "lookback": 8, "horizon": 4,
                     "split_ratios": [0.6, 0.15, 0.1, 0.15], "train_stride": 4,
                     "eval_stride": 4, "noise": 0.03},
        }
        cls.test_score_calls = 0
        original_evaluate = repaired.evaluate_predictions

        def require_lock_before_scoring(*args, **kwargs):
            lock = cls.out / "seed0_validation_lock.json"
            if not lock.is_file():
                raise AssertionError("test scoring happened before validation choices were saved")
            with lock.open() as stream:
                saved = json.load(stream)
            if "learned_selected" not in saved["choices"]:
                raise AssertionError("validation choice is missing from the lock")
            cls.test_score_calls += 1
            return original_evaluate(*args, **kwargs)

        with mock.patch.object(repaired, "evaluate_predictions", side_effect=require_lock_before_scoring):
            cls.result = run(cls.config, cls.out)
        cls.checkpoint = load_checkpoint(cls.out / "checkpoint.pt", seed=0)
        with np.load(cls.out / "predictions.npz") as archive:
            cls.arrays = {key: archive[key] for key in archive.files}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_artifacts_have_disjoint_earlystop_and_calibration_targets(self):
        self.assertEqual(self.result["schema_version"], 2)
        self.assertTrue(self.result["complete"])
        self.assertTrue(self.result["runs"][0]["checkpoint_reload_verified"])
        self.assertGreater(self.test_score_calls, 0)
        horizon = self.config["data"]["horizon"]
        early = self.arrays["earlystop__origins"]
        calibration = self.arrays["calibration__origins"]
        self.assertLess(early.max() + horizon, calibration.min() + 1)
        self.assertLess(early.max() + horizon, self.result["earlystop_calibration_split"]["cut_index"])
        self.assertGreaterEqual(calibration.min() + 1, self.result["earlystop_calibration_split"]["cut_index"])
        for split in ("earlystop", "calibration", "validation", "test"):
            self.assertEqual(self.arrays[split + "__truth"].shape[1:], (4, 2))
            self.assertEqual(self.arrays[split + "__history"].shape[1:], (8, 2))
        # Python's JSON loader accepts NaN by default; explicitly reject it here.
        def reject_constant(value):
            raise AssertionError("nonfinite JSON constant: " + value)
        for filename in ("config.json", "results.json", "seed0_validation_lock.json"):
            json.loads((self.out / filename).read_text(), parse_constant=reject_constant)

    def test_selected_and_unselected_model_outputs_reload(self):
        x = torch.from_numpy(self.arrays["test__history"])
        row = self.result["runs"][0]
        self.assertIn("learned_selected", row["methods"])
        self.assertIn(row["validation"]["choices"]["learned_selected"],
                      ("base", "projection", "projection_single"))
        for name in row["methods"]:
            actual = predict_from_checkpoint(self.checkpoint, x, name)
            if name == "direction_head":
                torch.testing.assert_close(actual["path"], torch.from_numpy(self.arrays["seed0__test__base"]))
                torch.testing.assert_close(actual["probabilities"], torch.from_numpy(self.arrays["seed0__test__direction_head__probs"]))
            else:
                torch.testing.assert_close(actual, torch.from_numpy(self.arrays["seed0__test__" + name]))
        raw_names = {name for name, spec in self.checkpoint["model_specs"].items() if spec["type"] != "head"}
        self.assertEqual(set(row["test_models"]), raw_names)
        for name in raw_names:
            actual = predict_from_checkpoint(self.checkpoint, x, name)
            torch.testing.assert_close(actual, torch.from_numpy(self.arrays["seed0__test_model__" + name]))
        with self.assertRaises(FileExistsError):
            run(self.config, self.out)

    def test_direction_control_diagnostics_are_finite_and_weights_are_not_closed_over(self):
        row = self.result["runs"][0]
        for weight in (0.01, 0.1):
            name = "residual_direction_weight_%g" % weight
            diagnostics = row["training"][name]["calibration_loss_diagnostics"]
            for key, value in diagnostics.items():
                self.assertIsNotNone(value, key)
                self.assertTrue(np.isfinite(value), key)
            self.assertAlmostEqual(diagnostics["weighted_direction_loss"], weight * diagnostics["direction_loss"])
        low = self.checkpoint["models"]["residual_direction_weight_0.01"]
        high = self.checkpoint["models"]["residual_direction_weight_0.1"]
        self.assertTrue(any(not torch.equal(low[key], high[key]) for key in low))
        self.assertTrue(any(not torch.equal(self.checkpoint["models"]["cosdir_weight_0.1"][key],
                                            self.checkpoint["models"]["cosdir_weight_1"][key])
                            for key in self.checkpoint["models"]["cosdir_weight_0.1"]))

    def test_changing_only_test_targets_cannot_change_training_or_validation_choices(self):
        poisoned_bundle = copy.deepcopy(load_data(self.config["data"]))
        poisoned_bundle.test.y += np.float32(100.0)
        poisoned_out = Path(self.temp.name) / "poisoned"
        with mock.patch.object(repaired, "load_data", return_value=poisoned_bundle):
            poisoned = run(self.config, poisoned_out)
        saved = load_checkpoint(poisoned_out / "checkpoint.pt", seed=0)
        self.assertEqual(poisoned["runs"][0]["validation"], self.result["runs"][0]["validation"])
        self.assertEqual(saved["entries"], self.checkpoint["entries"])
        self.assertEqual(saved["temperature"], self.checkpoint["temperature"])
        for name, state in self.checkpoint["models"].items():
            for key, value in state.items():
                torch.testing.assert_close(saved["models"][name][key], value, rtol=0, atol=0)
        self.assertGreater(poisoned["runs"][0]["methods"]["base"]["mse"],
                           self.result["runs"][0]["methods"]["base"]["mse"] + 1000)


if __name__ == "__main__":
    unittest.main()
