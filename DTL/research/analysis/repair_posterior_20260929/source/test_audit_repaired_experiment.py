"""Small numerical invariants for the independent audit and diagnostic summary."""
import unittest

import numpy as np

from audit_repaired_experiment import labels, path_metrics, posterior_indices
from summarize_repaired_experiment import aggregate_convergence, _convergence_row


class IndependentAuditTests(unittest.TestCase):
    def test_endpoints_are_promoted_before_subtraction(self):
        x = np.array([[[.2]]], dtype=np.float32)
        y = np.array([[[-.3]]], dtype=np.float32)
        # Float32 subtraction rounds to -0.5 exactly; true endpoint subtraction
        # in float64 is below -0.5 and therefore belongs to down, not flat.
        self.assertEqual(float(y[0, 0, 0] - x[0, 0, 0]), -.5)
        self.assertEqual(labels(x, y, [1], [[.5]]).item(), 0)

    def test_raw_zero_is_retained_and_first_step_uses_history(self):
        metric = path_metrics([[[0.]]], [[[0.], [1.]]], [[[0.], [-1.]]], [1], [[.1]])
        self.assertEqual(metric["mse"], 2.)
        self.assertEqual(metric["raw_sign_accuracy"], .5)
        self.assertEqual(metric["mean_3class_accuracy"], .5)
        self.assertAlmostEqual(metric["scales"]["1"]["macro_f1"], 1/3)
        self.assertEqual(metric["scales"]["1"]["confusion_matrix"], [[0, 0, 0], [0, 1, 0], [1, 0, 0]])

    def test_posterior_probability_cost_and_first_tie(self):
        x, base = np.zeros((1, 1, 1)), np.zeros((1, 1, 1))
        candidates = np.array([[[[0.]]], [[[1.]]], [[[-1.]]]])
        p = np.array([[[[[.2, .1, .7]]]]])
        self.assertEqual(posterior_indices(x, base, candidates, p, [1], [[.1]], 0)[0].item(), 1)
        self.assertEqual(posterior_indices(x, base, candidates, p, [1], [[.1]], 1)[0].item(), 0)
        self.assertEqual(posterior_indices(x, base, candidates, np.full_like(p, 1/3), [1], [[.1]], 0)[0].item(), 0)

    def test_base_fallback_does_not_count_as_solver_convergence(self):
        diagnostic = {"projected_gradient_rms": [1e-6, 2e-5], "iterations_per_path": [30, 400],
                      "cap_binding_per_path": [False, True], "convergence_tol": 1e-5,
                      "max_steps": 400, "perturbation_rms": [.1, .5]}
        rows = [_convergence_row("g", 0, "test", "m", "candidate", diagnostic),
                _convergence_row("g", 1, "test", "m", "base", {}, total=8)]
        row = aggregate_convergence(rows)[0]
        self.assertEqual(row["paths"], 10)
        self.assertEqual(row["projected_paths"], 2)
        self.assertEqual(row["not_converged_fraction"], .5)
        self.assertEqual(row["at_400_steps_fraction"], .5)
        self.assertEqual(row["cap_binding_fraction"], .5)


if __name__ == "__main__":
    unittest.main()
