"""Numerical guarantees and decision semantics independent of real test data."""

import unittest

import torch

from trend_lab.decoding import (project_trajectory, score_posterior_candidates,
                                select_posterior_candidates)


class DecodingRepairTests(unittest.TestCase):
    def test_default_40_step_path_matches_pre_repair_float32_fixture(self):
        x = torch.tensor([[[.1], [.2], [.15], [.05]]])
        b = torch.tensor([[[.01], [.2], [-.1]]])
        p = torch.tensor([[[[[.05, .15, .8]], [[.7, .2, .1]], [[.1, .7, .2]]],
                           [[[.2, .1, .7]], [[.4, .1, .5]], [[.1, .2, .7]]]]])
        z = project_trajectory(x, b, p, [1, 2], torch.tensor([[.02], [.03]]),
                               strength=10, max_rms=.15)
        # Captured from the unmodified decoder before this repair; not a solver oracle.
        self.assertEqual(z.view(torch.int32).reshape(-1).tolist(), [1034666370, 1034166959, 1025045430])

    def scalar_inputs(self, bases=(0.0,)):
        base = torch.tensor(bases, dtype=torch.float64).reshape(-1, 1, 1)
        history = torch.zeros(len(bases), 2, 1, dtype=torch.float64)
        probabilities = torch.tensor([0.05, 0.05, 0.90], dtype=torch.float64).reshape(1, 1, 1, 1, 3)
        return history, base, probabilities.expand(len(bases), -1, -1, -1, -1), torch.tensor([[0.2]], dtype=torch.float64)

    def test_unconstrained_scalar_matches_closed_form_optimum(self):
        x, b, p, eps = self.scalar_inputs()
        z, diag = project_trajectory(x, b, p, [1], eps, strength=10, steps=1000,
                                    max_rms=1, convergence_tol=1e-11, return_diagnostics=True)
        # f(z)=z^2/2 + 10*(.90-.05)*(.30-z)_+^2/2.
        expected = 8.5 * 0.3 / 9.5
        self.assertAlmostEqual(z.item(), expected, places=10)
        self.assertTrue(diag['all_converged'])
        self.assertLess(diag['projected_gradient_rms'][0], 1e-11)

    def test_active_rms_constraint_satisfies_projected_stationarity(self):
        x, b, p, eps = self.scalar_inputs()
        z, diag = project_trajectory(x, b, p, [1], eps, strength=10, steps=100,
                                    max_rms=.05, convergence_tol=1e-11, return_diagnostics=True)
        self.assertAlmostEqual(z.item(), .05, places=13)
        self.assertTrue(diag['all_converged'])
        self.assertTrue(diag['cap_binding_per_path'][0])

    def test_exhausting_budget_is_not_reported_as_convergence(self):
        x, b, p, eps = self.scalar_inputs()
        _, diag = project_trajectory(x, b, p, [1], eps, strength=10, steps=1,
                                    max_rms=1, convergence_tol=1e-12, return_diagnostics=True)
        self.assertFalse(diag['all_converged'])
        self.assertEqual(diag['iterations_per_path'], [1])

    def test_coupled_flat_problem_matches_independent_linear_solve(self):
        # epsilon=0 and a certain flat class make the soft objective quadratic.
        # Its exact minimizer solves (I + alpha*D.T@D) z = base.
        horizon = 8
        x = torch.zeros(1, 2, 1, dtype=torch.float64)
        b = torch.ones(1, horizon, 1, dtype=torch.float64)
        p = torch.zeros(1, 1, horizon, 1, 3, dtype=torch.float64)
        p[..., 1] = 1
        eps = torch.zeros(1, 1, dtype=torch.float64)
        difference = torch.eye(horizon, dtype=torch.float64)
        difference[torch.arange(1, horizon), torch.arange(horizon-1)] = -1
        expected = torch.linalg.solve(torch.eye(horizon, dtype=torch.float64) + 100*difference.T@difference,
                                      torch.ones(horizon, dtype=torch.float64))
        legacy = project_trajectory(x, b, p, [1], eps, strength=100, max_rms=10)
        z, diag = project_trajectory(x, b, p, [1], eps, strength=100, max_rms=10,
                                    steps=10000, convergence_tol=1e-10, return_diagnostics=True)
        self.assertGreater(float((legacy.reshape(-1)-expected).abs().max()), .05)
        torch.testing.assert_close(z.reshape(-1), expected, rtol=1e-9, atol=1e-10)
        self.assertTrue(diag['all_converged'])

    def test_per_sample_stopping_matches_separate_calls(self):
        x, b, p, eps = self.scalar_inputs((0., .5))
        z, diag = project_trajectory(x, b, p, [1], eps, strength=10, steps=1000,
                                    max_rms=1, convergence_tol=1e-10, return_diagnostics=True)
        separate = [project_trajectory(x[i:i+1], b[i:i+1], p[i:i+1], [1], eps,
                                      strength=10, steps=1000, max_rms=1, convergence_tol=1e-10)
                    for i in range(2)]
        self.assertTrue(torch.equal(z, torch.cat(separate)))
        self.assertEqual(diag['iterations_per_path'][1], 0)
        self.assertGreater(diag['iterations_per_path'][0], 0)

    def test_posterior_score_uses_probability_of_actual_path_class(self):
        x = torch.zeros(1, 2, 1, dtype=torch.float64)
        candidates = torch.tensor([-.2, .1, .2], dtype=torch.float64).reshape(3, 1, 1, 1)
        probabilities = torch.tensor([.4, .1, .5], dtype=torch.float64).reshape(1, 1, 1, 1, 3)
        eps = torch.tensor([[.1]], dtype=torch.float64)
        score = score_posterior_candidates(x, candidates, probabilities, [1], eps)
        torch.testing.assert_close(score[:, 0], torch.tensor([.4, .1, .5], dtype=torch.float64))
        base = candidates[0]
        chosen, indices = select_posterior_candidates(x, base, candidates, probabilities, [1], eps, penalty=1)
        # Utility(.2)=.5-.4^2=.34, so closeness changes the winner to base.
        self.assertEqual(indices.item(), 0)
        self.assertTrue(torch.equal(chosen, base))

    def test_identical_candidate_tie_prefers_first(self):
        x, b, p, eps = self.scalar_inputs()
        chosen, indices = select_posterior_candidates(x, b, torch.stack([b, b]), p, [1], eps)
        self.assertEqual(indices.item(), 0)
        self.assertTrue(torch.equal(chosen, b))

    def test_invalid_iteration_and_nonfinite_controls_rejected(self):
        x, b, p, eps = self.scalar_inputs()
        for controls in ({'steps': 1.5}, {'strength': float('nan')}, {'max_rms': float('inf')},
                         {'convergence_tol': 0}, {'convergence_tol': float('nan')}):
            with self.assertRaises(ValueError):
                project_trajectory(x, b, p, [1], eps, **controls)


if __name__ == '__main__':
    unittest.main()
