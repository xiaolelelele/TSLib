"""Independent finite enumeration and support checks for the decision layer."""
import itertools
import unittest

import numpy as np

from DTL.trend_lab.trajectory_decision import (decode_trajectory, make_common_grids,
                                             path_objective, raw_classes, _grid_dp, _grid_dp_batch, greedy_trajectory)


def independent_score(anchor, path, mean, probability, strength):
    previous = anchor
    squared, utility = 0., 0.
    for h, value in enumerate(path):
        cls = 0 if value < previous else (2 if value > previous else 1)
        utility += float(probability[h, cls])
        squared += float((value - mean[h]) ** 2)
        previous = value
    return (squared - strength * utility) / len(path)


class TrajectoryDecisionTests(unittest.TestCase):
    def test_prefix_suffix_dp_against_exhaustive_independent_enumeration(self):
        rng = np.random.default_rng(20260930)
        for trial in range(60):
            n, horizon, channels = 2, 3, 2
            x = rng.normal(size=(n, 2, channels))
            base = rng.normal(size=(n, horizon, channels))
            mean = rng.normal(size=base.shape)
            p = rng.dirichlet([.8, .8, .8], size=base.shape)
            strength = [0., .01, .3, 3.][trial % 4]
            grid = np.array([-1.7, -.3, .2, 1.6])
            grids = [[grid] * channels for _ in range(n)]
            decoded, diagnostic = decode_trajectory(x, base, mean, p, strength, grids)
            for sample in range(n):
                grid_value = 0.
                for channel in range(channels):
                    grid_value += min(independent_score(x[sample, -1, channel], path,
                                                       mean[sample, :, channel], p[sample, :, channel], strength)
                                      for path in itertools.product(grid, repeat=horizon)) / channels
                base_value = sum(independent_score(x[sample, -1, channel], base[sample, :, channel],
                                                   mean[sample, :, channel], p[sample, :, channel], strength)
                                 for channel in range(channels)) / channels
                self.assertAlmostEqual(diagnostic["objective"][sample], min(grid_value, base_value), places=12)
                self.assertLess(diagnostic["max_dp_objective_recompute_error"], 1e-12)
            self.assertEqual(decoded.dtype, np.float64)
            self.assertEqual(diagnostic["state_updates"], n * channels * horizon * len(grid))

    def test_whole_base_candidate_cannot_be_stitched_per_channel(self):
        x = np.zeros((1, 1, 2))
        base = np.full((1, 1, 2), .3)
        mean = np.array([[[0., 1.]]])
        p = np.full((1, 1, 2, 3), 1 / 3)
        grids = [[np.array([0.]), np.array([2.])]]
        z, diagnostic = decode_trajectory(x, base, mean, p, 0., grids)
        np.testing.assert_array_equal(z, base)
        self.assertTrue(diagnostic["selected_base"][0])
        z, _ = decode_trajectory(x, base, mean, p, 0., grids, allow_base=False)
        np.testing.assert_array_equal(z, [[[0., 2.]]])

    def test_zero_strength_keeps_true_conditional_mean_and_has_no_risk_guarantee(self):
        x = np.zeros((1, 1, 1))
        mean = np.zeros((1, 1, 1))
        p = np.array([[[[.2, .2, .6]]]])
        z, diagnostic = decode_trajectory(x, mean, mean, p, 0., np.array([-.1, 0., .1]))
        np.testing.assert_array_equal(z, mean)
        modified, result = decode_trajectory(x, mean, mean, p, 1., np.array([-.1, 0., .1]))
        self.assertGreater(result["estimated_utility_delta"][0], 0.)
        self.assertGreater(result["estimated_risk_delta"][0], 0.)
        self.assertEqual(float(modified[0, 0, 0]), .1)

    def test_exact_zero_and_first_anchor_use_no_tolerance(self):
        x = np.array([[[.37]]])
        z = np.array([[[.37], [.37 + 1e-12], [.37]]])
        np.testing.assert_array_equal(raw_classes(x, z), [[[1], [2], [0]]])
        p = np.tile([.05, .9, .05], (1, 3, 1, 1))
        mean = np.full((1, 3, 1), .37)
        decoded, _ = decode_trajectory(x, mean, mean, p, 1., np.array([0., .37, 1.]), allow_base=False)
        np.testing.assert_array_equal(decoded, mean)

    def test_native_grid_offset_normalizer_and_sparse_budget(self):
        normal_mean, normal_std = np.array([2.3]), np.array([3.7])
        raw_anchor = np.array([[[6.]]])
        x = (raw_anchor - normal_mean) / normal_std
        base = (np.array([[[4.], [5.]]]) - normal_mean) / normal_std
        grids, meta = make_common_grids(x, base, base, mode="native", native_step=1.,
                                       normalizer_mean=normal_mean, normalizer_std=normal_std,
                                       padding=1 / normal_std, max_states=257)
        self.assertTrue(meta["anchor_reachable"][0][0])
        recovered = grids[0][0] * normal_std[0] + normal_mean[0]
        np.testing.assert_allclose(recovered, np.round(recovered), atol=1e-14)
        large = np.array([[[-1000.], [1000.]]])
        grids, meta = make_common_grids(x, large, large, mode="native", native_step=1.,
                                       normalizer_mean=normal_mean, normalizer_std=normal_std,
                                       max_states=17)
        self.assertTrue(meta["sparse_native_lattice"][0][0])
        self.assertTrue(meta["anchor_reachable"][0][0])
        self.assertLessEqual(len(grids[0][0]), 17)
        recovered = grids[0][0] * normal_std[0] + normal_mean[0]
        np.testing.assert_allclose(recovered, np.round(recovered), atol=1e-12)
        with self.assertRaises(ValueError):
            make_common_grids(x, large, large, max_states=17, coarsen=False)

    def test_continuous_coarsening_is_observable_and_shared_across_controls(self):
        x = np.array([[[.123456789]]])
        base = np.array([[[-20.], [30.]]])
        mean = np.array([[[-10.], [25.]]])
        grids, meta = make_common_grids(x, base, mean, step=.01, max_states=21)
        self.assertTrue(meta["coarsened"][0][0])
        self.assertLessEqual(len(grids[0][0]), 21)
        self.assertIn(float(x[0, -1, 0]), grids[0][0])
        self.assertLessEqual(grids[0][0][0], -20.5)
        self.assertGreaterEqual(grids[0][0][-1], 30.5)
        p = np.array([[[[.1, .2, .7]], [[.7, .2, .1]]]])
        before = grids[0][0].copy()
        full, _ = decode_trajectory(x, base, mean, p, .1, grids)
        top, _ = decode_trajectory(x, base, base, p, .1, grids, reward="top1_gap")
        np.testing.assert_array_equal(grids[0][0], before)
        self.assertEqual(full.shape, top.shape)

    def test_batch_invariance_and_deterministic_exact_ties(self):
        rng = np.random.default_rng(74)
        x = rng.normal(size=(4, 2, 2))
        base = rng.normal(size=(4, 4, 2))
        mean = rng.normal(size=base.shape)
        p = rng.dirichlet([1., 1., 1.], size=base.shape)
        grids, _ = make_common_grids(x, base, mean, step=.3)
        batched, _ = decode_trajectory(x, base, mean, p, .3, grids)
        for i in range(4):
            alone, _ = decode_trajectory(x[i:i+1], base[i:i+1], mean[i:i+1], p[i:i+1], .3, [grids[i]])
            np.testing.assert_array_equal(alone, batched[i:i+1])
        zero = np.zeros((1, 2, 1))
        uniform = np.full((1, 2, 1, 3), 1 / 3)
        tied, _ = decode_trajectory(zero, zero + 2., zero, uniform, 0., np.array([-1., 1.]), allow_base=False)
        np.testing.assert_array_equal(tied, [[[-1.], [-1.]]])
        unchanged, result = decode_trajectory(zero, zero, zero, uniform, 0., np.array([0.]))
        self.assertTrue(result["selected_base"][0])
        np.testing.assert_array_equal(unchanged, zero)

    def test_batched_recurrence_matches_scalar_exactly_for_variable_state_counts(self):
        rng = np.random.default_rng(510)
        for size in [1, 2, 5, 17]:
            grids = np.tile(np.linspace(-2., 2., size), (9, 1))
            anchor = rng.normal(size=9)
            mean = rng.normal(size=(9, 7))
            p = rng.dirichlet([1., 1., 1.], size=(9, 7))
            batched, objective = _grid_dp_batch(anchor, mean, p, grids, .4)
            for i in range(9):
                scalar, value = _grid_dp(anchor[i], mean[i], p[i], grids[i], .4)
                np.testing.assert_array_equal(scalar, batched[i])
                self.assertEqual(value, objective[i])

    def test_greedy_same_library_stage_decisions_and_dp_no_worse(self):
        rng = np.random.default_rng(415)
        strict = 0
        for trial in range(15):
            x = np.zeros((3, 1, 1))
            base = rng.normal(size=(3, 4, 1))
            mean = rng.normal(size=base.shape)
            p = rng.dirichlet([.5, .5, .5], size=base.shape)
            grid = np.array([-2., -1., 0., 1., 2.])
            greedy, gd = greedy_trajectory(x, base, mean, p, 1., grid, allow_base=False)
            exact, dp = decode_trajectory(x, base, mean, p, 1., grid, allow_base=False)
            for sample in range(3):
                previous = 0.
                for h in range(4):
                    costs = []
                    for value in grid:
                        cls = 0 if value < previous else (2 if value > previous else 1)
                        costs.append((value - mean[sample, h, 0]) ** 2 - p[sample, h, 0, cls])
                    self.assertEqual(greedy[sample, h, 0], grid[int(np.argmin(costs))])
                    previous = greedy[sample, h, 0]
                self.assertLessEqual(dp["objective"][sample], gd["objective"][sample] + 1e-12)
                strict += int(dp["objective"][sample] < gd["objective"][sample] - 1e-9)
            self.assertFalse(gd["globally_optimal_finite_grid"])
            self.assertTrue(dp["globally_optimal_finite_grid"])
        self.assertGreater(strict, 0)

    def test_invalid_probabilities_and_unsorted_grids_rejected(self):
        zero = np.zeros((1, 1, 1))
        p = np.full((1, 1, 1, 3), 1 / 3)
        with self.assertRaises(ValueError):
            decode_trajectory(zero, zero, zero, p * 2., 1., np.array([0., 1.]))
        with self.assertRaises(ValueError):
            decode_trajectory(zero, zero, zero, p, 1., np.array([1., 0.]))
        with self.assertRaises(ValueError):
            decode_trajectory(zero, zero, zero, p, -1., np.array([0., 1.]))


if __name__ == "__main__":
    unittest.main()
