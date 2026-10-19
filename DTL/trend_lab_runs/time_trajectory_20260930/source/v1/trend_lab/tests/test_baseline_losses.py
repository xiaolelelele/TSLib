import unittest

import torch

from trend_lab.baseline_losses import cosdir_loss, tdt_loss


class BaselineLossTests(unittest.TestCase):
    def test_cosdir_first_forecast_step_uses_history_anchor(self):
        history = torch.tensor([[[4.0], [10.0]]])
        path = torch.tensor([[[9.0]]], requires_grad=True)
        truth = torch.tensor([[[11.0]]])
        # MSE = 4; opposite anchored increments have cosine -1.
        loss = cosdir_loss(history, path, truth, weight=0.5)
        self.assertAlmostEqual(loss.item(), 5.0)
        loss.backward()
        self.assertTrue(torch.isfinite(path.grad).all())

    def test_cosdir_averages_horizon_vectors_per_channel(self):
        history = torch.zeros(1, 1, 2, dtype=torch.float64)
        truth = torch.tensor([[[3.0, 3.0], [7.0, 7.0]]], dtype=torch.float64)
        path = torch.tensor([[[6.0, -0.3], [14.0, -0.7]]], dtype=torch.float64)
        # Channel 0 has parallel [6,8] vs [3,4]; channel 1 is antiparallel.
        # Averaging their penalties gives 1 despite unequal channel norms.
        expected_mse = (9.0 + 49.0 + 3.3**2 + 7.7**2) / 4
        self.assertAlmostEqual(cosdir_loss(history, path, truth, 0.7).item(), expected_mse + 0.7)
        torch.testing.assert_close(cosdir_loss(history, path, truth, 0), (path - truth).square().mean())

    def test_cosdir_flat_convention_and_gradients_stay_finite(self):
        history = torch.ones(2, 3, 2)
        truth = torch.ones(2, 4, 2)
        path = truth.clone().requires_grad_()
        loss = cosdir_loss(history, path, truth)
        # Match the official code: a zero vector has cosine 0, not 1.
        self.assertAlmostEqual(loss.item(), 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(path.grad).all())

    def test_tdt_uses_sample_channel_weights_not_global_batch_weight(self):
        history = torch.zeros(2, 1, 1)
        path = torch.tensor([[[2.0], [3.0]], [[-2.0], [-5.0]]])
        truth = torch.tensor([[[1.0], [2.0]], [[1.0], [2.0]]])
        # rho=[0,1]. First sample contributes diff-MSE=0.5;
        # second contributes value-MSE=(3^2+7^2)/2=29.
        loss = tdt_loss(history, path, truth)
        self.assertAlmostEqual(loss.item(), 14.75)
        pieces = torch.stack([tdt_loss(history[i:i+1], path[i:i+1], truth[i:i+1]) for i in range(2)])
        torch.testing.assert_close(loss, pieces.mean())
        # Reinterpret the two samples as channels; their weights remain separate.
        torch.testing.assert_close(
            loss,
            tdt_loss(history.permute(2, 1, 0), path.permute(2, 1, 0), truth.permute(2, 1, 0)),
        )

    def test_tdt_paper_zero_convention_is_explicit_and_distinct(self):
        history = torch.zeros(1, 1, 1)
        path = torch.tensor([[[0.0], [2.0]]])
        truth = torch.tensor([[[1.0], [2.0]]])
        # Official product<0 gives rho=0; sign inequality gives rho=1/2.
        self.assertAlmostEqual(tdt_loss(history, path, truth).item(), 1.0)
        self.assertAlmostEqual(tdt_loss(history, path, truth, variant="paper").item(), 0.75)
        self.assertEqual(tdt_loss(history, truth, truth).item(), 0.0)

    def test_both_losses_pass_autograd_check_away_from_sign_boundaries(self):
        history = torch.tensor([[[0.0]]], dtype=torch.float64)
        truth = torch.tensor([[[1.0], [2.0], [1.0]]], dtype=torch.float64)
        path = torch.tensor([[[0.7], [0.2], [-0.8]]], dtype=torch.float64, requires_grad=True)
        for loss in (cosdir_loss, tdt_loss):
            self.assertTrue(torch.autograd.gradcheck(lambda z: loss(history, z, truth), (path,)))

    def test_invalid_shapes_and_options_fail_before_broadcasting(self):
        history, truth = torch.zeros(2, 3, 1), torch.zeros(2, 4, 1)
        for loss in (cosdir_loss, tdt_loss):
            with self.assertRaises(ValueError):
                loss(history, truth[:1], truth)
            with self.assertRaises(ValueError):
                loss(history[:, :0], truth, truth)
        with self.assertRaises(ValueError):
            cosdir_loss(history, truth, truth, weight=-1)
        with self.assertRaises(ValueError):
            cosdir_loss(history, truth, truth, eps=0)
        with self.assertRaises(ValueError):
            tdt_loss(history, truth, truth, variant="global")


if __name__ == "__main__":
    unittest.main()
