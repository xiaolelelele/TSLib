"""Formula, causality and phase checks for the strong research baselines."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.special import ndtr
import torch
from torch.nn import functional as F

from trend_lab.trajectory_baselines import (CaReTSPhaseTransformer, OCETimeSeries,
                                          ordinal_cross_entropy)


class TrajectoryBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(391)

    def test_ordinal_loss_official_source_and_distance(self):
        source = Path(__file__).resolve().parents[2] / "research/analysis/time_baseline_sources_20260930/loss__OrdinalCrossEntropyLoss.py"
        spec = importlib.util.spec_from_file_location("official_oce_loss", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predicted = torch.tensor([[.3, .5, .2], [.4, .1, .5]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[.8, .1, .1], [.2, .1, .7]], dtype=torch.float64)
        actual = ordinal_cross_entropy(predicted, target)
        official = module.OrderedCrossEntropyLoss()(predicted, target)
        self.assertAlmostEqual(actual.item(), official.item(), places=6)
        actual.backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        truth = torch.tensor([[1., 0., 0.]])
        near = ordinal_cross_entropy(torch.tensor([[.01, .98, .01]]), truth)
        far = ordinal_cross_entropy(torch.tensor([[.01, .01, .98]]), truth)
        self.assertGreater(far, near)

    def test_soft_targets_independent_gaussian_cdf(self):
        model = OCETimeSeries(8, 2, 2, [-4., 2.], [6., 12.], bins=9, sigma=.13)
        values = torch.tensor([[[-4., 4.], [6., 11.]], [[1., 2.], [0., 12.]]], dtype=torch.float64)
        unit = 2*(values.numpy()-np.array([-4., 2.]))/10-1
        support = np.linspace(-1, 1, 10)
        cdf = ndtr((support-unit[..., None])/.13)
        expected = np.diff(cdf, axis=-1)
        expected /= expected.sum(-1, keepdims=True)
        np.testing.assert_allclose(model.target_probabilities(values).numpy(), expected, atol=2e-15, rtol=2e-13)
        np.testing.assert_allclose(model.target_probabilities(values).sum(-1).numpy(), 1, atol=2e-15)
        # Explicit boundary clamping in soft loss leaves supplied labels intact.
        outside = values.clone(); outside[0, 0, 0] = -20
        expected_boundary = model.target_probabilities(values)[0, 0, 0]
        torch.testing.assert_close(model.target_probabilities(outside)[0, 0, 0], expected_boundary)
        self.assertEqual(outside[0, 0, 0].item(), -20)

    def test_fixed_support_affine_and_no_future_dependence(self):
        model = OCETimeSeries(8, 2, 2, [-2., -1.], [3., 4.], bins=7)
        changed = OCETimeSeries(8, 2, 2, [4., 7.], [19., 22.], bins=7)
        state = {k: v for k, v in model.state_dict().items() if k.startswith(("seasonal", "trend"))}
        changed.load_state_dict(state, strict=False)
        x = torch.randn(3, 8, 2)
        y = torch.randn(3, 2, 2)
        torch.testing.assert_close(changed(x*3+10), model(x)*3+10, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(changed.loss(x*3+10, y*3+10), model.loss(x, y), atol=2e-6, rtol=2e-6)
        before = model(x).detach().clone()
        support_before = model.support.clone()
        lower_before = model.lower.clone()
        _ = model.target_probabilities(y*1000)
        torch.testing.assert_close(before, model(x))
        torch.testing.assert_close(support_before, model.support)
        torch.testing.assert_close(lower_before, model.lower)
        self.assertTrue(torch.isfinite(OCETimeSeries(8, 2, 1, [3.], [3.])(x[:, :, :1])).all())

    def test_official_decomposition_classifier_numeric_path(self):
        model = OCETimeSeries(4, 2, 2, [-2., -3.], [2., 7.], bins=3, moving_average=3).double()
        x = torch.tensor([[[1., 2.], [2., 3.], [0., -2.], [-1., 0.]]], dtype=torch.float64)
        unit = 2*(x.numpy()-np.array([-2., -3.]))/np.array([4., 10.])-1
        pad = np.concatenate((unit[:, :1], unit, unit[:, -1:]), axis=1)
        trend = (pad[:, :-2]+pad[:, 1:-1]+pad[:, 2:])/3
        seasonal = unit-trend
        def linear(v, layer):
            return np.transpose(v, (0, 2, 1)) @ layer.weight.detach().numpy().T+layer.bias.detach().numpy()
        logits = linear(seasonal, model.seasonal)+linear(trend, model.trend)
        logits = logits.reshape(1, 2, 2, 3).transpose(0, 2, 1, 3)
        probability = np.exp(logits-logits.max(-1, keepdims=True))
        probability /= probability.sum(-1, keepdims=True)
        expected_unit = (probability*np.array([-2/3, 0, 2/3])).sum(-1)
        expected = (expected_unit+1)*np.array([4., 10.])/2+np.array([-2., -3.])
        np.testing.assert_allclose(model(x).detach().numpy(), expected, atol=1e-15, rtol=1e-14)

    def test_carets_anchor_fusion_and_ground_truth_mask(self):
        model = CaReTSPhaseTransformer(4, 2, 1, 3, d_model=8, heads=2, layers=1, dropout=0.)
        for branch in (model.classifier, model.regressor):
            for parameter in branch.parameters():
                parameter.data.zero_()
        model.classifier[-1].bias.data.copy_(torch.tensor([np.log(3), 0., np.log(3), 0.]))
        model.regressor[-1].bias.data.copy_(torch.tensor([2., 4., 3., 5.]))
        x = torch.zeros(1, 4, 1); x[:, -1] = 7
        y = torch.tensor([[[9.], [2.]]])
        path, logits, magnitude = model.components(x)
        torch.testing.assert_close(path, torch.tensor([[[7.5], [8.]]]))
        losses = model.task_losses(x, y)
        expected_ce = -(np.log(.75)+np.log(.25))/2
        self.assertAlmostEqual(losses[0].item(), expected_ce, places=6)
        self.assertAlmostEqual(losses[1].item(), 0., places=6)
        self.assertAlmostEqual(losses[2].item(), (1.5**2+6**2)/2, places=6)
        # A zero deviation belongs to UP by the explicitly fixed binary tie.
        zero_y = x[:, -1:].repeat(1, 2, 1)
        self.assertAlmostEqual(model.task_losses(x, zero_y)[0].item(), -np.log(.75), places=6)

    def test_transformer_keeps_phase_and_branch_gradient(self):
        x = torch.tensor([[[0.], [1.], [4.], [-3.], [2.], [0.]]])
        reordered = x.clone(); reordered[:, 1:-1] = x[:, 1:-1].flip(1)
        for variant in (3, 4):
            model = CaReTSPhaseTransformer(6, 2, 1, variant, d_model=8, layers=1, heads=2, dropout=0.)
            model.eval()
            features = model.encoder(x.transpose(1, 2))
            changed_features = model.encoder(reordered.transpose(1, 2))
            self.assertGreater(torch.max(torch.abs(features-changed_features)).item(), 1e-4)
            # Equality of mean/last makes this a temporal-order rather than
            # anchoring test; classification and regression gradients exist.
            torch.testing.assert_close(x.mean(1), reordered.mean(1))
            y = torch.tensor([[[2.], [-1.]]])
            model.loss(x, y).backward()
            for branch in (model.classifier, model.regressor, model.encoder):
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in branch.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters()))

    def test_reload_and_batch_invariance(self):
        x = torch.randn(5, 8, 2)
        factories = [lambda: OCETimeSeries(8, 3, 2, [-4., -3.], [5., 6.]),
                     lambda: CaReTSPhaseTransformer(8, 3, 2, 3, d_model=8, layers=1, heads=2, dropout=0.),
                     lambda: CaReTSPhaseTransformer(8, 3, 2, 4, d_model=8, layers=1, heads=2, dropout=0.)]
        for factory in factories:
            original = factory().eval()
            reloaded = factory().eval()
            reloaded.load_state_dict(original.state_dict())
            with torch.no_grad():
                expected = original(x)
                torch.testing.assert_close(expected, reloaded(x))
                torch.testing.assert_close(expected, torch.cat([original(v[None]) for v in x]), atol=2e-6, rtol=2e-6)

    def test_invalid_ranges_and_ordinal_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            OCETimeSeries(8, 2, 1, [2.], [1.])
        with self.assertRaises(ValueError):
            OCETimeSeries(8, 2, 2, [1.], [2.])
        with self.assertRaises(ValueError):
            ordinal_cross_entropy(torch.tensor([[.1, .1]]), torch.tensor([[.5, .5]]))
        with self.assertRaises(ValueError):
            CaReTSPhaseTransformer(8, 2, 1, 3, d_model=7, heads=4)


if __name__ == "__main__":
    unittest.main()
