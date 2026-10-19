import unittest
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from trend_lab.decoding import project_trajectory
from trend_lab.models import DirectionHead, EventAdapter, ResidualAdapter, build_base


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.x = torch.randn(5, 6, 2)
        self.base = torch.randn(5, 4, 2)

    def test_base_and_adapters_have_horizon_outputs_and_gradients(self):
        for name in ("linear", "mlp", "dlinear"):
            model = build_base(name, 6, 4, 2)
            path = model(self.x)
            self.assertEqual(path.shape, self.base.shape)
            path.sum().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
        head = DirectionHead(6, 4, 2, [1, 3])
        logits = head(self.x, self.base)
        self.assertEqual(logits.shape, (5, 2, 4, 2, 3))
        logits.sum().backward()
        residual = ResidualAdapter(6, 4, 2)
        self.assertEqual(residual(self.x, self.base).shape, self.base.shape)
        for enabled in (True, False):
            event = EventAdapter(6, 4, 2, use_events=enabled)
            result = event(self.x, self.base)
            self.assertEqual(result["path"].shape, self.base.shape)
            self.assertEqual(result["turn_logits"].shape, self.base.shape)
            result["path"].sum().backward()
            self.assertTrue(any(p.grad is not None for p in event.parameters()))

    def test_dlinear_matches_local_reference_forward_and_state_dict(self):
        default_root = Path(__file__).resolve().parents[3] / "TSLib" / "Time-Series-Library"
        library_root = Path(os.environ.get("TSLIB_SOURCE_ROOT", default_root))
        reference_file = library_root / "models" / "DLinear.py"
        if not reference_file.is_file():
            self.skipTest(f"local Time-Series-Library reference not found at {reference_file}")

        sys.path.insert(0, str(library_root))
        try:
            reference_module = importlib.import_module("models.DLinear")
        finally:
            sys.path.remove(str(library_root))
        reference = reference_module.Model(SimpleNamespace(
            task_name="long_term_forecast", seq_len=6, pred_len=4,
            moving_avg=25, enc_in=2), individual=False)
        candidate = build_base("dlinear", 6, 4, 2)
        result = candidate.load_state_dict(reference.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(set(candidate.state_dict()), set(reference.state_dict()))

        x = torch.randn(7, 6, 2)
        with torch.no_grad():
            expected = reference(x, None, None, None)
            actual = candidate(x)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_projection_identity_bound_and_batch_invariance(self):
        eps = torch.full((2, 2), 0.05)
        probs = torch.zeros(5, 2, 4, 2, 3)
        probs[..., 2] = 0.9
        probs[..., 1] = 0.1
        ident = project_trajectory(self.x, self.base, probs, [1, 3], eps, strength=0)
        self.assertTrue(torch.equal(ident, self.base))
        projected = project_trajectory(self.x, self.base, probs, [1, 3], eps, steps=12, max_rms=0.08)
        rms = (projected - self.base).square().mean((1, 2)).sqrt()
        self.assertTrue(torch.all(rms <= 0.080001))
        pieces = torch.cat([
            project_trajectory(self.x[i:i + 2], self.base[i:i + 2], probs[i:i + 2], [1, 3], eps,
                               steps=12, max_rms=0.08)
            for i in (0, 2, 4)
        ])
        self.assertTrue(torch.allclose(projected, pieces, atol=1e-6, rtol=1e-6))

    def test_up_posterior_pushes_first_step_up(self):
        history = torch.zeros(1, 3, 1)
        base = torch.zeros(1, 2, 1)
        probs = torch.tensor([[[[[0.01, 0.01, 0.98]], [[0.01, 0.01, 0.98]]]]])
        out = project_trajectory(history, base, probs, [1], torch.tensor([[0.1]]),
                                 strength=2, steps=30, max_rms=10)
        self.assertGreater(out[0, 0, 0].item(), 0)

    def test_projection_works_under_no_grad_without_input_gradients(self):
        history = self.x.detach().requires_grad_()
        base = self.base.detach().requires_grad_()
        probs = torch.zeros(5, 2, 4, 2, 3, requires_grad=True)
        with torch.no_grad():
            probs[..., 2] = 0.9
            probs[..., 1] = 0.1
            out = project_trajectory(history, base, probs, [1, 3], torch.full((2, 2), 0.05),
                                     steps=3)
        self.assertEqual(out.shape, self.base.shape)
        self.assertFalse(out.requires_grad)
        self.assertIsNone(history.grad)
        self.assertIsNone(base.grad)
        self.assertIsNone(probs.grad)


if __name__ == "__main__":
    unittest.main()
