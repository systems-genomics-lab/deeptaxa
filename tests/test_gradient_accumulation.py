"""Regression test for the gradient-accumulation window.

Splitting a batch into micro-batches and accumulating the gradients has to give
the same result as running the whole batch at once. The earlier code cleared
the gradients on every micro-batch, so only the last one survived; this test
locks in the corrected window behavior.
"""

import unittest

try:
    import torch
    import torch.nn as nn

    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def _fresh_model():
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))


@unittest.skipUnless(HAVE_TORCH, "requires torch")
class TestGradientAccumulation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.criterion = nn.CrossEntropyLoss()
        self.inputs = torch.randn(8, 8)
        self.targets = torch.randint(0, 3, (8,))
        self.accum_steps = 4  # four micro-batches of two == one batch of eight

    def _full_batch_grads(self):
        model = _fresh_model()
        model.zero_grad()
        self.criterion(model(self.inputs), self.targets).backward()
        return [p.grad.clone() for p in model.parameters()]

    def _micro_batches(self):
        step = self.inputs.shape[0] // self.accum_steps
        return [
            (self.inputs[i * step:(i + 1) * step], self.targets[i * step:(i + 1) * step])
            for i in range(self.accum_steps)
        ]

    def test_accumulated_grads_match_full_batch(self):
        reference = self._full_batch_grads()
        model = _fresh_model()
        for i, (xb, yb) in enumerate(self._micro_batches()):
            if i % self.accum_steps == 0:
                model.zero_grad(set_to_none=True)
            (self.criterion(model(xb), yb) / self.accum_steps).backward()
        accumulated = [p.grad.clone() for p in model.parameters()]

        for got, want in zip(accumulated, reference):
            self.assertTrue(torch.allclose(got, want, atol=1e-5))

    def test_clearing_every_micro_batch_would_be_wrong(self):
        # Guards against a regression to the old behavior: clearing on every
        # micro-batch leaves only the last one's gradient, which does not match.
        reference = self._full_batch_grads()
        model = _fresh_model()
        for xb, yb in self._micro_batches():
            model.zero_grad(set_to_none=True)  # cleared every step (the old bug)
            (self.criterion(model(xb), yb) / self.accum_steps).backward()
        last_only = [p.grad.clone() for p in model.parameters()]

        differs = any(
            not torch.allclose(got, want, atol=1e-3)
            for got, want in zip(last_only, reference)
        )
        self.assertTrue(differs)


if __name__ == "__main__":
    unittest.main()
