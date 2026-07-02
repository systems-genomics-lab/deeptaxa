"""Tests for the BERT masked mean-pooling helper, including the all-padding row."""

import unittest

try:
    import torch

    from deeptaxa.models.bert import masked_mean_pool

    HAVE_DEPS = True
except Exception:  # torch / transformers not installed
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "requires torch and the deeptaxa runtime deps")
class TestMaskedMeanPool(unittest.TestCase):
    def test_averages_only_valid_positions(self):
        # Row of four positions, two of them padding: the pooled value is the mean
        # of the first two, and the padded positions do not contribute.
        seq = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [99.0, 99.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0, 0]])
        pooled = masked_mean_pool(seq, mask)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[3.0, 6.0]])))

    def test_all_padding_row_is_zero_not_nan(self):
        # A row that is entirely padding must not divide by zero. It should pool to
        # zero and stay finite.
        seq = torch.randn(1, 5, 3)
        mask = torch.zeros(1, 5, dtype=torch.long)
        pooled = masked_mean_pool(seq, mask)
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.allclose(pooled, torch.zeros(1, 3)))

    def test_mixed_batch_stays_finite(self):
        seq = torch.randn(3, 6, 4)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        pooled = masked_mean_pool(seq, mask)
        self.assertTrue(torch.isfinite(pooled).all())


if __name__ == "__main__":
    unittest.main()
