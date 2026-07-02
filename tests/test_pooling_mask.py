"""Tests that padded positions can be kept out of the CNN max pooling.

Real one-hot padding is all zeros, so the leak the mask fixes is not in the
input but in the pooled output: after the convolution, padded positions carry a
bias-driven activation that can win the global max. These tests use realistic
zero padding and check that masking removes that contribution.
"""

import unittest

try:
    import torch

    from deeptaxa.models import CNNClassifier

    HAVE_DEPS = True
except Exception:  # torch / transformers not installed
    HAVE_DEPS = False


def _onehot(length, valid=6):
    """One real region of length ``valid`` followed by zero padding."""
    x = torch.zeros(1, 4, length)
    for pos in range(valid):
        x[:, pos % 4, pos] = 1.0
    mask = torch.zeros(1, length, dtype=torch.long)
    mask[:, :valid] = 1
    return x, mask


@unittest.skipUnless(HAVE_DEPS, "requires torch and the deeptaxa runtime deps")
class TestPoolingMask(unittest.TestCase):
    def _model(self, mask_padding=True):
        torch.manual_seed(0)
        model = CNNClassifier(
            tokenizer_name="unused-in-onehot",
            num_labels_per_level={0: 3},
            embed_dim=16,
            num_filters=8,
            kernel_sizes=[3],
            num_conv_layers=1,
            hidden_dropout_prob=0.0,
            input_mode="onehot",
            mask_padding=mask_padding,
        )
        model.eval()
        return model

    def test_masked_pooling_invariant_to_padding_amount(self):
        # The same sequence padded to two different lengths must classify the
        # same way once padding is masked out of the pool.
        model = self._model(mask_padding=True)
        x_short, m_short = _onehot(length=10)
        x_long, m_long = _onehot(length=24)
        with torch.no_grad():
            short = model(x_short, m_short)["0"]
            long = model(x_long, m_long)["0"]
        self.assertTrue(torch.allclose(short, long, atol=1e-5))

    def test_masking_excludes_padded_positions_from_max(self):
        # Force negative conv weights with a positive bias so padded positions
        # (all-zero input) produce a larger activation than the real region.
        # Masking must then change the pooled result.
        model = self._model(mask_padding=True)
        conv = model.conv_stacks[0][0].conv[0]
        with torch.no_grad():
            conv.weight.fill_(-1.0)
            conv.bias.fill_(0.5)
        x, mask = _onehot(length=12)
        with torch.no_grad():
            masked = model(x, mask)["0"]
            model.mask_padding = False
            unmasked = model(x, mask)["0"]
        self.assertFalse(torch.allclose(masked, unmasked, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
