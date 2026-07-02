"""Tests for the --save-best-only training option."""

import glob
import os
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch  # noqa: F401

    from deeptaxa.cli import parse_args
    from deeptaxa.train import train
    from tests._synth import make_dataset

    HAVE_DEPS = True
except Exception:
    HAVE_DEPS = False


def _train_args(argv):
    with mock.patch.object(sys, "argv", argv):
        return parse_args()


def _run(out_dir, fasta, taxonomy, epochs, extra):
    argv = [
        "deeptaxa", "train",
        "--fasta-file", fasta,
        "--taxonomy-file", taxonomy,
        "--model-type", "cnn",
        "--encoding", "onehot",
        "--output-dir", out_dir,
        "--epochs", str(epochs),
        "--batch-size", "8",
        "--max-length", "64",
        "--num-filters", "8",
        "--embed-dim", "16",
        "--num-conv-layers", "1",
        "--kernel-sizes", "3",
        "--num-workers", "0",
        "--eval-every", "1",
    ] + extra
    train(_train_args(argv))
    return len(glob.glob(os.path.join(out_dir, "checkpoints", "*.pt")))


@unittest.skipUnless(HAVE_DEPS, "requires torch and the deeptaxa runtime deps")
class TestSaveBestOnly(unittest.TestCase):
    def test_save_best_only_caps_checkpoint_count(self):
        epochs = 3
        with tempfile.TemporaryDirectory() as d:
            fasta, taxonomy = make_dataset(d, n_seqs=40, seed=3)
            baseline = _run(os.path.join(d, "all"), fasta, taxonomy, epochs, [])
            best = _run(os.path.join(d, "best"), fasta, taxonomy, epochs, ["--save-best-only"])
        # Every evaluated epoch saves by default; with the flag the first epoch
        # always saves and later epochs save only on improvement, so the count is
        # at least one and never more than the default run.
        self.assertEqual(baseline, epochs)
        self.assertGreaterEqual(best, 1)
        self.assertLessEqual(best, baseline)


if __name__ == "__main__":
    unittest.main()
