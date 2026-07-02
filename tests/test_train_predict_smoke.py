"""End-to-end smoke test: train a tiny one-hot CNN, then predict with it.

This runs the real train() and predict() entry points on a handful of synthetic
sequences. It is deliberately small (one epoch, a few filters) so it finishes in
seconds on CPU, but it still covers the training loop with gradient accumulation
turned on, checkpoint saving, and the prediction/metrics path including the
species-agreement and AUC handling.
"""

import glob
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

try:
    import torch  # noqa: F401

    from deeptaxa.cli import parse_args
    from deeptaxa.predict import predict
    from deeptaxa.train import train
    from tests._synth import make_dataset

    HAVE_DEPS = True
except Exception:
    HAVE_DEPS = False


def _args(argv):
    with mock.patch.object(sys, "argv", argv):
        return parse_args()


@unittest.skipUnless(HAVE_DEPS, "requires torch and the deeptaxa runtime deps")
class TestTrainPredictSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        cls.fasta, cls.taxonomy = make_dataset(root, n_seqs=48, seed=1)
        cls.train_out = os.path.join(root, "train")
        cls.pred_out = os.path.join(root, "pred")

        train_args = _args([
            "deeptaxa", "train",
            "--fasta-file", cls.fasta,
            "--taxonomy-file", cls.taxonomy,
            "--model-type", "cnn",
            "--encoding", "onehot",
            "--output-dir", cls.train_out,
            "--epochs", "1",
            "--batch-size", "8",
            "--accum-steps", "2",
            "--max-length", "64",
            "--num-filters", "8",
            "--embed-dim", "16",
            "--num-conv-layers", "1",
            "--kernel-sizes", "3",
            "--num-workers", "0",
            "--eval-every", "1",
            "--tokenizer-revision", "pinned-test-rev",
        ])
        train(train_args)

        checkpoints = glob.glob(os.path.join(cls.train_out, "checkpoints", "*.pt"))
        cls.checkpoint = max(checkpoints, key=os.path.getmtime) if checkpoints else None

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_training_wrote_a_checkpoint(self):
        self.assertIsNotNone(self.checkpoint, "training did not produce a checkpoint")
        self.assertTrue(os.path.exists(self.checkpoint))

    def test_checkpoint_records_tokenizer_revision(self):
        # The pinned revision has to survive into the checkpoint so prediction can
        # reload the exact same tokenizer.
        self.assertIsNotNone(self.checkpoint)
        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt.get("tokenizer_revision"), "pinned-test-rev")

    def test_predict_writes_outputs_and_metrics(self):
        self.assertIsNotNone(self.checkpoint, "no checkpoint to predict with")
        predict_args = _args([
            "deeptaxa", "predict",
            "--fasta-file", self.fasta,
            "--taxonomy-file", self.taxonomy,
            "--checkpoint", self.checkpoint,
            "--output-dir", self.pred_out,
            "--batch-size", "8",
            "--top-k", "3",
            "--num-workers", "0",
        ])
        predict(predict_args)

        predictions = glob.glob(os.path.join(self.pred_out, "*_deeptaxa_predictions.json"))
        self.assertEqual(len(predictions), 1)
        metrics_path = os.path.join(self.pred_out, "metrics.json")
        self.assertTrue(os.path.exists(metrics_path))

        with open(metrics_path) as fh:
            metrics = json.load(fh)
        # Every rank should carry the evaluation fields the metric path fills in.
        per_rank = metrics["performance_metrics"]
        self.assertTrue(per_rank)
        for stats in per_rank.values():
            for key in ("accuracy", "f1_score", "top_k_accuracy"):
                self.assertIn(key, stats)
            self.assertGreaterEqual(stats["accuracy"], 0.0)
            self.assertLessEqual(stats["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
