"""End-to-end tests for the classify and fit actions.

These exercise the full action wiring (namespace construction, the DeepTaxa
input/output file contract, checkpoint selection, state stripping, and artifact
output) with ``deeptaxa.predict``/``deeptaxa.train`` mocked, so no real model,
GPU, or network is needed. They require a QIIME 2 environment with torch, so
they are skipped where those are unavailable.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

try:
    import qiime2  # noqa: F401
    import torch
    from q2_types.feature_data import DNAFASTAFormat

    from q2_deeptaxa import _methods
    from q2_deeptaxa._formats import DeepTaxaModelDirectoryFormat

    HAVE_DEPS = True
except ImportError:  # pragma: no cover - runs only without the QIIME stack
    HAVE_DEPS = False

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
LABELS = ["Bacteria", "Firmicutes", "Bacilli", "Bacillales",
          "Bacillaceae", "Bacillus", "subtilis"]


def _prediction(scores):
    return {
        rank: {"label": label, "raw_score": score}
        for rank, label, score in zip(RANKS, LABELS, scores)
    }


@unittest.skipUnless(HAVE_DEPS, "requires a QIIME 2 environment with torch")
class TestClassifyAction(unittest.TestCase):
    def _model(self, tmpdir):
        d = os.path.join(tmpdir, "model")
        os.makedirs(d)
        torch.save({"taxonomic_ranks": RANKS}, os.path.join(d, "model.pt"))
        return DeepTaxaModelDirectoryFormat(d, mode="r")

    def _reads(self, tmpdir):
        fp = os.path.join(tmpdir, "seqs.fasta")
        with open(fp, "w") as fh:
            fh.write(">s1\nACGTACGT\n>s2\nTTTTGGGG\n")
        return DNAFASTAFormat(fp, mode="r")

    def test_parses_predictions_and_truncates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._model(tmpdir)
            reads = self._reads(tmpdir)

            def fake_predict(args):
                output = {
                    "sequence_ids": ["s1", "s2"],
                    "predictions": [
                        _prediction([0.99, 0.95, 0.4, 0.3, 0.2, 0.1, 0.05]),
                        _prediction([0.9] * 7),
                    ],
                }
                fp = os.path.join(args.output_dir, "stub_deeptaxa_predictions.json")
                with open(fp, "w") as fh:
                    json.dump(output, fh)

            with mock.patch("deeptaxa.predict.predict", fake_predict):
                df = _methods.classify(reads, model, confidence=0.5, num_workers=0)

        self.assertEqual(df.index.name, "Feature ID")
        self.assertEqual(list(df.index), ["s1", "s2"])
        self.assertEqual(list(df.columns), ["Taxon", "Confidence"])
        # s1 trims at class (0.4 < 0.5); confidence is the deepest kept (phylum).
        self.assertEqual(df.loc["s1", "Taxon"], "d__Bacteria; p__Firmicutes")
        self.assertAlmostEqual(df.loc["s1", "Confidence"], 0.95)
        self.assertTrue(df.loc["s2", "Taxon"].endswith("s__subtilis"))

    def test_builds_expected_namespace_for_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = self._model(tmpdir)
            reads = self._reads(tmpdir)
            captured = {}

            def fake_predict(args):
                captured["args"] = args
                fp = os.path.join(args.output_dir, "x_deeptaxa_predictions.json")
                with open(fp, "w") as fh:
                    json.dump({"sequence_ids": [], "predictions": []}, fh)

            with mock.patch("deeptaxa.predict.predict", fake_predict):
                _methods.classify(
                    reads, model, batch_size=8, top_k=2, num_workers=0, seed=11
                )

        args = captured["args"]
        self.assertEqual(args.fasta_file, str(reads))
        self.assertEqual(args.checkpoint, os.path.join(str(model), "model.pt"))
        self.assertIsNone(args.taxonomy_file)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.top_k, 2)
        self.assertEqual(args.seed, 11)
        self.assertFalse(args.tabular)
        # Attributes predict reads only through the seeded defaults must exist.
        for attr in ("level_temperatures", "confidence_threshold", "verbose"):
            self.assertTrue(hasattr(args, attr))


@unittest.skipUnless(HAVE_DEPS, "requires a QIIME 2 environment with torch")
class TestFitAction(unittest.TestCase):
    def test_selects_best_checkpoint_and_strips_state(self):
        import pandas as pd

        taxonomy = pd.Series(
            {
                "s1": "d__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; "
                "f__Bacillaceae; g__Bacillus; s__subtilis",
            }
        )

        def fake_train(args):
            ckpt_dir = os.path.join(args.output_dir, "checkpoints")
            metrics_dir = os.path.join(args.output_dir, "metrics")
            os.makedirs(ckpt_dir)
            os.makedirs(metrics_dir)
            # Epoch 2 has the lowest validation loss but is not the last epoch.
            for epoch, val_loss in {1: 0.5, 2: 0.2, 3: 0.4}.items():
                torch.save(
                    {
                        "state_dict": {},
                        "taxonomic_ranks": RANKS,
                        "marker": epoch,
                        "optimizer_state_dict": {"x": 1},
                        "scheduler_state_dict": {"x": 1},
                        "scaler_state_dict": {"x": 1},
                        "rng_state": {"x": 1},
                    },
                    os.path.join(ckpt_dir, f"deeptaxa_run_epoch{epoch}.pt"),
                )
                with open(
                    os.path.join(metrics_dir, f"deeptaxa_run_epoch{epoch}.json"),
                    "w",
                ) as fh:
                    json.dump(
                        {"performance_metrics": {"validation_loss": val_loss}}, fh
                    )

        with tempfile.TemporaryDirectory() as tmpdir:
            reads_fp = os.path.join(tmpdir, "ref.fasta")
            with open(reads_fp, "w") as fh:
                fh.write(">s1\nACGTACGT\n")
            reads = DNAFASTAFormat(reads_fp, mode="r")

            with mock.patch("deeptaxa.train.train", fake_train):
                result = _methods.fit(reads, taxonomy, epochs=3, num_workers=0)

            saved = torch.load(
                os.path.join(str(result), "model.pt"), weights_only=False
            )

        self.assertEqual(saved["marker"], 2)  # lowest-val-loss epoch chosen
        self.assertIn("state_dict", saved)
        self.assertIn("taxonomic_ranks", saved)
        for key in (
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "rng_state",
        ):
            self.assertNotIn(key, saved)

    def test_writes_taxonomy_table_for_train(self):
        import pandas as pd

        taxonomy = pd.Series({"s1": "k__Bacteria; p__Firmicutes"})
        captured = {}

        def fake_train(args):
            with open(args.taxonomy_file) as fh:
                captured["header"] = fh.readline().strip()
                captured["row"] = fh.readline().strip()
            ckpt_dir = os.path.join(args.output_dir, "checkpoints")
            os.makedirs(ckpt_dir)
            torch.save(
                {"state_dict": {}, "taxonomic_ranks": RANKS},
                os.path.join(ckpt_dir, "deeptaxa_run_epoch1.pt"),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            reads_fp = os.path.join(tmpdir, "ref.fasta")
            with open(reads_fp, "w") as fh:
                fh.write(">s1\nACGTACGT\n")
            reads = DNAFASTAFormat(reads_fp, mode="r")

            with mock.patch("deeptaxa.train.train", fake_train):
                _methods.fit(reads, taxonomy, epochs=1, num_workers=0)

        self.assertEqual(
            captured["header"],
            "sequence_id\tdomain\tphylum\tclass\torder\tfamily\tgenus\tspecies",
        )
        # k__ maps to domain; absent ranks are Unclassified.
        self.assertEqual(
            captured["row"],
            "s1\tBacteria\tFirmicutes\tUnclassified\tUnclassified\t"
            "Unclassified\tUnclassified\tUnclassified",
        )


if __name__ == "__main__":
    unittest.main()
