"""Tests for the fit checkpoint-selection helpers (no QIIME or torch needed)."""

import json
import os
import tempfile
import unittest

from q2_deeptaxa._checkpoint import (
    TRAINING_ONLY_KEYS,
    choose_checkpoint,
    epoch_of,
    read_validation_losses,
    strip_training_state,
)


def _write_metrics(directory, epoch, val_loss):
    fp = os.path.join(directory, f"deeptaxa_run_epoch{epoch}.json")
    with open(fp, "w") as fh:
        json.dump({"performance_metrics": {"validation_loss": val_loss}}, fh)
    return fp


class TestEpochOf(unittest.TestCase):
    def test_parses_epoch(self):
        self.assertEqual(epoch_of("/x/deeptaxa_run_epoch7.pt"), 7)
        self.assertEqual(epoch_of("deeptaxa_run_epoch12.json"), 12)

    def test_unparsable_is_minus_one(self):
        self.assertEqual(epoch_of("model.pt"), -1)


class TestStripTrainingState(unittest.TestCase):
    def test_removes_training_state_keeps_rest(self):
        ckpt = {
            "state_dict": {"w": 1},
            "taxonomic_ranks": ["domain"],
            "optimizer_state_dict": 1,
            "scheduler_state_dict": 1,
            "scaler_state_dict": 1,
            "rng_state": 1,
        }
        out = strip_training_state(ckpt)
        for key in TRAINING_ONLY_KEYS:
            self.assertNotIn(key, out)
        self.assertIn("state_dict", out)
        self.assertIn("taxonomic_ranks", out)


class TestChooseCheckpoint(unittest.TestCase):
    def test_picks_lowest_validation_loss(self):
        with tempfile.TemporaryDirectory() as d:
            ckpts = [f"deeptaxa_run_epoch{e}.pt" for e in (1, 2, 3)]
            metrics = [
                _write_metrics(d, 1, 0.5),
                _write_metrics(d, 2, 0.2),
                _write_metrics(d, 3, 0.4),
            ]
            self.assertTrue(choose_checkpoint(ckpts, metrics).endswith("epoch2.pt"))

    def test_fallback_to_most_trained_without_metrics(self):
        ckpts = [f"deeptaxa_run_epoch{e}.pt" for e in (1, 5, 3)]
        self.assertTrue(choose_checkpoint(ckpts, []).endswith("epoch5.pt"))

    def test_all_zero_loss_falls_back_to_most_trained(self):
        with tempfile.TemporaryDirectory() as d:
            ckpts = [f"deeptaxa_run_epoch{e}.pt" for e in (1, 2)]
            metrics = [_write_metrics(d, 1, 0.0), _write_metrics(d, 2, 0.0)]
            self.assertTrue(choose_checkpoint(ckpts, metrics).endswith("epoch2.pt"))

    def test_ignores_metrics_without_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            ckpts = [f"deeptaxa_run_epoch{e}.pt" for e in (1, 2)]
            metrics = [
                _write_metrics(d, 1, 0.5),
                _write_metrics(d, 2, 0.6),
                _write_metrics(d, 9, 0.1),  # best loss but no checkpoint
            ]
            self.assertTrue(choose_checkpoint(ckpts, metrics).endswith("epoch1.pt"))

    def test_tie_breaks_to_lower_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            ckpts = [f"deeptaxa_run_epoch{e}.pt" for e in (1, 2)]
            metrics = [_write_metrics(d, 1, 0.3), _write_metrics(d, 2, 0.3)]
            self.assertTrue(choose_checkpoint(ckpts, metrics).endswith("epoch1.pt"))


class TestReadValidationLosses(unittest.TestCase):
    def test_skips_malformed_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            good = _write_metrics(d, 1, 0.3)
            bad = os.path.join(d, "deeptaxa_run_epoch2.json")
            with open(bad, "w") as fh:
                fh.write("not json")
            no_key = os.path.join(d, "deeptaxa_run_epoch3.json")
            with open(no_key, "w") as fh:
                json.dump({"foo": 1}, fh)
            self.assertEqual(
                read_validation_losses([good, bad, no_key]), [(0.3, 1)]
            )


if __name__ == "__main__":
    unittest.main()
