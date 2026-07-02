"""Tests for per-rank AUC, including the novel-label alignment fix."""

import unittest

try:
    import numpy as np

    from deeptaxa.predict import rank_auc

    HAVE_DEPS = True
except Exception:
    HAVE_DEPS = False


def _probs(true_ids, seed=0):
    """One probability row per known-label sample, in loop order (novel skipped)."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in true_ids:
        if t < 0:
            continue  # novel labels never get a stored probability row
        p = rng.random(2)
        rows.append(p / p.sum())
    return rows


@unittest.skipUnless(HAVE_DEPS, "requires numpy and the deeptaxa runtime deps")
class TestRankAuc(unittest.TestCase):
    def test_computes_with_novel_labels(self):
        # A -1 sentinel sits among the true labels, but probabilities exist only
        # for the known rows. The old code compared the full true array against the
        # shorter score array and fell back to N/A; this must now return a value.
        true_ids = [0, 1, -1, 0, 1, 1]
        auc = rank_auc(true_ids, _probs(true_ids), max_classes=100)
        self.assertIsNotNone(auc)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_matches_manual_score_on_known_rows(self):
        from sklearn.metrics import roc_auc_score

        true_ids = [0, 1, -1, 0, 1, 1]
        rows = _probs(true_ids)
        known = np.array([t for t in true_ids if t >= 0])
        manual = roc_auc_score(known, np.stack(rows)[:, 1])
        self.assertAlmostEqual(rank_auc(true_ids, rows, 100), float(manual), places=6)

    def test_none_when_single_class(self):
        true_ids = [0, 0, 0]
        self.assertIsNone(rank_auc(true_ids, _probs(true_ids), max_classes=100))

    def test_none_when_too_many_classes(self):
        true_ids = [0, 1, 2, 3]
        self.assertIsNone(rank_auc(true_ids, _probs(true_ids), max_classes=2))

    def test_none_when_no_probability_rows(self):
        self.assertIsNone(rank_auc([0, 1, 0, 1], [], max_classes=100))


if __name__ == "__main__":
    unittest.main()
