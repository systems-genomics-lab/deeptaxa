"""Tests for the label-agreement rule shared by exact and top-k accuracy."""

import unittest

try:
    from deeptaxa.predict import labels_agree

    HAVE_DEPS = True
except Exception:  # torch / sklearn / pandas not installed
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "requires the deeptaxa runtime deps")
class TestLabelsAgree(unittest.TestCase):
    def test_exact_match_any_rank(self):
        self.assertTrue(labels_agree("Firmicutes", "Firmicutes", "phylum"))
        self.assertFalse(labels_agree("Firmicutes", "Proteobacteria", "phylum"))

    def test_species_epithet_matches_on_boundary(self):
        self.assertTrue(labels_agree("coli", "Escherichia coli", "species"))
        self.assertTrue(labels_agree("coli", "s__coli", "species"))

    def test_species_rejects_empty_and_arbitrary_suffix(self):
        self.assertFalse(labels_agree("", "Escherichia coli", "species"))
        self.assertFalse(labels_agree("cillus", "Bacillus", "species"))

    def test_boundary_rule_is_species_only(self):
        self.assertFalse(labels_agree("coli", "Escherichia coli", "genus"))

    def test_top_k_accepts_whatever_exact_accepts(self):
        # Top-k accuracy is a superset of exact accuracy, so any label the exact
        # rule counts must also count when it appears among the top-k labels.
        cases = [
            ("coli", "Escherichia coli", "species"),
            ("coli", "s__coli", "species"),
            ("Firmicutes", "Firmicutes", "phylum"),
        ]
        for pred, true, rank in cases:
            exact = labels_agree(pred, true, rank)
            top_k = any(labels_agree(lbl, true, rank) for lbl in ["other", pred])
            self.assertTrue(exact)
            self.assertTrue(top_k, f"top-k missed a match exact accepted: {pred!r}")


if __name__ == "__main__":
    unittest.main()
