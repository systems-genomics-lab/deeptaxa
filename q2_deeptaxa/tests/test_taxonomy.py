"""Tests for the QIIME <-> DeepTaxa taxonomy conversion helpers.

These tests are pure Python (pandas only) and do not require a QIIME 2
installation.
"""

import unittest

import pandas as pd

from q2_deeptaxa._taxonomy import (
    DEFAULT_RANKS,
    UNCLASSIFIED,
    assign_taxonomy,
    lineage_from_labels,
    predictions_to_dataframe,
    taxonomy_series_to_table,
)


class TestLineageFromLabels(unittest.TestCase):
    def test_full_lineage(self):
        labels = [
            "Bacteria",
            "Firmicutes",
            "Bacilli",
            "Bacillales",
            "Bacillaceae",
            "Bacillus",
            "subtilis",
        ]
        result = lineage_from_labels(DEFAULT_RANKS, labels)
        self.assertEqual(
            result,
            "d__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; "
            "f__Bacillaceae; g__Bacillus; s__subtilis",
        )

    def test_handles_none_and_whitespace(self):
        result = lineage_from_labels(["domain", "phylum"], [None, "  Firmicutes  "])
        self.assertEqual(result, "d__; p__Firmicutes")


class TestTaxonomySeriesToTable(unittest.TestCase):
    def test_prefixed_lineage(self):
        series = pd.Series(
            {
                "seq1": "d__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; "
                "f__Bacillaceae; g__Bacillus; s__subtilis",
            }
        )
        table = taxonomy_series_to_table(series)
        self.assertEqual(list(table.columns), ["sequence_id", *DEFAULT_RANKS])
        row = table.iloc[0]
        self.assertEqual(row["sequence_id"], "seq1")
        self.assertEqual(row["domain"], "Bacteria")
        self.assertEqual(row["species"], "subtilis")

    def test_missing_ranks_filled(self):
        series = pd.Series({"seq1": "d__Bacteria; p__Firmicutes"})
        table = taxonomy_series_to_table(series)
        row = table.iloc[0]
        self.assertEqual(row["phylum"], "Firmicutes")
        self.assertEqual(row["species"], UNCLASSIFIED)

    def test_empty_placeholder_is_unclassified(self):
        series = pd.Series({"seq1": "d__Bacteria; p__; c__Bacilli"})
        table = taxonomy_series_to_table(series)
        self.assertEqual(table.iloc[0]["phylum"], UNCLASSIFIED)

    def test_kingdom_prefix_maps_to_domain(self):
        # Legacy Greengenes 13_8 uses k__ for the top rank instead of d__.
        series = pd.Series(
            {"seq1": "k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; "
             "f__Bacillaceae; g__Bacillus; s__subtilis"}
        )
        table = taxonomy_series_to_table(series)
        row = table.iloc[0]
        self.assertEqual(row["domain"], "Bacteria")
        self.assertEqual(row["phylum"], "Firmicutes")

    def test_positional_lineage_without_prefixes(self):
        series = pd.Series(
            {"seq1": "Bacteria;Firmicutes;Bacilli;Bacillales;Bacillaceae;"
             "Bacillus;subtilis"}
        )
        table = taxonomy_series_to_table(series)
        row = table.iloc[0]
        self.assertEqual(row["domain"], "Bacteria")
        self.assertEqual(row["genus"], "Bacillus")
        self.assertEqual(row["species"], "subtilis")


class TestAssignTaxonomy(unittest.TestCase):
    RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    LABELS = ["Bacteria", "Firmicutes", "Bacilli", "Bacillales",
              "Bacillaceae", "Bacillus", "subtilis"]

    def test_no_threshold_keeps_full_lineage_and_min_confidence(self):
        scores = [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.30]
        taxon, conf = assign_taxonomy(self.RANKS, self.LABELS, scores, None)
        self.assertTrue(taxon.endswith("s__subtilis"))
        self.assertEqual(conf, 0.30)

    def test_threshold_below_all_keeps_full_lineage_deepest_score(self):
        scores = [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.30]
        taxon, conf = assign_taxonomy(self.RANKS, self.LABELS, scores, 0.2)
        self.assertTrue(taxon.endswith("s__subtilis"))
        self.assertEqual(conf, 0.30)

    def test_threshold_truncates_tail(self):
        scores = [0.99, 0.95, 0.90, 0.80, 0.70, 0.40, 0.20]
        taxon, conf = assign_taxonomy(self.RANKS, self.LABELS, scores, 0.5)
        self.assertEqual(
            taxon,
            "d__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; "
            "f__Bacillaceae",
        )
        self.assertEqual(conf, 0.70)

    def test_threshold_above_domain_is_unassigned(self):
        scores = [0.40, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
        taxon, conf = assign_taxonomy(self.RANKS, self.LABELS, scores, 0.5)
        self.assertEqual(taxon, "Unassigned")
        self.assertEqual(conf, 0.40)

    def test_missing_score_truncates_there(self):
        scores = [0.99, 0.95, None, 0.80, 0.70, 0.60, 0.30]
        taxon, conf = assign_taxonomy(self.RANKS, self.LABELS, scores, 0.5)
        self.assertEqual(taxon, "d__Bacteria; p__Firmicutes")
        self.assertEqual(conf, 0.95)


class TestPredictionsToDataframe(unittest.TestCase):
    RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    LABELS = ["Bacteria", "Firmicutes", "Bacilli", "Bacillales",
              "Bacillaceae", "Bacillus", "subtilis"]

    def _pred(self, scores):
        return {
            r: {"label": lbl, "raw_score": s}
            for r, lbl, s in zip(self.RANKS, self.LABELS, scores)
        }

    def test_full_lineage_no_threshold(self):
        preds = [self._pred([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.3])]
        df = predictions_to_dataframe(["s1"], preds, self.RANKS, None)
        self.assertEqual(df.index.name, "Feature ID")
        self.assertEqual(list(df.columns), ["Taxon", "Confidence"])
        self.assertTrue(df.loc["s1", "Taxon"].endswith("s__subtilis"))
        self.assertAlmostEqual(df.loc["s1", "Confidence"], 0.3)

    def test_truncates_at_threshold(self):
        preds = [self._pred([0.99, 0.95, 0.4, 0.3, 0.2, 0.1, 0.05])]
        df = predictions_to_dataframe(["s1"], preds, self.RANKS, 0.5)
        self.assertEqual(df.loc["s1", "Taxon"], "d__Bacteria; p__Firmicutes")
        self.assertAlmostEqual(df.loc["s1", "Confidence"], 0.95)

    def test_multiple_sequences_stay_aligned(self):
        preds = [self._pred([0.9] * 7), self._pred([0.8] * 7)]
        df = predictions_to_dataframe(["a", "b"], preds, self.RANKS, None)
        self.assertEqual(list(df.index), ["a", "b"])
        self.assertAlmostEqual(df.loc["b", "Confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
