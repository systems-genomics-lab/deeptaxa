"""Tests for TaxonomyDataset in one-hot mode (no tokenizer or network needed)."""

import tempfile
import unittest

try:
    import torch

    from deeptaxa.dataset import TaxonomyDataset, custom_collate_fn
    from tests._synth import make_dataset

    HAVE_DEPS = True
except Exception:  # torch / biopython / transformers not installed
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "requires torch and the deeptaxa runtime deps")
class TestTaxonomyDataset(unittest.TestCase):
    def _dataset(self, directory, **kwargs):
        fasta, taxonomy = make_dataset(directory, **kwargs)
        return TaxonomyDataset(
            fasta_file=fasta,
            taxonomy_file=taxonomy,
            max_length=64,
            encoding="onehot",
        )

    def test_length_matches_fasta(self):
        with tempfile.TemporaryDirectory() as d:
            ds = self._dataset(d, n_seqs=30)
            self.assertEqual(len(ds), 30)

    def test_onehot_item_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            ds = self._dataset(d)
            item = ds[0]
            # One-hot input is [channels, length] so Conv1d can read it directly.
            self.assertEqual(tuple(item["input_ids"].shape), (4, 64))
            self.assertEqual(tuple(item["attention_mask"].shape), (64,))
            self.assertEqual(item["labels"].shape[0], len(ds.taxonomic_ranks))

    def test_onehot_encoding_values(self):
        with tempfile.TemporaryDirectory() as d:
            fasta, taxonomy = make_dataset(d)
            # Overwrite the first record with a sequence we control, including an
            # ambiguous base that has no one-hot column.
            with open(fasta, "w") as fh:
                fh.write(">seq0\nACGTN\n")
            with open(taxonomy, "w") as fh:
                fh.write("sequence_id\tphylum\tgenus\tspecies\n")
                fh.write("seq0\tFirmicutes\tBacillus\tBacillus subtilis\n")
            ds = TaxonomyDataset(
                fasta_file=fasta, taxonomy_file=taxonomy, max_length=8, encoding="onehot"
            )
            x = ds[0]["input_ids"]
            # A, C, G, T land on channels 0..3 at positions 0..3.
            for pos, channel in enumerate(range(4)):
                self.assertEqual(x[channel, pos].item(), 1.0)
            # The ambiguous 'N' is dropped, so its column stays all zero.
            self.assertEqual(x[:, 4].sum().item(), 0.0)
            # Padding past the sequence is zero and masked out.
            self.assertEqual(x[:, 5:].sum().item(), 0.0)
            mask = ds[0]["attention_mask"]
            self.assertEqual(mask[:5].sum().item(), 5)
            self.assertEqual(mask[5:].sum().item(), 0)

    def test_partial_lineage_is_unclassified(self):
        # A blank species cell must not raise and should read back as
        # 'Unclassified' rather than the string 'nan'.
        with tempfile.TemporaryDirectory() as d:
            ds = self._dataset(d, n_seqs=48, blank_species=True)
            species_level = ds.taxonomic_ranks.index("species")
            self.assertIn("Unclassified", ds.level_label2id[species_level])
            for raw in ds.raw_labels:
                self.assertNotEqual(raw["species"].lower(), "nan")

    def test_collate_stacks_batch(self):
        with tempfile.TemporaryDirectory() as d:
            ds = self._dataset(d, n_seqs=10)
            batch = custom_collate_fn([ds[i] for i in range(4)])
            self.assertEqual(tuple(batch["input_ids"].shape), (4, 4, 64))
            self.assertEqual(tuple(batch["labels"].shape), (4, len(ds.taxonomic_ranks)))
            self.assertEqual(len(batch["seq_ids"]), 4)


if __name__ == "__main__":
    unittest.main()
