"""Helpers for building tiny synthetic inputs for the core deeptaxa tests.

Everything here stays in one-hot mode so the tests never need the DNABERT
tokenizer or any network access. The sequences are random but seeded, and one
row is left without a species so the partial-lineage handling gets exercised.
"""

import os
import random

_BASES = "ACGT"
_GENERA = ["Bacillus", "Escherichia", "Lactobacillus", "Streptococcus"]
_PHYLUM = {
    "Bacillus": "Firmicutes",
    "Lactobacillus": "Firmicutes",
    "Streptococcus": "Firmicutes",
    "Escherichia": "Proteobacteria",
}


def _sequence(rng, length):
    return "".join(rng.choice(_BASES) for _ in range(length))


def make_dataset(directory, n_seqs=48, seed=0, blank_species=True):
    """Write a FASTA file and a matching taxonomy TSV into ``directory``.

    Returns the (fasta_path, taxonomy_path) pair. When ``blank_species`` is set,
    every fourth record is left with an empty species cell so a reader has to
    cope with a lineage that only goes partway down.
    """
    rng = random.Random(seed)
    fasta_path = os.path.join(directory, "reads.fasta")
    taxonomy_path = os.path.join(directory, "taxonomy.tsv")

    rows = []
    with open(fasta_path, "w") as fasta:
        for i in range(n_seqs):
            seq_id = f"seq{i}"
            fasta.write(f">{seq_id}\n{_sequence(rng, rng.randint(120, 260))}\n")
            genus = rng.choice(_GENERA)
            species = "" if (blank_species and i % 4 == 0) else f"{genus} sp{i % 3}"
            rows.append((seq_id, _PHYLUM[genus], genus, species))

    with open(taxonomy_path, "w") as tax:
        tax.write("sequence_id\tphylum\tgenus\tspecies\n")
        for row in rows:
            tax.write("\t".join(row) + "\n")

    return fasta_path, taxonomy_path
