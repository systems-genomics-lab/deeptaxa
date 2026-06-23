"""
Helpers for converting between QIIME 2 taxonomy strings and DeepTaxa's
per-rank taxonomy table.

QIIME 2 ``FeatureData[Taxonomy]`` carries one semicolon-delimited lineage per
feature, e.g. ``d__Bacteria; p__Firmicutes; c__Bacilli; ...; s__subtilis``.
DeepTaxa instead consumes/produces a wide TSV with one column per rank
(``sequence_id``, ``domain``, ``phylum``, ..., ``species``). These helpers map
between the two representations.
"""

# Canonical single-letter prefixes used by Greengenes/SILVA-style lineages,
# keyed by the DeepTaxa rank name.
RANK_TO_PREFIX = {
    "domain": "d",
    "superkingdom": "d",
    "kingdom": "k",
    "phylum": "p",
    "class": "c",
    "order": "o",
    "family": "f",
    "genus": "g",
    "species": "s",
}

# Reverse lookup, prefix -> rank name, for parsing incoming lineages.
PREFIX_TO_RANK = {
    "d": "domain",
    "k": "kingdom",
    "p": "phylum",
    "c": "class",
    "o": "order",
    "f": "family",
    "g": "genus",
    "s": "species",
}

# Default rank schema DeepTaxa trains on when none can be inferred from the
# incoming lineages.
DEFAULT_RANKS = (
    "domain",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
)

UNCLASSIFIED = "Unclassified"


def prefix_for_rank(rank):
    """Return the single-letter lineage prefix for a DeepTaxa rank name."""
    return RANK_TO_PREFIX.get(rank.lower(), rank[:1].lower())


def lineage_from_labels(ranks, labels):
    """Build a QIIME lineage string from ordered ranks and predicted labels.

    Parameters
    ----------
    ranks : sequence of str
        Ordered taxonomic rank names (e.g. ``["domain", ..., "species"]``).
    labels : sequence of str
        Predicted label for each rank, aligned with ``ranks``.
    """
    parts = []
    for rank, label in zip(ranks, labels):
        label = "" if label is None else str(label).strip()
        parts.append(f"{prefix_for_rank(rank)}__{label}")
    return "; ".join(parts)


def _parse_lineage(taxon):
    """Parse one QIIME lineage string into ``{rank_name: label}``.

    Handles both prefixed lineages (``d__Bacteria; p__...``) and bare,
    positional lineages (``Bacteria;Firmicutes;...``). Empty placeholders such
    as ``g__`` map to ``Unclassified``.
    """
    tokens = [t.strip() for t in str(taxon).split(";") if t.strip() != ""]
    parsed = {}
    positional = []
    for token in tokens:
        if "__" in token:
            prefix, _, value = token.partition("__")
            rank = PREFIX_TO_RANK.get(prefix.strip().lower())
            value = value.strip()
            if rank is not None:
                parsed[rank] = value if value else UNCLASSIFIED
            else:
                positional.append(value if value else UNCLASSIFIED)
        else:
            positional.append(token)
    return parsed, positional


def taxonomy_series_to_table(taxonomy, ranks=DEFAULT_RANKS):
    """Convert a QIIME taxonomy Series to a DeepTaxa wide table.

    Parameters
    ----------
    taxonomy : pandas.Series
        Index is the feature/sequence id, values are lineage strings.
    ranks : sequence of str
        Rank columns to emit, in order.

    Returns
    -------
    pandas.DataFrame
        Columns ``["sequence_id", *ranks]``; missing ranks filled with
        ``Unclassified``.
    """
    import pandas as pd

    rows = []
    for seq_id, taxon in taxonomy.items():
        parsed, positional = _parse_lineage(taxon)
        row = {"sequence_id": seq_id}
        for i, rank in enumerate(ranks):
            if rank in parsed:
                row[rank] = parsed[rank]
            elif i < len(positional):
                # No prefixes present: fall back to positional assignment.
                row[rank] = positional[i]
            else:
                row[rank] = UNCLASSIFIED
        rows.append(row)
    return pd.DataFrame(rows, columns=["sequence_id", *ranks])
