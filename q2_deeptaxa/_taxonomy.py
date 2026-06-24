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

# Some references label the top rank kingdom (``k__``, e.g. legacy Greengenes
# 13_8) instead of domain (``d__``). Treat the two as interchangeable so the
# domain column is filled either way.
RANK_ALIASES = {
    "domain": ("domain", "kingdom"),
    "kingdom": ("kingdom", "domain"),
}


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


def assign_taxonomy(ranks, labels, scores, confidence=None):
    """Build a ``(taxon, confidence)`` pair from one sequence's per-rank output.

    Parameters
    ----------
    ranks : sequence of str
        Ordered taxonomic rank names.
    labels : sequence of str
        Predicted label per rank, aligned with ``ranks``.
    scores : sequence of float or None
        Softmax probability per rank, aligned with ``ranks``. ``None`` marks a
        rank with no score.
    confidence : float or None
        If ``None``, keep the whole lineage and report the lowest per-rank score
        as a cautious whole-lineage confidence. If a threshold is given, keep
        ranks from the top down only while their score stays at or above it, and
        report the score of the deepest kept rank. A sequence whose very first
        rank is below the threshold becomes ``Unassigned``.
    """
    if confidence is None:
        taxon = lineage_from_labels(ranks, labels)
        present = [s for s in scores if s is not None]
        return taxon, (min(present) if present else float("nan"))

    kept_labels, kept_scores = [], []
    for label, score in zip(labels, scores):
        if score is None or score < confidence:
            break
        kept_labels.append(label)
        kept_scores.append(score)

    if kept_labels:
        taxon = lineage_from_labels(ranks[: len(kept_labels)], kept_labels)
        return taxon, kept_scores[-1]

    first = scores[0] if scores else None
    return "Unassigned", (first if first is not None else float("nan"))


def predictions_to_dataframe(seq_ids, predictions, ranks, confidence=None):
    """Turn DeepTaxa per-sequence predictions into a taxonomy DataFrame.

    Parameters
    ----------
    seq_ids : sequence of str
        Feature ids, aligned with ``predictions``.
    predictions : sequence of dict
        One dict per sequence, keyed by rank name, each value a dict with at
        least ``label`` and ``raw_score``.
    ranks : sequence of str
        Ordered taxonomic rank names.
    confidence : float or None
        Passed through to :func:`assign_taxonomy` (None keeps the full lineage).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``Feature ID`` with ``Taxon`` and ``Confidence`` columns.
    """
    import pandas as pd

    taxa, confidences = [], []
    for pred in predictions:
        labels, scores = [], []
        for rank in ranks:
            entry = pred.get(rank, {})
            labels.append(entry.get("label", ""))
            score = entry.get("raw_score")
            scores.append(float(score) if score is not None else None)
        taxon, conf = assign_taxonomy(ranks, labels, scores, confidence)
        taxa.append(taxon)
        confidences.append(conf)

    result = pd.DataFrame(
        {"Taxon": taxa, "Confidence": confidences}, index=pd.Index(seq_ids)
    )
    result.index.name = "Feature ID"
    return result


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
            value = None
            for alias in RANK_ALIASES.get(rank, (rank,)):
                if alias in parsed:
                    value = parsed[alias]
                    break
            if value is not None:
                row[rank] = value
            elif not parsed and i < len(positional):
                # No prefixes anywhere: fall back to positional assignment.
                row[rank] = positional[i]
            else:
                row[rank] = UNCLASSIFIED
        rows.append(row)
    return pd.DataFrame(rows, columns=["sequence_id", *ranks])
