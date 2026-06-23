"""
Methods for the q2-deeptaxa plugin.

Both actions are wrappers around the existing DeepTaxa pipelines
(:func:`deeptaxa.predict.predict` and :func:`deeptaxa.train.train`). Those
pipelines take an :class:`argparse.Namespace` and read or write files on disk.
So each wrapper builds a namespace from the values in
:data:`deeptaxa.config.DEFAULT_CONFIG`, runs the pipeline inside a temporary
directory, and then converts the result to or from the QIIME 2 view it needs.
"""

import json
import os
import tempfile
from argparse import Namespace
from glob import glob

import pandas as pd
from q2_types.feature_data import DNAFASTAFormat

from deeptaxa.config import DEFAULT_CONFIG

from ._formats import DeepTaxaModelDirectoryFormat
from ._taxonomy import (
    DEFAULT_RANKS,
    assign_taxonomy,
    taxonomy_series_to_table,
)


def _base_namespace(**overrides):
    """Build an argparse.Namespace seeded from DEFAULT_CONFIG.

    DeepTaxa's train/predict pipelines read many attributes, most via
    ``getattr(args, name, DEFAULT_CONFIG[name])`` but several directly. Seeding
    from DEFAULT_CONFIG guarantees every config-backed attribute exists; the
    non-config flags accessed via getattr (e.g. ``resume``, ``no_level_weights``)
    are given explicit safe defaults here.
    """
    params = dict(DEFAULT_CONFIG)
    params.update(
        {
            "command": None,
            "verbose": False,
            "resume": None,
            "init_weights": None,
            "reset_lr_on_resume": False,
            "no_level_weights": False,
            "class_weights": "none",
            "loss_type": DEFAULT_CONFIG["loss_type"],
            "export_sequence_embeddings": False,
            "export_permutation_importance": False,
            "perm_region_size": None,
            "tabular": False,
            "tabular_fields": "predicted,raw_score,entropy,true,agreement",
            "use_raw_labels_for_true": False,
        }
    )
    params.update(overrides)
    return Namespace(**params)


def _checkpoint_path(classifier: DeepTaxaModelDirectoryFormat) -> str:
    """Absolute path to the model.pt inside a DeepTaxaModel artifact."""
    return os.path.join(str(classifier), "model.pt")


def _read_ranks(checkpoint_path: str):
    """Return the ordered taxonomic ranks stored in a checkpoint."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ranks = checkpoint.get("taxonomic_ranks")
    return list(ranks) if ranks else list(DEFAULT_RANKS)


def classify(
    reads: DNAFASTAFormat,
    classifier: DeepTaxaModelDirectoryFormat,
    confidence: float = "disable",
    batch_size: int = DEFAULT_CONFIG["batch_size"],
    top_k: int = DEFAULT_CONFIG["top_k"],
    num_workers: int = DEFAULT_CONFIG["num_workers"],
    seed: int = DEFAULT_CONFIG["seed"],
) -> pd.DataFrame:
    """Classify sequences with a trained DeepTaxa model.

    Parameters
    ----------
    reads : DNAFASTAFormat
        Sequences to classify (FeatureData[Sequence]).
    classifier : DeepTaxaModelDirectoryFormat
        Trained DeepTaxa model.
    confidence : float or "disable"
        Confidence threshold for limiting how deep a lineage is reported.
        ``"disable"`` (the default) keeps all ranks. A float between 0 and 1
        trims the lineage at the first rank whose score falls below it, so only
        the confident part of the assignment is kept.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``Feature ID`` with ``Taxon`` and ``Confidence`` columns,
        which q2-types turns into FeatureData[Taxonomy]. With ``confidence``
        disabled, ``Confidence`` is the lowest per-rank softmax probability
        along the lineage; with a threshold, it is the score of the deepest
        rank that was kept.
    """
    checkpoint = _checkpoint_path(classifier)
    ranks = _read_ranks(checkpoint)

    with tempfile.TemporaryDirectory() as tmpdir:
        args = _base_namespace(
            fasta_file=str(reads),
            checkpoint=checkpoint,
            taxonomy_file=None,
            output_dir=tmpdir,
            batch_size=batch_size,
            top_k=top_k,
            num_workers=num_workers,
            seed=seed,
        )

        # Imported lazily so plugin/format registration does not import torch.
        from deeptaxa.predict import predict

        predict(args)

        matches = glob(os.path.join(tmpdir, "*_deeptaxa_predictions.json"))
        if not matches:
            raise RuntimeError(
                "DeepTaxa prediction did not produce an output file."
            )
        with open(matches[0]) as fh:
            output = json.load(fh)

    seq_ids = output["sequence_ids"]
    predictions = output["predictions"]

    threshold = None if confidence == "disable" else float(confidence)

    taxa, confidences = [], []
    for pred in predictions:
        labels, scores = [], []
        for rank in ranks:
            entry = pred.get(rank, {})
            labels.append(entry.get("label", ""))
            score = entry.get("raw_score")
            scores.append(float(score) if score is not None else None)
        taxon, conf = assign_taxonomy(ranks, labels, scores, threshold)
        taxa.append(taxon)
        confidences.append(conf)

    result = pd.DataFrame(
        {"Taxon": taxa, "Confidence": confidences}, index=pd.Index(seq_ids)
    )
    result.index.name = "Feature ID"
    return result


def fit(
    reference_reads: DNAFASTAFormat,
    reference_taxonomy: pd.Series,
    model_type: str = DEFAULT_CONFIG["model_type"],
    epochs: int = DEFAULT_CONFIG["epochs"],
    batch_size: int = DEFAULT_CONFIG["batch_size"],
    learning_rate: float = DEFAULT_CONFIG["learning_rate"],
    max_length: int = DEFAULT_CONFIG["max_length"],
    val_split: float = DEFAULT_CONFIG["val_split"],
    loss_type: str = DEFAULT_CONFIG["loss_type"],
    encoding: str = DEFAULT_CONFIG["encoding"],
    early_stopping_patience: int = DEFAULT_CONFIG["early_stopping_patience"],
    num_workers: int = DEFAULT_CONFIG["num_workers"],
    seed: int = DEFAULT_CONFIG["seed"],
) -> DeepTaxaModelDirectoryFormat:
    """Train a DeepTaxa model from reference sequences and taxonomy.

    Parameters
    ----------
    reference_reads : DNAFASTAFormat
        Reference 16S rRNA sequences (FeatureData[Sequence]).
    reference_taxonomy : pandas.Series
        Lineage per feature (FeatureData[Taxonomy]); index must match the
        sequence ids in ``reference_reads``.

    Returns
    -------
    DeepTaxaModelDirectoryFormat
        The trained model (the final checkpoint), as a DeepTaxaModel artifact.
    """
    taxonomy_table = taxonomy_series_to_table(reference_taxonomy, ranks=DEFAULT_RANKS)

    result = DeepTaxaModelDirectoryFormat()

    with tempfile.TemporaryDirectory() as tmpdir:
        taxonomy_fp = os.path.join(tmpdir, "taxonomy.tsv")
        taxonomy_table.to_csv(taxonomy_fp, sep="\t", index=False)

        args = _base_namespace(
            fasta_file=str(reference_reads),
            taxonomy_file=taxonomy_fp,
            output_dir=tmpdir,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            val_split=val_split,
            loss_type=loss_type,
            encoding=encoding,
            early_stopping_patience=early_stopping_patience,
            num_workers=num_workers,
            seed=seed,
        )

        from deeptaxa.train import train

        train(args)

        checkpoints = glob(os.path.join(tmpdir, "checkpoints", "*.pt"))
        if not checkpoints:
            raise RuntimeError(
                "DeepTaxa training did not produce a checkpoint."
            )
        # Checkpoints are named ..._epoch<N>.pt; pick the highest epoch, i.e.
        # the most-trained model.
        def _epoch(path):
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                return int(stem.rsplit("epoch", 1)[1])
            except (IndexError, ValueError):
                return -1

        best = max(checkpoints, key=_epoch)

        # The training checkpoint also stores optimizer, scheduler, scaler, and
        # RNG state, which are only needed to resume training. A DeepTaxaModel
        # artifact is used for prediction, so drop those large keys to keep the
        # artifact small. The remaining metadata is what predict and describe
        # read.
        import torch

        checkpoint = torch.load(best, map_location="cpu", weights_only=False)
        for key in (
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "rng_state",
        ):
            checkpoint.pop(key, None)
        torch.save(checkpoint, os.path.join(str(result), "model.pt"))

    return result
