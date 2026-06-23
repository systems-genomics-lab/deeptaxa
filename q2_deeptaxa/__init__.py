"""
QIIME 2 plugin for DeepTaxa.

DeepTaxa is a deep learning model that classifies 16S rRNA gene sequences into
all taxonomic ranks at once. This plugin adds three QIIME 2 actions that match
the DeepTaxa command line:

    - classify : label FeatureData[Sequence] with FeatureData[Taxonomy] using a
                 trained model.
    - fit      : train a model from reference sequences and their taxonomy.
    - describe : show a summary of a trained model.

The trained model is stored as a QIIME 2 artifact (semantic type
``DeepTaxaModel``), so QIIME records its provenance the same way it does for
any other artifact.
"""

try:
    from deeptaxa import __version__
except Exception:  # pragma: no cover - deeptaxa should always be importable
    __version__ = "0.0.0"

__all__ = ["__version__"]
