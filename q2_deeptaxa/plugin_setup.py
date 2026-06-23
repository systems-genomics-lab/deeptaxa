"""
Plugin registration for q2-deeptaxa.

Registers the ``DeepTaxaModel`` semantic type and format, and the three actions
(``classify``, ``fit``, ``describe``) that wrap the DeepTaxa pipelines.
"""

from qiime2.plugin import (
    Choices,
    Citations,
    Float,
    Int,
    Plugin,
    Range,
    Str,
)
from q2_types.feature_data import FeatureData, Sequence, Taxonomy

import q2_deeptaxa
from q2_deeptaxa._formats import (
    DeepTaxaModelDirectoryFormat,
    DeepTaxaModelFormat,
)
from q2_deeptaxa._methods import classify, fit
from q2_deeptaxa._types import DeepTaxaModel
from q2_deeptaxa._visualizers import describe

citations = Citations.load("citations.bib", package="q2_deeptaxa")

plugin = Plugin(
    name="deeptaxa",
    version=q2_deeptaxa.__version__,
    website="https://github.com/systems-genomics-lab/deeptaxa",
    package="q2_deeptaxa",
    citations=[citations["deeptaxa"]],
    short_description=(
        "Deep learning hierarchical taxonomy classification of 16S rRNA "
        "sequences."
    ),
    description=(
        "QIIME 2 plugin for DeepTaxa, a deep learning framework that classifies "
        "16S rRNA gene sequences across all taxonomic ranks in a single forward "
        "pass using a hybrid CNN-BERT architecture."
    ),
)

# --- Types and formats -----------------------------------------------------
plugin.register_formats(DeepTaxaModelFormat, DeepTaxaModelDirectoryFormat)
plugin.register_semantic_types(DeepTaxaModel)
plugin.register_semantic_type_to_format(
    DeepTaxaModel, artifact_format=DeepTaxaModelDirectoryFormat
)

# --- Actions ---------------------------------------------------------------
plugin.methods.register_function(
    function=classify,
    inputs={
        "reads": FeatureData[Sequence],
        "classifier": DeepTaxaModel,
    },
    parameters={
        "batch_size": Int % Range(1, None),
        "top_k": Int % Range(1, None),
        "confidence_threshold": Float % Range(0, 1, inclusive_end=True),
        "num_workers": Int % Range(0, None),
        "seed": Int,
    },
    outputs=[("classification", FeatureData[Taxonomy])],
    input_descriptions={
        "reads": "The sequences to classify.",
        "classifier": "The trained DeepTaxa model.",
    },
    parameter_descriptions={
        "batch_size": "Number of sequences per inference batch.",
        "top_k": "Number of top candidate labels considered per rank.",
        "confidence_threshold": (
            "Softmax confidence below which a prediction is flagged as "
            "low-confidence."
        ),
        "num_workers": "Number of data-loading worker processes.",
        "seed": "Random seed for reproducibility.",
    },
    output_descriptions={
        "classification": (
            "Per-feature taxonomic lineage with a conservative joint "
            "confidence (minimum per-rank probability)."
        )
    },
    name="Classify reads with a DeepTaxa model",
    description=(
        "Assign hierarchical taxonomy to sequences using a trained DeepTaxa "
        "model. The model predicts every taxonomic rank in a single forward "
        "pass."
    ),
    citations=[citations["deeptaxa"]],
)

plugin.methods.register_function(
    function=fit,
    inputs={
        "reference_reads": FeatureData[Sequence],
        "reference_taxonomy": FeatureData[Taxonomy],
    },
    parameters={
        "model_type": Str % Choices(["cnn", "bert", "hybridcnnbert"]),
        "epochs": Int % Range(1, None),
        "batch_size": Int % Range(1, None),
        "learning_rate": Float % Range(0, None, inclusive_start=False),
        "max_length": Int % Range(1, None),
        "val_split": Float % Range(0, 1, inclusive_end=False),
        "loss_type": Str % Choices(["cross_entropy", "focal"]),
        "encoding": Str % Choices(["dnabert", "onehot"]),
        "early_stopping_patience": Int % Range(0, None),
        "num_workers": Int % Range(0, None),
        "seed": Int,
    },
    outputs=[("classifier", DeepTaxaModel)],
    input_descriptions={
        "reference_reads": "Reference sequences to train on.",
        "reference_taxonomy": (
            "Reference taxonomy whose feature ids match the reference "
            "sequences."
        ),
    },
    parameter_descriptions={
        "model_type": (
            "Architecture: 'cnn' (local motifs), 'bert' (long-range "
            "context), or 'hybridcnnbert' (both; published default)."
        ),
        "epochs": "Number of training epochs.",
        "batch_size": "Training batch size.",
        "learning_rate": "AdamW learning rate.",
        "max_length": "Maximum number of tokens per sequence.",
        "val_split": "Fraction of sequences held out for validation.",
        "loss_type": "Training loss function.",
        "encoding": (
            "Sequence encoding: 'dnabert' (DNABERT-2 tokenizer) or 'onehot'."
        ),
        "early_stopping_patience": (
            "Epochs without validation improvement before stopping "
            "(0 disables)."
        ),
        "num_workers": "Number of data-loading worker processes.",
        "seed": "Random seed for reproducibility.",
    },
    output_descriptions={
        "classifier": "The trained DeepTaxa model."
    },
    name="Train a DeepTaxa model",
    description=(
        "Train a hierarchical DeepTaxa taxonomy classifier from reference "
        "sequences and taxonomy. Training is GPU-intensive; a CUDA device is "
        "strongly recommended."
    ),
    citations=[citations["deeptaxa"]],
)

plugin.visualizers.register_function(
    function=describe,
    inputs={"classifier": DeepTaxaModel},
    parameters={},
    input_descriptions={"classifier": "The trained DeepTaxa model to inspect."},
    parameter_descriptions={},
    name="Describe a DeepTaxa model",
    description=(
        "Render a summary of a trained DeepTaxa model checkpoint: "
        "architecture, hyperparameters, dataset, and taxonomic ranks."
    ),
    citations=[citations["deeptaxa"]],
)
