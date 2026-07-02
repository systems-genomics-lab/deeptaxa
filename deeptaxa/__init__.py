"""
Package: deeptaxa

Description:
    DeepTaxa is a comprehensive framework for hierarchical taxonomy classification of 16S rRNA sequences,
    leveraging deep learning techniques to predict taxonomic labels across multiple ranks (e.g., genus,
    species). It integrates convolutional neural networks (CNNs), transformer-based BERT models, and a
    hybrid CNN-BERT approach to capture both local sequence patterns and global contextual relationships.
    Designed for flexibility and scalability, the package supports end-to-end workflows from data
    preprocessing to model training and inference, with a modular structure for easy extension.

Package Structure:
    - models: Defines neural network architectures (CNNClassifier, BERTClassifier, HybridCNNBERTClassifier)
      and custom loss functions (e.g., FocalLoss) for multi-level classification.
    - dataset: Implements TaxonomyDataset for loading, tokenizing, and aligning 16S rRNA sequences with
      taxonomy labels, optimized for PyTorch data pipelines.
    - utils: Provides utility functions for reproducibility (e.g., seed setting), device management, and
      file validation, ensuring robust experiment control.
    - train: Orchestrates the training pipeline, including data splitting, optimization with AdamW, and
      checkpointing for resuming and analysis.
    - predict: Handles inference, generating taxonomic predictions and performance metrics (e.g., F1, AUC)
      from trained models.
    - cli: Exposes a command-line interface with configurable parameters for training, prediction, and
      model inspection, integrating all components seamlessly.
    - config: Centralizes default hyperparameters and architectural settings, enabling consistent and
      tunable configurations across experiments.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Distribution name is "deeptaxa-rrna" (the bare "deeptaxa" name was already
    # taken on PyPI); the import package and CLI command remain "deeptaxa".
    __version__ = version("deeptaxa-rrna")
except PackageNotFoundError:
    # Raised when running from a source tree that was never installed as a
    # distribution, e.g. during local development or a checkout-only test run.
    __version__ = "0.0.0"
