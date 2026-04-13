"""
Module: __init__.py (inside the models folder)

Description:
    Re-exports all model classes and related components, providing a centralized import point
    for the DeepTaxa framework's modeling toolkit. This simplifies access to CNN, BERT, hybrid
    classifiers, and custom loss functions.
"""

# Import and re-export CNNClassifier and its helper ResidualConvBlock
from .cnn import CNNClassifier, ResidualConvBlock
# Import and re-export BERTClassifier
from .bert import BERTClassifier
# Import and re-export HybridCNNBERTClassifier
from .hybrid import HybridCNNBERTClassifier
# Import and re-export custom loss functions
from .losses import FocalLoss

# Define public API for wildcard imports
__all__ = ['CNNClassifier', 'ResidualConvBlock', 'BERTClassifier', 'HybridCNNBERTClassifier', 'FocalLoss']

