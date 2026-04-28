"""
Module: bert.py

Description:
    Implements the BERTClassifier, a transformer-based model for hierarchical taxonomy classification
    of 16S rRNA sequences. Configurable via CLI parameters, it leverages a BERT architecture with
    custom classification heads for each taxonomic level, optimized for DNA sequence data.
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class BERTClassifier(nn.Module):
    def __init__(self, tokenizer_name: str, num_labels_per_level: dict, hidden_dropout_prob: float,
                 max_length: int, hidden_size: int, num_hidden_layers: int, num_attention_heads: int,
                 intermediate_size: int) -> None:
        """
        Initialize the BERTClassifier with configurable architecture.

        Workflow:
            1. Validate input for number of labels per taxonomic level.
            2. Load tokenizer to determine vocab size and configure BERT with CLI parameters.
            3. Initialize BERT model with custom configuration.
            4. Set up dropout and classification heads for each taxonomic level.

        Args:
            tokenizer_name (str): Name of pretrained tokenizer (e.g., DNABERT).
            num_labels_per_level (dict): Number of classes per taxonomic rank.
            hidden_dropout_prob (float): Dropout probability for regularization.
            max_length (int): Maximum sequence length for positional embeddings.
            hidden_size (int): Size of hidden states in BERT.
            num_hidden_layers (int): Number of transformer layers.
            num_attention_heads (int): Number of attention heads per layer.
            intermediate_size (int): Size of feed-forward network in transformer layers.

        Raises:
            ValueError: If num_labels_per_level is None.
        """
        super().__init__()
        if num_labels_per_level is None:
            raise ValueError("num_labels_per_level must be provided to define classification heads.")

        self.num_labels_per_level = num_labels_per_level
        self.hidden_dropout_prob = hidden_dropout_prob
        
        # Load tokenizer to align vocab size with embedding layer
        vocab_size = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True).vocab_size
        
        # Configure BERT with CLI-driven parameters
        config = BertConfig(
            vocab_size=vocab_size,  # Matches DNA tokenizer's vocabulary
            hidden_size=hidden_size,  # Controls transformer capacity
            num_hidden_layers=num_hidden_layers,  # Depth for contextual learning
            num_attention_heads=num_attention_heads,  # Multi-head attention for diverse feature capture
            intermediate_size=intermediate_size,  # Feed-forward layer size for richer representations
            hidden_dropout_prob=hidden_dropout_prob,  # Regularizes hidden states
            attention_probs_dropout_prob=hidden_dropout_prob,  # Regularizes attention weights
            max_position_embeddings=max_length  # Supports sequence length up to max_length
        )
        self.bert = BertModel(config)
        self.config = config
        
        # Dropout layer post-BERT to prevent overfitting on pooled output
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Define classifiers for each taxonomic level
        self.classifiers = nn.ModuleDict()
        for level, num_labels in num_labels_per_level.items():
            self.classifiers[str(level)] = nn.Linear(config.hidden_size, num_labels)
        
        logger.info("Initialized BERTClassifier with %d hidden layers", config.num_hidden_layers)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """
        Forward pass for taxonomic classification.

        Workflow:
            1. Pass input through BERT to get contextualized sequence embeddings.
            2. Mask and pool sequence output to a fixed-size representation.
            3. Apply dropout for regularization.
            4. Compute logits for each taxonomic level using level-specific classifiers.

        Args:
            input_ids (torch.Tensor): Tokenized input sequences [batch_size, seq_len].
            attention_mask (torch.Tensor): Mask for padding tokens [batch_size, seq_len].

        Returns:
            dict: Logits per taxonomic level, keyed by level index (str).
        """
        # Get BERT outputs (last hidden state: [batch_size, seq_len, hidden_size])
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Mask padded tokens and compute mean pooling
        masked_output = sequence_output * attention_mask.unsqueeze(-1)  # Zero out padding
        pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)  # Average over valid tokens
        
        # Apply dropout to pooled representation
        pooled = self.dropout(pooled)
        
        # Generate logits for each taxonomic level
        logits = {level: classifier(pooled) for level, classifier in self.classifiers.items()}
        
        return logits

