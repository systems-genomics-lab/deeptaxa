"""
Module: cnn.py

Description:
    Implements the CNNClassifier and ResidualConvBlock for hierarchical taxonomy classification.
    This module uses a convolutional neural network with residual connections and multi-scale kernels,
    configurable via CLI parameters, to extract local sequence features from 16S rRNA data.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, hidden_dropout_prob: float) -> None:
        """
        Initialize a residual convolutional block.

        Workflow:
            1. Define a convolutional sequence with ReLU and dropout.
            2. Add a projection layer for residual connection if channel sizes differ.

        Args:
            in_channels (int): Input feature dimension.
            out_channels (int): Output feature dimension.
            kernel_size (int): Size of convolutional kernel.
            hidden_dropout_prob (float): Dropout probability for regularization.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),  # Preserves sequence length
            nn.ReLU(),  # Non-linearity for feature extraction
            nn.Dropout(hidden_dropout_prob)  # Regularizes to prevent overfitting
        )
        # Projection layer aligns dimensions for residual addition
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        logger.debug("Initialized ResidualConvBlock: in_channels=%d, out_channels=%d, kernel_size=%d",
                     in_channels, out_channels, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x (torch.Tensor): Input tensor [batch_size, in_channels, seq_len].

        Returns:
            torch.Tensor: Output tensor [batch_size, out_channels, seq_len].
        """
        # Apply convolution and add residual connection
        return self.conv(x) + self.proj(x)

class CNNClassifier(nn.Module):
    def __init__(self, tokenizer_name: str, num_labels_per_level: dict, embed_dim: int, num_filters: int,
                 kernel_sizes: list[int], num_conv_layers: int, hidden_dropout_prob: float,
                 input_mode: str = "dnabert", mask_padding: bool = False) -> None:
        """
        Initialize the CNNClassifier with configurable architecture.

        Workflow:
            1. Load tokenizer to set up embedding layer with vocab size and padding.
            2. Define embedding layer to convert tokens to dense vectors.
            3. Build convolutional stacks with multi-scale residual blocks.
            4. Set up dropout and per-level classifiers for taxonomic prediction.

        Args:
            tokenizer_name (str): Name of pretrained tokenizer.
            num_labels_per_level (dict): Number of classes per taxonomic rank.
            embed_dim (int): Embedding dimension for token representations.
            num_filters (int): Number of filters per convolutional branch.
            kernel_sizes (list[int]): List of kernel sizes for multi-scale feature extraction.
            num_conv_layers (int): Number of convolutional layers.
            hidden_dropout_prob (float): Dropout probability for regularization.
            input_mode (str): Input encoding mode ('dnabert' or 'onehot').
        """
        super().__init__()
        self.input_mode = input_mode
        self.mask_padding = mask_padding
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_labels_per_level = num_labels_per_level

        if input_mode == "onehot":
            self.embedding = None
            first_in_channels = 4
            vocab_size = 0
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            vocab_size = len(tokenizer.get_vocab())
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            # Embedding layer maps tokens to dense vectors
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
            first_in_channels = embed_dim

        # Build convolutional stacks with residual blocks
        self.conv_stacks = nn.ModuleList()
        in_channels = first_in_channels
        for layer in range(num_conv_layers):
            branches = nn.ModuleList()
            for k in kernel_sizes:
                branches.append(ResidualConvBlock(in_channels, num_filters, k, hidden_dropout_prob))
            self.conv_stacks.append(branches)
            in_channels = num_filters * len(kernel_sizes)  # Concatenate outputs from all kernels
        
        # Dropout for pooled features
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Define classifiers for each taxonomic level
        self.classifiers = nn.ModuleDict()
        for level, num_labels in num_labels_per_level.items():
            self.classifiers[str(level)] = nn.Sequential(
                nn.Linear(in_channels, in_channels),  # Intermediate layer for feature refinement
                nn.ReLU(),
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(in_channels, num_labels)  # Output logits for classification
            )
        
        # Store configuration for reference
        self.config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_filters": num_filters,
            "kernel_sizes": kernel_sizes,
            "num_conv_layers": num_conv_layers,
            "hidden_dropout_prob": hidden_dropout_prob,
            "input_mode": input_mode,
            "mask_padding": mask_padding
        }
        logger.info("Initialized CNNClassifier with %d conv layers, kernel sizes: %s, input_mode: %s",
                     num_conv_layers, kernel_sizes, input_mode)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> dict:
        """
        Forward pass for taxonomic classification.

        Workflow:
            1. Embed input tokens into dense vectors.
            2. Apply convolutional stacks with multi-scale kernels.
            3. Pool features globally using max pooling.
            4. Apply dropout and compute logits per taxonomic level.

        Args:
            input_ids (torch.Tensor): Tokenized input sequences [batch_size, seq_len].
            attention_mask (torch.Tensor, optional): Mask for padding tokens [batch_size, seq_len].

        Returns:
            dict: Logits per taxonomic level, keyed by level index (str).
        """
        if self.input_mode == "onehot":
            # input_ids is already [batch_size, 4, seq_len] float tensor
            x = input_ids.float()
        else:
            # Embed tokens: [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
            x = self.embedding(input_ids)
            # Transpose for Conv1d: [batch_size, embed_dim, seq_len]
            x = x.transpose(1, 2)
        
        # Apply convolutional stacks
        for branches in self.conv_stacks:
            branch_outputs = [branch(x) for branch in branches]
            x = torch.cat(branch_outputs, dim=1)  # Concatenate multi-scale features
        
        # Global max pooling: [batch_size, in_channels, seq_len] -> [batch_size, in_channels]
        if self.mask_padding and attention_mask is not None:
            # Push padded positions below every real activation so they cannot win
            # the max; a padded column would otherwise leak in through the conv bias.
            pad = (attention_mask == 0).unsqueeze(1)  # [batch_size, 1, seq_len]
            x = x.masked_fill(pad, torch.finfo(x.dtype).min)
        pooled = torch.max(x, dim=2)[0]
        pooled = self.dropout(pooled)
        
        # Compute logits for each level
        logits = {level: classifier(pooled) for level, classifier in self.classifiers.items()}
        return logits

