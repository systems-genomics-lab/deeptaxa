"""
Module: hybrid.py

Description:
    Implements the HybridCNNBERTClassifier, combining convolutional and transformer architectures
    for hierarchical taxonomy classification of 16S rRNA sequences. Fully configurable via CLI,
    it fuses CNN-based local feature extraction with BERT's contextual modeling.

    The hybrid approach leverages CNNs for efficient local pattern detection and BERT for global
    sequence context, with learnable weights balancing their contributions.
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, AutoTokenizer
import logging
from deeptaxa.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class HybridCNNBERTClassifier(nn.Module):
    def __init__(self, tokenizer_name: str, num_labels_per_level: dict,
                 hidden_dropout_prob: float = DEFAULT_CONFIG["hidden_dropout_prob"], 
                 num_filters: int = DEFAULT_CONFIG["num_filters"],
                 kernel_sizes: list[int] = DEFAULT_CONFIG["kernel_sizes"],
                 num_conv_layers: int = DEFAULT_CONFIG["num_conv_layers"],
                 max_length: int = DEFAULT_CONFIG["max_length"],
                 embed_dim: int = DEFAULT_CONFIG["embed_dim"],
                 hidden_size: int = DEFAULT_CONFIG["hidden_size"],
                 num_hidden_layers: int = DEFAULT_CONFIG["num_hidden_layers"],
                 num_attention_heads: int = DEFAULT_CONFIG["num_attention_heads"],
                 intermediate_size: int = DEFAULT_CONFIG["intermediate_size"],
                 output_attentions: bool = False, mask_padding: bool = False) -> None:
        """
        Initialize the HybridCNNBERTClassifier with configurable parameters.

        Workflow:
            1. Validate num_labels_per_level and load tokenizer for vocab size.
            2. Set up embedding layer for token-to-vector mapping.
            3. Build CNN stacks with multi-scale kernels for local feature extraction.
            4. Configure BERT with transformer parameters for contextual modeling.
            5. Define projection and fusion layers, plus per-level classifiers.

        Args:
            tokenizer_name (str): Pretrained tokenizer name.
            num_labels_per_level (dict): Number of classes per taxonomic rank.
            hidden_dropout_prob (float): Dropout probability.
            num_filters (int): Number of CNN filters.
            kernel_sizes (list[int]): CNN kernel sizes.
            num_conv_layers (int): Number of CNN layers.
            max_length (int): Maximum sequence length.
            embed_dim (int): Embedding dimension.
            hidden_size (int): BERT hidden state size.
            num_hidden_layers (int): Number of BERT layers.
            num_attention_heads (int): Number of attention heads.
            intermediate_size (int): BERT feed-forward size.
            output_attentions (bool): Whether to return attention weights.

        Raises:
            ValueError: If num_labels_per_level is None.
        """
        super().__init__()
        if num_labels_per_level is None:
            raise ValueError("num_labels_per_level must be provided.")
        
        self.num_labels_per_level = num_labels_per_level
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_attentions = output_attentions
        self.mask_padding = mask_padding
        
        # Load tokenizer and set up embedding
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        vocab_size = len(tokenizer.get_vocab())
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id or 0)
        
        # Build CNN stacks
        self.conv_stacks = nn.ModuleList()
        in_channels = embed_dim
        for _ in range(num_conv_layers):
            branches = nn.ModuleList()
            for k in kernel_sizes:
                branches.append(nn.Sequential(
                    nn.Conv1d(in_channels, num_filters, k, padding=k // 2),  # Preserve sequence length
                    nn.ReLU(),
                    nn.Dropout(hidden_dropout_prob)
                ))
            self.conv_stacks.append(branches)
            in_channels = num_filters * len(kernel_sizes)  # Concatenate multi-scale outputs
        
        # Configure BERT
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_length
        )
        self.bert = BertModel(config)
        self.config = config
        
        # Projection and fusion layers
        self.cnn_proj = nn.Linear(in_channels, config.hidden_size)  # Align CNN output with BERT
        self.w_cnn = nn.Parameter(torch.ones(1))  # Learnable weight for CNN contribution
        self.w_bert = nn.Parameter(torch.ones(1))  # Learnable weight for BERT contribution
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Classifiers for each taxonomic level
        self.classifiers = nn.ModuleDict()
        for level, num_labels in num_labels_per_level.items():
            self.classifiers[str(level)] = nn.Linear(config.hidden_size, num_labels)
        
        # Store CNN parameters for reference
        self.cnn_params = {
            'embed_dim': embed_dim,
            'num_filters': num_filters,
            'kernel_sizes': kernel_sizes,
            'num_conv_layers': num_conv_layers,
            'dropout_prob': hidden_dropout_prob,
            'mask_padding': mask_padding
        }
        logger.info("Initialized HybridCNNBERTClassifier with CNN: %s, BERT hidden_size=%d, output_attentions=%s", 
                    self.cnn_params, config.hidden_size, output_attentions)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """
        Forward pass combining CNN and BERT features.

        Workflow:
            1. Embed input tokens and process through CNN stacks.
            2. Pool CNN features and project to BERT hidden size.
            3. Pass input through BERT and extract CLS token representation.
            4. Fuse CNN and BERT outputs with learnable weights.
            5. Apply dropout and compute per-level logits.

        Args:
            input_ids (torch.Tensor): Tokenized input sequences [batch_size, seq_len].
            attention_mask (torch.Tensor): Mask for padding tokens [batch_size, seq_len].

        Returns:
            tuple: (logits dict, attentions if output_attentions is True).
        """
        # CNN path
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        cnn_input = embedded.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
        x = cnn_input
        for branches in self.conv_stacks:
            branch_outputs = [branch(x) for branch in branches]
            x = torch.cat(branch_outputs, dim=1)  # Concatenate multi-scale features
        if self.mask_padding and attention_mask is not None:
            # Keep padded positions out of the max so they cannot leak in through
            # the conv bias; the BERT path already masks through attention.
            pad = (attention_mask == 0).unsqueeze(1)  # [batch_size, 1, seq_len]
            x = x.masked_fill(pad, torch.finfo(x.dtype).min)
        cnn_pooled = torch.max(x, dim=2)[0]  # Global max pooling
        cnn_proj = self.cnn_proj(cnn_pooled)  # Project to BERT hidden size
        
        # BERT path
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=self.output_attentions)
        bert_pooled = bert_outputs.last_hidden_state[:, 0]  # CLS token representation
        attentions = bert_outputs.attentions if self.output_attentions else None
        
        # Fuse CNN and BERT with learnable weights
        fused = self.w_cnn * cnn_proj + self.w_bert * bert_pooled
        fused = fused + bert_pooled  # Residual-like connection with BERT
        fused = self.dropout(fused)
        
        # Compute logits for each level
        logits = {level: classifier(fused) for level, classifier in self.classifiers.items()}
        return logits, attentions

