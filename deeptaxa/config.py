"""
Module: config.py

Description:
    Centralizes configuration parameters for DeepTaxa models and training pipelines.
    Defaults match the compact HybridCNNBERT configuration used for the published
    checkpoint: 256 filters / 896 hidden / 4 transformer layers / 7 attention heads /
    3584 FFN / kernels 3,5,7, trained with cross-entropy loss at LR 5e-4, batch 64,
    dropout 0.20, weight decay 0.01.

    The earlier Optuna-optimized expanded values (512 filters / 1024 hidden /
    5 layers / 8 heads / 4096 FFN, kernels 5,7,9, LR 3.72e-4, batch 32,
    dropout 0.174, WD 0.042) are preserved in this file's git history.
"""

DEFAULT_CONFIG = {
    # General parameters (shared across all models)
    "tokenizer_name": "zhihan1996/DNABERT-2-117M",   # Pretrained DNA tokenizer from Hugging Face
    "model_type": "hybridcnnbert",                   # Default architecture: cnn, bert, or hybridcnnbert
    "max_length": 512,                               # Maximum sequence length post-tokenization
    "hidden_dropout_prob": 0.20,                     # Dropout probability (compact)
    "batch_size": 64,                                # Batch size for training and inference (compact)
    "epochs": 10,                                    # Number of training epochs
    "learning_rate": 5e-4,                           # Learning rate (compact)
    "val_split": 0.2,                                # Fraction of dataset reserved for validation
    "seed": 42,                                      # Random seed for reproducibility
    "output_dir": "./",                              # Directory for model checkpoints, metrics, or predictions
    "eval_every": 1,                                 # Frequency of validation (in epochs)
    "accum_steps": 1,                                # Gradient accumulation steps
    "focal_gamma": 2.0,                              # Gamma for focal loss
    "focal_reduction": "mean",                       # Reduction method for focal loss
    "focal_weight": None,                            # Class weights for focal loss (None for no weighting)
    "level_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Uniform weights per taxonomic rank
    "num_workers": 2,                                # Parallel workers for data loading
    "top_k": 3,                                      # Number of top predictions returned
    "confidence_threshold": 0.95,                    # Threshold for confident predictions
    "scheduler_warmup_ratio": 0.1,                   # Warmup ratio for learning rate scheduler
    "optimizer_betas": [0.9, 0.999],                 # AdamW momentum parameters
    "optimizer_eps": 1e-8,                           # Epsilon for AdamW numerical stability
    "optimizer_weight_decay": 0.01,                  # Weight decay (compact)
    "scaler_init_scale": 16384.0,                    # Initial scale for GradScaler in mixed precision training
    "level_temperatures": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Softmax temperature per rank

    # CNN-specific parameters (used by CNNClassifier and HybridCNNBERTClassifier)
    "embed_dim": 896,                                # Embedding dimension
    "num_filters": 256,                              # Filters per kernel size (compact)
    "kernel_sizes": [3, 5, 7],                       # CNN kernel sizes (compact)
    "num_conv_layers": 1,                            # Number of convolutional layers

    # BERT-specific parameters (used by BERTClassifier and HybridCNNBERTClassifier)
    "hidden_size": 896,                              # Hidden layer size (compact)
    "num_hidden_layers": 4,                          # Number of transformer layers (compact)
    "num_attention_heads": 7,                        # Number of attention heads (compact)
    "intermediate_size": 3584,                       # Feed-forward network size (compact)

    # HybridCNNBERTClassifier-specific parameters
    "output_attentions": False,                      # Toggle for returning attention weights

    # Ablation and training control parameters
    "encoding": "dnabert",                           # Sequence encoding: dnabert or onehot
    "loss_type": "cross_entropy",                    # Loss function: focal or cross_entropy
    "early_stopping_patience": 0,                    # Epochs without improvement before stopping (0 = disabled)
    "early_stopping_min_delta": 0.001,               # Minimum improvement to reset patience counter
}
