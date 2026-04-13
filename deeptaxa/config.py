"""
Module: config.py

Description:
    Centralizes configuration parameters for DeepTaxa models and training pipelines.
    Defaults match the published Optuna-optimized HuggingFace v1 checkpoint.
"""

DEFAULT_CONFIG = {
    # General parameters (shared across all models)
    "tokenizer_name": "zhihan1996/DNABERT-2-117M",   # Pretrained DNA tokenizer from Hugging Face
    "model_type": "hybridcnnbert",                   # Default architecture: cnn, bert, or hybridcnnbert
    "max_length": 512,                               # Maximum sequence length post-tokenization
    "hidden_dropout_prob": 0.17,                     # Dropout probability (Optuna-optimized)
    "batch_size": 32,                                # Batch size for training and inference
    "epochs": 10,                                    # Number of training epochs
    "learning_rate": 3.72e-4,                        # Learning rate (Optuna-optimized)
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
    "optimizer_weight_decay": 0.042,                 # Weight decay (Optuna-optimized)
    "scaler_init_scale": 16384.0,                    # Initial scale for GradScaler in mixed precision training
    "level_temperatures": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Softmax temperature per rank

    # CNN-specific parameters (used by CNNClassifier and HybridCNNBERTClassifier)
    "embed_dim": 896,                                # Embedding dimension (Optuna-optimized)
    "num_filters": 512,                              # Filters per kernel size (Optuna-optimized)
    "kernel_sizes": [5, 7, 9],                       # CNN kernel sizes (Optuna-optimized)
    "num_conv_layers": 1,                            # Number of convolutional layers

    # BERT-specific parameters (used by BERTClassifier and HybridCNNBERTClassifier)
    "hidden_size": 1024,                             # Hidden layer size (Optuna-optimized)
    "num_hidden_layers": 5,                          # Number of transformer layers (Optuna-optimized)
    "num_attention_heads": 8,                        # Number of attention heads
    "intermediate_size": 4096,                       # Feed-forward network size (Optuna-optimized)

    # HybridCNNBERTClassifier-specific parameters
    "output_attentions": False,                      # Toggle for returning attention weights

    # Ablation and training control parameters
    "encoding": "dnabert",                           # Sequence encoding: dnabert or onehot
    "loss_type": "cross_entropy",                    # Loss function: focal or cross_entropy
    "early_stopping_patience": 0,                    # Epochs without improvement before stopping (0 = disabled)
    "early_stopping_min_delta": 0.001,               # Minimum improvement to reset patience counter
}

