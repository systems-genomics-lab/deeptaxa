"""
Module: cli.py

Description:
    Defines the command-line interface for DeepTaxa, exposing parameters for training,
    prediction, and model description. Integrates with train, predict, and describe modules
    to provide a unified entry point for model workflows.
"""

import argparse
import sys
from deeptaxa.train import train
from deeptaxa.predict import predict
from deeptaxa.describe import describe
import logging
import torch
from deeptaxa import __version__
from deeptaxa.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse CLI arguments for DeepTaxa commands.

    Workflow:
        1. Initialize parser with version info and subcommand structure.
        2. Define shared arguments applicable to multiple commands (e.g., model architecture).
        3. Configure subparsers for train, predict, and describe with specific parameters.
        4. Validate and process arguments, converting types as needed (e.g., focal weights).

    Returns:
        argparse.Namespace: Parsed arguments ready for downstream functions.

    Raises:
        ValueError: If invalid fields or options are provided.
    """
    parser = argparse.ArgumentParser(description="DeepTaxa CLI: Train, predict, and describe taxonomy models.")
    parser.add_argument("--version", action="version", version=f"DeepTaxa {__version__}", help="Show DeepTaxa version and exit")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Shared arguments for model configuration
    def add_shared_args(subparser):
        subparser.add_argument(
            "--tokenizer-name", type=str, default=DEFAULT_CONFIG["tokenizer_name"],
            help=(
                f"Hugging Face model ID of the pretrained BPE tokenizer (default: {DEFAULT_CONFIG['tokenizer_name']}). "
                "Must match the tokenizer used at training time when running predict."
            )
        )
        subparser.add_argument(
            "--tokenizer-revision", type=str, default=DEFAULT_CONFIG["tokenizer_revision"],
            help=(
                "Pin the tokenizer to a specific Hugging Face commit hash, tag, or branch "
                "(default: none, which tracks the repository's default branch). Pinning guards "
                "against upstream changes to a tokenizer loaded with trust_remote_code. The value "
                "is recorded in the checkpoint so predict reuses the same revision."
            )
        )
        subparser.add_argument(
            "--max-length", type=int, default=DEFAULT_CONFIG["max_length"],
            help=(
                f"Maximum number of tokens per sequence (default: {DEFAULT_CONFIG['max_length']}). "
                "Sequences longer than this value are truncated; shorter ones are padded. "
                "Must match the value used at training time when running predict."
            )
        )
        subparser.add_argument(
            "--embed-dim", type=int, default=DEFAULT_CONFIG["embed_dim"],
            help=(
                f"Token embedding dimension fed into the CNN pathway (default: {DEFAULT_CONFIG['embed_dim']}). "
                "Must equal --hidden-size in the hybrid model so the CNN projection aligns with the Transformer output."
            )
        )
        subparser.add_argument(
            "--num-filters", type=int, default=DEFAULT_CONFIG["num_filters"],
            help=(
                f"Number of filters per CNN kernel size (default: {DEFAULT_CONFIG['num_filters']}). "
                "The total CNN output dimension is num-filters x len(kernel-sizes)."
            )
        )
        subparser.add_argument(
            "--kernel-sizes", type=int, nargs='+', default=DEFAULT_CONFIG["kernel_sizes"],
            help=(
                f"Space-separated CNN kernel widths in nucleotides (default: {' '.join(map(str, DEFAULT_CONFIG['kernel_sizes']))}). "
                "Each kernel size runs in parallel; smaller values capture short motifs, "
                "larger values capture longer conserved patterns."
            )
        )
        subparser.add_argument(
            "--num-conv-layers", type=int, default=DEFAULT_CONFIG["num_conv_layers"],
            help=(
                f"Number of stacked CNN layers (default: {DEFAULT_CONFIG['num_conv_layers']}). "
                "The published checkpoint uses 1. Additional layers increase depth but also memory and training time."
            )
        )
        subparser.add_argument(
            "--hidden-size", type=int, default=DEFAULT_CONFIG["hidden_size"],
            help=(
                f"Transformer hidden dimension (default: {DEFAULT_CONFIG['hidden_size']}). "
                "Must be divisible by --num-attention-heads. "
                "Must match the checkpoint when running predict."
            )
        )
        subparser.add_argument(
            "--num-hidden-layers", type=int, default=DEFAULT_CONFIG["num_hidden_layers"],
            help=(
                f"Number of Transformer encoder layers (default: {DEFAULT_CONFIG['num_hidden_layers']}). "
                "Must match the checkpoint when running predict."
            )
        )
        subparser.add_argument(
            "--num-attention-heads", type=int, default=DEFAULT_CONFIG["num_attention_heads"],
            help=(
                f"Number of self-attention heads (default: {DEFAULT_CONFIG['num_attention_heads']}). "
                "Must divide --hidden-size evenly (head dimension = hidden-size / num-attention-heads). "
                "Must match the checkpoint when running predict."
            )
        )
        subparser.add_argument(
            "--intermediate-size", type=int, default=DEFAULT_CONFIG["intermediate_size"],
            help=(
                f"Dimension of the feed-forward layer inside each Transformer block (default: {DEFAULT_CONFIG['intermediate_size']}). "
                "Typically 4x --hidden-size. Must match the checkpoint when running predict."
            )
        )
        subparser.add_argument(
            "--hidden-dropout-prob", type=float, default=DEFAULT_CONFIG["hidden_dropout_prob"],
            help=(
                f"Dropout probability applied after CNN activations and after the fusion layer (default: {DEFAULT_CONFIG['hidden_dropout_prob']}). "
                "Dropout is active during training and disabled at inference."
            )
        )
        subparser.add_argument(
            "--output-attentions", action="store_true",
            help=(
                "Return per-layer self-attention weight matrices from the Transformer. "
                "Attention weights are included in the model output dictionary and saved to the checkpoint. "
                "Increases memory usage; disabled by default."
            )
        )

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a DeepTaxa model")
    train_parser.add_argument("--fasta-file", type=str, required=True, help="Path to FASTA file")
    train_parser.add_argument("--taxonomy-file", type=str, required=True, help="Path to taxonomy file")
    train_parser.add_argument(
        "--model-type", type=str, choices=["cnn", "bert", "hybridcnnbert"],
        default=DEFAULT_CONFIG["model_type"],
        help=(
            "Model architecture. "
            "'cnn': convolutional network only; fast, captures local sequence motifs. "
            "'bert': Transformer encoder only; models long-range dependencies via self-attention. "
            "'hybridcnnbert': CNN and Transformer in parallel with learnable fusion weights; "
            "best accuracy and the architecture used for the published checkpoint."
        )
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
        help=f"Batch size (default: {DEFAULT_CONFIG['batch_size']})"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
        help=f"Number of training epochs (default: {DEFAULT_CONFIG['epochs']})"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
        help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']})"
    )
    train_parser.add_argument(
        "--val-split", type=float, default=DEFAULT_CONFIG["val_split"],
        help=(
            f"Fraction of training sequences reserved for validation (default: {DEFAULT_CONFIG['val_split']}). "
            "For example, 0.1 holds out 10%% for validation and trains on the remaining 90%%. "
            "Validation loss and F1 are reported after every --eval-every epochs."
        )
    )
    train_parser.add_argument(
        "--seed", type=int, default=DEFAULT_CONFIG["seed"],
        help=f"Random seed (default: {DEFAULT_CONFIG['seed']})"
    )
    train_parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"],
        help=f"Output directory (default: {DEFAULT_CONFIG['output_dir']})"
    )
    train_parser.add_argument(
        "--reset-lr-on-resume", action="store_true",
        help="Reset learning rate when resuming training"
    )
    checkpoint_init_group = train_parser.add_mutually_exclusive_group()
    checkpoint_init_group.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint for resuming"
    )
    checkpoint_init_group.add_argument(
        "--init-weights", type=str, default=None,
        help=(
            "Path to a checkpoint whose weights will initialize the model before training. "
            "Unlike --resume, this does NOT restore optimizer, scheduler, RNG, or dataset split "
            "and does NOT count the checkpoint's epochs; training starts at epoch 1 with a fresh "
            "optimizer. Weights are loaded with strict=False so classifier heads are randomly "
            "initialized when the target label space differs from the checkpoint's. Use this for "
            "fine-tuning across datasets (e.g., full-length 16S -> V3-V4 amplicons). The body "
            "layers (CNN, BERT, embeddings) must match the checkpoint architecture; a shape "
            "mismatch on body layers aborts training."
        )
    )
    train_parser.add_argument(
        "--eval-every", type=int, default=DEFAULT_CONFIG["eval_every"],
        help=f"Evaluate every N epochs (default: {DEFAULT_CONFIG['eval_every']})"
    )
    train_parser.add_argument(
        "--accum-steps", type=int, default=DEFAULT_CONFIG["accum_steps"],
        help=(
            f"Gradient accumulation steps (default: {DEFAULT_CONFIG['accum_steps']}). "
            "Gradients are accumulated over this many batches before a single optimizer step, "
            "making the effective batch size equal to --batch-size x --accum-steps. "
            "Useful for simulating larger batches when GPU memory is limited."
        )
    )
    train_parser.add_argument(
        "--focal-gamma", type=float, default=DEFAULT_CONFIG["focal_gamma"],
        help=(
            f"Focusing parameter for focal loss (default: {DEFAULT_CONFIG['focal_gamma']}). "
            "Only used when --loss-type focal is set. Higher values suppress well-classified "
            "examples more aggressively (gamma=0 reduces to cross-entropy)."
        )
    )
    train_parser.add_argument(
        "--level-weights", type=float, nargs='+', default=DEFAULT_CONFIG["level_weights"],
        help=(
            f"Seven space-separated loss weights, one per taxonomic rank in order "
            f"Domain Phylum Class Order Family Genus Species "
            f"(default: {' '.join(map(str, DEFAULT_CONFIG['level_weights']))}). "
            "Increase the weight for a rank to emphasize accuracy at that level. "
            "See also --no-level-weights to enforce uniform weights."
        )
    )
    train_parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"],
        help=f"Number of DataLoader workers (default: {DEFAULT_CONFIG['num_workers']})"
    )
    train_parser.add_argument(
        "--optimizer-betas", type=float, nargs=2, default=DEFAULT_CONFIG["optimizer_betas"],
        help=(
            f"AdamW beta coefficients for the first and second gradient moment estimates "
            f"(default: {' '.join(map(str, DEFAULT_CONFIG['optimizer_betas']))}). "
            "beta1 controls momentum; beta2 controls the adaptive learning rate. "
            "The defaults follow the original AdamW paper and are rarely changed."
        )
    )
    train_parser.add_argument(
        "--optimizer-eps", type=float, default=DEFAULT_CONFIG["optimizer_eps"],
        help=(
            f"AdamW numerical stability term added to the denominator (default: {DEFAULT_CONFIG['optimizer_eps']}). "
            "Prevents division by zero when gradient variance is near zero. Rarely needs adjustment."
        )
    )
    train_parser.add_argument(
        "--optimizer-weight-decay", type=float, default=DEFAULT_CONFIG["optimizer_weight_decay"],
        help=(
            f"AdamW decoupled weight decay (L2 regularization) coefficient (default: {DEFAULT_CONFIG['optimizer_weight_decay']}). "
            "Applied to all parameters except biases and layer-norm weights. "
            "Increase to reduce overfitting on small datasets."
        )
    )
    train_parser.add_argument(
        "--scheduler-warmup-ratio", type=float, default=DEFAULT_CONFIG["scheduler_warmup_ratio"],
        help=(
            f"Fraction of total training steps used for linear learning rate warmup (default: {DEFAULT_CONFIG['scheduler_warmup_ratio']}). "
            "The learning rate rises linearly from 0 to --learning-rate over this fraction, "
            "then decays linearly back to 0 over the remainder. "
            "Warmup stabilizes early training when classification head weights are randomly initialized."
        )
    )
    train_parser.add_argument(
        "--export-sequence-embeddings", action="store_true",
        help=(
            "Save the fused embedding vector for every validation sequence to the checkpoint directory. "
            "Embeddings can be used for downstream visualization (e.g., t-SNE) or transfer learning. "
            "Increases memory usage proportionally to the validation set size."
        )
    )
    train_parser.add_argument(
        "--export-permutation-importance", action="store_true",
        help=(
            "Compute and save permutation feature importance scores for the validation set. "
            "Sequence positions are shuffled in non-overlapping regions; the drop in accuracy "
            "estimates how much each region contributes to classification. "
            "Runtime scales with sequence length and validation set size."
        )
    )
    train_parser.add_argument(
        "--perm-region-size", type=int, default=None,
        help=(
            "Width of the shuffled regions used for permutation importance (in tokens). "
            "Smaller values give finer-grained position-level importance estimates at higher compute cost. "
            "Only used when --export-permutation-importance is set."
        )
    )
    train_parser.add_argument(
        "--focal-weight", type=str, default=None,
        help=(
            "Comma-separated per-class weights for focal loss (only used with --loss-type focal). "
            "Must have exactly as many values as there are classes. Typically set to inverse class "
            "frequencies to further down-weight abundant classes. See --class-weights for a "
            "built-in automatic weighting strategy."
        )
    )
    train_parser.add_argument(
        "--focal-reduction", type=str, choices=["mean", "sum"], default=DEFAULT_CONFIG["focal_reduction"],
        help=(
            f"How to reduce focal loss across the batch (default: {DEFAULT_CONFIG['focal_reduction']}). "
            "Only used with --loss-type focal. "
            "'mean': divide by the number of sequences in the batch (scale-invariant to batch size). "
            "'sum': sum over the batch (loss magnitude scales with batch size)."
        )
    )
    train_parser.add_argument(
        "--encoding", type=str, choices=["dnabert", "onehot"], default=DEFAULT_CONFIG["encoding"],
        help=(
            f"Sequence encoding method (default: {DEFAULT_CONFIG['encoding']}). "
            "'dnabert': byte-pair encoding via the DNABERT-2 tokenizer; captures subword patterns "
            "and benefits from pretraining. "
            "'onehot': encodes each nucleotide as a 4-channel one-hot vector (A/C/G/T); "
            "simpler but ignores higher-order sequence patterns."
        )
    )
    train_parser.add_argument(
        "--loss-type", type=str, choices=["focal", "cross_entropy"], default=DEFAULT_CONFIG["loss_type"],
        help=(
            f"Loss function used during training (default: {DEFAULT_CONFIG['loss_type']}). "
            "'cross_entropy': standard cross-entropy loss, suitable for most datasets. "
            "'focal': focal loss (Lin et al. 2017), which down-weights well-classified examples "
            "and concentrates gradient updates on harder cases; recommended when class imbalance "
            "is severe. Controlled by --focal-gamma."
        )
    )
    train_parser.add_argument(
        "--no-level-weights", action="store_true",
        help="Use uniform level weights (all 1.0) instead of hierarchical weighting"
    )
    train_parser.add_argument(
        "--no-mask-padding", action="store_false", dest="mask_padding",
        default=DEFAULT_CONFIG["mask_padding"],
        help=(
            "Include padded positions in the CNN and hybrid max pooling. By default "
            "padding is masked out so it cannot leak into the pooled features. This "
            "flag restores the older unmasked behavior for comparison; the setting is "
            "recorded in the checkpoint so prediction reproduces how the model was trained."
        )
    )
    train_parser.add_argument(
        "--early-stopping-patience", type=int, default=DEFAULT_CONFIG["early_stopping_patience"],
        help=f"Epochs without improvement before early stopping (0 = disabled, default: {DEFAULT_CONFIG['early_stopping_patience']})"
    )
    train_parser.add_argument(
        "--early-stopping-min-delta", type=float, default=DEFAULT_CONFIG["early_stopping_min_delta"],
        help=f"Minimum improvement to reset patience counter (default: {DEFAULT_CONFIG['early_stopping_min_delta']})"
    )
    train_parser.add_argument(
        "--class-weights", type=str, choices=["none", "inverse", "sqrt_inverse"],
        default="none",
        help="Per-class weighting strategy for loss function: "
             "'none' (default), 'inverse' (weight inversely proportional to class frequency), "
             "'sqrt_inverse' (square root of inverse frequency, less aggressive)"
    )
    train_parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    add_shared_args(train_parser)

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Predict taxonomy using a trained model")
    predict_parser.add_argument("--fasta-file", type=str, required=True, help="Path to FASTA file")
    predict_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    predict_parser.add_argument(
        "--taxonomy-file", type=str, default=None,
        help="Path to taxonomy file for true labels"
    )
    predict_parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"],
        help=f"Output directory (default: {DEFAULT_CONFIG['output_dir']})"
    )
    predict_parser.add_argument(
        "--tabular", action="store_true",
        help=(
            "Write predictions to a TSV file (one row per sequence) in addition to the default "
            "JSON output. Column names are prefixed by rank (e.g., species_predicted, "
            "species_raw_score). Use --tabular-fields to select which columns to include."
        )
    )
    predict_parser.add_argument(
        "--tabular-fields", type=str, default="predicted,raw_score,entropy,true,agreement",
        help=(
            "Comma-separated fields to include in TSV output (default: predicted,raw_score,entropy,true,agreement). "
            "Available fields: 'predicted' (top predicted label), 'raw_score' (softmax confidence), "
            "'entropy' (Shannon entropy of the full distribution), "
            "'true' (ground-truth label, requires --taxonomy-file), "
            "'agreement' (whether prediction matches ground truth)."
        )
    )
    predict_parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
        help=f"Batch size (default: {DEFAULT_CONFIG['batch_size']})"
    )
    predict_parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"],
        help=f"Number of DataLoader workers (default: {DEFAULT_CONFIG['num_workers']})"
    )
    predict_parser.add_argument(
        "--seed", type=int, default=DEFAULT_CONFIG["seed"],
        help=f"Random seed (default: {DEFAULT_CONFIG['seed']})"
    )
    predict_parser.add_argument(
        "--top-k", type=int, default=DEFAULT_CONFIG["top_k"],
        help=(
            f"Number of top-ranked classes counted when scoring top-k accuracy (default: {DEFAULT_CONFIG['top_k']}). "
            "A prediction counts as correct when the true label is among the k highest-probability classes. "
            "This only affects the top-k accuracy metric reported when a taxonomy file is provided; "
            "the per-sequence output always reports the single highest-probability class per rank."
        )
    )
    predict_parser.add_argument(
        "--confidence-threshold", type=float, default=DEFAULT_CONFIG["confidence_threshold"],
        help=(
            f"Softmax confidence threshold below which a prediction is flagged as low-confidence "
            f"(default: {DEFAULT_CONFIG['confidence_threshold']}). "
            "Flagged predictions are marked in the output but still reported; "
            "they can be filtered in downstream analysis."
        )
    )
    predict_parser.add_argument(
        "--level-temperatures", type=float, nargs='+', default=DEFAULT_CONFIG["level_temperatures"],
        help=(
            f"Seven space-separated temperature scalars, one per rank in order "
            f"Domain Phylum Class Order Family Genus Species "
            f"(default: {' '.join(map(str, DEFAULT_CONFIG['level_temperatures']))}). "
            "Each logit vector is divided by its temperature before softmax: values above 1.0 "
            "soften the distribution (lower confidence); values below 1.0 sharpen it. "
            "Use this to recalibrate confidence scores without retraining."
        )
    )
    predict_parser.add_argument(
        "--use-raw-labels-for-true", action="store_true",
        help=(
            "Match predictions against the raw taxonomy strings from --taxonomy-file without "
            "normalizing to the training vocabulary. By default, true labels are looked up in "
            "the label encoder stored in the checkpoint; labels absent from the training vocabulary "
            "are treated as novel and marked accordingly."
        )
    )
    predict_parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    add_shared_args(predict_parser)

    # Describe subcommand
    describe_parser = subparsers.add_parser("describe", help="Describe a trained model checkpoint")
    describe_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    describe_parser.add_argument(
        "--export-metrics", type=str, default=None,
        help="Path to export metrics as JSON"
    )
    describe_parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Process focal_weight if provided for training
    if hasattr(args, "focal_weight") and args.command == "train" and args.focal_weight:
        args.focal_weight = torch.tensor([float(w) for w in args.focal_weight.split(",")])

    return args

def main():
    """
    Execute the DeepTaxa CLI based on parsed arguments.

    Workflow:
        1. Parse CLI arguments and configure logging verbosity.
        2. Dispatch to appropriate module (train, predict, describe) based on command.
        3. Handle invalid commands with user feedback.

    Raises:
        SystemExit: If no command is specified.
    """
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "describe":
        describe(args)
    else:
        print("No command specified. Use --help for usage.")
        sys.exit(1)

if __name__ == "__main__":
    main()

