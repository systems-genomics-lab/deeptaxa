"""
Module: predict.py

Description:
    Implements the prediction pipeline for DeepTaxa models, enabling inference on new 16S rRNA sequences.
    Loads a trained model checkpoint, processes input sequences, and generates taxonomic predictions across
    multiple hierarchical ranks. Optionally evaluates against ground truth, computing metrics such as
    accuracy, F1, and AUC. Configurable via CLI, it outputs detailed predictions in JSON and optionally
    TSV formats, leveraging PyTorch for efficient computation and scikit-learn for robust evaluation.

    This pipeline performs inference via a forward pass through a trained model, converting raw logits to
    probabilities using softmax with temperature scaling for rank-specific sharpness control.
"""

import os
import json
import logging
import time
import torch
import uuid
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from deeptaxa.dataset import TaxonomyDataset, custom_collate_fn
from deeptaxa.utils import set_seed, get_device_info, check_file
from deeptaxa.models import CNNClassifier, BERTClassifier, HybridCNNBERTClassifier
from deeptaxa import __version__
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

def save_metrics_json(args, checkpoint, performance_stats, prediction_time, post_process_time, total_sequences, run_uuid, taxonomic_ranks, num_labels_per_level, model):
    """
    Save performance metrics to a JSON file in the output directory.

    Args:
        args: CLI arguments containing file paths and inference settings.
        checkpoint: Loaded checkpoint with model metadata.
        performance_stats: Dictionary of per-rank performance metrics.
        prediction_time: Time taken for prediction (seconds).
        post_process_time: Time taken for post-processing (seconds).
        total_sequences: Number of sequences processed.
        run_uuid: Unique identifier for the prediction run.
        taxonomic_ranks: List of taxonomic ranks (e.g., ["phylum", "class", ...]).
        num_labels_per_level: Dictionary mapping level indices to number of labels.
        model: Loaded DeepTaxa model instance.

    Saves:
        A JSON file named 'metrics.json' in args.output_dir, overwriting any existing file.
    """
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    metrics_data = {
        "version": f"deeptaxa.v{__version__}",
        "run_uuid": run_uuid,
        "timestamp": datetime.now().isoformat(),
        "model_details": {
            "model_type": checkpoint["model_type"],
            "tokenizer_name": checkpoint["tokenizer_name"],
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "max_length": checkpoint.get("max_length"),
            "dropout_prob": checkpoint.get("hidden_dropout_prob", 0.2)
        },
        "prediction_parameters": {
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "confidence_threshold": args.confidence_threshold,
            "level_temperatures": checkpoint.get("level_temperatures", args.level_temperatures)
        },
        "dataset_info": {
            "total_sequences": total_sequences,
            "fasta_file": args.fasta_file,
            "taxonomy_file": args.taxonomy_file or "N/A"
        },
        "taxonomic_levels": {
            str(level): {"rank": rank, "labels": num_labels_per_level[str(level)]}
            for level, rank in enumerate(taxonomic_ranks)
        },
        "timing": {
            "prediction_time_seconds": prediction_time,
            "post_process_time_seconds": post_process_time
        },
        "performance_metrics": {
            str(level): {
                "accuracy": stats["accuracy"],
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1_score": stats["f1_score"],
                "top_k_accuracy": stats.get("top_k_accuracy"),
                "auc": stats.get("auc"),
                "ece": stats.get("ece")
            } for level, (rank, stats) in enumerate(performance_stats.items()) if args.taxonomy_file
        },
        "system_info": {
            "cuda_available": str(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }
    }
    
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)
        logger.info("Saved metrics to %s", metrics_path)
    except Exception as e:
        logger.error("Failed to save metrics to %s: %s", metrics_path, str(e))
        raise

def predict(args):
    """
    Execute the prediction pipeline for taxonomic classification of DNA sequences.

    Workflow:
        1. Configure logging verbosity to balance detail and brevity based on user preference.
        2. Validate input files (FASTA, checkpoint, optional taxonomy) to ensure data integrity.
        3. Set up computation device (GPU/CPU) and load model from checkpoint with exact architecture.
        4. Initialize dataset and DataLoader with checkpoint-consistent tokenization parameters.
        5. Perform inference: forward pass to logits, softmax to probabilities, and top-k label extraction.
        6. If ground truth is provided, compute per-rank metrics (accuracy, F1, AUC) for evaluation.
        7. Post-process predictions with normalized scores and save results in JSON and optionally TSV.

    Args:
        args: CLI arguments containing file paths (fasta_file, checkpoint, taxonomy_file), inference
              settings (batch_size, top_k, confidence_threshold), and output options (output_dir, tabular).

    Raises:
        ValueError: If checkpoint lacks required keys (e.g., 'model_type', 'state_dict') or input files are invalid.
        FileNotFoundError: If any required input file is missing or inaccessible.

    Design Notes:
        - Stateless design relies on checkpoint for model architecture, ensuring reproducibility but requiring
          a fully self-contained checkpoint. CLI overrides are limited to inference behavior (e.g., top_k),
          not architecture, to prevent mismatches with trained state.
        - Optimized for GPU throughput with pin_memory and multi-worker data loading, though I/O may bottleneck
          for very large datasets. Consider streaming outputs for scalability in future iterations.
        - Multi-model support (CNN, BERT, Hybrid) uses checkpoint-driven initialization, avoiding ambiguity
          in parameter sourcing (e.g., max_length vs. max_position_embeddings).
    """
    # Configure logging verbosity for user feedback and debugging
    if args.verbose:
        logger.setLevel(logging.DEBUG)  # Detailed tracing for inference steps, e.g., batch timing
    else:
        logger.setLevel(logging.INFO)   # Concise output for production use, focusing on key milestones

    start_time = time.time()
    start_timestamp = datetime.fromtimestamp(start_time).isoformat()
    banner = f"""
{'=' * 70}
          DeepTaxa Prediction Session (v{__version__})
{'-' * 50}
          Started: {start_timestamp}
{'=' * 70}
    """
    logger.info(banner)

    # Validate input files to prevent runtime errors
    check_file(args.fasta_file)    # FASTA file with 16S rRNA sequences
    check_file(args.checkpoint)    # Checkpoint with trained model state and config
    if args.taxonomy_file:
        check_file(args.taxonomy_file)  # Optional TSV with ground truth labels

    # Set random seed and computation device
    set_seed(args.seed)  # Controls randomness for reproducibility (e.g., NumPy ops, potential stochastic inference)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_device_info(verbose=args.verbose)  # Logs GPU details if verbose, aiding hardware optimization

    # Load checkpoint with full metadata (not just weights)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    required_keys = ['model_type', 'num_labels_per_level', 'tokenizer_name', 'level_label2id', 'taxonomic_ranks']
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing required key: {key}")

    # Extract state dictionary for model weights
    state_dict = checkpoint.get('state_dict')
    if state_dict is None:
        raise ValueError("Checkpoint missing required key: 'state_dict'")

    model_type = checkpoint['model_type'].lower()
    num_labels_per_level = checkpoint['num_labels_per_level']
    tokenizer_name = checkpoint.get('tokenizer_name')
    if tokenizer_name is None:
        raise ValueError("Checkpoint missing required key: 'tokenizer_name'")

    # Extract architecture parameters from checkpoint
    hidden_dropout_prob = checkpoint.get('hidden_dropout_prob', 0.2)  # Fallback for legacy checkpoints
    model_config = checkpoint.get('model_config', {})  # Contains model-specific configs (CNN, BERT)
    # Determine encoding: check checkpoint metadata, then infer from CNN weights if missing
    encoding = checkpoint.get('encoding', None)
    if encoding is None and model_type == 'cnn':
        # Infer from conv weight shape: onehot uses 4 input channels, dnabert uses 512
        first_conv_key = next((k for k in state_dict if 'conv_stacks.0.0.conv.0.weight' in k), None)
        if first_conv_key and state_dict[first_conv_key].shape[1] == 4:
            encoding = 'onehot'
            logger.info("Inferred one-hot encoding from checkpoint conv weight shape")
        else:
            encoding = 'dnabert'
    elif encoding is None:
        encoding = 'dnabert'

    # Initialize model with exact checkpoint parameters, avoiding CLI overrides for architecture
    if model_type == 'hybridcnnbert':
        cnn_config = model_config.get('cnn', {})
        bert_config = model_config.get('bert', {}).__dict__ if hasattr(model_config.get('bert', {}), '__dict__') else model_config.get('bert', {})
        
        # Extract parameters ensuring compatibility with trained state
        embed_dim = cnn_config['embed_dim']  # CNN embedding size
        num_filters = cnn_config['num_filters']  # Filters per CNN branch
        kernel_sizes = cnn_config['kernel_sizes']  # Multi-scale kernel sizes
        num_conv_layers = cnn_config['num_conv_layers']  # CNN layer count
        # Use max_position_embeddings for BERT architecture, falling back to top-level max_length
        max_length = bert_config.get('max_position_embeddings', checkpoint.get('max_length'))
        if max_length is None:
            logger.error("max_position_embeddings and max_length missing from checkpoint; cannot determine BERT sequence length")
            raise ValueError("Checkpoint lacks max_length information for HybridCNNBERTClassifier")
        hidden_size = bert_config['hidden_size']  # BERT hidden state size
        num_hidden_layers = bert_config['num_hidden_layers']  # Transformer layers
        num_attention_heads = bert_config['num_attention_heads']  # Attention heads
        intermediate_size = bert_config['intermediate_size']  # Feed-forward size
        output_attentions = bert_config.get('output_attentions', False)  # Optional attention outputs

        # Log configuration for transparency and debugging
        logger.debug("Loading HybridCNNBERTClassifier with checkpoint parameters:")
        logger.debug(f"  embed_dim: {embed_dim}")
        logger.debug(f"  num_filters: {num_filters}")
        logger.debug(f"  kernel_sizes: {kernel_sizes}")
        logger.debug(f"  num_conv_layers: {num_conv_layers}")
        logger.debug(f"  max_length (max_position_embeddings): {max_length}")
        logger.debug(f"  hidden_size: {hidden_size}")
        logger.debug(f"  num_hidden_layers: {num_hidden_layers}")
        logger.debug(f"  num_attention_heads: {num_attention_heads}")
        logger.debug(f"  intermediate_size: {intermediate_size}")
        logger.debug(f"  output_attentions: {output_attentions}")

        model = HybridCNNBERTClassifier(
            tokenizer_name=tokenizer_name,
            num_labels_per_level=num_labels_per_level,
            hidden_dropout_prob=hidden_dropout_prob,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            num_conv_layers=num_conv_layers,
            max_length=max_length,  # Sets max_position_embeddings in BertConfig
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            output_attentions=output_attentions
        )
    elif model_type == 'cnn':
        cnn_config = model_config if isinstance(model_config, dict) else model_config.__dict__
        embed_dim = cnn_config.get('embed_dim')
        num_filters = cnn_config.get('num_filters')
        kernel_sizes = cnn_config.get('kernel_sizes')
        num_conv_layers = cnn_config.get('num_conv_layers')
        if embed_dim is None or num_filters is None or kernel_sizes is None or num_conv_layers is None:
            raise ValueError("Checkpoint missing required CNN parameters in model_config")
        max_length = checkpoint.get('max_length')
        input_mode = "onehot" if encoding == "onehot" else "dnabert"
        model = CNNClassifier(
            tokenizer_name=tokenizer_name,
            num_labels_per_level=num_labels_per_level,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            num_conv_layers=num_conv_layers,
            hidden_dropout_prob=hidden_dropout_prob,
            input_mode=input_mode
        )
    elif model_type == 'bert':
        bert_config = model_config.__dict__ if hasattr(model_config, '__dict__') else model_config
        max_length = bert_config.get('max_position_embeddings', checkpoint.get('max_length'))
        if max_length is None:
            logger.error("max_position_embeddings and max_length missing from checkpoint; cannot determine BERT sequence length")
            raise ValueError("Checkpoint lacks max_length information for BERTClassifier")
        hidden_size = bert_config.get('hidden_size')
        num_hidden_layers = bert_config.get('num_hidden_layers')
        num_attention_heads = bert_config.get('num_attention_heads')
        intermediate_size = bert_config.get('intermediate_size')
        if hidden_size is None or num_hidden_layers is None or num_attention_heads is None or intermediate_size is None:
            raise ValueError("Checkpoint missing required BERT parameters in model_config")
        model = BERTClassifier(
            tokenizer_name=tokenizer_name,
            num_labels_per_level=num_labels_per_level,
            hidden_dropout_prob=hidden_dropout_prob,
            max_length=max_length,  # Sets max_position_embeddings in BertConfig
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size
        )
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    # Load state dictionary and prepare model for inference
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Disables dropout and batch norm updates for deterministic inference
    logger.info("Model loaded from checkpoint: %s with type: %s", args.checkpoint, model_type)

    # Prediction-specific parameters (CLI overrides allowed for inference behavior)
    level_temperatures = checkpoint.get('level_temperatures', args.level_temperatures)
    top_k = args.top_k  # CLI override controls number of top predictions returned
    confidence_threshold = args.confidence_threshold  # CLI override filters low-confidence outputs

    # Prepare dataset and data loader with checkpoint-consistent max_length
    level_label2id = checkpoint['level_label2id']
    taxonomic_ranks = checkpoint['taxonomic_ranks']
    id2label = {lvl: {v: k for k, v in level_label2id[lvl].items()} for lvl in level_label2id}

    dataset = TaxonomyDataset(
        fasta_file=args.fasta_file,
        taxonomy_file=args.taxonomy_file,
        tokenizer_name=tokenizer_name,
        max_length=max_length,  # Ensures tokenization matches trained model’s sequence length
        encoding=encoding,  # Must match encoding used during training
        use_raw_labels_for_true=bool(args.taxonomy_file)  # Always include raw labels when evaluating against ground truth
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # CLI-controlled for inference batching flexibility
        shuffle=False,  # Preserves sequence order for consistent output mapping
        num_workers=args.num_workers or min(os.cpu_count(), 4),  # Parallelizes data loading
        pin_memory=True,  # Accelerates GPU data transfer by pinning CPU memory
        collate_fn=custom_collate_fn  # Custom collation for batched tensor assembly
    )

    # Storage for predictions and evaluation metrics
    predictions = []  # List of per-sequence prediction dicts
    true_labels = [] if args.taxonomy_file else None  # Ground truth labels if provided
    score_distributions = {rank: [] for rank in taxonomic_ranks}  # Raw confidence scores per rank
    correct_counts = {rank: 0 for rank in taxonomic_ranks} if args.taxonomy_file else {}  # Exact matches
    top_k_correct_counts = {rank: 0 for rank in taxonomic_ranks} if args.taxonomy_file else {}  # Top-k matches
    all_true_labels = {rank: [] for rank in taxonomic_ranks} if args.taxonomy_file else {}  # True label IDs
    all_pred_labels = {rank: [] for rank in taxonomic_ranks} if args.taxonomy_file else {}  # Predicted label IDs
    all_pred_probs = {rank: [] for rank in taxonomic_ranks} if args.taxonomy_file else {}  # Probability distributions (AUC-eligible ranks only)
    ece_pred_confs = {rank: [] for rank in taxonomic_ranks} if args.taxonomy_file else {}  # Max confidence per prediction (scalars for ECE)
    ece_pred_correct = {rank: [] for rank in taxonomic_ranks} if args.taxonomy_file else {}  # Correctness flags (scalars for ECE)
    MAX_CLASSES_FOR_AUC = 100  # Caps AUC computation and full-prob storage to manage memory

    # Inference loop: process batches without gradient computation
    with torch.no_grad():  # Disables gradient tracking, reducing memory footprint
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Predicting")):
            input_ids = batch['input_ids'].to(device)  # Tokenized sequences
            attention_mask = batch.get('attention_mask')  # Padding mask, optional for CNN
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass to obtain logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs  # Handle Hybrid’s (logits, attentions) tuple
            batch_size = len(batch['input_ids'])

            # Process each sequence in the batch
            for i in range(batch_size):
                pred_dict = {}
                # Read the id and length straight from the batch instead of
                # recomputing a position, so they stay tied to the right sequence
                # even if the loader order or batch size ever changes.
                seq_id = batch['seq_ids'][i]
                sequence_length = int(batch['seq_lengths'][i])

                # Compute predictions per taxonomic level
                for lvl_str, level_logits in logits.items():
                    lvl_idx = int(lvl_str)
                    rank = taxonomic_ranks[lvl_idx]
                    # Convert logits to probabilities with temperature scaling
                    probs = softmax(level_logits[i] / level_temperatures[lvl_idx], dim=0)
                    pred_id = torch.argmax(probs).item()
                    pred_label = id2label[lvl_idx][pred_id]
                    pred_score = probs[pred_id].item()
                    score_distributions[rank].append(pred_score)

                    # Compute entropy as an uncertainty metric
                    entropy = max(0, -torch.sum(probs * torch.log(probs + 1e-10)).item())
                    entropy = round(entropy, 4)

                    # Extract top-k predictions for confidence analysis
                    num_classes = probs.shape[0]
                    k = min(top_k, num_classes)  # Caps k to avoid index errors with small class counts
                    top_probs, top_ids = torch.topk(probs, k=k)
                    top_k_predictions = [
                        {"label": id2label[lvl_idx][idx.item()], "raw_score": round(score.item(), 4)}
                        for idx, score in zip(top_ids, top_probs)
                    ]
                    top_k_confidence_gap = round(top_probs[0].item() - top_probs[1].item(), 4) if len(top_probs) > 1 else 1.0

                    # Store prediction details in a structured dict
                    # Note: top_k list is NOT stored per-prediction to avoid OOM on large datasets
                    # (277K seqs × 7 ranks × 5 entries = ~4GB of Python dicts)
                    pred_dict[rank] = {
                        "label": pred_label,
                        "raw_score": pred_score,
                        "confidence_threshold_flag": pred_score >= confidence_threshold,
                        "entropy": entropy,
                    }

                    # Evaluate against ground truth if provided
                    if args.taxonomy_file:
                        true_label = batch['raw_labels'][i][rank]
                        # At species rank a model may emit a bare epithet while the
                        # reference stores a full binomial or a prefixed label. Allow
                        # that only when the epithet lines up on a name boundary, so an
                        # empty prediction or an arbitrary suffix does not count as a hit.
                        agreement = pred_label == true_label or (
                            rank == "species"
                            and pred_label != ""
                            and (true_label.endswith(" " + pred_label) or true_label.endswith("__" + pred_label))
                        )
                        pred_dict[rank]["agreement"] = agreement
                        if not agreement:
                            pred_dict[rank]["mismatch_detail"] = {"predicted": pred_label, "true": true_label}
                        if agreement:
                            correct_counts[rank] += 1
                        top_k_labels = [pred["label"] for pred in top_k_predictions]
                        if true_label in top_k_labels:
                            top_k_correct_counts[rank] += 1

                        true_id = level_label2id[lvl_idx].get(true_label, -1)
                        # Novel labels (true_id == -1) are included in F1 so it stays
                        # consistent with accuracy, which already counts those sequences
                        # as wrong. ECE and AUC still require valid training-class IDs,
                        # so we skip their accumulators for novel-label sequences.
                        all_true_labels[rank].append(true_id)
                        all_pred_labels[rank].append(pred_id)
                        if true_id == -1:
                            continue
                        # Lightweight ECE accumulators: store only scalars to avoid OOM on large datasets
                        ece_pred_confs[rank].append(pred_score)
                        ece_pred_correct[rank].append(float(pred_id == true_id))
                        # Only store full probability vectors for AUC-eligible ranks (low class count)
                        n_labels_for_rank = int(num_labels_per_level[str(lvl_idx)])
                        if n_labels_for_rank <= MAX_CLASSES_FOR_AUC:
                            all_pred_probs[rank].append(probs.cpu().numpy())

                pred_dict["sequence_length"] = sequence_length
                predictions.append(pred_dict)

                if args.taxonomy_file:
                    true_labels.append(batch['raw_labels'][i])

    # Post-process predictions for output consistency
    prediction_time = time.time() - start_time
    post_process_start_time = time.time()

    # Normalize scores relative to maximum per rank
    score_max_per_rank = {rank: max(scores) for rank, scores in score_distributions.items() if scores}
    for pred in predictions:
        for rank in pred:
            if rank == "sequence_length":
                continue
            raw = pred[rank]['raw_score']
            norm = round(raw / score_max_per_rank[rank], 4) if score_max_per_rank[rank] else raw
            pred[rank]['score'] = norm

    total_sequences = len(dataset)
    end_timestamp = datetime.fromtimestamp(time.time()).isoformat()

    # Define fasta_base before logging to avoid UnboundLocalError
    fasta_path = Path(args.fasta_file)
    fasta_base = fasta_path.name
    while fasta_base != Path(fasta_base).stem:
        fasta_base = Path(fasta_base).stem  # Iteratively strips extensions (e.g., .fna.gz -> .fna)

    run_uuid = str(uuid.uuid4()).replace('-', '_')
    os.makedirs(args.output_dir, exist_ok=True)
    uuid_file = os.path.join(args.output_dir, "deeptaxa_uuid.txt")
    try:
        with open(uuid_file, "w") as f:
            f.write(run_uuid)
        logger.info("Saved run UUID to %s", uuid_file)
    except Exception as e:
        logger.error("Failed to save UUID to %s: %s", uuid_file, str(e))
        raise

    logger.info(f"""
{'=' * 70}
          DeepTaxa Prediction Summary (v{__version__})
{'-' * 50}
          Here's a summary of your prediction results:
          - Total Sequences Processed: {total_sequences:,}
          - Prediction Time: {prediction_time:.2f} seconds
          - Completed At: {end_timestamp}
{'=' * 70}
    """)

    # Compute and log performance metrics if ground truth is available
    performance_stats = {}
    table_header = f"{'Rank':<10} | {'Mean Score':<12} | {'Std Score':<12} | {'Accuracy':<12} | {f'Top-{top_k} Acc':<12} | {'F1':<8} | {'Precision':<12} | {'Recall':<12} | {'AUC':<12}"
    table_separator = '-' * len(table_header)
    logger.info("Performance Metrics by Taxonomic Rank:")
    logger.info(table_header)
    logger.info(table_separator)

    for rank in taxonomic_ranks:
        scores = score_distributions[rank]
        mean_score = np.mean(scores) if scores else 0
        std_score = np.std(scores) if scores else 0

        if args.taxonomy_file and all_true_labels[rank]:
            accuracy = correct_counts[rank] / total_sequences
            top_k_accuracy = top_k_correct_counts[rank] / total_sequences
            y_true = np.array(all_true_labels[rank], dtype=int).flatten()
            y_pred = np.array(all_pred_labels[rank], dtype=int)

            # Weighted metrics mitigate class imbalance effects
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

            # Compute AUC with class count limitation for scalability
            # Exclude the novel-label sentinel (-1) from AUC; F1 uses the full arrays.
            # Probability rows are only stored for non-novel sequences, so the true
            # labels must be trimmed the same way to keep the two arrays row-aligned.
            unique_true_labels = np.unique(y_true[y_true >= 0])
            y_true_auc = y_true[y_true >= 0]
            auc = None
            if len(unique_true_labels) > 1:
                if all_pred_probs[rank] and len(unique_true_labels) <= MAX_CLASSES_FOR_AUC:
                    y_score = np.stack(all_pred_probs[rank])
                    label_ids = sorted(unique_true_labels)
                    y_score_filtered = y_score[:, label_ids]
                    row_sums = y_score_filtered.sum(axis=1, keepdims=True)
                    y_score_filtered = y_score_filtered / np.where(row_sums == 0, 1, row_sums)

                    try:
                        if len(unique_true_labels) == 2:
                            auc = roc_auc_score(y_true_auc, y_score_filtered[:, 1])
                        else:
                            auc = roc_auc_score(y_true_auc, y_score_filtered, multi_class='ovr', average='weighted', labels=label_ids)
                    except ValueError as e:
                        logger.warning(f"Rank {rank}: AUC calculation failed: {str(e)}")
                else:
                    logger.info(f"Rank {rank}: Skipping AUC (too many classes: {len(unique_true_labels)} > {MAX_CLASSES_FOR_AUC})")
            auc_value = f"{auc:.4f}" if auc is not None else "N/A"
            row = f"{rank:<10} | {mean_score:<12.4f} | {std_score:<12.4f} | {accuracy:<12.4f} | {top_k_accuracy:<12.4f} | {f1:<8.4f} | {precision:<12.4f} | {recall:<12.4f} | {auc_value:<12}"
            performance_stats[rank] = {
                "mean_raw_score": float(mean_score),
                "std_raw_score": float(std_score),
                "correct": correct_counts[rank],
                "total": total_sequences,
                "accuracy": float(accuracy),
                "top_k_accuracy": float(top_k_accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "auc": float(auc) if auc is not None else None
            }
        else:
            row = f"{rank:<10} | {mean_score:<12.4f} | {std_score:<12.4f} | {'N/A':<12} | {'N/A':<12} | {'N/A':<8} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12}"
            performance_stats[rank] = {
                "mean_raw_score": float(mean_score),
                "std_raw_score": float(std_score)
            }
        logger.info(row)

    # Compute Expected Calibration Error (ECE) per rank if ground truth is available
    # Uses lightweight accumulators (scalars) instead of full probability vectors
    if args.taxonomy_file:
        n_bins = 10
        logger.info("\nCalibration Metrics (Expected Calibration Error):")
        logger.info(f"{'Rank':<10} | {'ECE':<10}")
        logger.info('-' * 25)
        for rank in taxonomic_ranks:
            if not ece_pred_confs[rank]:
                continue
            pred_confs = np.array(ece_pred_confs[rank])
            correct = np.array(ece_pred_correct[rank])
            n_total = len(pred_confs)
            ece = 0.0
            for bin_i in range(n_bins):
                lo = bin_i / n_bins
                hi = (bin_i + 1) / n_bins
                mask = (pred_confs > lo) & (pred_confs <= hi)
                if mask.sum() == 0:
                    continue
                bin_acc = correct[mask].mean()
                bin_conf = pred_confs[mask].mean()
                ece += mask.sum() / n_total * abs(bin_acc - bin_conf)
            logger.info(f"{rank:<10} | {ece:<10.4f}")
            if rank in performance_stats:
                performance_stats[rank]["ece"] = float(ece)

    # Free evaluation data structures to reclaim memory before JSON output
    del all_true_labels, all_pred_labels, all_pred_probs
    del ece_pred_confs, ece_pred_correct, score_distributions
    import gc; gc.collect()

    post_process_time = time.time() - post_process_start_time
    logger.debug(f"\n{'=' * 70}")
    logger.debug(f"Post-Processing Time (excluding I/O): {post_process_time:.2f} seconds")

    # Save metrics if ground truth is provided
    if args.taxonomy_file:
        save_metrics_json(args, checkpoint, performance_stats, prediction_time, post_process_time, total_sequences, run_uuid, taxonomic_ranks, num_labels_per_level, model)

    # Package and save output in structured formats
    output_data = {
        "sequence_ids": dataset.seq_ids,
        "predictions": predictions,
        "summary": {
            "correct_count_per_rank": correct_counts if args.taxonomy_file else {},
            "total_sequences": total_sequences,
            "top_k_correct_count_per_rank": top_k_correct_counts if args.taxonomy_file else {},
            "performance_stats": performance_stats,
            "prediction_time_seconds": prediction_time,
            "post_process_time_seconds": post_process_time,
            "prediction_parameters": {
                "top_k": top_k,
                "confidence_threshold": confidence_threshold,
                "level_temperatures": level_temperatures
            }
        }
    }
    if args.taxonomy_file:
        output_data["true_labels"] = true_labels

    os.makedirs(args.output_dir, exist_ok=True)
    json_file = os.path.join(args.output_dir, f"{fasta_base}_deeptaxa_predictions.json")
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info("Saved prediction results to: %s", json_file)

    if args.tabular:
        tabular_data = []
        fields = args.tabular_fields.split(',')
        valid_fields = {'predicted', 'raw_score', 'entropy', 'true', 'agreement'}
        for f in fields:
            if f not in valid_fields:
                raise ValueError(f"Invalid tabular field: {f}. Choose from {valid_fields}")

        # Build tabular output row-by-row
        for seq_id, pred in zip(dataset.seq_ids, predictions):
            row = {"sequence_id": seq_id}
            for rank in taxonomic_ranks:
                if 'predicted' in fields:
                    row[f"{rank}_predicted"] = pred[rank]["label"]
                if 'raw_score' in fields:
                    row[f"{rank}_raw_score"] = pred[rank]["raw_score"]
                if 'entropy' in fields:
                    row[f"{rank}_entropy"] = pred[rank]["entropy"]
                if args.taxonomy_file:
                    if 'true' in fields:
                        row[f"{rank}_true"] = true_labels[len(tabular_data)][rank]
                    if 'agreement' in fields:
                        row[f"{rank}_agreement"] = pred[rank]["agreement"]
            row["sequence_length"] = pred["sequence_length"]
            tabular_data.append(row)

        df = pd.DataFrame(tabular_data)
        tabular_file = os.path.join(args.output_dir, f"{fasta_base}_deeptaxa_predictions.tsv")
        df.to_csv(tabular_file, sep='\t', index=False)
        logger.info("Saved tabular prediction results to: %s", tabular_file)
    
    logger.info(f"\n{'=' * 70}\nThank you for using DeepTaxa\n{'=' * 70}")

if __name__ == "__main__":
    pass

