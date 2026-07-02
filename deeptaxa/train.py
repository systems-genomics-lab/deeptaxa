"""
Module: train.py

Description:
    Implements the training pipeline for DeepTaxa models, supporting CNN, BERT, and hybrid architectures
    for hierarchical taxonomy classification of 16S rRNA sequences. Manages data loading, model optimization,
    evaluation, and checkpointing, with options for resuming training and exporting diagnostics. Uses mixed
    precision to optimize GPU utilization, ensuring efficient and robust training for genomic data. Training
    minimizes a summed per-rank classification loss across taxonomic ranks (cross-entropy by default, with
    focal loss available for class imbalance), using AdamW with a linear warmup scheduler for stable convergence.
"""

import torch
import torch.nn as nn
import json
import os
import uuid
import h5py
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.amp import GradScaler, autocast
from deeptaxa.dataset import TaxonomyDataset, custom_collate_fn
from deeptaxa.models import CNNClassifier, BERTClassifier, HybridCNNBERTClassifier, FocalLoss
from deeptaxa.utils import set_seed, get_device_info, check_file
from deeptaxa.config import DEFAULT_CONFIG
import logging
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from deeptaxa import __version__
import optuna
import random
import hashlib
import math
import pickle

import warnings
warnings.simplefilter("ignore", UserWarning)  # Suppresses minor warnings for cleaner logs

logger = logging.getLogger(__name__)

def setup_model(args, num_labels_per_level):
    """Initialize a DeepTaxa model based on specified architecture."""
    model_classes = {
        "cnn": CNNClassifier,
        "bert": BERTClassifier,
        "hybridcnnbert": HybridCNNBERTClassifier
    }
    model_class = model_classes.get(args.model_type)
    if not model_class:
        logger.error("Unsupported model type: %s", args.model_type)
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model_args = {}
    common_args = ["hidden_dropout_prob"]
    for arg in common_args:
        model_args[arg] = getattr(args, arg, DEFAULT_CONFIG[arg])
    # Every architecture loads the tokenizer, so a pinned revision applies to all.
    model_args["tokenizer_revision"] = getattr(args, "tokenizer_revision", DEFAULT_CONFIG["tokenizer_revision"])

    if args.model_type in ["cnn", "hybridcnnbert"]:
        cnn_args = ["embed_dim", "num_filters", "kernel_sizes", "num_conv_layers"]
        for arg in cnn_args:
            model_args[arg] = getattr(args, arg, DEFAULT_CONFIG[arg])
        model_args["mask_padding"] = getattr(args, "mask_padding", DEFAULT_CONFIG["mask_padding"])

    if args.model_type in ["bert", "hybridcnnbert"]:
        bert_args = ["max_length", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"]
        for arg in bert_args:
            model_args[arg] = getattr(args, arg, DEFAULT_CONFIG[arg])

    if args.model_type == "hybridcnnbert":
        model_args["output_attentions"] = getattr(args, 'output_attentions', DEFAULT_CONFIG["output_attentions"])

    # Pass input_mode for one-hot encoding support (CNN only)
    if args.model_type == "cnn":
        encoding = getattr(args, 'encoding', 'dnabert')
        model_args["input_mode"] = "onehot" if encoding == "onehot" else "dnabert"

    model = model_class(args.tokenizer_name, num_labels_per_level, **model_args)
    return model

def setup_optimizer(model, args):
    """Configure AdamW optimizer for model training."""
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=getattr(args, "optimizer_betas", DEFAULT_CONFIG["optimizer_betas"]),
        eps=getattr(args, "optimizer_eps", DEFAULT_CONFIG["optimizer_eps"]),
        weight_decay=getattr(args, "optimizer_weight_decay", DEFAULT_CONFIG["optimizer_weight_decay"])
    )
    optimizer_config = {
        "lr": args.learning_rate,
        "betas": getattr(args, "optimizer_betas", DEFAULT_CONFIG["optimizer_betas"]),
        "eps": getattr(args, "optimizer_eps", DEFAULT_CONFIG["optimizer_eps"]),
        "weight_decay": getattr(args, "optimizer_weight_decay", DEFAULT_CONFIG["optimizer_weight_decay"])
    }
    logger.info("Optimizer initialized with config: %s", optimizer_config)
    return optimizer

def evaluate_model(model, val_loader, criterion, device, taxonomic_ranks, criteria=None, level_weights=None):
    """Evaluate model performance on the validation set.

    When level_weights is given, the validation loss weights each rank the same
    way the training loss does, so early stopping and checkpoint selection track
    the objective the model is actually trained on.
    """
    model.eval()
    total_loss = 0
    all_preds = {level: [] for level in range(len(taxonomic_ranks))}
    all_labels = {level: [] for level in range(len(taxonomic_ranks))}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast('cuda', enabled=torch.cuda.is_available()):
                output = model(input_ids, attention_mask)
                logits = output[0] if isinstance(output, tuple) else output
                loss = sum(
                    (level_weights[level] if level_weights is not None else 1.0)
                    * (criteria[level] if criteria else criterion)(logits[str(level)], labels[:, level])
                    for level in range(len(taxonomic_ranks))
                )
            total_loss += loss.item()
            
            for level in range(len(taxonomic_ranks)):
                preds = torch.argmax(logits[str(level)], dim=1).cpu().numpy()
                all_preds[level].extend(preds)
                all_labels[level].extend(labels[:, level].cpu().numpy())
    
    metrics = {}
    for level in range(len(taxonomic_ranks)):
        metrics[level] = {
            "accuracy": accuracy_score(all_labels[level], all_preds[level]),
            "precision": precision_score(all_labels[level], all_preds[level], average="weighted", zero_division=0),
            "recall": recall_score(all_labels[level], all_preds[level], average="weighted", zero_division=0),
            "f1_score": f1_score(all_labels[level], all_preds[level], average="weighted", zero_division=0)
        }
    avg_loss = total_loss / len(val_loader)
    logger.info("Evaluation metrics and loss computed for %d levels", len(taxonomic_ranks))
    return metrics, avg_loss

def compute_permutation_importance(model, val_loader, device, taxonomic_ranks, args):
    """Calculate permutation importance to assess feature relevance across sequence regions."""
    model.eval()
    baseline_preds = {level: [] for level in range(len(taxonomic_ranks))}
    baseline_labels = {level: [] for level in range(len(taxonomic_ranks))}
    region_size = getattr(args, 'perm_region_size', args.max_length // 4) or args.max_length // 4
    num_regions = args.max_length // region_size
    importance_scores = {level: np.zeros(num_regions) for level in range(len(taxonomic_ranks))}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Baseline Prediction"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast('cuda', enabled=torch.cuda.is_available()):
                output = model(input_ids, attention_mask)
                logits = output[0] if isinstance(output, tuple) else output
            for level in range(len(taxonomic_ranks)):
                preds = torch.argmax(logits[str(level)], dim=1).cpu().numpy()
                baseline_preds[level].extend(preds)
                baseline_labels[level].extend(labels[:, level].cpu().numpy())

    baseline_accuracy = {level: accuracy_score(baseline_labels[level], baseline_preds[level]) for level in range(len(taxonomic_ranks))}

    for region in range(num_regions):
        permuted_preds = {level: [] for level in range(len(taxonomic_ranks))}
        start = region * region_size
        end = min(start + region_size, args.max_length)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Permuting Region {region+1}/{num_regions}"):
                input_ids = batch["input_ids"].clone().to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                perm_indices = torch.randperm(input_ids.size(0))
                input_ids[:, start:end] = input_ids[perm_indices, start:end]
                with autocast('cuda', enabled=torch.cuda.is_available()):
                    output = model(input_ids, attention_mask)
                    logits = output[0] if isinstance(output, tuple) else output
                for level in range(len(taxonomic_ranks)):
                    preds = torch.argmax(logits[str(level)], dim=1).cpu().numpy()
                    permuted_preds[level].extend(preds)
        
        permuted_accuracy = {level: accuracy_score(baseline_labels[level], permuted_preds[level]) for level in range(len(taxonomic_ranks))}
        for level in range(len(taxonomic_ranks)):
            importance_scores[level][region] = baseline_accuracy[level] - permuted_accuracy[level]

    return importance_scores

def save_checkpoint(model, optimizer, scheduler, epoch, run_uuid, args, dataset_split, metrics, val_loss, taxonomic_ranks, session_total_time, checkpoint_total_time, train_loader, val_loader, device, total_steps, total_parameters, scaler=None, trial=None):
    """Save model checkpoint with comprehensive metadata."""
    logger.debug("Saving checkpoint for epoch %d, trial: %s", epoch, trial.number if trial else "None")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    metrics_dir = os.path.join(args.output_dir, "metrics")
    weights_dir = os.path.join(args.output_dir, "weights")
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
    except Exception as e:
        logger.error("Failed to create directories: %s", e)
        raise
    
    checkpoint_end_time = datetime.now().isoformat()

    # CPU random state
    cpu_rng_state = torch.get_rng_state().cpu()
    if not isinstance(cpu_rng_state, torch.ByteTensor):
        logger.warning("CPU RNG state is not a ByteTensor, converting to ByteTensor")
        cpu_rng_state = cpu_rng_state.to(dtype=torch.uint8)
    logger.debug("CPU RNG state hash: %s", hashlib.sha256(cpu_rng_state.cpu().numpy().tobytes()).hexdigest()[:16])
    
    # CUDA random state
    cuda_rng_state = None
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state().cpu()
        if not isinstance(cuda_rng_state, torch.ByteTensor):
            logger.warning("CUDA RNG state is not a ByteTensor, converting to ByteTensor")
            cuda_rng_state = cuda_rng_state.to(dtype=torch.uint8)
    if cuda_rng_state is not None:
        logger.debug("CUDA RNG state hash: %s", hashlib.sha256(cuda_rng_state.cpu().numpy().tobytes()).hexdigest()[:16])
    
    # Python random state
    python_rng_state = random.getstate()
    logger.debug("Python RNG state hash: %s", hashlib.sha256(pickle.dumps(python_rng_state)).hexdigest()[:16])

    # NumPy random state
    np_rng_state = np.random.get_state()
    logger.debug("NumPy RNG state hash: %s", hashlib.sha256(np_rng_state[1].tobytes()).hexdigest()[:16])
    
    checkpoint = {
        "model_type": args.model_type,
        "encoding": getattr(args, 'encoding', 'dnabert'),
        "state_dict": model.state_dict(),
        "hidden_dropout_prob": model.hidden_dropout_prob,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "scaler_init_scale": getattr(args, "scaler_init_scale", DEFAULT_CONFIG["scaler_init_scale"]),
        "epoch": epoch,
        "run_uuid": run_uuid,
        "session_start_time": args.session_start_time,
        "session_end_time": checkpoint_end_time,
        "session_total_time": session_total_time,
        "checkpoint_start_time": args.checkpoint_start_time,
        "checkpoint_end_time": checkpoint_end_time,
        "current_train_time": getattr(args, "current_train_time", 0.0),
        "current_eval_time": getattr(args, "current_eval_time", 0.0),
        "current_train_loss": getattr(args, "current_train_loss", 0.0),
        "val_loss": val_loss if val_loss is not None else 0.0,
        "metrics": metrics if metrics is not None else {},
        "num_labels_per_level": {str(k): v for k, v in model.num_labels_per_level.items()},
        "taxonomic_ranks": taxonomic_ranks,
        "model_config": (
            {"cnn": model.cnn_params, "bert": model.config.__dict__} if args.model_type == "hybridcnnbert"
            else model.config if args.model_type == "cnn"
            else model.config.__dict__
        ),
        "optimizer_config": {
            "lr": args.learning_rate,
            "betas": getattr(args, "optimizer_betas", DEFAULT_CONFIG["optimizer_betas"]),
            "eps": getattr(args, "optimizer_eps", DEFAULT_CONFIG["optimizer_eps"]),
            "weight_decay": getattr(args, "optimizer_weight_decay", DEFAULT_CONFIG["optimizer_weight_decay"])
        },
        "scheduler": {
            "num_warmup_steps": int(getattr(args, "scheduler_warmup_ratio", DEFAULT_CONFIG["scheduler_warmup_ratio"]) * total_steps),
            "num_training_steps": total_steps
        },
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "focal_gamma": getattr(args, "focal_gamma", DEFAULT_CONFIG["focal_gamma"]),
        "focal_reduction": getattr(args, "focal_reduction", DEFAULT_CONFIG["focal_reduction"]),
        "focal_weight": getattr(args, "focal_weight", DEFAULT_CONFIG["focal_weight"]),
        "level_weights": args.level_weights,
        "dataset_split": dataset_split,
        "dataset_size": args.dataset_size,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "fasta_file": args.fasta_file,
        "taxonomy_file": args.taxonomy_file,
        "max_length": args.max_length,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_revision": getattr(args, "tokenizer_revision", DEFAULT_CONFIG["tokenizer_revision"]),
        "level_label2id": args.level_label2id,
        "rng_state": {
            "torch": cpu_rng_state,
            "numpy": np_rng_state,
            "cuda": cuda_rng_state,
            "python": python_rng_state
        },
        "total_parameters": total_parameters
    }

    metrics_data = {
        "version": f"deeptaxa.v{__version__}",
        "run_uuid": run_uuid,
        "checkpoint_path": os.path.join(checkpoint_dir, f"deeptaxa_{run_uuid}_epoch{epoch}.pt"),
        "timestamp": checkpoint_end_time,
        "model_details": {
            "model_type": args.model_type,
            "tokenizer_name": args.tokenizer_name,
            "current_epoch": epoch,
            "total_parameters": total_parameters,
            "max_length": args.max_length,
            "dropout_prob": model.hidden_dropout_prob
        },
        "training_hyperparameters": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "target_epochs": args.epochs,
            "focal_gamma": getattr(args, "focal_gamma", DEFAULT_CONFIG["focal_gamma"]),
            "focal_reduction": getattr(args, "focal_reduction", DEFAULT_CONFIG["focal_reduction"]),
            "focal_weight": getattr(args, "focal_weight", DEFAULT_CONFIG["focal_weight"]),
            "level_weights": args.level_weights,
            "optimizer_summary": {
                "lr": args.learning_rate,
                "betas": getattr(args, "optimizer_betas", DEFAULT_CONFIG["optimizer_betas"]),
                "eps": getattr(args, "optimizer_eps", DEFAULT_CONFIG["optimizer_eps"]),
                "weight_decay": getattr(args, "optimizer_weight_decay", DEFAULT_CONFIG["optimizer_weight_decay"])
            },
            "scheduler_steps": total_steps
        },
        "dataset_info": {
            "total_sequences": args.dataset_size,
            "training_sequences": args.train_size,
            "validation_sequences": args.val_size,
            "fasta_file": args.fasta_file,
            "taxonomy_file": args.taxonomy_file
        },
        "taxonomic_levels": {
            str(level): {"rank": rank, "labels": model.num_labels_per_level[level]}
            for level, rank in enumerate(taxonomic_ranks)
        },
        "timing": {
            "training_time_seconds": getattr(args, "current_train_time", 0.0),
            "evaluation_time_seconds": getattr(args, "current_eval_time", 0.0),
            "session_total_time": session_total_time,
            "checkpoint_total_time": checkpoint_total_time,
            "session_start_time": args.session_start_time,
            "checkpoint_start_time": args.checkpoint_start_time,
            "checkpoint_end_time": checkpoint_end_time
        },
        "performance_metrics": {
            "training_loss": getattr(args, "current_train_loss", 0.0),
            "validation_loss": val_loss if val_loss is not None else 0.0,
            "validation_metrics": {str(k): v for k, v in metrics.items()} if metrics is not None else {}
        },
        "system_info": {
            "cuda_available": str(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }
    }
    metrics_path = os.path.join(metrics_dir, f"deeptaxa_{run_uuid}_epoch{epoch}.json")
    try:
        with open(metrics_path + ".tmp", "w") as f:
            json.dump(metrics_data, f, indent=4)
        os.replace(metrics_path + ".tmp", metrics_path)
        logger.info("Saved metrics to %s", metrics_path)
    except Exception as e:
        logger.error("Failed to save metrics to %s: %s", metrics_path, str(e))
        raise

    if trial is None:
        checkpoint_path = os.path.join(checkpoint_dir, f"deeptaxa_{run_uuid}_epoch{epoch}.pt")
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved model checkpoint to %s", checkpoint_path)
        except Exception as e:
            logger.error("Failed to save checkpoint to %s: %s", checkpoint_path, str(e))
            raise

        weights_path = os.path.join(weights_dir, f"deeptaxa_{run_uuid}_epoch{epoch}_layer_weights.h5")
        key_layers = []
        if args.model_type in ["cnn", "hybridcnnbert"]:
            key_layers.append("embedding.weight")
        if args.model_type == "bert":
            key_layers.append("bert.embeddings.word_embeddings.weight")
        if args.model_type == "cnn":
            key_layers.append("conv_stacks.0.0.conv.0.weight")
        if args.model_type == "hybridcnnbert":
            key_layers.append("conv_stacks.0.0.0.weight")
        if args.model_type in ["bert", "hybridcnnbert"]:
            key_layers.append("bert.encoder.layer.0.attention.self.query.weight")
        
        for level in model.num_labels_per_level.keys():
            if args.model_type == "cnn":
                key_layers.append(f"classifiers.{level}.3.weight")
            elif args.model_type in ["bert", "hybridcnnbert"]:
                key_layers.append(f"classifiers.{level}.weight")

        state_dict = model.state_dict()
        try:
            with h5py.File(weights_path, 'w') as f:
                for name in key_layers:
                    if name in state_dict:
                        f.create_dataset(name, data=state_dict[name].cpu().numpy(), compression='gzip', compression_opts=4)
                    else:
                        logger.warning("Layer %s not found in model state_dict", name)
                f.attrs['epoch'] = epoch
                f.attrs['run_uuid'] = run_uuid
            logger.info("Exported selected layer weights to %s", weights_path)
        except Exception as e:
            logger.error("Failed to export weights to %s: %s", weights_path, str(e))
            raise
        
        if getattr(args, 'export_sequence_embeddings', False) and args.model_type in ["cnn", "hybridcnnbert"]:
            embeddings_path = os.path.join(weights_dir, f"deeptaxa_{run_uuid}_epoch{epoch}_sequence_embeddings.h5")
            model.eval()
            max_sequences = getattr(args, 'max_sequences', 10000)  # Configurable via args
            seq_count = 0
            seq_ids = []
            embeddings_list = []

            try:
                with h5py.File(embeddings_path, 'w') as f:
                    emb_group = f.create_group('embeddings')
                    for batch in tqdm(train_loader, desc=f"Exporting embeddings for epoch {epoch}", total=min(max_sequences // args.batch_size + 1, len(train_loader))):
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        seq_ids_batch = batch["seq_ids"]
                        with torch.no_grad():
                            embeddings = model.embedding(input_ids)
                            pooled_emb = embeddings.mean(dim=1).cpu().numpy()
                        for seq_id, emb in zip(seq_ids_batch, pooled_emb):
                            if seq_count >= max_sequences:
                                break
                            embeddings_list.append((seq_id, emb))
                            seq_ids.append(seq_id)
                            seq_count += 1
                        if seq_count >= max_sequences:
                            break
                    
                    for seq_id, emb in embeddings_list:
                        emb_group.create_dataset(seq_id, data=emb, compression='gzip', compression_opts=4)
                    f.attrs['num_sequences'] = len(seq_ids)
                    f.attrs['embed_dim'] = model.cnn_params["embed_dim"] if args.model_type == "hybridcnnbert" else model.config["embed_dim"]
                    f.attrs['epoch'] = epoch
                    f.attrs['run_uuid'] = run_uuid
                    f.create_dataset('seq_ids', data=np.array(seq_ids, dtype='S'))
                logger.info("Exported %d sequence embeddings to %s", len(seq_ids), embeddings_path)
            except Exception as e:
                logger.error("Failed to export embeddings to %s: %s", embeddings_path, str(e))
                raise

        if getattr(args, 'export_permutation_importance', False):
            perm_importance_path = os.path.join(weights_dir, f"deeptaxa_{run_uuid}_epoch{epoch}_permutation_importance.tsv")
            try:
                importance_scores = compute_permutation_importance(model, val_loader, device, taxonomic_ranks, args)
                with open(perm_importance_path, 'w') as f:
                    f.write("Level\tRegion\tStart\tEnd\tImportance\n")
                    region_size = getattr(args, 'perm_region_size', args.max_length // 4) or args.max_length // 4
                    for level in range(len(taxonomic_ranks)):
                        for region, score in enumerate(importance_scores[level]):
                            start = region * region_size
                            end = min(start + region_size, args.max_length)
                            f.write(f"{taxonomic_ranks[level]}\t{region}\t{start}\t{end}\t{score:.6f}\n")
                logger.info("Exported permutation importance to %s", perm_importance_path)
            except Exception as e:
                logger.error("Failed to export permutation importance to %s: %s", perm_importance_path, str(e))
                raise

def train(args, trial=None):
    """Train a DeepTaxa model on 16S rRNA sequence data."""
    logger.debug("Starting training with trial: %s", trial.number if trial else "None")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Enabled deterministic CuDNN behavior for reproducibility")
    get_device_info(args.verbose)
    
    start_time = datetime.now().isoformat()
    banner = f"""
{'=' * 70}
          DeepTaxa Training Session (v{__version__})
{'-' * 50}
          Started: {start_time}
{'=' * 70}
    """
    logger.info(banner)

    params = vars(args)
    param_groups = {
        "General Parameters": [
            "command", "fasta_file", "taxonomy_file", "model_type", "output_dir", "resume",
            "batch_size", "epochs", "val_split", "seed", "eval_every", "accum_steps", "num_workers", "verbose"
        ],
        "Model Architecture": [
            "tokenizer_name", "max_length", "embed_dim", "num_filters", "kernel_sizes", "num_conv_layers",
            "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "hidden_dropout_prob",
            "output_attentions"
        ],
        "Loss Configuration": [
            "loss_type", "class_weights", "focal_gamma", "focal_reduction", "focal_weight", "level_weights"
        ],
        "Early Stopping": [
            "early_stopping_patience", "early_stopping_min_delta"
        ],
        "Encoding": [
            "encoding"
        ],
        "Optimizer and Scheduler": [
            "learning_rate", "optimizer_betas", "optimizer_eps", "optimizer_weight_decay", "scheduler_warmup_ratio",
            "scaler_init_scale"
        ],
        "Export Options": [
            "export_sequence_embeddings", "export_permutation_importance", "perm_region_size"
        ]
    }

    param_output = "Training Parameters:\n"
    param_output += "-" * 50 + "\n"
    for group_name, keys in param_groups.items():
        param_output += f"{group_name}:\n"
        for key in keys:
            if key in params:
                value = params[key]
                if isinstance(value, list):
                    value_str = f"[{', '.join(map(str, value))}]"
                elif isinstance(value, torch.Tensor):
                    value_str = str(value.tolist())
                else:
                    value_str = str(value)
                param_output += f"  {key.replace('_', '-'):>25}: {value_str}\n"
        param_output += "-" * 50 + "\n"
    logger.info(param_output.rstrip("\n-"))

    output_dir = args.output_dir.rstrip('/')
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory set to: %s", output_dir)

    args.session_start_time = start_time
    resume = getattr(args, 'resume', None)
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        start_epoch = checkpoint["epoch"] + 1
        run_uuid = checkpoint["run_uuid"]
        args.checkpoint_start_time = checkpoint.get("checkpoint_start_time", args.session_start_time)
        logger.info("Resuming training from epoch %d with run UUID: %s", start_epoch, run_uuid)

        # Log checkpoint fields for debugging
        logger.info("Checkpoint fields: %s", list(checkpoint.keys()))

        # Load hyperparameters from checkpoint.  Preserve CLI --epochs and
        # --learning-rate; restore everything else so the model, optimizer,
        # and scheduler are consistent with the checkpoint.
        args.model_type = checkpoint.get("model_type", args.model_type)
        args.encoding = checkpoint.get("encoding", getattr(args, "encoding", "dnabert"))
        args.batch_size = checkpoint.get("batch_size", args.batch_size)
        args.focal_gamma = checkpoint.get("focal_gamma", getattr(args, "focal_gamma", DEFAULT_CONFIG["focal_gamma"]))
        args.focal_reduction = checkpoint.get("focal_reduction", getattr(args, "focal_reduction", DEFAULT_CONFIG["focal_reduction"]))
        args.focal_weight = checkpoint.get("focal_weight", getattr(args, "focal_weight", DEFAULT_CONFIG["focal_weight"]))
        args.level_weights = checkpoint.get("level_weights", args.level_weights)
        args.max_length = checkpoint.get("max_length", args.max_length)
        args.tokenizer_name = checkpoint.get("tokenizer_name", args.tokenizer_name)
        args.tokenizer_revision = checkpoint.get("tokenizer_revision", getattr(args, "tokenizer_revision", DEFAULT_CONFIG["tokenizer_revision"]))
        args.optimizer_betas = checkpoint["optimizer_config"].get("betas", getattr(args, "optimizer_betas", DEFAULT_CONFIG["optimizer_betas"]))
        args.optimizer_eps = checkpoint["optimizer_config"].get("eps", getattr(args, "optimizer_eps", DEFAULT_CONFIG["optimizer_eps"]))
        args.optimizer_weight_decay = checkpoint["optimizer_config"].get("weight_decay", getattr(args, "optimizer_weight_decay", DEFAULT_CONFIG["optimizer_weight_decay"]))
        args.scheduler_warmup_ratio = checkpoint["scheduler"].get("num_warmup_steps", int(getattr(args, "scheduler_warmup_ratio", DEFAULT_CONFIG["scheduler_warmup_ratio"]) * checkpoint["scheduler"]["num_training_steps"])) / checkpoint["scheduler"]["num_training_steps"]
        args.scaler_init_scale = checkpoint.get("scaler_init_scale", getattr(args, "scaler_init_scale", DEFAULT_CONFIG["scaler_init_scale"]))

        # Restore architecture parameters so model shape matches checkpoint
        model_config = checkpoint.get("model_config", {})
        if args.model_type == "hybridcnnbert" and "cnn" in model_config:
            cnn_cfg = model_config["cnn"]
            args.embed_dim = cnn_cfg.get("embed_dim", args.embed_dim)
            args.num_filters = cnn_cfg.get("num_filters", args.num_filters)
            args.kernel_sizes = cnn_cfg.get("kernel_sizes", args.kernel_sizes)
            args.num_conv_layers = cnn_cfg.get("num_conv_layers", args.num_conv_layers)
            # Legacy checkpoints predate the flag and were trained unmasked.
            args.mask_padding = cnn_cfg.get("mask_padding", False)
            bert_cfg = model_config.get("bert", {})
            args.hidden_size = bert_cfg.get("hidden_size", args.hidden_size)
            args.num_hidden_layers = bert_cfg.get("num_hidden_layers", args.num_hidden_layers)
            args.num_attention_heads = bert_cfg.get("num_attention_heads", args.num_attention_heads)
            args.intermediate_size = bert_cfg.get("intermediate_size", args.intermediate_size)
            args.hidden_dropout_prob = bert_cfg.get("hidden_dropout_prob", args.hidden_dropout_prob)
        elif args.model_type == "cnn" and isinstance(model_config, dict):
            args.embed_dim = model_config.get("embed_dim", args.embed_dim)
            args.num_filters = model_config.get("num_filters", args.num_filters)
            args.kernel_sizes = model_config.get("kernel_sizes", args.kernel_sizes)
            args.num_conv_layers = model_config.get("num_conv_layers", args.num_conv_layers)
            # Legacy checkpoints predate the flag and were trained unmasked.
            args.mask_padding = model_config.get("mask_padding", False)
        elif args.model_type == "bert" and isinstance(model_config, dict):
            args.hidden_size = model_config.get("hidden_size", args.hidden_size)
            args.num_hidden_layers = model_config.get("num_hidden_layers", args.num_hidden_layers)
            args.num_attention_heads = model_config.get("num_attention_heads", args.num_attention_heads)
            args.intermediate_size = model_config.get("intermediate_size", args.intermediate_size)
            args.hidden_dropout_prob = model_config.get("hidden_dropout_prob", args.hidden_dropout_prob)

        checkpoint_lr = checkpoint["optimizer_config"].get("lr", "not specified")
        logger.info("Restored from checkpoint: model_type=%s, batch_size=%d, focal_gamma=%.6f, learning_rate=%.6f (CLI), checkpoint_lr=%.6f, warmup_ratio=%.4f",
                    args.model_type, args.batch_size, getattr(args, "focal_gamma", DEFAULT_CONFIG["focal_gamma"]), args.learning_rate, checkpoint_lr, args.scheduler_warmup_ratio)
        logger.info("Restored architecture: embed_dim=%d, num_filters=%d, num_conv_layers=%d, hidden_size=%d, num_hidden_layers=%d, num_attention_heads=%d, intermediate_size=%d",
                    args.embed_dim, args.num_filters, args.num_conv_layers, args.hidden_size, args.num_hidden_layers, args.num_attention_heads, args.intermediate_size)
    else:
        args.checkpoint_start_time = args.session_start_time
        timestamp = args.session_start_time.split('.')[0].replace('-', '_').replace(':', '_')
        run_uuid = f"{timestamp}_{str(uuid.uuid4()).replace('-', '_')}"
        start_epoch = 1
        logger.info("Training started with run UUID: %s", run_uuid)

    uuid_file = os.path.join(output_dir, "deeptaxa_uuid.txt")
    try:
        with open(uuid_file, "w") as f:
            f.write(run_uuid)
        logger.info("Saved run UUID to %s", uuid_file)
    except Exception as e:
        logger.error("Failed to save UUID to %s: %s", uuid_file, str(e))
        raise

    check_file(args.fasta_file)
    check_file(args.taxonomy_file)

    encoding = getattr(args, 'encoding', DEFAULT_CONFIG['encoding'])
    dataset = TaxonomyDataset(args.fasta_file, args.taxonomy_file, args.tokenizer_name, args.max_length, encoding=encoding,
                              tokenizer_revision=getattr(args, "tokenizer_revision", DEFAULT_CONFIG["tokenizer_revision"]))
    num_labels_per_level = {level: len(dataset.level_label2id[level]) for level in dataset.level_label2id}
    logger.info("Number of labels per taxonomic level: %s", num_labels_per_level)

    metadata_dir = os.path.join(args.output_dir, "metadata")
    try:
        os.makedirs(metadata_dir, exist_ok=True)
        label2id_filename = os.path.join(metadata_dir, f"deeptaxa_{run_uuid}_label2id.json")
        with open(label2id_filename, "w") as f:
            json.dump(dataset.level_label2id, f, indent=4)
        logger.info("Saved level_label2id to %s", label2id_filename)
    except Exception as e:
        logger.error("Failed to save label2id to %s: %s", label2id_filename, str(e))
        raise

    if getattr(args, 'no_level_weights', False):
        level_weights = torch.ones(len(dataset.taxonomic_ranks), dtype=torch.float32).to(device)
        logger.info("Using uniform level weights (--no-level-weights)")
    else:
        level_weights = torch.tensor(args.level_weights[:len(dataset.taxonomic_ranks)], dtype=torch.float32).to(device)
    taxonomic_ranks = dataset.taxonomic_ranks
    logger.info("Retrieved taxonomic ranks: %s", taxonomic_ranks)

    try:
        model = setup_model(args, num_labels_per_level)
        model.to(device)
        if args.model_type == "hybridcnnbert":
            logger.info("Model setup complete with configuration: %s", {"cnn": model.cnn_params, "bert": model.config.__dict__})
        else:
            logger.info("Model setup complete with configuration: %s", model.config)
    except Exception as e:
        logger.error("Failed to setup model: %s", str(e))
        raise

    # Weight-only init for fine-tuning across datasets; see --init-weights help.
    # --resume and --init-weights are argparse-mutex, so at most one is set.
    init_weights = getattr(args, 'init_weights', None)
    if init_weights:
        check_file(init_weights)
        logger.info("Initializing model weights from checkpoint: %s", init_weights)
        init_ckpt = torch.load(init_weights, map_location='cpu', weights_only=False)
        init_state_dict = init_ckpt.get("state_dict", init_ckpt)

        # load_state_dict(strict=False) silences missing/unexpected-key errors
        # but STILL raises on shape mismatches. For cross-dataset fine-tuning the
        # classifier heads always have different output shapes (source label
        # space != target label space), so we must pre-filter them out before
        # calling load_state_dict. Body layers (CNN, BERT, embeddings) must
        # match the checkpoint's architecture and are loaded verbatim.
        target_state = model.state_dict()
        filtered_state = {}
        shape_mismatch = []
        for k, v in init_state_dict.items():
            if k in target_state and v.shape != target_state[k].shape:
                shape_mismatch.append((k, tuple(v.shape), tuple(target_state[k].shape)))
            else:
                filtered_state[k] = v

        load_result = model.load_state_dict(filtered_state, strict=False)
        logger.info(
            "Init-weights transfer: %d/%d checkpoint keys loaded "
            "(%d skipped due to shape mismatch, %d unexpected); "
            "%d target-model keys left at random init.",
            len(filtered_state) - len(load_result.unexpected_keys),
            len(init_state_dict),
            len(shape_mismatch),
            len(load_result.unexpected_keys),
            len(load_result.missing_keys),
        )
        if shape_mismatch:
            logger.info(
                "Shape-mismatched keys (re-initialized) — first 5: %s",
                [f"{k} src={s} dst={d}" for k, s, d in shape_mismatch[:5]],
            )
        if load_result.missing_keys:
            logger.info("Missing keys (first 10): %s", load_result.missing_keys[:10])
        if load_result.unexpected_keys:
            logger.info("Unexpected keys (first 10): %s", load_result.unexpected_keys[:10])

        # Safety: after pre-filtering, the only keys that should be missing or
        # shape-mismatched are classifier heads. If any body layer (CNN, BERT,
        # embedding) ended up in either pool, the architecture differs from the
        # checkpoint and fine-tuning would silently start from mostly-random
        # body weights — exactly the scenario --init-weights exists to prevent.
        def _is_classifier(k):
            return k.startswith("classifiers.")
        body_missing = [k for k in load_result.missing_keys if not _is_classifier(k)]
        body_shape_mismatch = [k for (k, _, _) in shape_mismatch if not _is_classifier(k)]
        if body_missing or body_shape_mismatch:
            offenders = body_missing + body_shape_mismatch
            raise ValueError(
                f"--init-weights: {len(offenders)} non-classifier body parameters "
                f"did not transfer from the checkpoint ({len(body_missing)} missing, "
                f"{len(body_shape_mismatch)} shape mismatch). Likely architecture "
                f"mismatch. First 5 offenders: {offenders[:5]}. Ensure the CLI "
                f"architecture flags (--embed-dim, --hidden-size, --num-filters, "
                f"--kernel-sizes, --num-hidden-layers, --num-attention-heads, "
                f"--intermediate-size, --num-conv-layers) match the values used when "
                f"the checkpoint was trained."
            )

    if resume:
        try:
            checkpoint = torch.load(resume, map_location=device, weights_only=False)
            required_keys = ["state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state", "dataset_split"]
            for key in required_keys:
                if key not in checkpoint:
                    raise ValueError(f"Checkpoint missing required key: {key}")
            logger.info("Checkpoint contains all required keys: %s", required_keys)
            
            if torch.cuda.is_available() and checkpoint["rng_state"].get("cuda") is None:
                logger.warning("Checkpoint lacks CUDA RNG state, which may affect reproducibility on CUDA devices")
            
            model.load_state_dict(checkpoint["state_dict"])
            model.train()
            dataset_split = checkpoint["dataset_split"]
            train_idx, val_idx = dataset_split["train_idx"], dataset_split["val_idx"]
            train_size, val_size = len(train_idx), len(val_idx)
            logger.info("Restored dataset split from checkpoint: Training: %d, Validation: %d", train_size, val_size)
            
            cpu_rng_state = checkpoint["rng_state"]["torch"]
            logger.debug("Loaded CPU RNG state type: %s, device: %s, hash: %s", 
                        type(cpu_rng_state), 
                        cpu_rng_state.device if isinstance(cpu_rng_state, torch.Tensor) else "N/A",
                        hashlib.sha256(cpu_rng_state.cpu().numpy().tobytes()).hexdigest()[:16])
            if not isinstance(cpu_rng_state, torch.ByteTensor):
                logger.warning("CPU RNG state is not a ByteTensor, attempting to convert")
                cpu_rng_state = cpu_rng_state.cpu().to(dtype=torch.uint8)
            torch.set_rng_state(cpu_rng_state)
            
            np_rng_state = checkpoint["rng_state"]["numpy"]
            logger.debug("Loaded NumPy RNG state hash: %s", 
                        hashlib.sha256(np_rng_state[1].tobytes()).hexdigest()[:16])
            np.random.set_state(np_rng_state)
            
            if "python" in checkpoint["rng_state"]:
                python_rng_state = checkpoint["rng_state"]["python"]
                python_rng_hash = hashlib.sha256(pickle.dumps(python_rng_state)).hexdigest()[:16]
                logger.debug("Loaded Python RNG state hash: %s", python_rng_hash)
                random.setstate(python_rng_state)
                logger.info("Restored Python RNG state from checkpoint")
            else:
                logger.warning("Checkpoint lacks Python RNG state, using default seed-based initialization")
            
            if torch.cuda.is_available() and checkpoint["rng_state"].get("cuda") is not None:
                cuda_rng_state = checkpoint["rng_state"]["cuda"]
                logger.debug("Loaded CUDA RNG state type: %s, device: %s, hash: %s", 
                            type(cuda_rng_state), 
                            cuda_rng_state.device if isinstance(cuda_rng_state, torch.Tensor) else "N/A",
                            hashlib.sha256(cuda_rng_state.cpu().numpy().tobytes()).hexdigest()[:16])
                if not isinstance(cuda_rng_state, torch.ByteTensor):
                    logger.warning("CUDA RNG state is not a ByteTensor, attempting to convert")
                    cuda_rng_state = cuda_rng_state.cpu().to(dtype=torch.uint8)
                torch.cuda.set_rng_state(cuda_rng_state)
                logger.info("Restored CUDA RNG state from checkpoint")
            else:
                logger.warning("No CUDA RNG state found in checkpoint or CUDA not available")
        except Exception as e:
            logger.error("Failed to resume from checkpoint: %s", str(e))
            raise
    else:
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        split = int(len(dataset) * (1 - getattr(args, "val_split", DEFAULT_CONFIG["val_split"])))
        train_idx, val_idx = indices[:split], indices[split:]
        train_size, val_size = len(train_idx), len(val_idx)
        dataset_split = {"train_idx": train_idx, "val_idx": val_idx}
        logger.info("Generated new dataset split: Total: %d, Training: %d, Validation: %d", len(dataset), train_size, val_size)

    try:
        def worker_init_fn(worker_id):
            worker_seed = args.seed + worker_id
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(worker_seed)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=getattr(args, "num_workers", DEFAULT_CONFIG["num_workers"]),
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if getattr(args, "num_workers", DEFAULT_CONFIG["num_workers"]) > 0 else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=getattr(args, "num_workers", DEFAULT_CONFIG["num_workers"]),
            collate_fn=custom_collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if getattr(args, "num_workers", DEFAULT_CONFIG["num_workers"]) > 0 else False
        )
    except Exception as e:
        logger.error("Failed to create data loaders: %s", str(e))
        raise

    optimizer = setup_optimizer(model, args)
    accum_steps = getattr(args, "accum_steps", DEFAULT_CONFIG["accum_steps"])

    if resume:
        # --- Resume path ---
        # 1. Restore optimizer state (Adam momentum/variance + base LR).
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.error("Failed to load optimizer state: %s", str(e))
            raise

        # 2. Restore scheduler by re-creating it and fast-forwarding to the
        #    checkpoint step.  We cannot use load_state_dict because it
        #    restores the step counter but does NOT update pg['lr'] (the
        #    LambdaLR.__init__ already set pg['lr'] = base_lr * lambda(0),
        #    and load_state_dict leaves that stale value in place).
        #    Fast-forwarding calls scheduler.step() repeatedly, which both
        #    advances the counter and correctly sets pg['lr'] each time.
        old_total = checkpoint["scheduler"]["num_training_steps"]
        old_warmup = checkpoint["scheduler"]["num_warmup_steps"]
        old_step = checkpoint.get("scheduler_state_dict", {}).get("last_epoch", 0)
        original_base_lr = optimizer.param_groups[0].get('initial_lr', optimizer.param_groups[0]['lr'])

        # If extending training beyond the original epoch count, use the
        # new total_steps so the LR decays over the longer schedule.
        # The loop also steps on the final partial window, so a leftover batch
        # adds one more optimizer step per epoch. Round up so the schedule
        # covers every step and the LR does not bottom out early.
        steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
        new_total = steps_per_epoch * args.epochs
        sched_total = max(new_total, old_total)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=old_warmup, num_training_steps=sched_total
        )
        # Fast-forward to checkpoint position.
        for _ in range(old_step):
            scheduler.step()
        resumed_lr = optimizer.param_groups[0]['lr']
        total_steps = sched_total
        logger.info("Resume: base_lr=%.8f, old_step=%d/%d, sched_total=%d, resumed_lr=%.8f",
                    original_base_lr, old_step, old_total, sched_total, resumed_lr)

        # 4. Restore GradScaler state.
        scaler_init_scale = getattr(args, "scaler_init_scale", DEFAULT_CONFIG["scaler_init_scale"])
        scaler = GradScaler('cuda', init_scale=scaler_init_scale) if torch.cuda.is_available() else None
        if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info("Restored GradScaler state from checkpoint")
        else:
            logger.warning("No scaler state found in checkpoint, initializing new GradScaler")
        logger.info("Initialized GradScaler for mixed precision training with init_scale=%.1f", scaler_init_scale)
    else:
        # --- Fresh training path ---
        total_steps = math.ceil(len(train_loader) / accum_steps) * args.epochs
        warmup_steps = int(getattr(args, "scheduler_warmup_ratio", DEFAULT_CONFIG["scheduler_warmup_ratio"]) * total_steps)
        logger.info("Initialized scheduler with total_steps=%d, warmup_steps=%d", total_steps, warmup_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        logger.info("Initial learning rate: %.8f", args.learning_rate)
        scaler_init_scale = getattr(args, "scaler_init_scale", DEFAULT_CONFIG["scaler_init_scale"])
        scaler = GradScaler('cuda', init_scale=scaler_init_scale) if torch.cuda.is_available() else None
        logger.info("Initialized GradScaler for mixed precision training with init_scale=%.1f", scaler_init_scale)

    try:
        loss_type = getattr(args, 'loss_type', DEFAULT_CONFIG['loss_type'])
        class_weights_mode = getattr(args, 'class_weights', 'none')

        # Compute per-rank class weights from training data if requested
        per_rank_weights = {}
        if class_weights_mode != 'none':
            for level in range(len(dataset.taxonomic_ranks)):
                counts = Counter(dataset.labels[i][level] for i in train_idx)
                n_classes = len(dataset.level_label2id[level])
                n_samples = sum(counts.values())
                weights = torch.zeros(n_classes, dtype=torch.float32)
                for cls_id, count in counts.items():
                    if class_weights_mode == 'inverse':
                        weights[cls_id] = n_samples / (n_classes * count)
                    elif class_weights_mode == 'sqrt_inverse':
                        weights[cls_id] = math.sqrt(n_samples / (n_classes * count))
                per_rank_weights[level] = weights.to(device)
                rank_name = dataset.taxonomic_ranks[level]
                logger.info("Class weights for %s: min=%.4f, max=%.4f, mean=%.4f",
                            rank_name, weights.min().item(), weights.max().item(), weights.mean().item())

        # Build per-rank criteria (or single shared criterion if no class weights)
        if per_rank_weights:
            criteria = {}
            for level in range(len(dataset.taxonomic_ranks)):
                if loss_type == 'cross_entropy':
                    criteria[level] = nn.CrossEntropyLoss(weight=per_rank_weights[level])
                else:
                    criteria[level] = FocalLoss(
                        gamma=getattr(args, "focal_gamma", DEFAULT_CONFIG["focal_gamma"]),
                        reduction=getattr(args, "focal_reduction", DEFAULT_CONFIG["focal_reduction"]),
                        weight=per_rank_weights[level]
                    )
                criteria[level].to(device)
            criterion = None
            logger.info("Initialized per-rank %s with %s class weights", loss_type, class_weights_mode)
        else:
            criteria = None
            if loss_type == 'cross_entropy':
                criterion = nn.CrossEntropyLoss(
                    weight=getattr(args, "focal_weight", DEFAULT_CONFIG["focal_weight"])
                )
                criterion.to(device)
                logger.info("Initialized CrossEntropyLoss")
            else:
                criterion = FocalLoss(
                    gamma=getattr(args, "focal_gamma", DEFAULT_CONFIG["focal_gamma"]),
                    reduction=getattr(args, "focal_reduction", DEFAULT_CONFIG["focal_reduction"]),
                    weight=getattr(args, "focal_weight", DEFAULT_CONFIG["focal_weight"])
                )
                criterion.to(device)
                logger.info("Initialized FocalLoss with gamma=%.6f, reduction=%s, weight=%s",
                            getattr(args, "focal_gamma", DEFAULT_CONFIG["focal_gamma"]), getattr(args, "focal_reduction", DEFAULT_CONFIG["focal_reduction"]),
                            getattr(args, "focal_weight", DEFAULT_CONFIG["focal_weight"]))
    except Exception as e:
        logger.error("Failed to initialize loss function: %s", str(e))
        raise

    args.dataset_size = len(dataset)
    args.train_size = train_size
    args.val_size = val_size
    args.level_label2id = dataset.level_label2id

    final_avg_f1 = 0
    early_stopping_patience = getattr(args, 'early_stopping_patience', 0)
    early_stopping_min_delta = getattr(args, 'early_stopping_min_delta', 0.001)
    best_val_loss = None
    best_epoch = 0
    patience_counter = 0
    # With --save-best-only, only write a checkpoint on an epoch that improves the
    # validation loss, so a long run does not leave one full checkpoint per epoch.
    save_best_only = getattr(args, 'save_best_only', False)
    best_saved_val_loss = None

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_banner = f"""
{'=' * 70}
          DeepTaxa Training (v{__version__}) - Epoch {epoch}/{args.epochs}
{'-' * 50}
          Start Time: {datetime.now().isoformat()}
{'=' * 70}
        """
        logger.info(epoch_banner)

        if epoch == start_epoch and not resume:
            logger.info("Learning rate at start of epoch %d: %.8f", epoch, args.learning_rate)
        else:
            logger.info("Learning rate at start of epoch %d: %.8f", epoch, optimizer.param_groups[0]['lr'])
        
        model.train()
        total_loss = 0
        epoch_start_time = datetime.now()
        consecutive_nan_count = 0
        max_consecutive_nan = 50  # Abort epoch if 50+ consecutive NaN losses

        try:
            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}/{args.epochs}")):
                batch_start = datetime.now()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                                
                # Clear gradients only at the start of an accumulation window.
                # Everything backward()-ed until the next window boundary adds up,
                # which is what makes the effective batch size batch_size x accum_steps.
                if i % accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)

                with autocast('cuda', enabled=torch.cuda.is_available()):
                    output = model(input_ids, attention_mask)
                    logits = output[0] if isinstance(output, tuple) else output
                    loss = sum(level_weights[level] * (criteria[level] if criteria else criterion)(logits[str(level)], labels[:, level]) for level in range(len(taxonomic_ranks)))
                    loss = loss / accum_steps

                    # Skip any batch whose loss is NaN or infinite
                    loss_value = loss.item()
                    if math.isnan(loss_value) or math.isinf(loss_value):
                        consecutive_nan_count += 1
                        if consecutive_nan_count <= 3 or consecutive_nan_count == max_consecutive_nan:
                            logger.warning(f"Invalid loss at epoch {epoch}, batch {i} ({consecutive_nan_count} consecutive). Skipping backward.")
                        if consecutive_nan_count >= max_consecutive_nan:
                            logger.error(f"Aborting epoch {epoch}: {consecutive_nan_count} consecutive NaN/Inf losses — model has diverged.")
                            break
                        optimizer.zero_grad()
                        continue
                    consecutive_nan_count = 0  # Reset on valid loss
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += loss.item()

                if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(args, "max_grad_norm", DEFAULT_CONFIG.get("max_grad_norm", 0.5)))

                    has_nan = any(torch.isnan(param.grad).any() for param in model.parameters() if param.grad is not None)
                    if has_nan:
                        consecutive_nan_count += 1
                        if consecutive_nan_count <= 3 or consecutive_nan_count == max_consecutive_nan:
                            logger.warning("NaN gradients at batch %d of epoch %d (%d consecutive). Skipping update.", i, epoch, consecutive_nan_count)
                        if consecutive_nan_count >= max_consecutive_nan:
                            logger.error(f"Aborting epoch {epoch}: {consecutive_nan_count} consecutive NaN gradients — model has diverged.")
                            break
                        if scaler is not None:
                            scaler.update()
                        optimizer.zero_grad()
                        continue
                    consecutive_nan_count = 0  # Reset on valid gradients

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
        except Exception as e:
            logger.error("Training loop failed at epoch %d: %s", epoch, str(e))
            raise
        
        train_time = (datetime.now() - epoch_start_time).total_seconds()
        avg_train_loss = total_loss / len(train_loader)
        args.current_train_loss = avg_train_loss
        args.current_train_time = train_time
        total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("Epoch %d training completed. Training Loss: %.4f", epoch, avg_train_loss)
        logger.info("Learning rate at end of epoch %d: %.8f", epoch, optimizer.param_groups[0]['lr'])
        
        metrics = None
        val_loss = None
        eval_time = 0
        if (resume and epoch == start_epoch) or (epoch % getattr(args, "eval_every", DEFAULT_CONFIG["eval_every"]) == 0):
            is_resume_eval = resume and epoch == start_epoch
            logger.info("Starting evaluation for epoch %d%s", epoch, " (fresh evaluation after resume)" if is_resume_eval else "")
            eval_start_time = datetime.now()
            try:
                metrics, val_loss = evaluate_model(model, val_loader, criterion, device, taxonomic_ranks, criteria=criteria, level_weights=level_weights)
                logger.info("Evaluation completed for epoch %d. Validation loss: %.4f", epoch, val_loss)
                
                metrics_output = "Validation Metrics:\n"
                metrics_output += f"{'Level':<8} {'Rank':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
                metrics_output += "-" * 58 + "\n"
                for level, rank in enumerate(taxonomic_ranks):
                    m = metrics[level]
                    metrics_output += f"{level:<8} {rank:<10} {m['accuracy']:.5f}    {m['precision']:.5f}     {m['recall']:.5f}    {m['f1_score']:.5f}\n"
                logger.info(metrics_output.rstrip("\n"))
                
                logger.info("Validation Loss: %.4f", val_loss)
            except Exception as e:
                logger.error("Evaluation failed at epoch %d: %s", epoch, str(e))
                raise
            eval_time = (datetime.now() - eval_start_time).total_seconds()
            args.current_eval_time = eval_time
            
            checkpoint_total_time = train_time + eval_time
            session_total_time = (datetime.fromisoformat(checkpoint_end_time := datetime.now().isoformat()) - datetime.fromisoformat(args.session_start_time)).total_seconds()
            
            avg_f1 = sum(m["f1_score"] for m in metrics.values()) / len(metrics)
            final_avg_f1 = avg_f1
            
            if trial:
                logger.info("Trial %d reporting F1-score: %.4f at epoch %d", trial.number, avg_f1, epoch)
                try:
                    trial.report(-avg_f1, epoch)
                    if trial.should_prune():
                        logger.info("Trial %d pruned at epoch %d", trial.number, epoch)
                        raise optuna.TrialPruned()
                except NameError as e:
                    logger.error("Optuna import error during pruning: %s", str(e))
                    raise RuntimeError("Failed to prune trial due to missing optuna module")
                except Exception as e:
                    logger.error("Pruning failed for trial %d at epoch %d: %s", trial.number, epoch, str(e))
                    raise
            
            is_best = val_loss is not None and (best_saved_val_loss is None or val_loss < best_saved_val_loss)
            if is_best:
                best_saved_val_loss = val_loss
            if save_best_only and not is_best:
                logger.info("save-best-only: keeping the previous checkpoint; epoch %d val_loss %.4f did not beat %.4f",
                            epoch, val_loss, best_saved_val_loss)
            else:
                logger.info("Saving checkpoint for epoch %d...", epoch)
                try:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, run_uuid, args, dataset_split, metrics,
                        val_loss, taxonomic_ranks, session_total_time, checkpoint_total_time,
                        train_loader, val_loader, device, total_steps, total_parameters, scaler=scaler, trial=trial
                    )
                except Exception as e:
                    logger.error("Checkpoint saving failed for epoch %d: %s", epoch, str(e))
                    raise
                logger.info("Checkpoint save completed for epoch %d", epoch)

            if early_stopping_patience > 0 and val_loss is not None:
                if best_val_loss is None or val_loss < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    logger.info("Early stopping: new best val_loss=%.4f at epoch %d", val_loss, epoch)
                else:
                    patience_counter += 1
                    logger.info("Early stopping: no improvement for %d epoch(s) (best=%.4f at epoch %d)",
                                patience_counter, best_val_loss, best_epoch)
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d (patience=%d, best epoch=%d, best val_loss=%.4f)",
                                epoch, early_stopping_patience, best_epoch, best_val_loss)
                    break

            args.checkpoint_start_time = checkpoint_end_time
        else:
            logger.info("Skipping evaluation for epoch %d (eval_every=%d)", epoch, getattr(args, "eval_every", DEFAULT_CONFIG["eval_every"]))
            checkpoint_total_time = train_time
            session_total_time = (datetime.fromisoformat(checkpoint_end_time := datetime.now().isoformat()) - datetime.fromisoformat(args.session_start_time)).total_seconds()

            # No validation loss on a non-eval epoch, so there is nothing to rank
            # against; skip the write under --save-best-only.
            if save_best_only:
                logger.info("save-best-only: not saving on non-eval epoch %d", epoch)
            else:
                logger.info("Saving checkpoint for epoch %d...", epoch)
                try:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, run_uuid, args, dataset_split, metrics,
                        val_loss, taxonomic_ranks, session_total_time, checkpoint_total_time,
                        train_loader, val_loader, device, total_steps, total_parameters, scaler=scaler, trial=trial
                    )
                except Exception as e:
                    logger.error("Checkpoint saving failed for epoch %d: %s", epoch, str(e))
                    raise
                logger.info("Checkpoint save completed for epoch %d", epoch)

            args.checkpoint_start_time = checkpoint_end_time

    logger.info("Training completed successfully.")
    return final_avg_f1

