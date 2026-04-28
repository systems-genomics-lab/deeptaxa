"""
Module: describe.py

Description:
    Provides a detailed description of a trained DeepTaxa model checkpoint, extracting architecture,
    hyperparameters, training metadata, and performance metrics. Outputs to console and optionally
    exports to JSON, serving as a diagnostic tool for model inspection and reproducibility.
"""

import torch
import json
import logging
from datetime import datetime
from typing import Dict, Any
from deeptaxa import __version__

logger = logging.getLogger(__name__)

def describe(args) -> None:
    """
    Describe a DeepTaxa model checkpoint in detail.

    Workflow:
        1. Load checkpoint and extract key metadata (model type, epoch, weights).
        2. Parse model-specific configuration (e.g., CNN filters, BERT layers).
        3. Retrieve total parameter count from checkpoint for complexity assessment.
        4. Optionally export structured data to JSON for further analysis.

    Args:
        args: CLI arguments with checkpoint path and export options.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Log checkpoint contents for debugging
    logger.debug("Checkpoint keys: %s", list(checkpoint.keys()))

    # Extract core metadata
    run_uuid = checkpoint.get("run_uuid", "Missing")
    epoch = checkpoint.get("epoch", "Missing")
    model_type = checkpoint.get("model_type", "Missing")
    tokenizer_name = checkpoint.get("tokenizer_name", "Missing")
    max_length = checkpoint.get("max_length", "Missing")
    hidden_dropout_prob = checkpoint.get("hidden_dropout_prob", "Missing")
    taxonomic_ranks = checkpoint.get("taxonomic_ranks", [])
    num_labels_per_level = checkpoint.get("num_labels_per_level", {})
    total_parameters = checkpoint.get("total_parameters", "N/A")
    optimizer_config = checkpoint.get("optimizer_config", {})
    scheduler_config = checkpoint.get("scheduler", {})

    # Parse model configuration based on type
    model_config = checkpoint.get("model_config", {})
    if model_type == "hybridcnnbert":
        cnn_config = model_config.get("cnn", {})
        bert_config = model_config.get("bert", {})
    elif model_type == "cnn":
        cnn_config = model_config
        bert_config = {}
    else:  # bert
        cnn_config = {}
        bert_config = model_config

    # Build the entire output as one string
    timestamp = datetime.now().isoformat()
    output = f"""
{'=' * 70}
          DeepTaxa Model Description (v{__version__})
{'-' * 50}
          Checkpoint: {checkpoint_path}
          Timestamp: {timestamp}
{'=' * 70}

Model Details:
{'-' * 50}
"""
    model_details = {
        "run-uuid": run_uuid,
        "model-type": model_type,
        "tokenizer": tokenizer_name,
        "epoch": epoch,
        "total-parameters": total_parameters,
        "max-length": max_length,
        "embed-dim": cnn_config.get("embed_dim"),
        "num-filters": cnn_config.get("num_filters"),
        "kernel-sizes": cnn_config.get("kernel_sizes"),
        "num-conv-layers": cnn_config.get("num_conv_layers"),
        "hidden-size": bert_config.get("hidden_size"),
        "num-hidden-layers": bert_config.get("num_hidden_layers"),
        "num-attention-heads": bert_config.get("num_attention_heads"),
        "intermediate-size": bert_config.get("intermediate_size"),
        "output-attentions": bert_config.get("output_attentions"),
        "hidden-dropout-prob": hidden_dropout_prob
    }
    for key, value in model_details.items():
        if value is None or (model_type not in ["cnn", "hybridcnnbert"] and key in ["embed-dim", "num-filters", "kernel-sizes", "num-conv-layers"]) or (model_type not in ["bert", "hybridcnnbert"] and key in ["hidden-size", "num-hidden-layers", "num-attention-heads", "intermediate-size", "output-attentions"]):
            continue
        value_str = f"{value:,}" if isinstance(value, int) else str(value)
        output += f"  {key:>25}: {value_str}\n"

    output += f"""
Training Hyperparameters:
{'-' * 50}
"""
    training_hyperparams = {
        "learning-rate": optimizer_config.get("lr", "Missing"),
        "batch-size": checkpoint.get("batch_size", "Missing"),
        "target-epochs": checkpoint.get("epochs", "Missing"),
        "focal-gamma": checkpoint.get("focal_gamma", "Missing"),
        "level-weights": checkpoint.get("level_weights", "Missing"),
        "optimizer": optimizer_config if optimizer_config else "Missing",
        "scheduler-steps": scheduler_config.get("num_training_steps", "Missing")
    }
    for key, value in training_hyperparams.items():
        value_str = f"{value:,}" if isinstance(value, int) else str(value)
        output += f"  {key:>25}: {value_str}\n"

    output += f"""
Dataset Info:
{'-' * 50}
"""
    dataset_size = checkpoint.get("dataset_size", "Missing")
    train_size = checkpoint.get("train_size", "Missing")
    val_size = checkpoint.get("val_size", "Missing")
    fasta_file = checkpoint.get("fasta_file", "Missing")
    taxonomy_file = checkpoint.get("taxonomy_file", "Missing")
    dataset_info = {
        "total-sequences": f"{dataset_size:,}" if isinstance(dataset_size, int) else str(dataset_size),
        "training": f"{train_size:,}" if isinstance(train_size, int) else str(train_size),
        "validation": f"{val_size:,}" if isinstance(val_size, int) else str(val_size),
        "fasta-file": fasta_file,
        "taxonomy-file": taxonomy_file
    }
    for key, value in dataset_info.items():
        output += f"  {key:>25}: {value}\n"

    output += f"""
Taxonomic Levels:
{'-' * 50}
"""
    if taxonomic_ranks and num_labels_per_level:
        for level, rank in enumerate(taxonomic_ranks):
            labels = num_labels_per_level.get(str(level), "Missing")
            output += f"  Level {level} - {rank:>15}: {labels} labels\n"
    else:
        output += f"  {'taxonomic-levels':>25}: Missing\n"

    output += f"""
Timing:
{'-' * 50}
"""
    checkpoint_total_time = checkpoint.get("checkpoint_total_time", 0)
    evaluation_time = checkpoint.get("current_eval_time", 0)
    training_time = checkpoint.get("current_train_time", 0)
    if training_time == 0 and evaluation_time == 0 and checkpoint_total_time != 0:
        training_time = checkpoint_total_time
    timing = {
        "training-time": f"{training_time} seconds" if training_time != 0 else "Missing",
        "evaluation-time": f"{evaluation_time} seconds" if evaluation_time != 0 else "Missing"
    }
    for key, value in timing.items():
        output += f"  {key:>25}: {value}\n"

    output += f"""
System Info:
{'-' * 50}
"""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    system_info = {
        "cuda": "Available" if cuda_available else "Not Available",
        "gpu": gpu_name
    }
    for key, value in system_info.items():
        output += f"  {key:>25}: {value}\n"

    # Log the entire output as one string
    logger.info(output.rstrip("\n"))

    # Export to JSON if requested
    if args.export_metrics:
        export_data = {
            "version": f"deeptaxa.v{__version__}",
            "checkpoint_path": checkpoint_path,
            "timestamp": timestamp,
            "model_details": {k.replace("-", "_"): v for k, v in model_details.items()},
            "training_hyperparameters": {k.replace("-", "_"): v for k, v in training_hyperparams.items()},
            "dataset_info": {
                "total_sequences": dataset_size,
                "training_sequences": train_size,
                "validation_sequences": val_size,
                "fasta_file": fasta_file,
                "taxonomy_file": taxonomy_file
            },
            "taxonomic_levels": {
                str(level): {"rank": rank, "labels": num_labels_per_level.get(str(level), "Missing")}
                for level, rank in enumerate(taxonomic_ranks)
            } if taxonomic_ranks else "Missing",
            "timing": {k.replace("-", "_"): v.split()[0] if "seconds" in v else v for k, v in timing.items()},
            "performance_metrics": {
                "training_loss": checkpoint.get("current_train_loss", "Missing"),
                "validation_loss": checkpoint.get("val_loss", "Missing"),
                "validation_metrics": checkpoint.get("metrics", {})
            },
            "system_info": system_info
        }
        with open(args.export_metrics, 'w') as f:
            json.dump(export_data, f, indent=4)
        logger.info(f"\n{'-' * 50}\nMetrics exported to: {args.export_metrics}")

if __name__ == "__main__":
    pass