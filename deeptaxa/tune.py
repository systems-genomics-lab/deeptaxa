"""
Module: tune.py

Description:
    Implements Bayesian hyperparameter tuning for DeepTaxa models using Optuna.
"""

import optuna
import argparse
import logging
import os
import json
from deeptaxa.train import train
from deeptaxa.utils import set_seed
from deeptaxa.config import DEFAULT_CONFIG
from copy import deepcopy

logger = logging.getLogger(__name__)

def objective(trial, base_args):
    """
    Optuna objective function for hyperparameter tuning.

    Args:
        trial (optuna.trial.Trial): Current trial object.
        base_args (argparse.Namespace): Base CLI arguments.

    Returns:
        float: Metric to minimize (e.g., negative F1-score).
    """
    # Copy base arguments
    args = deepcopy(base_args)

    # Explicitly set resume to None
    args.resume = None

    # Define hyperparameter search space
    args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    args.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    args.hidden_dropout_prob = trial.suggest_float("hidden_dropout_prob", 0.1, 0.5)
    if getattr(args, 'loss_type', 'cross_entropy') == 'focal':
        args.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 5.0)
    args.optimizer_weight_decay = trial.suggest_float("optimizer_weight_decay", 0.0, 0.1)
    # max_length is tuned only when the CLI did not pin a fixed value.
    if base_args.max_length is None:
        args.max_length = trial.suggest_categorical("max_length", [192, 256, 384, 512])

    # Model-specific hyperparameters
    if args.model_type in ["cnn", "hybridcnnbert"]:
        args.num_filters = trial.suggest_int("num_filters", 256, 1024, step=128)
        args.num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        args.embed_dim = trial.suggest_int("embed_dim", 256, 1024, step=128)
        # Kernel sizes: encode as strings for Optuna compatibility (lists not supported in persistent storage)
        kernel_options = ["3", "5", "7", "3_5", "3_7", "5_7", "3_5_7", "3_5_9", "3_7_9", "5_7_9"]
        kernel_str = trial.suggest_categorical("kernel_sizes", kernel_options)
        args.kernel_sizes = [int(k) for k in kernel_str.split("_")]
    if args.model_type in ["bert", "hybridcnnbert"]:
        # Encode (hidden_size, num_attention_heads) as a single categorical to guarantee
        # hidden_size % num_attention_heads == 0 without dynamic distributions
        bert_configs = [
            "256_4", "320_4", "384_4", "448_4", "512_4", "576_4", "640_4", "704_4",
            "768_4", "832_4", "896_4", "960_4", "1024_4",
            "512_8", "640_8", "768_8", "896_8", "1024_8",
            "768_12", "960_12",
            "896_16", "1024_16",
        ]
        bert_config = trial.suggest_categorical("bert_config", bert_configs)
        args.hidden_size, args.num_attention_heads = [int(x) for x in bert_config.split("_")]
        args.num_hidden_layers = trial.suggest_int("num_hidden_layers", 2, 12)
        args.intermediate_size = args.hidden_size * 4

    # Set output directory for this trial and create subdirectories
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    args.output_dir = trial_dir
    try:
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(trial_dir, "metadata"), exist_ok=True)
    except Exception as e:
        logger.error("Failed to create directories for trial %d: %s", trial.number, str(e))
        raise

    # Set seed
    set_seed(args.seed)

    # Run training with trial object for early stopping
    try:
        avg_f1 = train(args, trial=trial)
    except Exception as e:
        trial_number = trial.number if trial else "unknown"
        logger.error("Trial %s failed: %s", trial_number, str(e))
        raise optuna.TrialPruned()

    return -avg_f1  # Maximize F1-score

def main():
    """
    Execute Bayesian hyperparameter tuning for a DeepTaxa model.
    """
    parser = argparse.ArgumentParser(description="DeepTaxa Hyperparameter Tuning with Optuna")
    parser.add_argument("--fasta-file", type=str, required=True, help="Path to FASTA file")
    parser.add_argument("--taxonomy-file", type=str, required=True, help="Path to taxonomy file")
    parser.add_argument("--model-type", type=str, choices=["cnn", "bert", "hybridcnnbert"], required=True)
    parser.add_argument("--output-dir", type=str, default="./tune_results", help="Base output directory")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of trials to run")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="Epochs per trial")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="Random seed")
    parser.add_argument("--encoding", type=str, choices=["dnabert", "onehot"], default=DEFAULT_CONFIG["encoding"],
                        help="Sequence encoding (default: dnabert)")
    parser.add_argument("--loss-type", type=str, choices=["cross_entropy", "focal"], default=DEFAULT_CONFIG["loss_type"],
                        help="Loss function (default: cross_entropy)")
    parser.add_argument("--max-length", type=int, default=None,
                        help=(
                            "Pin max_length to a fixed value (e.g. 512 for full-length 16S). "
                            "If omitted, max_length is added to the Optuna search space with "
                            "values [192, 256, 384, 512] — appropriate for short amplicons "
                            "where a smaller token budget saves compute without truncating."
                        ))
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Set default arguments (epochs, encoding, loss_type come from CLI; rest from defaults)
    args.val_split = DEFAULT_CONFIG["val_split"]
    args.eval_every = DEFAULT_CONFIG["eval_every"]
    args.accum_steps = DEFAULT_CONFIG["accum_steps"]
    args.num_workers = DEFAULT_CONFIG["num_workers"]
    args.tokenizer_name = DEFAULT_CONFIG["tokenizer_name"]
    # args.max_length is None when the CLI did not pin it; objective() will
    # tune it via trial.suggest_categorical in that case.
    args.level_weights = DEFAULT_CONFIG["level_weights"]
    args.scheduler_warmup_ratio = DEFAULT_CONFIG["scheduler_warmup_ratio"]
    args.optimizer_betas = DEFAULT_CONFIG["optimizer_betas"]
    args.optimizer_eps = DEFAULT_CONFIG["optimizer_eps"]
    args.focal_reduction = DEFAULT_CONFIG["focal_reduction"]
    args.resume = None  # Explicitly set resume to None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create Optuna study with MedianPruner for early stopping
    study_name = f"deeptaxa_{args.model_type}_tuning"
    storage = f"sqlite:///{os.path.join(args.output_dir, f'deeptaxa_optuna_{args.model_type}.db')}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,  # Reduced from 5
            n_warmup_steps=2,   # Increased to 2 for stability
            interval_steps=1     # Check every epoch
        ),
        load_if_exists=True
    )

    # Optimize
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, n_jobs=1)

    # Log best trial
    try:
        best_trial = study.best_trial
    except ValueError:
        logger.error("No completed trials for %s. All %d trials failed or were pruned.", args.model_type, args.n_trials)
        return

    logger.info("Best trial for %s:", args.model_type)
    logger.info("  F1-Score: %.4f", -study.best_value)
    logger.info("  Parameters: %s", study.best_params)

    # Extract metrics and run_uuid for the best trial
    best_trial_dir = os.path.join(args.output_dir, f"trial_{best_trial.number}")
    metrics_dir = os.path.join(best_trial_dir, "metrics")
    if not os.path.isdir(metrics_dir):
        logger.error("Metrics directory not found for best trial %d: %s (may have been cleaned up from a previous run)", best_trial.number, metrics_dir)
        logger.error("Delete the study DB and re-run from scratch, or remove the optuna_search_* directory.")
        return
    latest_metrics_file = max(
        [os.path.join(metrics_dir, f) for f in os.listdir(metrics_dir) if f.endswith(".json")],
        key=os.path.getmtime,
        default=None
    )
    if not latest_metrics_file:
        logger.error("No metrics file found for best trial %d in %s", best_trial.number, metrics_dir)
        return

    with open(latest_metrics_file, "r") as f:
        metrics_data = json.load(f)

    # Extract run_uuid and performance metrics safely
    try:
        run_uuid = metrics_data.get("run_uuid", metrics_data.get("model_details", {}).get("run_uuid", "unknown"))
        val_metrics = metrics_data["performance_metrics"]["validation_metrics"]
        val_loss = metrics_data["performance_metrics"]["validation_loss"]
        avg_f1 = sum(m["f1_score"] for m in val_metrics.values()) / len(val_metrics)
    except (KeyError, TypeError, ZeroDivisionError) as e:
        logger.error("Failed to parse metrics for best trial %d: %s", best_trial.number, e)
        logger.error("Metrics file may be corrupt: %s", latest_metrics_file)
        return

    # Expand encoded params back to individual fields for readability
    best_params = dict(study.best_params)
    if "bert_config" in best_params:
        config = best_params.pop("bert_config")
        h, a = [int(x) for x in config.split("_")]
        best_params["hidden_size"] = h
        best_params["num_attention_heads"] = a
        best_params["intermediate_size"] = h * 4
    if "kernel_sizes" in best_params and isinstance(best_params["kernel_sizes"], str):
        best_params["kernel_sizes"] = [int(k) for k in best_params["kernel_sizes"].split("_")]

    # Record encoding and loss_type for reproducibility
    best_params["encoding"] = getattr(args, "encoding", "dnabert")
    best_params["loss_type"] = getattr(args, "loss_type", "cross_entropy")

    # Prepare output data
    output_data = {
        "model_type": args.model_type,
        "run_uuid": run_uuid,
        "trial_number": best_trial.number,
        "f1_score": -study.best_value,  # Positive F1-score
        "hyperparameters": best_params,
        "performance_metrics": {
            "validation_loss": val_loss,
            "validation_metrics": val_metrics,
            "average_f1_score": avg_f1,
        }
    }

    # Save best hyperparameters, run_uuid, and metrics
    best_params_file = os.path.join(args.output_dir, f"deeptaxa_best_params_{args.model_type}.json")
    with open(best_params_file, "w") as f:
        json.dump(output_data, f, indent=4)
    logger.info("Saved best parameters, run_uuid, and metrics to %s", best_params_file)

    # Log study location
    logger.info("Saved Optuna study to %s", storage.replace("sqlite:///", ""))

if __name__ == "__main__":
    main()

