#!/bin/bash
# Script to train, resume, describe, and predict with a DeepTaxa HybridCNNBERT model, logging everything to a single file.
#
# Usage:
#   ./deeptaxa_workflow.sh [--verbose] [--export-sequence-embeddings] [--export-permutation-importance] [FRESH_EPOCHS] [RESUME_EPOCH] [END_EPOCHS]
#   Defaults: verbose off, export-sequence-embeddings off, export-permutation-importance off, FRESH_EPOCHS=2, RESUME_EPOCH=1, END_EPOCHS=3
#
# Options:
#   --verbose: Enable verbose logging (default: off)
#   --export-sequence-embeddings: Export sequence-specific embeddings and attention weights (default: off)
#   --export-permutation-importance: Export permutation importance scores (default: on)

set -euo pipefail
set -x

# Default values
VERBOSE=false
EXPORT_SEQUENCE_EMBEDDINGS=false
EXPORT_PERMUTATION_IMPORTANCE=false
FRESH_EPOCHS_DEFAULT=1
RESUME_EPOCH_DEFAULT=1
END_EPOCHS_DEFAULT=2

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --export-sequence-embeddings)
            EXPORT_SEQUENCE_EMBEDDINGS=true
            shift
            ;;
        --export-permutation-importance)
            EXPORT_PERMUTATION_IMPORTANCE=true
            shift
            ;;
        --level-temperatures)
            LEVEL_TEMPERATURES="$2"
            shift 2
            ;;
        *)
            if [ -z "${FRESH_EPOCHS:-}" ]; then
                FRESH_EPOCHS="$1"
            elif [ -z "${RESUME_EPOCH:-}" ]; then
                RESUME_EPOCH="$1"
            elif [ -z "${END_EPOCHS:-}" ]; then
                END_EPOCHS="$1"
            else
                echo "Error: Too many arguments."
                echo "Usage: $0 [--verbose] [--export-sequence-embeddings] [--export-permutation-importance] [FRESH_EPOCHS] [RESUME_EPOCH] [END_EPOCHS]"
                exit 1
            fi
            shift
            ;;
    esac
done

# Assign defaults if not provided
FRESH_EPOCHS="${FRESH_EPOCHS:-$FRESH_EPOCHS_DEFAULT}"
RESUME_EPOCH="${RESUME_EPOCH:-$RESUME_EPOCH_DEFAULT}"
END_EPOCHS="${END_EPOCHS:-$END_EPOCHS_DEFAULT}"

# Validate epoch inputs
for EPOCH_VAR in "$FRESH_EPOCHS" "$RESUME_EPOCH" "$END_EPOCHS"; do
    if ! [[ "$EPOCH_VAR" =~ ^[0-9]+$ ]] || [ "$EPOCH_VAR" -lt 1 ]; then
        echo "Error: Epochs must be positive integers, got '$EPOCH_VAR'."
        exit 1
    fi
done

if [ "$RESUME_EPOCH" -gt "$FRESH_EPOCHS" ]; then
    echo "Error: RESUME_EPOCH ($RESUME_EPOCH) must be <= FRESH_EPOCHS ($FRESH_EPOCHS)."
    exit 1
fi
if [ "$FRESH_EPOCHS" -ge "$END_EPOCHS" ]; then
    echo "Error: FRESH_EPOCHS ($FRESH_EPOCHS) must be < END_EPOCHS ($END_EPOCHS)."
    exit 1
fi

rm -fr ../deeptaxa-logs ../deeptaxa-outputs
mkdir -p ../deeptaxa-logs ../deeptaxa-outputs

# Generate timestamp
TIMESTAMP=$(date +"%Y_%m_%dT%H_%M_%S")
LOG_FILE="../deeptaxa-logs/deeptaxa_workflow_${TIMESTAMP}.log"

# Base directory for outputs
OUTPUT_DIR="../deeptaxa-outputs"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"

# Common training arguments
COMMON_ARGS="--fasta-file ../deeptaxa-data/greengenes/gg_2024_09_training.fna.gz \
  --taxonomy-file ../deeptaxa-data/greengenes/gg_2024_09_training.tsv.gz \
  --model-type cnn \
  --max-length 128 \
  --embed-dim 128 \
  --batch-size 8 \
  --learning-rate 0.0001 \
  --hidden-dropout-prob 0.2 \
  --num-filters 128 \
  --kernel-sizes 3 \
  --num-conv-layers 1 \
  --focal-gamma 2.0 \
  --output-dir ${OUTPUT_DIR}"
[ "$VERBOSE" = true ] && COMMON_ARGS="$COMMON_ARGS --verbose"
[ "$EXPORT_SEQUENCE_EMBEDDINGS" = true ] && COMMON_ARGS="$COMMON_ARGS --export-sequence-embeddings"
[ "$EXPORT_PERMUTATION_IMPORTANCE" = true ] && COMMON_ARGS="$COMMON_ARGS --export-permutation-importance"

# Redirect all output to the log file
exec > "$LOG_FILE" 2>&1

echo "Workflow started at $(date)"
echo "DeepTaxa version: $(python -c 'from deeptaxa import __version__; print(__version__)')"
echo "Verbose mode: $VERBOSE"
echo "Export sequence embeddings: $EXPORT_SEQUENCE_EMBEDDINGS"
echo "Export permutation importance: $EXPORT_PERMUTATION_IMPORTANCE"

# Step 1: Train fresh with warning suppression
echo "Starting fresh training for $FRESH_EPOCHS epochs..."
time PYTHONWARNINGS=ignore python -m deeptaxa.cli train \
  $COMMON_ARGS \
  --epochs "$FRESH_EPOCHS"

# Wait briefly to ensure log is flushed
sleep 2

# Extract the run UUID with debug
echo "Extracting RUN_UUID from $LOG_FILE..."
RUN_UUID=$(awk '/Saved model checkpoint to/ {for (i=1; i<=NF; i++) if ($i ~ /deeptaxa_.*_epoch[0-9]+\.pt/) {split($i, a, "deeptaxa_"); sub(/_epoch[0-9]+\.pt/, "", a[2]); print a[2]; exit}}' "$LOG_FILE")
if [ -z "$RUN_UUID" ]; then
    echo "Warning: Could not extract run UUID from $LOG_FILE. Checking for checkpoints manually..."
    LATEST_CHECKPOINT=$(ls -t "${CHECKPOINT_DIR}/deeptaxa_*_epoch${FRESH_EPOCHS}.pt" 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        RUN_UUID=$(basename "$LATEST_CHECKPOINT" | sed 's/deeptaxa_\(.*\)_epoch[0-9]\+\.pt/\1/')
        echo "Fallback: Extracted RUN_UUID '$RUN_UUID' from checkpoint file $LATEST_CHECKPOINT"
    else
        echo "Error: No checkpoints found in $CHECKPOINT_DIR. Last 10 lines of log:"
        tail -n 10 "$LOG_FILE"
        exit 1
    fi
else
    echo "Extracted run UUID: $RUN_UUID"
fi

# Define checkpoint paths
CHECKPOINT_RESUME="${CHECKPOINT_DIR}/deeptaxa_${RUN_UUID}_epoch${RESUME_EPOCH}.pt"
CHECKPOINT_END="${CHECKPOINT_DIR}/deeptaxa_${RUN_UUID}_epoch${END_EPOCHS}.pt"

# Verify resume checkpoint
if [ ! -f "$CHECKPOINT_RESUME" ]; then
    echo "Error: Checkpoint $CHECKPOINT_RESUME not found."
    exit 1
fi

# Step 2: Resume training
echo "Resuming training from $CHECKPOINT_RESUME to $END_EPOCHS epochs..."
time PYTHONWARNINGS=ignore python -m deeptaxa.cli train \
  $COMMON_ARGS \
  --epochs "$END_EPOCHS" \
  --resume "$CHECKPOINT_RESUME"

# Verify end checkpoint
if [ ! -f "$CHECKPOINT_END" ]; then
    echo "Error: Checkpoint $CHECKPOINT_END not found after resumption."
    exit 1
fi

# Step 3: Describe the checkpoint
echo "Describing checkpoint $CHECKPOINT_END..."
time PYTHONWARNINGS=ignore python -m deeptaxa.cli describe \
  --checkpoint "$CHECKPOINT_END" \
  --export-metrics "${TIMESTAMP}_epoch${END_EPOCHS}_metrics.json" \
  $( [ "$VERBOSE" = true ] && echo "--verbose" )

# Step 4: Predict
echo "Predicting with checkpoint $CHECKPOINT_END..."
time PYTHONWARNINGS=ignore python -m deeptaxa.cli predict \
  --fasta-file ../deeptaxa-data/greengenes/gg_2024_09_testing.fna.gz \
  --checkpoint "$CHECKPOINT_END" \
  --taxonomy-file ../deeptaxa-data/greengenes/gg_2024_09_testing.tsv.gz \
  --use-raw-labels-for-true \
  --tabular \
  --top-k 3 \
  --confidence-threshold 0.95 \
  --output-dir "${OUTPUT_DIR}" \
  $( [ "$VERBOSE" = true ] && echo "--verbose" )

echo "Workflow completed."
echo "Log file: $LOG_FILE"
echo "Checkpoints: $CHECKPOINT_RESUME, $CHECKPOINT_END"
echo "Metrics: ${TIMESTAMP}_epoch${END_EPOCHS}_metrics.json"
echo "Predictions: ${OUTPUT_DIR}/predictions_gg_2024_09_testing.tsv"
