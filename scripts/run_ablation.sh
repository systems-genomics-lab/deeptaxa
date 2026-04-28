#!/usr/bin/env bash
# Ablation study with manuscript hyperparameters (Table 1)
#
# Baseline: Hybrid CNN+BERT with cross-entropy loss, uniform level weights
#
# Variants (each changes exactly one factor from baseline):
#   1. baseline_hybrid   - Full hybrid CNN+BERT, CE loss (default)
#   2. cnn_only          - CNN branch only, no BERT
#   3. bert_only         - BERT branch only, no CNN
#   4. onehot_cnn        - CNN with one-hot encoding instead of BPE tokenization
#   5. focal             - Hybrid CNN+BERT with focal loss (gamma=3.5)

set -euo pipefail

DATA_DIR="/work/deeptaxa-data/greengenes"
OUTPUT_BASE="/work/deeptaxa-outputs/ablation"

TRAIN_FASTA="${DATA_DIR}/gg_2024_09_training.fna.gz"
TRAIN_TAX="${DATA_DIR}/gg_2024_09_training.tsv.gz"
TEST_FASTA="${DATA_DIR}/gg_2024_09_testing.fna.gz"
TEST_TAX="${DATA_DIR}/gg_2024_09_testing.tsv.gz"

# Manuscript Table 1 parameters (shared)
COMMON_ARGS=(
    --learning-rate 0.0005
    --batch-size 64
    --epochs 10
    --embed-dim 896
    --num-filters 256
    --num-conv-layers 1
    --seed 42
    --eval-every 1
)

# BERT/Hybrid architecture parameters (manuscript Table 1)
BERT_ARGS=(
    --hidden-size 896
    --num-hidden-layers 4
    --num-attention-heads 7
    --intermediate-size 3584
)

train_and_predict() {
    local name="$1"
    local step="$2"
    local total="$3"
    shift 3
    local extra_args=("$@")

    echo ""
    echo "[${step}/${total}] Training ${name}..."
    echo ""

    deeptaxa train \
        --fasta-file "${TRAIN_FASTA}" \
        --taxonomy-file "${TRAIN_TAX}" \
        --output-dir "${OUTPUT_BASE}/${name}" \
        "${COMMON_ARGS[@]}" \
        "${extra_args[@]}"

    echo ""
    echo "[${step}/${total}] Predicting ${name}..."
    echo ""

    local ckpt
    ckpt=$(ls -t "${OUTPUT_BASE}/${name}/checkpoints/"*epoch10.pt 2>/dev/null | head -1)
    if [ -z "${ckpt}" ]; then
        echo "WARNING: No epoch 10 checkpoint found for ${name}, skipping prediction"
        return
    fi

    deeptaxa predict \
        --fasta-file "${TEST_FASTA}" \
        --taxonomy-file "${TEST_TAX}" \
        --checkpoint "${ckpt}" \
        --output-dir "${OUTPUT_BASE}/${name}/predictions" \
        --batch-size 64
}

echo "=========================================="
echo "Ablation: Manuscript hyperparameters"
echo "=========================================="
echo "Baseline: Hybrid CNN+BERT, cross-entropy loss, uniform weights"
echo "Variants: 5 total"

# 1. Baseline Hybrid (CE loss, uniform weights - default)
train_and_predict "baseline_hybrid" 1 5 \
    --model-type hybridcnnbert \
    --loss-type cross_entropy \
    "${BERT_ARGS[@]}"

# 2. CNN-only (removes BERT branch)
train_and_predict "cnn_only" 2 5 \
    --model-type cnn \
    --loss-type cross_entropy

# 3. BERT-only (removes CNN branch)
train_and_predict "bert_only" 3 5 \
    --model-type bert \
    --loss-type cross_entropy \
    "${BERT_ARGS[@]}"

# 4. One-hot CNN (replaces BPE with one-hot encoding; necessarily removes BERT)
train_and_predict "onehot_cnn" 4 5 \
    --model-type cnn \
    --loss-type cross_entropy \
    --encoding onehot

# 5. Focal loss (replaces CE with focal loss, gamma=3.5)
train_and_predict "focal" 5 5 \
    --model-type hybridcnnbert \
    --loss-type focal \
    --focal-gamma 3.5 \
    "${BERT_ARGS[@]}"

echo ""
echo "=========================================="
echo "Ablation complete."
echo "Results in: ${OUTPUT_BASE}"
echo "=========================================="
