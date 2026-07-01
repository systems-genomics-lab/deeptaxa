#!/usr/bin/env bash
# Step 3. Classify the ASVs from 01_dada2_asv.R with DeepTaxa, the primary taxonomy
# used in the case study. DeepTaxa inference is deterministic (a fixed model with no
# sampling), so re-running reproduces the committed table.
# Inputs:  $DATA/asv_seqs.fasta and the V4 checkpoint at $CHECKPOINT
# Output:  $DATA/asv_taxonomy.tsv
set -euo pipefail

DATA="${DATA:-../data}"
CHECKPOINT="${CHECKPOINT:-deeptaxa-v4-v1.pt}"

# Download the V4 checkpoint if it is not already present.
if [ ! -f "$CHECKPOINT" ]; then
  curl -L -o "$CHECKPOINT" \
    "https://huggingface.co/systems-genomics-lab/deeptaxa/resolve/main/deeptaxa-v4-v1.pt"
fi

deeptaxa predict --fasta-file "$DATA/asv_seqs.fasta" --checkpoint "$CHECKPOINT" \
  --output-dir out --tabular --batch-size 128 --num-workers 4

cp out/asv_seqs_deeptaxa_predictions.tsv "$DATA/asv_taxonomy.tsv"
