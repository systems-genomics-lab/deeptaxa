#!/usr/bin/env bash
# End-to-end similarity-stratified evaluation pipeline
#
# 1. Computes nearest-neighbor similarity between test and training sequences
# 2. Splits test set into similarity buckets
# 3. Runs DeepTaxa prediction on each bucket
# 4. Reports per-bucket accuracy
#
# Usage:
#   bash scripts/run_similarity_eval.sh <train_fasta> <test_fasta> <test_tax> <checkpoint> <output_dir> [threads]

set -euo pipefail

TRAIN_FASTA="${1:?Usage: $0 <train_fasta> <test_fasta> <test_tax> <checkpoint> <output_dir> [threads]}"
TEST_FASTA="${2:?Missing test FASTA}"
TEST_TAX="${3:?Missing test taxonomy}"
CHECKPOINT="${4:?Missing checkpoint}"
OUTPUT_DIR="${5:?Missing output directory}"
THREADS="${6:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Similarity-Stratified Evaluation"
echo "=========================================="
echo ""

# Step 1: Compute similarity and split FASTA
echo "[Step 1/3] Computing sequence similarity..."
python3 "${SCRIPT_DIR}/sequence_similarity.py" \
    --train-fasta "${TRAIN_FASTA}" \
    --test-fasta "${TEST_FASTA}" \
    --output-dir "${OUTPUT_DIR}/similarity" \
    --threads "${THREADS}"

echo ""

# Step 2: Create per-bucket taxonomy files by filtering the full taxonomy
echo "[Step 2/3] Creating per-bucket taxonomy files..."
python3 -c "
import csv, gzip, sys, os

tax_file = '${TEST_TAX}'
sim_dir = '${OUTPUT_DIR}/similarity'

# Load full taxonomy (first line is header)
opener = gzip.open if tax_file.endswith('.gz') else open
tax = {}
header = None
with opener(tax_file, 'rt') as f:
    for line in f:
        stripped = line.strip()
        if header is None:
            header = stripped
            continue
        parts = stripped.split('\t', 1)
        if len(parts) >= 2:
            tax[parts[0]] = stripped

# Load bucket assignments
buckets = {}
with open(os.path.join(sim_dir, 'similarity_results.tsv')) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        buckets[row['test_seq_id']] = row['bucket']

# Write per-bucket taxonomy
bucket_names = set(buckets.values())
for bucket in bucket_names:
    ids_in_bucket = [sid for sid, b in buckets.items() if b == bucket]
    tax_path = os.path.join(sim_dir, f'test_{bucket}.tsv')
    with open(tax_path, 'w') as f:
        f.write(header + '\n')
        for sid in ids_in_bucket:
            if sid in tax:
                f.write(tax[sid] + '\n')
    print(f'  {bucket}: {len(ids_in_bucket)} sequences')
"

echo ""

# Step 3: Run predictions on each bucket
echo "[Step 3/3] Running predictions per bucket..."
for BUCKET_FASTA in "${OUTPUT_DIR}"/similarity/test_*.fasta; do
    BUCKET_NAME=$(basename "${BUCKET_FASTA}" .fasta | sed 's/test_//')
    BUCKET_TAX="${OUTPUT_DIR}/similarity/test_${BUCKET_NAME}.tsv"

    if [ ! -f "${BUCKET_TAX}" ]; then
        echo "  Skipping ${BUCKET_NAME}: no taxonomy file"
        continue
    fi

    SEQ_COUNT=$(grep -c "^>" "${BUCKET_FASTA}" 2>/dev/null || echo 0)
    if [ "${SEQ_COUNT}" -eq 0 ]; then
        echo "  Skipping ${BUCKET_NAME}: empty FASTA"
        continue
    fi

    echo ""
    echo "  Predicting ${BUCKET_NAME} (${SEQ_COUNT} sequences)..."
    deeptaxa predict \
        --fasta-file "${BUCKET_FASTA}" \
        --taxonomy-file "${BUCKET_TAX}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${OUTPUT_DIR}/predictions/${BUCKET_NAME}" \
        --batch-size 64
done

echo ""
echo "=========================================="
echo "Similarity-stratified evaluation complete."
echo "Results in: ${OUTPUT_DIR}"
echo "=========================================="
