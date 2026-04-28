#!/bin/bash
# Calibration diagnosis: test whether level_temperatures cause phylum/class miscalibration
# Runs prediction twice on the same checkpoint with different temperature settings
#
# Usage:
#   bash scripts/calibration_diagnosis.sh <checkpoint> <test_fasta> <test_tax> <output_base> [temps_a] [temps_b]
#
# Example:
#   bash scripts/calibration_diagnosis.sh /work/checkpoint.pt /work/test.fna.gz /work/test.tsv.gz /work/output \
#       "0.5 5.0 5.0 1.0 1.0 1.0 1.0" "1.0 1.0 1.0 1.0 1.0 1.0 1.0"

set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint> <test_fasta> <test_tax> <output_base> [temps_a] [temps_b]}"
TEST_FASTA="${2:?Missing test FASTA file}"
TEST_TAX="${3:?Missing test taxonomy file}"
OUTPUT_BASE="${4:?Missing output directory}"
TEMPS_A="${5:-0.5 5.0 5.0 1.0 1.0 1.0 1.0}"
TEMPS_B="${6:-1.0 1.0 1.0 1.0 1.0 1.0 1.0}"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "=========================================="
echo "Calibration Diagnosis"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo ""

echo "[1/2] Run A: temperatures [${TEMPS_A}]"
echo ""

deeptaxa predict \
    --fasta-file "${TEST_FASTA}" \
    --taxonomy-file "${TEST_TAX}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_BASE}/run_a" \
    --batch-size 64 \
    --level-temperatures ${TEMPS_A}

echo ""
echo "[2/2] Run B: temperatures [${TEMPS_B}]"
echo ""

deeptaxa predict \
    --fasta-file "${TEST_FASTA}" \
    --taxonomy-file "${TEST_TAX}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_BASE}/run_b" \
    --batch-size 64 \
    --level-temperatures ${TEMPS_B}

echo ""
echo "=========================================="
echo "Done. Compare ECE values:"
echo "  Run A: ${OUTPUT_BASE}/run_a/"
echo "  Run B: ${OUTPUT_BASE}/run_b/"
echo "=========================================="
