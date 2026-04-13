#!/bin/bash
# Calibration temperature sweep: test multiple temperature configurations
# Runs prediction for each configuration and collects ECE values
#
# Usage:
#   bash scripts/calibration_sweep.sh <checkpoint> <test_fasta> <test_tax> <output_base>

set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint> <test_fasta> <test_tax> <output_base>}"
TEST_FASTA="${2:?Missing test FASTA file}"
TEST_TAX="${3:?Missing test taxonomy file}"
OUTPUT_BASE="${4:?Missing output directory}"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# Define temperature configurations: name and values
declare -A CONFIGS
CONFIGS=(
    ["1_uniform_1.0"]="1.0 1.0 1.0 1.0 1.0 1.0 1.0"
    ["2_old_default"]="0.5 5.0 5.0 1.0 1.0 1.0 1.0"
    ["3_mild_sharp"]="0.8 0.8 0.8 0.8 0.8 0.8 0.8"
    ["4_hierarchical"]="0.7 0.8 0.8 0.9 0.9 1.0 1.0"
    ["5_species_soft"]="1.0 1.0 1.0 1.0 1.0 1.0 1.2"
)

TOTAL=${#CONFIGS[@]}
COUNT=0

echo "=========================================="
echo "Calibration Temperature Sweep"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Configurations: ${TOTAL}"
echo ""

for NAME in $(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort); do
    TEMPS="${CONFIGS[$NAME]}"
    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}] ${NAME}: temperatures [${TEMPS}]"
    echo ""

    deeptaxa predict \
        --fasta-file "${TEST_FASTA}" \
        --taxonomy-file "${TEST_TAX}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${OUTPUT_BASE}/${NAME}" \
        --batch-size 64 \
        --level-temperatures ${TEMPS}

    echo ""
done

echo "=========================================="
echo "Sweep complete. Results in:"
for NAME in $(echo "${!CONFIGS[@]}" | tr ' ' '\n' | sort); do
    echo "  ${NAME}: ${OUTPUT_BASE}/${NAME}/"
done
echo "=========================================="
