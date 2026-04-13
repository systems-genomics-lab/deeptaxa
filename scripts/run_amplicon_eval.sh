#!/usr/bin/env bash
# Amplicon evaluation v2 (R1.3) — V3-V4 in silico PCR + CE checkpoint
#
# Uses simulate_amplicons.py for primer-based extraction from full-length
# Greengenes 16S reference sequences. Predicts with the CE baseline checkpoint.
#
# Usage:
#   bash revision/scripts/run_amplicon_eval_v2.sh

set -euo pipefail

# Paths
TEST_FASTA="/work/deeptaxa-data/greengenes/gg_2024_09_testing.fna.gz"
TEST_TAX="/work/deeptaxa-data/greengenes/gg_2024_09_testing.tsv.gz"
CHECKPOINT="/work/deeptaxa-outputs/baseline_clean_10ep/checkpoints/deeptaxa_2026_03_16T10_40_07_25aca8c0_27fb_4a93_8596_2b1390bcf1f2_epoch10.pt"
OUTDIR="/work/deeptaxa-outputs/amplicon_eval_v2"
LOGFILE="${OUTDIR}/amplicon_eval_v2.log"

# DeepTaxa public repo (for simulate_amplicons.py)
DEEPTAXA_DIR="/work/deeptaxa"

mkdir -p "${OUTDIR}"

# Clear previous log
> "${LOGFILE}"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "=========================================="
echo "Amplicon Evaluation v2 — V3-V4 + CE checkpoint"
echo "Date: $(date)"
echo "=========================================="
echo ""
echo "Test FASTA:  ${TEST_FASTA}"
echo "Test tax:    ${TEST_TAX}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Output:      ${OUTDIR}"
echo ""

# Step 1: Extract V3-V4 amplicons via in silico PCR
# 341F (CCTACGGGNGGCWGCAG) / 805R (GACTACHVGGGTATCTAATCC)
# Degenerate base matching, up to 2 mismatches, length filter 200-600 bp
echo "[Step 1/3] Extracting V3-V4 amplicons (in silico PCR)..."
python3 "${DEEPTAXA_DIR}/scripts/simulate_amplicons.py" \
    --input-fasta "${TEST_FASTA}" \
    --output-fasta "${OUTDIR}/test_v3v4.fasta" \
    --region V3-V4 \
    --min-length 200 \
    --max-length 600 \
    --max-mismatches 2 \
    --error-rate 0.0 \
    --seed 42

N_EXTRACTED=$(grep -c '^>' "${OUTDIR}/test_v3v4.fasta")
echo ""
echo "Extracted ${N_EXTRACTED} V3-V4 amplicons"
echo ""

# Step 2: Create matching taxonomy file
echo "[Step 2/3] Creating taxonomy subset..."
python3 -c "
import gzip

ids = set()
with open('${OUTDIR}/test_v3v4.fasta') as f:
    for line in f:
        if line.startswith('>'):
            ids.add(line[1:].strip().split()[0])

n = 0
with gzip.open('${TEST_TAX}', 'rt') as fin, open('${OUTDIR}/test_v3v4.tsv', 'w') as fout:
    header = next(fin)
    fout.write(header)
    for line in fin:
        if line.split('\t')[0] in ids:
            fout.write(line)
            n += 1

print(f'Wrote {n} taxonomy entries for {len(ids)} sequence IDs')
"
echo ""

# Step 3: Predict with CE checkpoint
echo "[Step 3/3] Running DeepTaxa predictions on V3-V4 amplicons..."
deeptaxa predict \
    --fasta-file "${OUTDIR}/test_v3v4.fasta" \
    --taxonomy-file "${OUTDIR}/test_v3v4.tsv" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTDIR}/predictions_v3v4" \
    --batch-size 64

echo ""
echo "=========================================="
echo "Amplicon evaluation v2 complete."
echo "Results: ${OUTDIR}/predictions_v3v4/metrics.json"
echo "Log:     ${LOGFILE}"
echo "=========================================="
