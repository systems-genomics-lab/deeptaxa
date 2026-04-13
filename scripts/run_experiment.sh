#!/usr/bin/env bash
# ==============================================================================
# Script: run_experiment.sh
#
# Central experiment runner for DeepTaxa. Handles system info, logging,
# timing, training, test-set prediction, metric summaries, and cleanup.
#
# All deeptaxa train parameters are passed explicitly by the caller.
# This script manages data paths, output directory, logging, and the
# final metrics report.
#
# Features:
#   - Multi-seed support: pass multiple seeds (e.g., --seed 42 123 456) to
#     run the same experiment with different seeds and get mean/std summary
#   - Uses the final epoch checkpoint for prediction
#   - Keeps all checkpoints by default; pass --cleanup-checkpoints to
#     remove intermediate checkpoints and save disk space
#   - Saves a rerun.sh in the output directory for reproducibility
#   - Logs GPU info and data file sizes before starting
#   - Configurable prediction batch size via --predict-batch-size
#
# Usage:
#   # Single seed (standard)
#   bash scripts/run_experiment.sh --name baseline \
#       --model-type hybridcnnbert --loss-type cross_entropy \
#       --batch-size 64 --epochs 10 --learning-rate 0.0005 \
#       --seed 42 --eval-every 1 \
#       --hidden-size 896 --num-hidden-layers 4 --num-attention-heads 7 \
#       --intermediate-size 3584 --embed-dim 896 \
#       --num-filters 256 --num-conv-layers 1
#
#   # Multiple seeds (runs 3 experiments, reports mean/std)
#   bash scripts/run_experiment.sh --name baseline_multiseed \
#       --model-type hybridcnnbert --loss-type cross_entropy \
#       --batch-size 64 --epochs 10 --learning-rate 0.0005 \
#       --seed 42 123 456 --eval-every 1 \
#       --hidden-size 896 --num-hidden-layers 4 --num-attention-heads 7 \
#       --intermediate-size 3584 --embed-dim 896 \
#       --num-filters 256 --num-conv-layers 1
#
#   # If --name is omitted, one is generated automatically:
#       20260404_143022_bs64_ep10_lr0.0005
#
# Environment Variables:
#   DEEPTAXA_DATA_DIR   - Data directory (default: /workspace/deeptaxa-data/greengenes)
#   DEEPTAXA_OUTPUT_DIR - Base output directory (default: /workspace/deeptaxa-output)
#
# Output (single seed):
#   $DEEPTAXA_OUTPUT_DIR/<name>/
#       experiment.log
#       rerun.sh
#       checkpoints/
#       metrics/
#       predictions/
#
# Output (multi-seed):
#   $DEEPTAXA_OUTPUT_DIR/<name>/
#       experiment.log
#       rerun.sh
#       summary.json
#       seed_42/   (checkpoints/, metrics/, predictions/)
#       seed_123/  (checkpoints/, metrics/, predictions/)
#       seed_456/  (checkpoints/, metrics/, predictions/)
# ==============================================================================

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,/^# ==/p' "$0" | sed 's/^# \?//'
    exit 0
fi

# ---------------------------------------------------------------------------
# Parse arguments:
#   --name                experiment name (optional, auto-generated if omitted)
#   --seed N [N ...]      one or more seeds (must be positive integers)
#   --predict-batch-size  batch size for prediction (default: 64)
#   --cleanup-checkpoints remove intermediate checkpoints after training
#   everything else       forwarded to deeptaxa train
# ---------------------------------------------------------------------------
EXPERIMENT_NAME=""
SEEDS=()
PREDICT_BS=64
CLEANUP_CHECKPOINTS=false
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --name requires a value" >&2
                exit 1
            fi
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --predict-batch-size)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --predict-batch-size requires a value" >&2
                exit 1
            fi
            PREDICT_BS="$2"
            shift 2
            ;;
        --cleanup-checkpoints)
            CLEANUP_CHECKPOINTS=true
            shift
            ;;
        --seed)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default to a single seed if none provided
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=(42)
fi

# Validate that all seeds are positive integers
for s in "${SEEDS[@]}"; do
    if ! [[ "$s" =~ ^[0-9]+$ ]]; then
        echo "ERROR: seed must be a positive integer, got: $s" >&2
        exit 1
    fi
done

# Auto-generate experiment name if not provided
if [[ -z "$EXPERIMENT_NAME" ]]; then
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    PARAM_PARTS=""
    i=0
    while [[ $i -lt ${#TRAIN_ARGS[@]} ]]; do
        arg="${TRAIN_ARGS[$i]}"
        next="${TRAIN_ARGS[$((i+1))]:-}"
        case "$arg" in
            --batch-size)                PARAM_PARTS+="_bs${next}";  i=$((i + 2)) ;;
            --epochs)                    PARAM_PARTS+="_ep${next}";  i=$((i + 2)) ;;
            --learning-rate)             PARAM_PARTS+="_lr${next}";  i=$((i + 2)) ;;
            --early-stopping-patience)   PARAM_PARTS+="_es${next}";  i=$((i + 2)) ;;
            *)                           i=$((i + 1)) ;;
        esac
    done
    if [[ ${#SEEDS[@]} -gt 1 ]]; then
        PARAM_PARTS+="_x${#SEEDS[@]}seeds"
    fi
    EXPERIMENT_NAME="${TIMESTAMP}${PARAM_PARTS:-_default}"
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Auto-detect default workspace if env vars not set
if [[ -d "/workspace/deeptaxa-data" ]]; then
    _DEFAULT_BASE="/workspace"
elif [[ -d "/work/deeptaxa-data" ]]; then
    _DEFAULT_BASE="/work"
else
    _DEFAULT_BASE="/workspace"
fi
DATA_DIR="${DEEPTAXA_DATA_DIR:-${_DEFAULT_BASE}/deeptaxa-data/greengenes}"
BASE_OUTPUT_DIR="${DEEPTAXA_OUTPUT_DIR:-${_DEFAULT_BASE}/deeptaxa-output}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"

TRAIN_FASTA="${DATA_DIR}/gg_2024_09_training.fna.gz"
TRAIN_TAX="${DATA_DIR}/gg_2024_09_training.tsv.gz"
TEST_FASTA="${DATA_DIR}/gg_2024_09_testing.fna.gz"
TEST_TAX="${DATA_DIR}/gg_2024_09_testing.tsv.gz"

LOG="${OUTPUT_DIR}/experiment.log"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
    local level="$1"; shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] $*" | tee -a "$LOG"
}

fmt_duration() {
    local secs=$1
    printf '%dh %dm %ds' $((secs / 3600)) $(( (secs % 3600) / 60 )) $((secs % 60))
}

separator() {
    echo "========================================================================" | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# Failure trap: log a message if the script exits unexpectedly
# ---------------------------------------------------------------------------
COMPLETED=false
on_exit() {
    if [[ "$COMPLETED" != "true" ]]; then
        log ERROR "Script exited unexpectedly. Check the log for details: ${LOG}"
    fi
}
trap on_exit EXIT

# ---------------------------------------------------------------------------
# Validate data files (existence and non-zero size)
# ---------------------------------------------------------------------------
for f in "$TRAIN_FASTA" "$TRAIN_TAX" "$TEST_FASTA" "$TEST_TAX"; do
    if [[ ! -f "$f" ]]; then
        log ERROR "Data file not found: $f"
        exit 1
    fi
    if [[ ! -s "$f" ]]; then
        log ERROR "Data file is empty: $f"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Validate training arguments
# ---------------------------------------------------------------------------
if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
    log ERROR "No training arguments provided. Pass at least --model-type and --epochs."
    exit 1
fi

# ---------------------------------------------------------------------------
# Save rerun.sh for reproducibility
# ---------------------------------------------------------------------------
RERUN_FILE="${OUTPUT_DIR}/rerun.sh"
{
    echo "#!/usr/bin/env bash"
    echo "# Auto-generated rerun command for experiment: ${EXPERIMENT_NAME}"
    echo "# Created: $(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo ""
    printf "bash %s \\\\\n" "$SCRIPT_PATH"
    if [[ -n "$EXPERIMENT_NAME" ]]; then
        printf "    --name %s \\\\\n" "$EXPERIMENT_NAME"
    fi
    printf "    --seed %s \\\\\n" "${SEEDS[*]}"
    printf "    --predict-batch-size %s \\\\\n" "$PREDICT_BS"
    if [[ "$CLEANUP_CHECKPOINTS" == "true" ]]; then
        printf "    --cleanup-checkpoints \\\\\n"
    fi
    # Pair flags with their values on the same line
    # Handles multi-value flags like --kernel-sizes 3 5 7
    i=0
    while [[ $i -lt ${#TRAIN_ARGS[@]} ]]; do
        arg="${TRAIN_ARGS[$i]}"
        if [[ "$arg" =~ ^-- ]]; then
            # Collect all non-flag values after this flag
            vals=""
            j=$((i + 1))
            while [[ $j -lt ${#TRAIN_ARGS[@]} && ! "${TRAIN_ARGS[$j]}" =~ ^-- ]]; do
                vals+=" ${TRAIN_ARGS[$j]}"
                j=$((j + 1))
            done
            if [[ $j -ge ${#TRAIN_ARGS[@]} ]]; then
                printf "    %s%s\n" "$arg" "$vals"
            else
                printf "    %s%s \\\\\n" "$arg" "$vals"
            fi
            i=$j
        else
            # Standalone value (shouldn't happen, but handle gracefully)
            if [[ $((i + 1)) -ge ${#TRAIN_ARGS[@]} ]]; then
                printf "    %s\n" "$arg"
            else
                printf "    %s \\\\\n" "$arg"
            fi
            i=$((i + 1))
        fi
    done
} > "$RERUN_FILE"
chmod +x "$RERUN_FILE"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
OVERALL_START=$(date +%s)

separator
log INFO "EXPERIMENT: ${EXPERIMENT_NAME}"
log INFO "Started:    $(date '+%Y-%m-%dT%H:%M:%S%z')"
log INFO "Output:     ${OUTPUT_DIR}"
log INFO "Seeds:      ${SEEDS[*]}"
log INFO "Predict BS: ${PREDICT_BS}"
separator

# System and hardware details
log INFO "Hostname:   $(hostname)"
log INFO "OS:         $(uname -srm)"
log INFO "Python:     $(python3 --version 2>&1)"
log INFO "PyTorch:    $(python3 -c 'import torch; print(torch.__version__)' 2>&1)"
log INFO "CUDA:       $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>&1)"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap \
        --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
        log INFO "GPU:        $line"
    done
fi
log INFO "DeepTaxa:   $(deeptaxa --version 2>&1 || echo 'unknown')"
log INFO "Disk free:  $(df -h "${OUTPUT_DIR}" 2>/dev/null | tail -1 | awk '{print $4}')"

# Data file sizes
for f in "$TRAIN_FASTA" "$TRAIN_TAX" "$TEST_FASTA" "$TEST_TAX"; do
    log INFO "Data:       $(basename "$f") ($(du -h "$f" | cut -f1))"
done

separator
log INFO "Training arguments: ${TRAIN_ARGS[*]:-<none>}"
separator

# ---------------------------------------------------------------------------
# print_val_metrics: print validation metrics table from a metrics JSON
# Args: $1 = path to metrics JSON
# ---------------------------------------------------------------------------
print_val_metrics() {
    local output
    output=$(python3 - "$1" <<'PYEOF'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
vm = data.get("performance_metrics", {}).get("validation_metrics", {})
val_loss = data.get("performance_metrics", {}).get("validation_loss", "N/A")
ranks = data.get("taxonomic_levels", {})
levels = sorted([k for k in vm.keys() if k.isdigit()], key=int)
print(f"  {'Rank':<10} | {'Accuracy':>8} | {'F1':>8} | {'Precision':>8} | {'Recall':>8}")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
for level in levels:
    m = vm[level]
    rank_name = ranks.get(level, {}).get("rank", f"level_{level}")
    print(f"  {rank_name:<10} | {m['accuracy']:.4f}   | {m['f1_score']:.4f}   | {m['precision']:.4f}   | {m['recall']:.4f}")
print()
print(f"  Validation Loss: {val_loss}")
PYEOF
    )
    while IFS= read -r line; do
        log INFO "$line"
    done <<< "$output"
}

# ---------------------------------------------------------------------------
# print_test_metrics: print test-set metrics table from predictions JSON
# Args: $1 = path to predictions metrics JSON
# ---------------------------------------------------------------------------
print_test_metrics() {
    local output
    output=$(python3 - "$1" <<'PYEOF'
import json, sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
pm = data.get("performance_metrics", {})
ranks = data.get("taxonomic_levels", {})
levels = sorted([k for k in pm.keys() if k.isdigit()], key=int)
print(f"  {'Rank':<10} | {'Accuracy':>8} | {'F1':>8} | {'Precision':>8} | {'Recall':>8} | {'ECE':>8}")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
for level in levels:
    m = pm[level]
    rank_name = ranks.get(level, {}).get("rank", f"level_{level}")
    ece = m.get("ece", 0)
    print(f"  {rank_name:<10} | {m['accuracy']:.4f}   | {m['f1_score']:.4f}   | {m['precision']:.4f}   | {m['recall']:.4f}   | {ece:.4f}")
PYEOF
    )
    while IFS= read -r line; do
        log INFO "$line"
    done <<< "$output"
}

# ---------------------------------------------------------------------------
# cleanup_checkpoints: remove all checkpoints except the specified one
# Args: $1 = run output directory, $2 = checkpoint to keep
# ---------------------------------------------------------------------------
cleanup_checkpoints() {
    local run_dir="$1"
    local keep="$2"
    local removed=0
    for f in "${run_dir}/checkpoints/"*.pt; do
        if [[ -f "$f" && "$f" != "$keep" ]]; then
            rm -f "$f"
            removed=$((removed + 1))
        fi
    done
    if [[ $removed -gt 0 ]]; then
        log INFO "Checkpoint cleanup: removed ${removed} checkpoint(s), kept $(basename "$keep")"
    fi
}

# ---------------------------------------------------------------------------
# run_single_seed: train + predict + summarize for one seed
# Args: $1 = seed, $2 = output directory for this run
# ---------------------------------------------------------------------------
run_single_seed() {
    local seed="$1"
    local run_dir="$2"

    log INFO "Starting training (seed=${seed})..."
    local t_start
    t_start=$(date +%s)

    deeptaxa train \
        --fasta-file "$TRAIN_FASTA" \
        --taxonomy-file "$TRAIN_TAX" \
        --output-dir "$run_dir" \
        --seed "$seed" \
        "${TRAIN_ARGS[@]}" \
        2>&1 | tee -a "$LOG"

    local t_end
    t_end=$(date +%s)
    local train_dur=$(( t_end - t_start ))
    log INFO "Training completed (seed=${seed}) in $(fmt_duration $train_dur)"

    # Use the final epoch checkpoint (highest epoch number)
    local ckpt
    ckpt=$(python3 - "$run_dir" <<'PYEOF'
import glob, re, sys, os
run_dir = sys.argv[1]
pts = glob.glob(os.path.join(run_dir, 'checkpoints', '*.pt'))
if not pts:
    sys.exit(0)
def epoch_num(p):
    m = re.search(r'epoch(\d+)', p)
    return int(m.group(1)) if m else 0
print(max(pts, key=epoch_num))
PYEOF
)
    if [[ -z "$ckpt" ]]; then
        log ERROR "No checkpoint found (seed=${seed}). Skipping prediction."
        return 1
    fi
    log INFO "Using checkpoint: $(basename "$ckpt")"

    # Test-set prediction
    separator
    log INFO "Starting prediction on test set (seed=${seed})..."
    local p_start
    p_start=$(date +%s)

    deeptaxa predict \
        --fasta-file "$TEST_FASTA" \
        --taxonomy-file "$TEST_TAX" \
        --checkpoint "$ckpt" \
        --batch-size "$PREDICT_BS" \
        --output-dir "${run_dir}/predictions" \
        2>&1 | tee -a "$LOG"

    local p_end
    p_end=$(date +%s)
    local pred_dur=$(( p_end - p_start ))
    log INFO "Prediction completed (seed=${seed}) in $(fmt_duration $pred_dur)"

    # Metrics summary for this seed
    separator
    log INFO "METRICS (seed=${seed})"
    separator

    # Validation metrics from the final epoch
    local val_metrics_file
    val_metrics_file=$(python3 - "$run_dir" <<'PYEOF'
import glob, re, sys, os
run_dir = sys.argv[1]
jsons = glob.glob(os.path.join(run_dir, 'metrics', '*.json'))
if not jsons:
    sys.exit(0)
def epoch_num(p):
    m = re.search(r'epoch(\d+)', p)
    return int(m.group(1)) if m else 0
print(max(jsons, key=epoch_num))
PYEOF
)
    if [[ -n "$val_metrics_file" ]]; then
        log INFO ""
        log INFO "Validation (final epoch):"
        log INFO "  Source: $(basename "$val_metrics_file")"
        log INFO ""
        print_val_metrics "$val_metrics_file"
    fi

    # Test-set metrics
    local test_metrics="${run_dir}/predictions/metrics.json"
    if [[ -f "$test_metrics" ]]; then
        log INFO ""
        log INFO "Test Set:"
        log INFO "  Source: $(basename "$test_metrics")"
        log INFO ""
        print_test_metrics "$test_metrics"
    fi

    # Checkpoint cleanup
    if [[ "$CLEANUP_CHECKPOINTS" == "true" ]]; then
        cleanup_checkpoints "$run_dir" "$ckpt"
    fi

    log INFO ""
    log INFO "Seed ${seed} complete: train=$(fmt_duration $train_dur), predict=$(fmt_duration $pred_dur)"
}

# ===========================================================================
# Main: single seed vs. multi-seed
# ===========================================================================

if [[ ${#SEEDS[@]} -eq 1 ]]; then
    # ----- Single seed: run directly in OUTPUT_DIR -----
    run_single_seed "${SEEDS[0]}" "$OUTPUT_DIR"
else
    # ----- Multi-seed: run each in a subdirectory, then summarize -----
    log INFO "Running ${#SEEDS[@]} seeds: ${SEEDS[*]}"
    separator

    for seed in "${SEEDS[@]}"; do
        separator
        log INFO "===== SEED ${seed} ====="
        separator
        run_single_seed "$seed" "${OUTPUT_DIR}/seed_${seed}"
        echo "" | tee -a "$LOG"
    done

    # Cross-seed summary: compute mean and std for each rank
    separator
    log INFO "CROSS-SEED SUMMARY (${#SEEDS[@]} seeds: ${SEEDS[*]})"
    separator

    local_summary=$(python3 - "$OUTPUT_DIR" "${SEEDS[@]}" <<'PYEOF'
import json, sys, os, math

output_dir = sys.argv[1]
seeds = sys.argv[2:]

all_metrics = []
found_seeds = []
for seed in seeds:
    path = os.path.join(output_dir, f"seed_{seed}", "predictions", "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            all_metrics.append(json.load(f))
        found_seeds.append(seed)

if not all_metrics:
    print("  No prediction metrics found for any seed.")
    sys.exit(0)

if len(found_seeds) < len(seeds):
    missing = set(seeds) - set(found_seeds)
    print(f"  WARNING: Missing results for seeds: {', '.join(sorted(missing))}")

if len(found_seeds) < 5:
    print(f"  Note: {len(found_seeds)} seeds -- standard deviations should be interpreted with caution.")
    print()

# Collect rank names from first result
ranks = all_metrics[0].get("taxonomic_levels", {})
levels = sorted([k for k in all_metrics[0].get("performance_metrics", {}).keys() if k.isdigit()], key=int)

def mean_std(values):
    n = len(values)
    if n == 0:
        return 0, 0
    mu = sum(values) / n
    if n == 1:
        return mu, 0
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return mu, math.sqrt(var)

print(f"  {'Rank':<10} | {'Accuracy':>14} | {'F1':>14} | {'Precision':>14} | {'Recall':>14} | {'ECE':>14}")
print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")

summary_data = {}
for level in levels:
    rank_name = ranks.get(level, {}).get("rank", f"level_{level}")
    accs = [m["performance_metrics"][level]["accuracy"] for m in all_metrics if level in m.get("performance_metrics", {})]
    f1s = [m["performance_metrics"][level]["f1_score"] for m in all_metrics if level in m.get("performance_metrics", {})]
    precs = [m["performance_metrics"][level]["precision"] for m in all_metrics if level in m.get("performance_metrics", {})]
    recs = [m["performance_metrics"][level]["recall"] for m in all_metrics if level in m.get("performance_metrics", {})]
    eces = [m["performance_metrics"][level].get("ece", 0) for m in all_metrics if level in m.get("performance_metrics", {})]

    a_m, a_s = mean_std(accs)
    f_m, f_s = mean_std(f1s)
    p_m, p_s = mean_std(precs)
    r_m, r_s = mean_std(recs)
    e_m, e_s = mean_std(eces)

    print(f"  {rank_name:<10} | {a_m:.4f}+-{a_s:.4f} | {f_m:.4f}+-{f_s:.4f} | {p_m:.4f}+-{p_s:.4f} | {r_m:.4f}+-{r_s:.4f} | {e_m:.4f}+-{e_s:.4f}")

    summary_data[level] = {
        "rank": rank_name,
        "accuracy": {"mean": a_m, "std": a_s},
        "f1_score": {"mean": f_m, "std": f_s},
        "precision": {"mean": p_m, "std": p_s},
        "recall": {"mean": r_m, "std": r_s},
        "ece": {"mean": e_m, "std": e_s},
    }

# Save summary JSON
summary_path = os.path.join(output_dir, "summary.json")
summary = {
    "seeds": found_seeds,
    "num_seeds": len(found_seeds),
    "seeds_requested": seeds,
    "metrics_per_rank": summary_data,
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)
print()
print(f"  Summary saved to: {summary_path}")
PYEOF
    )
    while IFS= read -r line; do
        log INFO "$line"
    done <<< "$local_summary"
fi

# ---------------------------------------------------------------------------
# Wrap-up
# ---------------------------------------------------------------------------
OVERALL_END=$(date +%s)
OVERALL_DUR=$(( OVERALL_END - OVERALL_START ))

log INFO ""
separator
log INFO "EXPERIMENT COMPLETE: ${EXPERIMENT_NAME}"
log INFO "  Seeds:           ${SEEDS[*]}"
log INFO "  Total time:      $(fmt_duration $OVERALL_DUR)"
log INFO "  Output:          ${OUTPUT_DIR}"
log INFO "  Log:             ${LOG}"
log INFO "  Rerun:           ${RERUN_FILE}"
log INFO "Finished: $(date '+%Y-%m-%dT%H:%M:%S%z')"
separator

COMPLETED=true
