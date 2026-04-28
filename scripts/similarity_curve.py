#!/usr/bin/env python3
"""Plot classification accuracy as a function of train-test sequence similarity.

Reads per-sequence similarity scores (from sequence_similarity.py) and
per-bucket prediction outputs, then bins sequences by 1% identity intervals
and computes per-bin accuracy at each taxonomic rank.

Produces a TSV of binned results and a multi-panel PDF figure.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]


def load_similarity(tsv_path):
    """Load per-sequence similarity scores.

    Returns dict: seq_id -> percent_identity
    """
    similarities = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            similarities[row["test_seq_id"]] = float(row["percent_identity"])
    return similarities


def load_predictions(pred_dir):
    """Load predictions from all bucket subdirectories.

    Returns dict: seq_id -> {rank: (predicted, true)}
    """
    results = {}
    for bucket in os.listdir(pred_dir):
        bucket_dir = os.path.join(pred_dir, bucket)
        if not os.path.isdir(bucket_dir):
            continue
        pred_file = None
        for f in os.listdir(bucket_dir):
            if f.endswith("_predictions.json"):
                pred_file = os.path.join(bucket_dir, f)
                break
        if pred_file is None:
            continue

        with open(pred_file) as fh:
            data = json.load(fh)

        seq_ids = data["sequence_ids"]
        predictions = data["predictions"]
        true_labels = data["true_labels"]

        for i, seq_id in enumerate(seq_ids):
            entry = {}
            for rank in RANKS:
                pred_label = predictions[i][rank]["label"]
                true_label = true_labels[i][rank]
                entry[rank] = (pred_label, true_label)
            results[seq_id] = entry

    return results


def assign_bin(pident, adaptive_threshold=90.0, fine_width=1.0, coarse_width=5.0):
    """Assign a bin key (low, high) for a given percent identity.

    Uses fine bins (1%) above the threshold and coarse bins (5%) below.
    """
    if pident >= adaptive_threshold:
        low = int(pident / fine_width) * fine_width
        return (low, low + fine_width)
    else:
        low = int(pident / coarse_width) * coarse_width
        return (low, low + coarse_width)


def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bin_results(similarities, predictions, bin_width=1.0, min_bin_size=20,
                adaptive=False, adaptive_threshold=90.0, coarse_width=5.0):
    """Bin sequences by identity and compute per-bin accuracy.

    If adaptive=True, uses coarse bins below adaptive_threshold and
    fine (bin_width) bins above it.

    Returns list of dicts with bin info and per-rank accuracy.
    """
    bins = defaultdict(list)
    for seq_id, pident in similarities.items():
        if seq_id not in predictions:
            continue
        if adaptive:
            key = assign_bin(pident, adaptive_threshold, bin_width, coarse_width)
        else:
            idx = int(pident / bin_width)
            key = (idx * bin_width, idx * bin_width + bin_width)
        bins[key].append((pident, predictions[seq_id]))

    results = []
    for (bin_low, bin_high) in sorted(bins.keys()):
        entries = bins[(bin_low, bin_high)]
        n = len(entries)
        if n < min_bin_size:
            continue

        bin_center = sum(e[0] for e in entries) / n

        row = {
            "bin_low": bin_low,
            "bin_high": bin_high,
            "bin_center": round(bin_center, 2),
            "count": n,
        }
        for rank in RANKS:
            correct = sum(1 for _, pred in entries if pred[rank][0] == pred[rank][1])
            acc = correct / n
            row[f"{rank}_accuracy"] = round(acc, 4)
            # Wilson score 95% CI
            ci_lo, ci_hi = wilson_ci(correct, n)
            row[f"{rank}_ci_lo"] = round(ci_lo, 4)
            row[f"{rank}_ci_hi"] = round(ci_hi, 4)

        results.append(row)

    return results


def write_tsv(binned, output_path):
    """Write binned results to TSV."""
    fieldnames = ["bin_low", "bin_high", "bin_center", "count"]
    for r in RANKS:
        fieldnames += [f"{r}_accuracy", f"{r}_ci_lo", f"{r}_ci_hi"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(binned)


RANK_COLORS = {
    "species": "#d62728",
    "genus": "#ff7f0e",
    "family": "#2ca02c",
    "order": "#1f77b4",
    "class": "#9467bd",
    "phylum": "#8c564b",
    "domain": "#7f7f7f",
}


def plot_curve(binned, output_path, plot_ranks=None, x_min=None):
    """Plot accuracy vs similarity curve with Wilson CIs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot", file=sys.stderr)
        return

    if plot_ranks is None:
        plot_ranks = ["species", "genus", "family", "order", "class", "phylum", "domain"]

    centers = [r["bin_center"] for r in binned]
    counts = [r["count"] for r in binned]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: accuracy curves with CIs
    ax = axes[0]
    for rank in plot_ranks:
        color = RANK_COLORS.get(rank, "#333333")
        acc = [r[f"{rank}_accuracy"] for r in binned]
        ci_lo = [r[f"{rank}_ci_lo"] for r in binned]
        ci_hi = [r[f"{rank}_ci_hi"] for r in binned]
        ax.plot(centers, acc, "o-", color=color, label=rank, markersize=3, linewidth=1.5)
        ax.fill_between(centers, ci_lo, ci_hi, color=color, alpha=0.12)

    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Classification accuracy vs. train-test sequence similarity")
    ax.grid(True, alpha=0.3)

    # Bottom panel: histogram of sequence counts
    ax2 = axes[1]
    bin_widths = [r["bin_high"] - r["bin_low"] for r in binned]
    ax2.bar(centers, counts, width=bin_widths, color="#1f77b4", alpha=0.6, edgecolor="none")
    ax2.set_xlabel("Nearest-neighbor sequence identity (%)")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # Shared x-axis range
    if x_min is None:
        all_lows = [r["bin_low"] for r in binned]
        x_min = max(50, min(all_lows) - 2)
    for ax_i in axes:
        ax_i.set_xlim(x_min, 101)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot classification accuracy vs. train-test sequence similarity"
    )
    parser.add_argument(
        "--similarity-tsv",
        required=True,
        help="Per-sequence similarity TSV from sequence_similarity.py",
    )
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory containing per-bucket prediction subdirectories",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--bin-width",
        type=float,
        default=1.0,
        help="Bin width in percent identity (default: 1.0)",
    )
    parser.add_argument(
        "--min-bin-size",
        type=int,
        default=20,
        help="Minimum sequences per bin to include (default: 20)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive binning: coarse bins below threshold, fine bins above",
    )
    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=90.0,
        help="Identity threshold for switching from coarse to fine bins (default: 90.0)",
    )
    parser.add_argument(
        "--coarse-width",
        type=float,
        default=5.0,
        help="Bin width below adaptive threshold (default: 5.0)",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        default=None,
        choices=RANKS,
        help="Taxonomic ranks to plot (default: all)",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Minimum x-axis value in percent identity (default: auto)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading similarity scores...")
    similarities = load_similarity(args.similarity_tsv)
    print(f"  {len(similarities)} sequences")

    print("Loading predictions...")
    predictions = load_predictions(args.predictions_dir)
    print(f"  {len(predictions)} sequences with predictions")

    matched = sum(1 for s in similarities if s in predictions)
    print(f"  {matched} sequences matched")

    if args.adaptive:
        print(f"Adaptive binning: {args.coarse_width}% below {args.adaptive_threshold}%, "
              f"{args.bin_width}% above (min {args.min_bin_size} seqs)...")
    else:
        print(f"Binning with {args.bin_width}% bins (min {args.min_bin_size} seqs)...")
    binned = bin_results(similarities, predictions, args.bin_width, args.min_bin_size,
                         args.adaptive, args.adaptive_threshold, args.coarse_width)
    print(f"  {len(binned)} bins")

    tsv_path = os.path.join(args.output_dir, "similarity_curve.tsv")
    write_tsv(binned, tsv_path)
    print(f"Wrote {tsv_path}")

    pdf_path = os.path.join(args.output_dir, "similarity_curve.pdf")
    plot_curve(binned, pdf_path, plot_ranks=args.ranks, x_min=args.x_min)


if __name__ == "__main__":
    main()
