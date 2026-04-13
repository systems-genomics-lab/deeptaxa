#!/usr/bin/env python3
"""Compute nearest-neighbor sequence similarity between test and training sets.

Uses vsearch --usearch_global to find the closest training sequence for each
test sequence. Outputs a TSV with per-sequence identity and similarity bucket,
plus per-bucket FASTA files for stratified evaluation.

Addresses Reviewer 1, Comment 4:
  "When splitting the training and test sets, the authors did not consider the
   similarity between the sequences in the test set and those in the training set."
"""

import argparse
import csv
import gzip
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


SIMILARITY_BUCKETS = [
    ("high", 97.0, 100.0),
    ("medium", 90.0, 97.0),
    ("low", 0.0, 90.0),
]


def decompress_if_needed(fasta_path, tmpdir):
    """Decompress gzipped FASTA to a temp file if needed."""
    if fasta_path.endswith(".gz"):
        decompressed = os.path.join(tmpdir, os.path.basename(fasta_path).replace(".gz", ""))
        with gzip.open(fasta_path, "rb") as f_in, open(decompressed, "wb") as f_out:
            for chunk in iter(lambda: f_in.read(1024 * 1024), b""):
                f_out.write(chunk)
        return decompressed
    return fasta_path


def run_vsearch(train_fasta, test_fasta, output_tsv, threads=4, identity_threshold=0.5):
    """Run vsearch usearch_global to find nearest neighbors."""
    cmd = [
        "vsearch",
        "--usearch_global", test_fasta,
        "--db", train_fasta,
        "--blast6out", output_tsv,
        "--id", str(identity_threshold),
        "--maxaccepts", "1",
        "--maxrejects", "32",
        "--threads", str(threads),
        "--top_hits_only",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"vsearch stderr:\n{result.stderr}", file=sys.stderr)
        result.check_returncode()
    return result


def parse_blast6(blast6_path):
    """Parse BLAST6 format output from vsearch.

    Returns dict: test_seq_id -> (train_seq_id, percent_identity)
    """
    hits = {}
    with open(blast6_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            query_id = row[0]
            target_id = row[1]
            pident = float(row[2])
            if query_id not in hits or pident > hits[query_id][1]:
                hits[query_id] = (target_id, pident)
    return hits


def get_bucket(pident):
    """Assign a similarity bucket based on percent identity."""
    for name, low, high in SIMILARITY_BUCKETS:
        if low <= pident < high or (name == "high" and pident == 100.0):
            return name
    return "low"


def read_fasta_ids(fasta_path):
    """Read sequence IDs from a FASTA file (plain or gzipped)."""
    ids = []
    opener = gzip.open if fasta_path.endswith(".gz") else open
    with opener(fasta_path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                seq_id = line[1:].strip().split()[0]
                ids.append(seq_id)
    return ids


def split_fasta_by_bucket(fasta_path, bucket_assignments, output_dir):
    """Split a FASTA file into per-bucket files."""
    handles = {}
    for bucket_name, _, _ in SIMILARITY_BUCKETS:
        bucket_path = os.path.join(output_dir, f"test_{bucket_name}.fasta")
        handles[bucket_name] = open(bucket_path, "w")

    # Handle no-hit sequences
    handles["no_hit"] = open(os.path.join(output_dir, "test_no_hit.fasta"), "w")

    opener = gzip.open if fasta_path.endswith(".gz") else open
    current_bucket = None
    with opener(fasta_path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                seq_id = line[1:].strip().split()[0]
                current_bucket = bucket_assignments.get(seq_id, "no_hit")
            if current_bucket in handles:
                handles[current_bucket].write(line)

    for h in handles.values():
        h.close()

    # Remove empty files
    for bucket_name in list(handles.keys()):
        path = os.path.join(output_dir, f"test_{bucket_name}.fasta")
        if os.path.getsize(path) == 0:
            os.remove(path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute train-test sequence similarity using vsearch"
    )
    parser.add_argument("--train-fasta", required=True, help="Training FASTA (plain or .gz)")
    parser.add_argument("--test-fasta", required=True, help="Test FASTA (plain or .gz)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads (default: 4)")
    parser.add_argument(
        "--min-identity",
        type=float,
        default=0.5,
        help="Minimum identity threshold for vsearch (default: 0.5)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Decompress if needed (vsearch needs plain FASTA)
        print("Preparing input files...")
        train_fasta = decompress_if_needed(args.train_fasta, tmpdir)
        test_fasta = decompress_if_needed(args.test_fasta, tmpdir)

        # Run vsearch
        blast6_out = os.path.join(tmpdir, "vsearch_hits.tsv")
        print("Running vsearch nearest-neighbor search...")
        run_vsearch(train_fasta, test_fasta, blast6_out, args.threads, args.min_identity)

        # Parse results
        print("Parsing results...")
        hits = parse_blast6(blast6_out)
        all_test_ids = read_fasta_ids(args.test_fasta)

        # Assign buckets
        bucket_assignments = {}
        results = []
        for seq_id in all_test_ids:
            if seq_id in hits:
                train_id, pident = hits[seq_id]
                bucket = get_bucket(pident)
            else:
                train_id, pident, bucket = "no_hit", 0.0, "no_hit"
            bucket_assignments[seq_id] = bucket
            results.append((seq_id, train_id, pident, bucket))

        # Write per-sequence results
        output_tsv = os.path.join(args.output_dir, "similarity_results.tsv")
        with open(output_tsv, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["test_seq_id", "nearest_train_seq_id", "percent_identity", "bucket"])
            writer.writerows(results)
        print(f"Wrote per-sequence results to {output_tsv}")

        # Summary statistics
        bucket_counts = Counter(r[3] for r in results)
        print("\nSimilarity Distribution:")
        print(f"{'Bucket':<15} {'Count':>8} {'Percent':>8}")
        print("-" * 33)
        total = len(results)
        for bucket_name, low, high in SIMILARITY_BUCKETS:
            count = bucket_counts.get(bucket_name, 0)
            pct = 100.0 * count / total if total > 0 else 0
            print(f"{bucket_name:<15} {count:>8} {pct:>7.1f}%")
        no_hit = bucket_counts.get("no_hit", 0)
        if no_hit > 0:
            print(f"{'no_hit':<15} {no_hit:>8} {100.0 * no_hit / total:>7.1f}%")
        print(f"{'total':<15} {total:>8}")

        # Write summary
        summary_path = os.path.join(args.output_dir, "similarity_summary.tsv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["bucket", "identity_range", "count", "percent"])
            for bucket_name, low, high in SIMILARITY_BUCKETS:
                count = bucket_counts.get(bucket_name, 0)
                pct = 100.0 * count / total if total > 0 else 0
                writer.writerow([bucket_name, f"{low}-{high}%", count, f"{pct:.1f}"])
            if no_hit > 0:
                writer.writerow(["no_hit", "<50%", no_hit, f"{100.0 * no_hit / total:.1f}"])

        # Split FASTA by bucket
        print("\nSplitting test FASTA by similarity bucket...")
        split_fasta_by_bucket(args.test_fasta, bucket_assignments, args.output_dir)
        print("Done.")


if __name__ == "__main__":
    main()
