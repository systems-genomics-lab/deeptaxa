#!/usr/bin/env python3
"""Simulate amplicon sequencing by extracting variable regions from full-length 16S sequences.

Extracts regions between primer binding sites using degenerate primer matching.
Optionally introduces substitution errors to simulate sequencing noise.

Supported regions:
    V3-V4: 341F (CCTACGGGNGGCWGCAG) / 805R (GACTACHVGGGTATCTAATCC)
    V4:    515F (GTGYCAGCMGCCGCGGTAA) / 806R (GGACTACNVGGGTWTCTAAT)

Usage:
    python scripts/simulate_amplicons.py \
        --input-fasta data/gg_testing.fna.gz \
        --output-fasta data/gg_testing_v3v4.fasta \
        --region V3-V4 \
        --error-rate 0.005 \
        --seed 42
"""

import argparse
import gzip
import random
import re
import sys
from pathlib import Path

# IUPAC degenerate base codes
IUPAC = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
    'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
    'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
    'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]',
}

# Primer sequences for supported regions
PRIMERS = {
    'V3-V4': {
        'forward': 'CCTACGGGNGGCWGCAG',    # 341F
        'reverse': 'GACTACHVGGGTATCTAATCC',  # 805R
    },
    'V4': {
        'forward': 'GTGYCAGCMGCCGCGGTAA',   # 515F
        'reverse': 'GGACTACNVGGGTWTCTAAT',   # 806R
    },
}


def degenerate_to_regex(primer):
    """Convert a degenerate primer sequence to a regex pattern."""
    return ''.join(IUPAC.get(base, base) for base in primer.upper())


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans('ACGTRYSWKMBDHVNacgtryswkmbdhvn',
                                'TGCAYRSWMKVHDBNtgcayrswmkvhdbn')
    return seq.translate(complement)[::-1]


def find_primer(seq, primer, max_mismatches=2):
    """Find primer binding site in sequence, allowing degenerate bases.

    First tries exact regex match. Falls back to approximate matching
    with up to max_mismatches mismatches.

    Returns (start, end) tuple or None if not found.
    """
    pattern = degenerate_to_regex(primer)
    match = re.search(pattern, seq, re.IGNORECASE)
    if match:
        return match.start(), match.end()

    # Approximate matching with mismatches
    primer_len = len(primer)
    expanded = degenerate_to_regex(primer)
    best_pos = None
    best_mm = max_mismatches + 1

    for i in range(len(seq) - primer_len + 1):
        subseq = seq[i:i + primer_len]
        mismatches = 0
        for j, (s, p) in enumerate(zip(subseq.upper(), primer.upper())):
            pat = IUPAC.get(p, p)
            if len(pat) == 1:
                if s != pat:
                    mismatches += 1
            else:
                if s not in pat.strip('[]'):
                    mismatches += 1
            if mismatches > best_mm:
                break
        if mismatches < best_mm:
            best_mm = mismatches
            best_pos = (i, i + primer_len)

    if best_mm <= max_mismatches:
        return best_pos
    return None


def add_errors(seq, error_rate, rng):
    """Introduce random substitution errors into a sequence."""
    if error_rate <= 0:
        return seq
    bases = list(seq)
    alternatives = {'A': 'CGT', 'C': 'AGT', 'G': 'ACT', 'T': 'ACG'}
    for i in range(len(bases)):
        if rng.random() < error_rate:
            base = bases[i].upper()
            if base in alternatives:
                bases[i] = rng.choice(alternatives[base])
    return ''.join(bases)


def read_fasta(filepath):
    """Read sequences from a FASTA file (gzip supported)."""
    opener = gzip.open if str(filepath).endswith('.gz') else open
    sequences = []
    current_id = None
    current_seq = []

    with opener(filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            sequences.append((current_id, ''.join(current_seq)))

    return sequences


def extract_amplicons(sequences, region, error_rate, max_mismatches, rng,
                      min_length=None, max_length=None):
    """Extract amplicon regions from sequences.

    Returns list of (seq_id, amplicon_seq) tuples and extraction statistics.
    """
    primers = PRIMERS[region]
    fwd_primer = primers['forward']
    rev_primer_rc = reverse_complement(primers['reverse'])

    amplicons = []
    stats = {'total': 0, 'extracted': 0, 'no_fwd': 0, 'no_rev': 0,
             'too_short': 0, 'out_of_range': 0}

    for seq_id, seq in sequences:
        stats['total'] += 1

        # Find forward primer
        fwd_hit = find_primer(seq, fwd_primer, max_mismatches)
        if fwd_hit is None:
            stats['no_fwd'] += 1
            continue

        # Find reverse primer (as reverse complement in the sequence)
        rev_hit = find_primer(seq, rev_primer_rc, max_mismatches)
        if rev_hit is None:
            stats['no_rev'] += 1
            continue

        # Extract region between primers (excluding primer sequences)
        start = fwd_hit[1]  # after forward primer
        end = rev_hit[0]    # before reverse primer

        if end <= start or (end - start) < 50:
            stats['too_short'] += 1
            continue

        amplicon_len = end - start
        if (min_length and amplicon_len < min_length) or \
           (max_length and amplicon_len > max_length):
            stats['out_of_range'] += 1
            continue

        amplicon = seq[start:end]

        # Add sequencing errors
        if error_rate > 0:
            amplicon = add_errors(amplicon, error_rate, rng)

        amplicons.append((seq_id, amplicon))
        stats['extracted'] += 1

    return amplicons, stats


def main():
    parser = argparse.ArgumentParser(
        description='Simulate amplicon sequencing from full-length 16S sequences')
    parser.add_argument('--input-fasta', required=True,
                        help='Input FASTA file (gzip supported)')
    parser.add_argument('--output-fasta', required=True,
                        help='Output FASTA file with extracted amplicons')
    parser.add_argument('--region', default='V3-V4', choices=list(PRIMERS.keys()),
                        help='Variable region to extract (default: V3-V4)')
    parser.add_argument('--error-rate', type=float, default=0.0,
                        help='Per-base substitution error rate (default: 0.0)')
    parser.add_argument('--max-mismatches', type=int, default=2,
                        help='Max primer mismatches allowed (default: 2)')
    parser.add_argument('--min-length', type=int, default=None,
                        help='Minimum amplicon length in bp (discard shorter)')
    parser.add_argument('--max-length', type=int, default=None,
                        help='Maximum amplicon length in bp (discard longer)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for error simulation (default: 42)')
    args = parser.parse_args()

    if args.region not in PRIMERS:
        print(f"ERROR: Unknown region '{args.region}'. Choose from: {list(PRIMERS.keys())}")
        sys.exit(1)

    rng = random.Random(args.seed)

    print(f"Reading sequences from {args.input_fasta}...")
    sequences = read_fasta(args.input_fasta)
    print(f"  Loaded {len(sequences)} sequences")

    print(f"Extracting {args.region} amplicons...")
    primers = PRIMERS[args.region]
    print(f"  Forward primer: {primers['forward']}")
    print(f"  Reverse primer: {primers['reverse']}")
    if args.error_rate > 0:
        print(f"  Error rate: {args.error_rate}")

    if args.min_length or args.max_length:
        print(f"  Length filter: {args.min_length or 50}-{args.max_length or '∞'} bp")

    amplicons, stats = extract_amplicons(
        sequences, args.region, args.error_rate, args.max_mismatches, rng,
        min_length=args.min_length, max_length=args.max_length)

    # Write output
    output_path = Path(args.output_fasta)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for seq_id, amplicon in amplicons:
            f.write(f'>{seq_id}\n{amplicon}\n')

    # Report statistics
    print(f"\nExtraction Summary:")
    print(f"  Total sequences:     {stats['total']:,}")
    print(f"  Amplicons extracted: {stats['extracted']:,} ({100*stats['extracted']/stats['total']:.1f}%)")
    print(f"  No forward primer:   {stats['no_fwd']:,}")
    print(f"  No reverse primer:   {stats['no_rev']:,}")
    print(f"  Too short (<50bp):   {stats['too_short']:,}")
    print(f"  Out of length range: {stats['out_of_range']:,}")

    if amplicons:
        lengths = [len(s) for _, s in amplicons]
        print(f"\nAmplicon Length Distribution:")
        print(f"  Min:    {min(lengths)} bp")
        print(f"  Max:    {max(lengths)} bp")
        print(f"  Mean:   {sum(lengths)/len(lengths):.0f} bp")
        print(f"  Median: {sorted(lengths)[len(lengths)//2]} bp")

    print(f"\nWrote {len(amplicons)} amplicons to {args.output_fasta}")


if __name__ == '__main__':
    main()
