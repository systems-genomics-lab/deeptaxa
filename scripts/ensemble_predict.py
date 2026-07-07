#!/usr/bin/env python3
"""Evaluate a soft-voting ensemble of the five DeepTaxa seed checkpoints.

The script loads the five seed checkpoints for one region, runs each over the
held-out test set, averages the per-rank softmax probabilities across the five
models, and takes the argmax at each rank. It reports per-rank accuracy and
weighted F1 for the ensemble next to the single seed-42 model, so the ensemble
gain is visible directly. Accuracy and F1 follow the same definitions as
`deeptaxa predict` (label agreement for accuracy, weighted F1 over label ids).

The five seed checkpoints for a region must be named
deeptaxa-<region>-v2-seed<N>.pt inside --release-dir. Amplicon regions expect
region-matched test sequences (for example V4 amplicons for --region v4).

Example:
    python scripts/ensemble_predict.py \
        --region full-length \
        --fasta-file ../deeptaxa-data/greengenes/gg_2024_09_testing.fna.gz \
        --taxonomy-file ../deeptaxa-data/greengenes/gg_2024_09_testing.tsv.gz \
        --release-dir ../deeptaxa-data/models
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.metrics import f1_score
from tqdm import tqdm

from deeptaxa.dataset import TaxonomyDataset, custom_collate_fn
from deeptaxa.models import HybridCNNBERTClassifier
from deeptaxa.predict import labels_agree

SEEDS = [42, 123, 456, 789, 1011]


def build_hybrid(checkpoint, device):
    """Rebuild a HybridCNNBERTClassifier from a checkpoint, mirroring predict.py."""
    model_config = checkpoint.get("model_config", {})
    cnn = model_config.get("cnn", {})
    bert = model_config.get("bert", {})
    bert = bert.__dict__ if hasattr(bert, "__dict__") else bert
    max_length = bert.get("max_position_embeddings", checkpoint.get("max_length"))
    model = HybridCNNBERTClassifier(
        tokenizer_name=checkpoint["tokenizer_name"],
        num_labels_per_level=checkpoint["num_labels_per_level"],
        hidden_dropout_prob=checkpoint.get("hidden_dropout_prob", 0.2),
        num_filters=cnn["num_filters"],
        kernel_sizes=cnn["kernel_sizes"],
        num_conv_layers=cnn["num_conv_layers"],
        max_length=max_length,
        embed_dim=cnn["embed_dim"],
        hidden_size=bert["hidden_size"],
        num_hidden_layers=bert["num_hidden_layers"],
        num_attention_heads=bert["num_attention_heads"],
        intermediate_size=bert["intermediate_size"],
        output_attentions=bert.get("output_attentions", False),
        mask_padding=cnn.get("mask_padding", False),
        tokenizer_revision=checkpoint.get("tokenizer_revision", None),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True, help="Region label, e.g. full-length, v3v4, v4")
    ap.add_argument("--fasta-file", required=True)
    ap.add_argument("--taxonomy-file", required=True)
    ap.add_argument("--release-dir", required=True, help="Directory holding the seed checkpoints")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = [
        os.path.join(args.release_dir, f"deeptaxa-{args.region}-v2-seed{s}.pt")
        for s in SEEDS
    ]
    for p in ckpt_paths:
        if not os.path.isfile(p):
            raise SystemExit(f"missing checkpoint: {p}")

    print(f"Loading {len(ckpt_paths)} seed checkpoints for region '{args.region}' ...", flush=True)
    checkpoints = [torch.load(p, map_location=device, weights_only=False) for p in ckpt_paths]
    ref = checkpoints[0]
    taxonomic_ranks = ref["taxonomic_ranks"]
    level_label2id = ref["level_label2id"]
    id2label = {lvl: {v: k for k, v in level_label2id[lvl].items()} for lvl in level_label2id}
    max_length = ref.get("model_config", {}).get("bert", {})
    max_length = (max_length.__dict__ if hasattr(max_length, "__dict__") else max_length).get(
        "max_position_embeddings", ref.get("max_length")
    )
    temps = ref.get("level_temperatures") or [1.0] * len(taxonomic_ranks)

    models = [build_hybrid(c, device) for c in checkpoints]

    dataset = TaxonomyDataset(
        fasta_file=args.fasta_file,
        taxonomy_file=args.taxonomy_file,
        tokenizer_name=ref["tokenizer_name"],
        max_length=max_length,
        encoding=ref.get("encoding", "dnabert"),
        use_raw_labels_for_true=True,
        tokenizer_revision=ref.get("tokenizer_revision", None),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn,
    )

    # Accumulators: "ens" = five-seed soft-vote, "one" = seed 42 alone.
    correct = {m: {r: 0 for r in taxonomic_ranks} for m in ("ens", "one")}
    y_true = {m: {r: [] for r in taxonomic_ranks} for m in ("ens", "one")}
    y_pred = {m: {r: [] for r in taxonomic_ranks} for m in ("ens", "one")}
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"ensemble {args.region}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Per-model logits, then per-rank averaged probabilities.
            per_model_logits = []
            for model in models:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                per_model_logits.append(out[0] if isinstance(out, tuple) else out)

            bsz = len(batch["input_ids"])
            total += bsz
            for lvl_str in per_model_logits[0].keys():
                lvl = int(lvl_str)
                rank = taxonomic_ranks[lvl]
                t = temps[lvl]
                probs_one = softmax(per_model_logits[0][lvl_str] / t, dim=1)
                probs_ens = sum(
                    softmax(pl[lvl_str] / t, dim=1) for pl in per_model_logits
                ) / len(models)
                pred_one = torch.argmax(probs_one, dim=1).tolist()
                pred_ens = torch.argmax(probs_ens, dim=1).tolist()
                for i in range(bsz):
                    true_label = batch["raw_labels"][i][rank]
                    true_id = level_label2id[lvl].get(true_label, -1)
                    for tag, pid in (("one", pred_one[i]), ("ens", pred_ens[i])):
                        pred_label = id2label[lvl][pid]
                        if labels_agree(pred_label, true_label, rank):
                            correct[tag][rank] += 1
                        y_true[tag][rank].append(true_id)
                        y_pred[tag][rank].append(pid)

    # Report
    print(f"\nRegion: {args.region}   test sequences: {total}\n")
    head = f"{'Rank':<9} | {'seed42 acc':>10} {'seed42 F1':>10} | {'ens acc':>9} {'ens F1':>9} | {'dF1':>7}"
    print(head)
    print("-" * len(head))
    for rank in taxonomic_ranks:
        a1 = correct["one"][rank] / total
        aE = correct["ens"][rank] / total
        f1_one = f1_score(y_true["one"][rank], y_pred["one"][rank], average="weighted")
        f1_ens = f1_score(y_true["ens"][rank], y_pred["ens"][rank], average="weighted")
        print(f"{rank:<9} | {a1:>10.4f} {f1_one:>10.4f} | {aE:>9.4f} {f1_ens:>9.4f} | {f1_ens - f1_one:>+7.4f}")


if __name__ == "__main__":
    main()
