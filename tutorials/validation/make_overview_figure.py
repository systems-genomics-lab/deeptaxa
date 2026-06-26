#!/usr/bin/env python3
"""Generate the species figure shown on the validation overview page.

Nothing on the figure is hardcoded except the experimental definitions: the eight
expected ZymoBIOMICS species and the list of close relatives that 16S cannot separate
from them (the accepted siblings). Every plotted value is computed from the input
tables in ./data/, which hold, for each dataset, the read counts, the DeepTaxa
predictions, and the DADA2 + SILVA reference taxonomy. The DeepTaxa predictions are the
output of running the notebooks; the scoring here matches the notebooks' definitions.

The figure reports the percentage of reads correctly identified to species: for SILVA,
exact matches only; for DeepTaxa, exact matches plus close relatives 16S cannot
separate. The genus rank is omitted because both methods assign the correct genus to
essentially every read.

Run from this directory:  python make_overview_figure.py
Writes: overview-species.png
"""
import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# --- experimental definitions (ground truth and sibling policy) ---
# the eight ZymoBIOMICS species, as genus -> expected species epithet
EXPECTED = {
    "pseudomonas": "aeruginosa", "escherichia": "coli", "salmonella": "enterica",
    "lactobacillus": "fermentum", "enterococcus": "faecalis", "staphylococcus": "aureus",
    "listeria": "monocytogenes", "bacillus": "subtilis",
}
# Close relatives that 16S genuinely cannot separate from the expected species (credited).
# Salmonella bongori and Staphylococcus argenteus are deliberately excluded: on the ASVs
# that carry enough signal, the reference resolves the true species (enterica, aureus),
# so DeepTaxa's calls there are counted as misses, not indistinguishable siblings.
SIBLINGS = {
    "pseudomonas": {"aeruginosa", "paraeruginosa"},
    "escherichia": {"coli", "flexneri", "sonnei", "boydii", "dysenteriae", "albertii"},
    "salmonella": {"enterica"},
    "lactobacillus": {"fermentum"},
    "enterococcus": {"faecalis"},
    "staphylococcus": {"aureus"},
    "listeria": {"monocytogenes"},
    "bacillus": {"subtilis", "spizizenii", "inaquosorum", "halotolerans"},
}
GENUS_SYNONYMS = {"limosilactobacillus": "lactobacillus", "shigella": "escherichia"}


def base_genus(name):
    if not isinstance(name, str):
        return None
    g = re.sub(r"[ _-].*", "", name).lower()
    return GENUS_SYNONYMS.get(g, g) or None


def parse_species(genus_field, label):
    """Return (genus, epithet) from a species label, or (None, None)."""
    if not isinstance(label, str) or label.strip() in ("", "NA", "nan"):
        return (None, None)
    if " " in label:                       # a binomial, e.g. DeepTaxa output
        parts = label.split()
        g, ep = base_genus(parts[0]), parts[-1]
    else:                                  # an epithet only, e.g. SILVA Species column
        g, ep = base_genus(genus_field), label
    if g is None:
        return (None, None)
    return (g, re.sub(r"_[A-Za-z0-9]+", "", ep).lower())


def tier(genus, epithet):
    if genus is None or epithet is None:
        return "abstain"
    if genus not in EXPECTED:
        return "wrong"
    if epithet == EXPECTED[genus]:
        return "exact"
    if epithet in SIBLINGS[genus]:
        return "sibling"
    return "wrong"


def percent(counts, tiers, keep):
    selected = [a for a in counts.index if tiers[a] in keep]
    return round(counts[selected].sum() / counts.sum() * 100)


def score(prefix):
    counts = pd.read_csv(f"{DATA_DIR}/{prefix}_counts.tsv", sep="\t", index_col=0)["count"]
    dt = pd.read_csv(f"{DATA_DIR}/{prefix}_deeptaxa.tsv", sep="\t").set_index("sequence_id").reindex(counts.index)
    silva = pd.read_csv(f"{DATA_DIR}/{prefix}_silva.tsv", sep="\t").set_index("ASV").reindex(counts.index)

    silva_tiers = {a: tier(*parse_species(silva.loc[a, "Genus"], silva.loc[a, "Species"]))
                   for a in counts.index}
    dt_tiers = {a: ("abstain" if dt.loc[a, "species_raw_score"] < 0.5
                    else tier(*parse_species(None, dt.loc[a, "species_predicted"])))
                for a in counts.index}

    silva_exact = percent(counts, silva_tiers, {"exact"})
    dt_exact = percent(counts, dt_tiers, {"exact"})
    dt_total = percent(counts, dt_tiers, {"exact", "sibling"})
    return silva_exact, dt_exact, dt_total


DATASETS = [
    ("ZymoBIOMICS V3-V4\n(Illumina)",     "V3-V4",       "v3v4"),
    ("ZymoBIOMICS full-length\n(PacBio)", "full-length", "fulllength"),
]

fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.3), sharey=True)
for ax, (title, ckpt, prefix) in zip(axes, DATASETS):
    silva_exact, dt_exact, dt_total = score(prefix)
    ax.bar(0, silva_exact, 0.6, color="#9aa0a6")
    ax.bar(1, dt_exact, 0.6, color="#9aa0a6", label="exact species")
    ax.bar(1, dt_total - dt_exact, 0.6, bottom=dt_exact,
           color="#d1495b", label="16S-indistinguishable sibling")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["DADA2\n+ SILVA", f"DeepTaxa\n({ckpt})"])
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0, silva_exact + 1.5, str(silva_exact), ha="center", fontsize=9)
    ax.text(1, dt_exact / 2, str(dt_exact), ha="center", va="center", color="white", fontsize=9)
    ax.text(1, dt_total + 1.5, str(dt_total), ha="center", fontsize=9)
    print(f"{prefix}: SILVA exact {silva_exact}, DeepTaxa exact {dt_exact}, DeepTaxa total {dt_total}")

axes[0].set_ylabel("% of reads correctly identified to species")
axes[0].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig("overview-species.png", dpi=150, bbox_inches="tight")
print("wrote overview-species.png")
