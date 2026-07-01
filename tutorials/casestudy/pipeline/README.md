# Reproducing the case study from raw reads

The analysis notebook (`../analysis.qmd`) renders from the committed tables in `../data/`,
so it reproduces without any of the steps below. This pipeline regenerates those tables
from the public raw reads, so the case study is reproducible from step zero if you want it
to be.

## Data

The 18 sequencing runs are 16S rRNA V4 amplicons from BioProject PRJNA566436 (Hertzberg et
al., 2021). The run accessions and the ALS/control pairing are in `../data/samples.tsv`.

The deposited runs are already primer-trimmed: the 515F/806R primers have been removed, so the
reads begin at the conserved bases internal to the primers and the resulting ASVs are 253 bp
(the primer-free V4 insert). `01_dada2_asv.R` relies on this and does no primer removal of its
own. If you substitute reads that still carry the primers, strip them first (for example with
cutadapt), or the ASVs will be about 292 bp and will misclassify.

## Steps

Downloading the raw reads with the SRA Toolkit is a single command shown in the notebook's
"Reproducing the inputs" section. The other three steps are run as scripts:

| Script | Tools | Produces |
|--------|-------|----------|
| `01_dada2_asv.R` | R, DADA2 1.40.0 | `../data/asv_seqs.fasta`, `../data/asv_counts.tsv`, `../data/track_reads.tsv` |
| `classify_deeptaxa.sh` | `deeptaxa-rrna` 1.1.0, V4 checkpoint | `../data/asv_taxonomy.tsv` |
| `02_classify_silva.R` | R, DADA2, SILVA v138 reference | `../data/asv_taxonomy_silva.tsv` |

The full order is: download the reads, run `01_dada2_asv.R`, run `classify_deeptaxa.sh`,
run `02_classify_silva.R`. Then render the analysis:

```bash
quarto render ../analysis.qmd
```

## Requirements

- SRA Toolkit 3.x and `pigz` for the download.
- R with DADA2 1.40.0. DADA2 performs its own quality filtering on the raw paired reads,
  so `fastp` is not part of this route.
- `deeptaxa-rrna` 1.1.0 (`pip install deeptaxa-rrna`, CLI `deeptaxa`). The V4 checkpoint is
  downloaded from <https://huggingface.co/systems-genomics-lab/deeptaxa>. A CUDA GPU is
  recommended but the CLI also runs on CPU.
- The SILVA v138 training set (`silva_nr99_v138_train_set.fa.gz`) from the DADA2 reference
  downloads, <https://benjjneb.github.io/dada2/training.html>. Point `02_classify_silva.R`
  at it with the `SILVA_REF` environment variable.

## Determinism

DADA2 is seeded (`set.seed(42)`), DeepTaxa inference is deterministic (a fixed model with no
sampling), and the SILVA assignment is seeded (`set.seed(100)`). Re-running reproduces the
committed tables; small differences are possible only across different tool or reference
versions.
