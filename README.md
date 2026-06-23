# DeepTaxa

[![PyPI](https://img.shields.io/pypi/v/deeptaxa-rrna)](https://pypi.org/project/deeptaxa-rrna/)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/deeptaxa-rrna?label=bioconda)](https://anaconda.org/bioconda/deeptaxa-rrna)
[![DOI](https://img.shields.io/badge/DOI-10.1093%2Fbioadv%2Fvbag166-blue)](https://doi.org/10.1093/bioadv/vbag166)
[![bio.tools](https://img.shields.io/badge/bio.tools-deeptaxa-2c5985)](https://bio.tools/deeptaxa)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/deeptaxa-rrna/)
[![License](https://img.shields.io/github/license/systems-genomics-lab/deeptaxa)](https://github.com/systems-genomics-lab/deeptaxa/blob/main/LICENSE)
[![PyPI Downloads](https://img.shields.io/pypi/dm/deeptaxa-rrna?label=PyPI%20downloads)](https://pypi.org/project/deeptaxa-rrna/)
[![Conda Downloads](https://img.shields.io/conda/dn/bioconda/deeptaxa-rrna?label=conda%20downloads)](https://anaconda.org/bioconda/deeptaxa-rrna)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/systems-genomics-lab/deeptaxa)
[![Tutorials](https://img.shields.io/badge/Tutorials-GitHub%20Pages-green)](https://systems-genomics-lab.github.io/deeptaxa/)
[![Last Commit](https://img.shields.io/github/last-commit/systems-genomics-lab/deeptaxa)](https://github.com/systems-genomics-lab/deeptaxa/commits/main)
[![Issues](https://img.shields.io/github/issues/systems-genomics-lab/deeptaxa)](https://github.com/systems-genomics-lab/deeptaxa/issues)
[![GitHub Stars](https://img.shields.io/github/stars/systems-genomics-lab/deeptaxa?style=social)](https://github.com/systems-genomics-lab/deeptaxa/stargazers)

**DeepTaxa** is a deep learning framework for hierarchical taxonomic classification of 16S rRNA gene sequences. It classifies sequences into all seven taxonomic ranks (Domain through Species) in a single forward pass, achieving 92.96% species-level accuracy (3-seed mean) on the Greengenes2 2024.09 test set.

---

## Table of Contents

1. [Performance](#performance)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data and Pre-Trained Models](#data-and-pre-trained-models)
5. [Training](#training)
6. [Experimentation](#experimentation)
7. [Scripts](#scripts)
8. [Tutorials](#tutorials)
9. [License](#license)
10. [Citation](#citation)
11. [Contact](#contact)
12. [Acknowledgements](#acknowledgements)

---

## Performance

The published HybridCNNBERT checkpoint achieves the following on 69,335 held-out test sequences from Greengenes2 2024.09 (3-seed mean across seeds 42, 123, 456):

| Rank | Accuracy | F1 | ECE |
|------|----------|-----|-----|
| Domain | 99.98% | 99.98% | 0.0001 |
| Phylum | 99.69% | 99.68% | 0.0023 |
| Class | 99.63% | 99.59% | 0.0024 |
| Order | 99.07% | 98.97% | 0.0056 |
| Family | 98.61% | 98.41% | 0.0075 |
| Genus | 96.90% | 96.48% | 0.0144 |
| Species | 92.96% | 92.12% | 0.0242 |

Cross-seed standard deviation is at most 0.0008 F1 at every rank (species std 0.0008 F1 / 0.07 percentage points accuracy), demonstrating high reproducibility.

### Architecture

| Component | Configuration |
|-----------|--------------|
| CNN | embed_dim=896, 256 filters, kernels [3, 5, 7], 1 conv layer |
| BERT | 4 layers, 7 heads, hidden=896, FFN=3584, GELU, random init |
| Fusion | Learnable alpha/beta weights + BERT residual connection |
| Training | Cross-entropy loss, LR=5e-4, batch=64, dropout=0.20, 10 epochs |

Three architectures are available:

- **HybridCNNBERTClassifier** (default): Fuses CNN local motif features with BERT global context. Used for the published checkpoints.
- **CNNClassifier**: Multi-kernel convolutional network only. Faster training, slightly lower species accuracy.
- **BERTClassifier**: Transformer encoder only. On its own, a from-scratch transformer underperforms substantially at the species rank; provided mainly for ablation.

### Pre-Trained Checkpoints

Two checkpoints are hosted on [Hugging Face](https://huggingface.co/systems-genomics-lab/deeptaxa):

| Checkpoint | Training data | Species accuracy | Parameters |
|-----------|--------------|-----------------|------------|
| `deeptaxa-full-length-v1.pt` | Full-length 16S (277,336 sequences, ~1,500 bp) | 92.96% (3-seed mean) | 76.4 M |
| `deeptaxa-v3v4-v1.pt` | In-silico V3-V4 amplicons (~420 bp, 273,003 amplicons) | 87.55% (seed 42) | 75.8 M |

Both checkpoints share the same compact architecture (the small parameter difference reflects smaller per-rank classifier heads on the V3-V4 model, which has a smaller species vocabulary: 8,347 vs 16,909). A `config.json` with full model metadata is also available.

---

## Installation

DeepTaxa requires Python 3.10 or later. It is distributed as **`deeptaxa-rrna`** on PyPI and Bioconda (the bare `deeptaxa` name was already taken on PyPI by an unrelated tool); the import package and the command-line tool are both `deeptaxa`.

### From PyPI

```bash
pip install deeptaxa-rrna
deeptaxa --version
```

### From Bioconda

```bash
conda install -c bioconda deeptaxa-rrna
deeptaxa --version
```

### From source

For the latest development version, or to modify the code:

```bash
git clone https://github.com/systems-genomics-lab/deeptaxa.git
cd deeptaxa
conda create --name deeptaxa_env python=3.10 -y
conda activate deeptaxa_env
pip install .
deeptaxa --version
```

Dependencies (torch, transformers, pandas, numpy, scikit-learn, biopython, h5py, optuna, etc.) are specified in [`pyproject.toml`](https://github.com/systems-genomics-lab/deeptaxa/blob/main/pyproject.toml) and installed automatically.

> **Note**: For GPU support, install a CUDA-compatible PyTorch build before installing DeepTaxa. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## Quick Start

**Predict** with the pre-trained model (no training data needed):

```bash
# Download the checkpoint
mkdir -p ../deeptaxa-data/models
wget -P ../deeptaxa-data/models \
  https://huggingface.co/systems-genomics-lab/deeptaxa/resolve/main/deeptaxa-full-length-v1.pt

# Classify sequences
deeptaxa predict \
  --fasta-file your_sequences.fna \
  --checkpoint ../deeptaxa-data/models/deeptaxa-full-length-v1.pt \
  --output-dir ../deeptaxa-outputs/predictions
```

**Evaluate** against known labels (adds per-rank accuracy, F1, ECE to the output):

```bash
deeptaxa predict \
  --fasta-file ../deeptaxa-data/greengenes/gg_2024_09_testing.fna.gz \
  --taxonomy-file ../deeptaxa-data/greengenes/gg_2024_09_testing.tsv.gz \
  --checkpoint ../deeptaxa-data/models/deeptaxa-full-length-v1.pt \
  --output-dir ../deeptaxa-outputs/evaluation
```

**Inspect** a checkpoint:

```bash
deeptaxa describe \
  --checkpoint ../deeptaxa-data/models/deeptaxa-full-length-v1.pt
```

> **Tip**: Run `deeptaxa train --help` or `deeptaxa predict --help` for a full list of options.

---

## Data and Pre-Trained Models

Datasets and checkpoints are hosted on [Hugging Face](https://huggingface.co/systems-genomics-lab/deeptaxa). Store them in a sibling directory outside the codebase:

```
working_directory/
├── deeptaxa/              # This repository
├── deeptaxa-data/         # Datasets and checkpoints
│   ├── greengenes/
│   │   ├── gg_2024_09_training.fna.gz    (277,336 sequences, ~96 MB)
│   │   ├── gg_2024_09_training.tsv.gz    (taxonomy labels, ~2.6 MB)
│   │   ├── gg_2024_09_testing.fna.gz     (69,335 sequences, ~24 MB)
│   │   └── gg_2024_09_testing.tsv.gz     (taxonomy labels, ~0.8 MB)
│   └── models/
│       ├── deeptaxa-full-length-v1.pt
│       └── deeptaxa-v3v4-v1.pt
└── deeptaxa-outputs/      # Training and prediction outputs
```

DeepTaxa uses the [Greengenes2](https://greengenes2.ucsd.edu/) database (2024.09 release), reformatted and hosted on [Hugging Face](https://huggingface.co/datasets/systems-genomics-lab/greengenes).

### Download

```bash
# Dataset
mkdir -p ../deeptaxa-data/greengenes && cd ../deeptaxa-data/greengenes
for f in gg_2024_09_training.fna.gz gg_2024_09_training.tsv.gz \
         gg_2024_09_testing.fna.gz gg_2024_09_testing.tsv.gz; do
  wget https://huggingface.co/datasets/systems-genomics-lab/greengenes/resolve/main/$f
done

# Checkpoints
mkdir -p ../models && cd ../models
wget https://huggingface.co/systems-genomics-lab/deeptaxa/resolve/main/deeptaxa-full-length-v1.pt
wget https://huggingface.co/systems-genomics-lab/deeptaxa/resolve/main/deeptaxa-v3v4-v1.pt
wget https://huggingface.co/systems-genomics-lab/deeptaxa/resolve/main/config.json
```

> **Tip**: If `wget` is unavailable (for example, on macOS), substitute `curl -L -O` from within the target directory to download each file.

> **Note**: Checkpoint files use PyTorch's `pickle`-based serialization. Download them only from the official Hugging Face repository.

---

## Training

All architecture hyperparameters default to the published (compact) configuration, so a minimal training command reproduces the published checkpoint:

```bash
deeptaxa train \
  --fasta-file ../deeptaxa-data/greengenes/gg_2024_09_training.fna.gz \
  --taxonomy-file ../deeptaxa-data/greengenes/gg_2024_09_training.tsv.gz \
  --model-type hybridcnnbert \
  --output-dir ../deeptaxa-outputs/
```

Training takes approximately 1 h 20 m on an NVIDIA RTX 4090 (or 2 h 35 m on an NVIDIA A40) for 10 epochs.

### Output

Each training run produces:

- `checkpoints/deeptaxa_<uuid>_epoch<N>.pt`: Model weights, optimizer state, scheduler state, and label encoders for each epoch.
- `metrics/deeptaxa_<uuid>_epoch<N>.json`: Per-epoch validation loss, accuracy, F1, precision, and recall at each rank.
- `deeptaxa_uuid.txt`: The unique run identifier.

### Early Stopping

To stop training when validation loss plateaus:

```bash
deeptaxa train \
  --fasta-file ../deeptaxa-data/greengenes/gg_2024_09_training.fna.gz \
  --taxonomy-file ../deeptaxa-data/greengenes/gg_2024_09_training.tsv.gz \
  --model-type hybridcnnbert \
  --epochs 20 \
  --early-stopping-patience 3 \
  --output-dir ../deeptaxa-outputs/
```

Setting `--early-stopping-patience 0` (the default) disables early stopping.

---

## Experimentation

The default configuration uses DNABERT-2 tokenization, cross-entropy loss, and uniform rank weighting. Each choice can be varied independently for ablation studies.

### Encoding comparison

```bash
# Default: DNABERT-2 BPE tokenization
deeptaxa train --model-type cnn --encoding dnabert ...

# Ablation: one-hot nucleotide encoding (4-channel, no pretrained tokenizer)
deeptaxa train --model-type cnn --encoding onehot ...
```

### Loss function comparison

```bash
# Default: cross-entropy
deeptaxa train --model-type hybridcnnbert --loss-type cross_entropy ...

# Ablation: focal loss (gamma=2.0)
deeptaxa train --model-type hybridcnnbert --loss-type focal --focal-gamma 2.0 ...
```

### Architecture comparison

Train CNN-only, BERT-only, or the hybrid under the same data and hyperparameters using `--model-type cnn`, `--model-type bert`, or `--model-type hybridcnnbert`.

### Calibration

When `--taxonomy-file` is provided at prediction time, DeepTaxa computes Expected Calibration Error (ECE) alongside accuracy, F1, precision, recall, and AUC. ECE measures the gap between predicted confidence and observed accuracy across 10 equal-width bins. All metrics are saved to `metrics.json`.

---

## Scripts

The `scripts/` directory contains reusable tools for common workflows:

| Script | Purpose |
|--------|---------|
| `deeptaxa_workflow.sh` | End-to-end workflow: train, resume, describe, predict |
| `run_experiment.sh` | Central experiment runner with logging and timing |
| `run_ablation.sh` | Ablation study: architecture, encoding, and loss variants |
| `run_amplicon_eval.sh` | Simulated amplicon evaluation (V3-V4, V4) |
| `run_similarity_eval.sh` | Similarity-stratified evaluation using vsearch |
| `calibration_diagnosis.sh` | A/B comparison of temperature configurations |
| `calibration_sweep.sh` | Multi-configuration temperature sweep |
| `simulate_amplicons.py` | Extract amplicon regions via in-silico PCR |
| `sequence_similarity.py` | Compute train-test nearest-neighbor identity |
| `similarity_curve.py` | Plot accuracy stratified by train-test similarity |

---

## Tutorials

Interactive tutorials with executable code are published at [systems-genomics-lab.github.io/deeptaxa](https://systems-genomics-lab.github.io/deeptaxa/):

- [Prediction](https://systems-genomics-lab.github.io/deeptaxa/prediction.html): Classify sequences with the pre-trained model
- [Training](https://systems-genomics-lab.github.io/deeptaxa/training.html): Train from scratch on Greengenes2
- [Analysis](https://systems-genomics-lab.github.io/deeptaxa/analysis.html): Evaluate performance, calibration, and error patterns
- [Architecture](https://systems-genomics-lab.github.io/deeptaxa/architecture.html): Model internals and extensibility

---

## License

- **Code and models**: [MIT License](https://github.com/systems-genomics-lab/deeptaxa/blob/main/LICENSE)
- **Greengenes dataset**: [Modified BSD License](https://huggingface.co/datasets/systems-genomics-lab/greengenes)

---

## Citation

If DeepTaxa contributes to your research, please cite our paper in *Bioinformatics Advances*: [https://doi.org/10.1093/bioadv/vbag166](https://doi.org/10.1093/bioadv/vbag166)

```bibtex
@article{salah2026deeptaxa,
  title={{DeepTaxa}: A Hybrid {CNN}-{BERT} Framework for {16S} {rRNA} Taxonomic Classification},
  author={Salah, Rana and AbdElaal, Khlood R. and Ghonaim, Lobna and Awe, Olaitan I. and Moustafa, Ahmed},
  journal={Bioinformatics Advances},
  year={2026},
  doi={10.1093/bioadv/vbag166},
  publisher={Oxford University Press}
}
```

For the Greengenes dataset:

```bibtex
@article{mcdonald2024greengenes,
  title={Greengenes2 unifies microbial data in a single reference tree},
  author={McDonald, Daniel and Jiang, Yueyu and Balaban, Metin and others},
  journal={Nature Biotechnology},
  volume={42},
  pages={715--718},
  year={2024},
  doi={10.1038/s41587-023-01845-1}
}
```

---

## Contact

To report bugs, suggest features, or contribute code, open an issue on [GitHub](https://github.com/systems-genomics-lab/deeptaxa/issues).

---

## Acknowledgements

- **[Ahmed A. El Hosseiny](https://github.com/ahmedelhosseiny)** and the High-Performance Computing Team of the [School of Sciences and Engineering](https://sse.aucegypt.edu/) at the [American University in Cairo](https://www.aucegypt.edu/) for GPU access that enabled this work.
- **[Hugging Face](https://huggingface.co/)** for hosting datasets and models.
