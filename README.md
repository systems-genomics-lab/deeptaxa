# DeepTaxa

[![License](https://img.shields.io/github/license/systems-genomics-lab/deeptaxa)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/systems-genomics-lab/deeptaxa)
[![Tutorials](https://img.shields.io/badge/Tutorials-GitHub%20Pages-green)](https://systems-genomics-lab.github.io/deeptaxa/)
[![Last Commit](https://img.shields.io/github/last-commit/systems-genomics-lab/deeptaxa)](https://github.com/systems-genomics-lab/deeptaxa/commits/main)
[![Issues](https://img.shields.io/github/issues/systems-genomics-lab/deeptaxa)](https://github.com/systems-genomics-lab/deeptaxa/issues)
[![GitHub Stars](https://img.shields.io/github/stars/systems-genomics-lab/deeptaxa?style=social)](https://github.com/systems-genomics-lab/deeptaxa/stargazers)

**DeepTaxa** is a deep learning framework for hierarchical taxonomic classification of 16S rRNA gene sequences. It classifies sequences into all seven taxonomic ranks (Domain through Species) in a single forward pass, achieving 92.97% species-level accuracy on the Greengenes2 2024.09 test set.

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

The published HybridCNNBERT checkpoint, optimized via Optuna hyperparameter search (20 trials), achieves the following on 69,335 held-out test sequences:

| Rank | Accuracy | F1 | ECE |
|------|----------|-----|-----|
| Domain | 99.98% | 99.98% | 0.0002 |
| Phylum | 99.70% | 99.69% | 0.0023 |
| Class | 99.63% | 99.59% | 0.0026 |
| Order | 99.05% | 98.95% | 0.0066 |
| Family | 98.63% | 98.43% | 0.0085 |
| Genus | 96.88% | 96.43% | 0.0166 |
| Species | 92.97% | 92.00% | 0.0268 |

The table above reports the canonical seed-42 checkpoint. Across three independent training runs (seeds 42, 123, 456), species accuracy is 93.00% +/- 0.04 pp (mean +/- std). Full multi-seed results are in the [submission repository](https://github.com/ahmedmoustafa/deeptaxa-submission).

### Architecture

| Component | Configuration |
|-----------|--------------|
| CNN | embed_dim=896, 512 filters, kernels [5, 7, 9], 1 conv layer |
| BERT | 5 layers, 8 heads, hidden=1024, FFN=4096, GELU, random init |
| Fusion | Learnable alpha/beta weights + BERT residual connection |
| Training | Cross-entropy loss, LR=3.72e-4, batch=32, dropout=0.17, 10 epochs |

Three architectures are available:

- **HybridCNNBERTClassifier** (default): Fuses CNN local motif features with BERT global context. Used for the published checkpoints.
- **CNNClassifier**: Multi-kernel convolutional network only. Faster training, slightly lower species accuracy.
- **BERTClassifier**: Transformer encoder only. Captures long-range dependencies but requires more data to match CNN performance.

### Pre-Trained Checkpoints

Two checkpoints are hosted on [Hugging Face](https://huggingface.co/systems-genomics-lab/deeptaxa):

| Checkpoint | Training data | Species accuracy | Parameters |
|-----------|--------------|-----------------|------------|
| `deeptaxa-full-length-v1.pt` | Full-length 16S | 92.97% | 112.3M |
| `deeptaxa-v3v4-v1.pt` | In-silico V3-V4 amplicons | 87.52% | 99.5M |

Both use the same architecture. A `config.json` with full model metadata is also available.

---

## Installation

DeepTaxa requires Python 3.10 or later. We recommend using a Conda environment:

```bash
git clone https://github.com/systems-genomics-lab/deeptaxa.git
cd deeptaxa
conda create --name deeptaxa_env python=3.10 -y
conda activate deeptaxa_env
pip install .
deeptaxa --version
```

Dependencies (torch, transformers, pandas, numpy, scikit-learn, h5py, etc.) are specified in [`pyproject.toml`](pyproject.toml) and installed automatically.

> **Note**: For GPU support, install a CUDA-compatible PyTorch build before running `pip install .`. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

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
mkdir -p deeptaxa-data/greengenes && cd deeptaxa-data/greengenes
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

> **Note**: Checkpoint files use PyTorch's `pickle`-based serialization. Download them only from the official Hugging Face repository.

---

## Training

All architecture hyperparameters default to the Optuna-optimized values, so a minimal training command reproduces the published checkpoint:

```bash
deeptaxa train \
  --fasta-file ../deeptaxa-data/greengenes/gg_2024_09_training.fna.gz \
  --taxonomy-file ../deeptaxa-data/greengenes/gg_2024_09_training.tsv.gz \
  --model-type hybridcnnbert \
  --output-dir ../deeptaxa-outputs/
```

Training takes approximately 2.5 hours (10 epochs) on an NVIDIA A40 GPU.

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

---

## Tutorials

Interactive tutorials with executable code are published at [systems-genomics-lab.github.io/deeptaxa](https://systems-genomics-lab.github.io/deeptaxa/):

- [Prediction](https://systems-genomics-lab.github.io/deeptaxa/prediction.html): Classify sequences with the pre-trained model
- [Training](https://systems-genomics-lab.github.io/deeptaxa/training.html): Train from scratch on Greengenes2
- [Analysis](https://systems-genomics-lab.github.io/deeptaxa/analysis.html): Evaluate performance, calibration, and error patterns
- [Architecture](https://systems-genomics-lab.github.io/deeptaxa/architecture.html): Model internals and extensibility

---

## License

- **Code and models**: [MIT License](LICENSE)
- **Greengenes dataset**: [Modified BSD License](https://huggingface.co/datasets/systems-genomics-lab/greengenes)

---

## Citation

If DeepTaxa contributes to your research, please cite:

```bibtex
@article{salahkhalel2026deeptaxa,
  title={DeepTaxa: Deep Learning Framework for Taxonomic Classification},
  author={Salah Khalel, Rana and Abdelaal, Khlood and Ghonaim, Lobna and Awe, Olaitan I. and Moustafa, Ahmed},
  year={2026},
  note={Under review}
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
