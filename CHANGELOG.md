# Changelog

All notable changes to DeepTaxa are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
version number itself comes from the latest git tag through `setuptools_scm`.

Releases up to and including 1.2.0 predate this file; their notes live in the
[git history](https://github.com/systems-genomics-lab/deeptaxa/commits/main) and
the [GitHub releases](https://github.com/systems-genomics-lab/deeptaxa/releases).

## [Unreleased]

## [1.3.0] - 2026-07-07

### Added

- `--tokenizer-revision` pins the tokenizer to a specific Hugging Face commit,
  tag, or branch. It is recorded in the checkpoint and reused at prediction
  time, which guards against upstream changes to a tokenizer loaded with
  `trust_remote_code`. Defaults to none, so nothing changes unless you set it.
- `--mask-padding` masks padded positions out of the CNN and hybrid max pooling
  so they cannot leak into the pooled features. It is off by default, so training
  and prediction are unchanged unless you enable it; the setting is recorded in
  the checkpoint and reused at prediction time.
- `--save-best-only` writes a checkpoint only on an epoch that improves the
  validation loss, instead of one per evaluated epoch.
- A `batch_relative_score` field in the prediction output, alongside the existing
  `score`, that names the value for what it is: `raw_score` normalized against the
  rest of the input. `score` is kept as a deprecated alias of it, so existing
  consumers keep working. `raw_score` remains the stable per-sequence confidence.

### Changed

- The validation loss now applies the same per-rank `--level-weights` as the
  training loss, so early stopping and checkpoint selection track the objective
  the model is trained on.
- `--top-k` is documented as controlling the top-k accuracy metric (reported
  when a taxonomy file is provided); per-sequence output always reports the
  single highest-probability class per rank.
- Taxonomy alignment and one-hot encoding are faster: the sequence-to-taxonomy
  map is built in one pass and the one-hot encoder uses a byte lookup table.
- The per-epoch layer-weights HDF5 export drops from gzip level 9 to 4, which is
  much faster to write for a negligible size difference.
- The prediction summary drops a redundant introductory line, leaving the header
  and the run details.

### Fixed

- The top-k accuracy column header in the per-rank metrics table now shows the
  configured value of `k` instead of a literal placeholder.

- Gradient accumulation now accumulates across the whole window instead of
  clearing gradients on every micro-batch, so `--accum-steps` greater than 1
  produces the intended effective batch size. The scheduler step count rounds up
  so the final partial window is included.
- AUC no longer falls back to N/A when the evaluation set contains labels absent
  from the training vocabulary; the true labels are aligned with the stored
  probability rows before scoring.
- Species accuracy and top-k accuracy share one boundary-aware agreement rule,
  so top-k accuracy can no longer count fewer hits than exact accuracy.
- Training on a reference taxonomy with blank (partial) lineages no longer
  crashes with a `KeyError`; missing ranks fill with `Unclassified`, matching
  the QIIME 2 path.
- The BERT model sizes its embedding from the full tokenizer vocabulary and
  guards its mean pooling against an all-padding row.
