# Changelog

All notable changes to DeepTaxa are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
version number itself comes from the latest git tag through `setuptools_scm`.

Releases up to and including 1.2.0 predate this file; their notes live in the
[git history](https://github.com/systems-genomics-lab/deeptaxa/commits/main) and
the [GitHub releases](https://github.com/systems-genomics-lab/deeptaxa/releases).

## [Unreleased]

### Added

- `--tokenizer-revision` pins the tokenizer to a specific Hugging Face commit,
  tag, or branch. It is recorded in the checkpoint and reused at prediction
  time, which guards against upstream changes to a tokenizer loaded with
  `trust_remote_code`. Defaults to none, so nothing changes unless you set it.
- `--no-mask-padding` restores the older pooling that includes padded positions,
  for comparison against the new default.
- `--save-best-only` writes a checkpoint only on an epoch that improves the
  validation loss, instead of one per evaluated epoch.

### Changed

- The CNN and hybrid models now mask padded positions out of the global max
  pooling by default, so a prediction no longer shifts with the amount of
  padding. Existing checkpoints load and run exactly as before, since the
  setting is read from the checkpoint and defaults to off when absent. Note that
  `deeptaxa train` on defaults therefore no longer reproduces the published
  checkpoint's unmasked pooling; pass `--no-mask-padding` to match it.
- The per-prediction `score` field in the prediction output is renamed to
  `batch_relative_score`, since it is normalized against the rest of the input.
  `raw_score` stays as the stable per-sequence confidence.
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

### Fixed

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
