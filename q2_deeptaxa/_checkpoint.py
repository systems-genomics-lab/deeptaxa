"""
Checkpoint selection and trimming helpers for the q2-deeptaxa ``fit`` action.

These are kept free of QIIME and torch imports so the selection logic (which is
easy to get wrong, especially with early stopping) can be unit tested on its
own.
"""

import json
import os

# Keys a training checkpoint carries only so training can be resumed. They are
# dead weight in a DeepTaxaModel artifact, which is used for prediction.
TRAINING_ONLY_KEYS = (
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "rng_state",
)


def epoch_of(path):
    """Parse the epoch number from a ``..._epoch<N>.(pt|json)`` filename.

    Returns -1 when the name has no parsable epoch, so such files sort last.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem.rsplit("epoch", 1)[1])
    except (IndexError, ValueError):
        return -1


def read_validation_losses(metrics_paths):
    """Read ``(validation_loss, epoch)`` pairs from training metrics JSON files.

    Files that cannot be read, lack the validation-loss field, or recorded a
    non-positive loss (the sentinel for "not evaluated this epoch") are skipped.
    """
    scored = []
    for fp in metrics_paths:
        try:
            with open(fp) as fh:
                val_loss = json.load(fh)["performance_metrics"]["validation_loss"]
        except (KeyError, ValueError, OSError):
            continue
        if isinstance(val_loss, (int, float)) and not isinstance(val_loss, bool):
            if val_loss > 0:
                scored.append((val_loss, epoch_of(fp)))
    return scored


def choose_checkpoint(checkpoint_paths, metrics_paths):
    """Pick the best checkpoint to package as the model.

    Prefers the epoch with the lowest validation loss (which matters under early
    stopping, where the last epoch is past the best one). Falls back to the
    most-trained checkpoint when no validation loss was recorded.
    """
    by_epoch = {epoch_of(c): c for c in checkpoint_paths}
    scored = [
        (loss, epoch)
        for loss, epoch in read_validation_losses(metrics_paths)
        if epoch in by_epoch
    ]
    if scored:
        return by_epoch[min(scored)[1]]
    return by_epoch[max(by_epoch)]


def strip_training_state(checkpoint):
    """Remove resume-only state from a checkpoint dict in place and return it."""
    for key in TRAINING_ONLY_KEYS:
        checkpoint.pop(key, None)
    return checkpoint
