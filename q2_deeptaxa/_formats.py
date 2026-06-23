"""
File formats for the q2-deeptaxa plugin.

A trained DeepTaxa model is a single PyTorch checkpoint (``.pt``) written by
``deeptaxa train`` (:func:`deeptaxa.train.train`). The checkpoint holds
everything needed to run the model again: the weights, the architecture, the
tokenizer name, the label encoders (``level_label2id``), and the ordered list
of taxonomic ranks. We wrap this one file in a directory format so QIIME 2 can
import it as an artifact of type ``DeepTaxaModel``.
"""

from qiime2.plugin import model

# Keys that every usable DeepTaxa inference checkpoint must contain. These are
# exactly the keys deeptaxa.predict.predict validates before loading a model.
REQUIRED_CHECKPOINT_KEYS = (
    "model_type",
    "state_dict",
    "num_labels_per_level",
    "tokenizer_name",
    "level_label2id",
    "taxonomic_ranks",
)


class DeepTaxaModelFormat(model.BinaryFileFormat):
    """A single DeepTaxa model checkpoint serialized with ``torch.save``."""

    def _validate_(self, level):
        # Level "min" keeps import cheap: just confirm the file is a non-empty
        # ZIP-based archive, which is what modern torch.save produces.
        with self.open() as fh:
            magic = fh.read(4)
        if not magic:
            raise model.ValidationError(
                "DeepTaxa model checkpoint is empty."
            )
        # torch.save uses a ZIP container ("PK\x03\x04"); legacy pickle archives
        # start with "\x80". Accept either rather than rejecting older files.
        if not (magic.startswith(b"PK") or magic[:1] == b"\x80"):
            raise model.ValidationError(
                "File does not look like a PyTorch checkpoint produced by "
                "DeepTaxa (unexpected magic bytes). Provide the .pt file "
                "written by `deeptaxa train`."
            )

        if level != "max":
            return

        # Level "max" does a deeper check: load just the metadata and confirm the
        # required keys are present. torch is imported lazily so format
        # registration never pays the torch import cost.
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard dependency
            return
        try:
            checkpoint = torch.load(
                str(self), map_location="cpu", weights_only=False
            )
        except Exception as exc:  # noqa: BLE001 - surface a clean validation error
            raise model.ValidationError(
                f"Unable to load DeepTaxa checkpoint: {exc}"
            )
        if not isinstance(checkpoint, dict):
            raise model.ValidationError(
                "DeepTaxa checkpoint must deserialize to a dictionary."
            )
        missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in checkpoint]
        if missing:
            raise model.ValidationError(
                "DeepTaxa checkpoint is missing required keys: "
                + ", ".join(missing)
            )


# Single-file directory format: importing a .pt file places it at ``model.pt``.
DeepTaxaModelDirectoryFormat = model.SingleFileDirectoryFormat(
    "DeepTaxaModelDirectoryFormat", "model.pt", DeepTaxaModelFormat
)
