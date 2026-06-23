"""
Visualizers for the q2-deeptaxa plugin.

``describe`` summarizes a trained DeepTaxa model checkpoint. It reuses
:func:`deeptaxa.describe.describe` to export the checkpoint metadata as JSON,
then renders that metadata as a simple HTML report.
"""

import html
import json
import os
import tempfile
from argparse import Namespace

from ._formats import DeepTaxaModelDirectoryFormat


def _render_section(title, items):
    """Render a titled key/value table from a flat dict."""
    rows = []
    for key, value in items.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        rows.append(
            "<tr><th style='text-align:left;padding:4px 16px 4px 0;'>{}</th>"
            "<td style='padding:4px 0;'>{}</td></tr>".format(
                html.escape(str(key)), html.escape(str(value))
            )
        )
    return (
        f"<h2>{html.escape(title)}</h2>"
        "<table style='border-collapse:collapse;margin-bottom:24px;'>"
        + "".join(rows)
        + "</table>"
    )


def describe(output_dir: str, classifier: DeepTaxaModelDirectoryFormat) -> None:
    """Summarize a DeepTaxa model checkpoint as an HTML report.

    Parameters
    ----------
    output_dir : str
        Directory where the visualization is written (QIIME 2 supplies this).
    classifier : DeepTaxaModelDirectoryFormat
        Trained DeepTaxa model to describe.
    """
    checkpoint = os.path.join(str(classifier), "model.pt")

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_fp = os.path.join(tmpdir, "metrics.json")
        args = Namespace(
            checkpoint=checkpoint, export_metrics=metrics_fp, verbose=False
        )

        from deeptaxa.describe import describe as deeptaxa_describe

        deeptaxa_describe(args)
        with open(metrics_fp) as fh:
            metadata = json.load(fh)

    sections = ["<h1>DeepTaxa Model</h1>"]
    sections.append(
        f"<p><strong>Version:</strong> {html.escape(str(metadata.get('version', 'N/A')))}</p>"
    )

    if isinstance(metadata.get("model_details"), dict):
        sections.append(_render_section("Model details", metadata["model_details"]))
    if isinstance(metadata.get("training_hyperparameters"), dict):
        sections.append(
            _render_section(
                "Training hyperparameters", metadata["training_hyperparameters"]
            )
        )
    if isinstance(metadata.get("dataset_info"), dict):
        sections.append(_render_section("Dataset", metadata["dataset_info"]))

    levels = metadata.get("taxonomic_levels")
    if isinstance(levels, dict):
        flat = {
            f"Level {lvl} ({info.get('rank', '?')})": f"{info.get('labels', '?')} labels"
            for lvl, info in levels.items()
        }
        sections.append(_render_section("Taxonomic levels", flat))

    if isinstance(metadata.get("performance_metrics"), dict):
        sections.append(
            _render_section("Performance", metadata["performance_metrics"])
        )

    body = "\n".join(sections)
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>DeepTaxa Model</title></head>"
        "<body style='font-family:sans-serif;margin:24px;'>"
        f"{body}</body></html>"
    )

    with open(os.path.join(output_dir, "index.html"), "w") as fh:
        fh.write(page)
