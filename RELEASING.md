# Releasing DeepTaxa

DeepTaxa is published across several places that need to stay in step: the
GitHub repository, the PyPI package, the Hugging Face model card, and the
tutorials site. This checklist keeps them consistent when cutting a release.

The version number comes from the latest git tag through `setuptools_scm`, so
tagging is what sets the released version.

## Before tagging

1. Land all changes for the release on `main` and make sure CI is green.
2. If performance numbers, checkpoints, or the checkpoint list changed, update
   the `README.md` tables, the Hugging Face model card, and the QIIME 2 section
   so they tell the same story.
3. Run the pure-Python tests locally:
   ```bash
   python -m unittest q2_deeptaxa.tests.test_taxonomy q2_deeptaxa.tests.test_checkpoint -v
   ```
4. Validate the citation file:
   ```bash
   cffconvert --validate
   ```

## Tag and GitHub release

5. Tag the release and push the tag:
   ```bash
   git tag -a vX.Y.Z -m "DeepTaxa X.Y.Z"
   git push origin vX.Y.Z
   ```
6. Create the GitHub release from the tag with short notes on what changed.

## PyPI

7. Build from the tagged commit and upload. The long description on PyPI is the
   `README.md`, so rebuild whenever the README changed:
   ```bash
   python -m build
   python -m twine check dist/deeptaxa_rrna-X.Y.Z*
   python -m twine upload dist/deeptaxa_rrna-X.Y.Z*
   ```

## Hugging Face

8. If checkpoints or their numbers changed, update the model card and, for new
   checkpoint files, the SHA-256 table under "Download."

## Tutorials

9. Render the tutorials and deploy the site. The prediction and training pages
   install DeepTaxa from the latest release tag, so render after tagging to keep
   the `deeptaxa --version` output on the released version:
   ```bash
   bash tutorials/render_tutorials.sh
   ```

## After release

10. Check that the same version shows up everywhere: the GitHub release, the
    PyPI page, the `deeptaxa --version` output in the tutorials, and the model
    card. Fix any surface that lagged behind.
