# Bioconda recipe for DeepTaxa

DeepTaxa is distributed as **`deeptaxa-rrna`** on PyPI and Bioconda (the bare
`deeptaxa` name was already taken on PyPI by an unrelated tool). The import
package and the command-line tool remain `deeptaxa`.

This directory holds the working copy of the Bioconda recipe. The authoritative
recipe lives in the [bioconda-recipes](https://github.com/bioconda/bioconda-recipes)
repository under `recipes/deeptaxa-rrna/`.

## Release checklist

1. **Tag the release** (the version is derived from the git tag via `setuptools_scm`):

   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Build and publish to PyPI** (in a Python >=3.10 environment):

   ```bash
   python -m pip install --upgrade build twine
   python -m build                 # creates dist/deeptaxa_rrna-1.0.0.tar.gz and the wheel
   python -m twine upload dist/*   # requires a PyPI API token
   ```

3. **Compute the sdist checksum** and put it in `meta.yaml` (`source.sha256`):

   ```bash
   openssl sha256 dist/deeptaxa_rrna-1.0.0.tar.gz
   ```

   Or regenerate the whole recipe from the published PyPI release with grayskull:

   ```bash
   pip install grayskull
   grayskull pypi deeptaxa-rrna
   ```

4. **Submit to Bioconda:**

   ```bash
   # fork github.com/bioconda/bioconda-recipes, then:
   git clone https://github.com/<your-fork>/bioconda-recipes
   mkdir -p bioconda-recipes/recipes/deeptaxa-rrna
   cp conda-recipe/meta.yaml bioconda-recipes/recipes/deeptaxa-rrna/
   # commit on a branch and open a PR against bioconda/bioconda-recipes
   ```

   Bioconda CI builds and tests the recipe automatically. Address any review
   comments; once merged, the package is available via:

   ```bash
   conda install -c bioconda deeptaxa-rrna
   ```

## Notes

- `noarch: python` — the package is pure Python; dependencies carry their own
  platform builds.
- `SETUPTOOLS_SCM_PRETEND_VERSION` is exported during the build so the version
  resolves correctly when building outside a git checkout.
- Update `extra.recipe-maintainers` with the GitHub handle(s) responsible for
  the recipe.
