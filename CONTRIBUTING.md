# Contributing to DeepTaxa

Thank you for your interest in DeepTaxa. Contributions of all kinds are welcome,
including bug reports, documentation fixes, new features, and improvements to the
QIIME 2 plugin. This document explains how to get involved.

## Reporting issues

Please open an issue on the [issue tracker](https://github.com/systems-genomics-lab/deeptaxa/issues)
for bugs, unexpected behavior, or feature requests. A useful bug report includes:

- The DeepTaxa version (`deeptaxa --version`) and the Python and PyTorch versions.
- The command or code that triggered the problem.
- The full error message and traceback.
- A short description of what you expected to happen.

If the problem depends on a specific input, a minimal example that reproduces it is
the single most helpful thing you can provide.

## Development setup

DeepTaxa requires Python 3.10 or later and PyTorch 2.4 or later. Install the package
in editable mode inside a clean virtual environment:

```bash
git clone https://github.com/systems-genomics-lab/deeptaxa.git
cd deeptaxa
pip install -e .
```

Editable mode links the installed package to your working copy, so changes to the
source take effect without reinstalling.

## Running the tests

The pure-Python tests do not require a GPU or the full QIIME 2 stack. They run with
the standard library test runner:

```bash
pip install pandas
python -m unittest q2_deeptaxa.tests.test_taxonomy q2_deeptaxa.tests.test_checkpoint -v
```

The complete suite, including the QIIME 2 plugin tests, runs inside a QIIME 2
environment. Continuous integration executes both sets on every pull request, so
please make sure the pure-Python tests pass locally before you open one.

## Submitting changes

1. Create a branch for your change rather than working on `main`.
2. Keep each pull request focused on a single concern. Smaller pull requests are
   easier to review and merge.
3. Add or update tests when you change behavior, and update the documentation and
   tutorials if your change affects how the tool is used.
4. Confirm that the tests pass and that continuous integration is green.
5. Describe what the change does and why in the pull request text.

## Coding style

Please follow the style of the surrounding code: clear names, small functions, and
comments only where the intent is not obvious from the code itself. Match the
existing formatting rather than reformatting unrelated lines, which keeps the diff
readable and the review focused on the actual change.

## Citation

If you use DeepTaxa in your research, please cite the manuscript listed in
[`CITATION.cff`](CITATION.cff) and in the project [README](README.md).
