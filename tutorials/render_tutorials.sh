#!/bin/bash
# Render all DeepTaxa tutorials from a clean start and deploy to gh-pages.
# Usage: bash render_tutorials.sh 2>&1 | tee render_tutorials_$(date +%Y%m%d_%H%M%S).log
#
# Run this script from the tutorials/ directory.

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$SCRIPT_DIR"

SCRIPT_START=$(date +%s)
echo "=============================================="
echo "DeepTaxa Tutorial Rendering"
echo "Started: $(date)"
echo "=============================================="
echo ""

# Clean previous state
echo "--- Cleaning previous state ---"
rm -rf ~/deeptaxa-workspace
rm -rf _freeze _site
echo "Done"
echo ""

# Render each tutorial. Order: cheap/safe first so a failure in a
# heavyweight tutorial (training, analysis) doesn't strand the others.
# Per-tutorial failures are reported but do not abort the whole run.
TUTORIALS="index architecture prediction validation analysis training"
FAILED=""

for tutorial in $TUTORIALS; do
    echo "=============================================="
    echo "Rendering: $tutorial"
    echo "Started: $(date)"
    echo "=============================================="

    START=$(date +%s)
    if ! make "$tutorial"; then
        FAILED="$FAILED $tutorial"
        echo ""
        echo "$tutorial FAILED — continuing with remaining tutorials"
        echo ""
        continue
    fi
    END=$(date +%s)

    ELAPSED=$((END - START))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))

    echo ""
    echo "$tutorial completed in ${MINS}m ${SECS}s"
    echo ""
done

if [ -n "$FAILED" ]; then
    echo "=============================================="
    echo "Tutorials that failed to render:$FAILED"
    echo "=============================================="
    echo ""
fi

# Summary
SCRIPT_END=$(date +%s)
TOTAL=$((SCRIPT_END - SCRIPT_START))
TOTAL_MINS=$((TOTAL / 60))
TOTAL_SECS=$((TOTAL % 60))

echo "=============================================="
echo "All tutorials rendered"
echo "Finished: $(date)"
echo "Total time: ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "=============================================="
echo ""

# Show what was produced
echo "--- Frozen outputs ---"
find _freeze -name "*.json" -o -name "*.png" | sort
echo ""
echo "--- Site files ---"
find _site -name "*.html" | sort
echo ""

# Deploy _site/ to gh-pages
echo "--- Deploying _site/ to gh-pages ---"
TMPDIR=$(mktemp -d)
ORIGIN=$(git -C "$REPO_ROOT" remote get-url origin)
git -C "$TMPDIR" init
git -C "$TMPDIR" checkout -b gh-pages
cp -r "$SCRIPT_DIR/_site/." "$TMPDIR/"
git -C "$TMPDIR" add -A
git -C "$TMPDIR" commit -m "Deploy tutorials ($(date +%Y-%m-%d))"
git -C "$TMPDIR" remote add origin "$ORIGIN"
git -C "$TMPDIR" push --force origin gh-pages
rm -rf "$TMPDIR"
echo "Deployed to gh-pages"
