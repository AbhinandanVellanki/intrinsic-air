#!/usr/bin/env bash
# Run from the repository root. Fetches and overlays docs assets from another
# branch so Sphinx can render them. Avoids doing this inside Docker (no
# root-owned files or full-repo mount).
#
# Supports two branch layouts:
#   - Root-level 'media/' dir: it is moved to docs/sphinx/media/
#   - No media/ dir (files at branch root): branch root is extracted into docs/sphinx/media/
# If the branch also has docs/sphinx/source/_static/assets, that path is overlayed.
set -e

BRANCH="${1:-media}"
REPO_ROOT="$(git rev-parse --show-toplevel)"

echo "Fetching origin/${BRANCH}..."
git fetch origin "$BRANCH"

# No media/ on branch: all files are at branch root; extract root into docs/sphinx/media/
echo "Extracting root of origin/${BRANCH} into ./docs/sphinx/source/_static/assets/..."
rm -rf ${REPO_ROOT}/docs/sphinx/source/_static/assets
mkdir -p ${REPO_ROOT}/docs/sphinx/source/_static/assets
cd ${REPO_ROOT} && git archive "origin/${BRANCH}" | tar -x -C docs/sphinx/source/_static/assets

echo "Done. Run: cd docker && docker compose up docs"
