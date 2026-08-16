#!/usr/bin/env bash
# Installs MissionMind's git hooks in this checkout.
#
# Run once after cloning (or after pulling a change to .githooks/):
#   bash scripts/install-hooks.sh
#
# It points core.hooksPath at the tracked .githooks/ directory, so the
# pre-commit clean-sync hook becomes active for every commit made here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

git config core.hooksPath .githooks
echo "hooks installed: core.hooksPath -> $(git config core.hooksPath)"
echo "active hooks: $(ls .githooks)"
