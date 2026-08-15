#!/usr/bin/env bash
# Portable launcher for the agent `skills` CLI (https://skills.sh).
#
# Runs `npx skills <args>` even when node/npx are not on PATH. Resolution order:
#   1. bundled Node shipped under .freebuff/node/ (this dev machine)
#   2. common install locations (Program Files, nvm, scoop)
#   3. whatever `node`/`npx` resolves on PATH
#
# Usage:
#   bash scripts/skills.sh find readme
#   bash scripts/skills.sh find --owner garrytan
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

find_node_dir() {
  # 1. bundled Node inside the workspace (.freebuff/ is gitignored, so this is
  #    a dev-machine convenience, not something a fresh clone can rely on).
  local bundled
  bundled="$(ls -d "$REPO_ROOT"/.freebuff/node/node-* 2>/dev/null | head -1 || true)"
  if [ -n "$bundled" ] && [ -x "$bundled/node.exe" ]; then
    echo "$bundled"
    return 0
  fi

  # 2a. plain Windows installs
  for d in "/c/Program Files/nodejs" "/c/Program Files (x86)/nodejs"; do
    if [ -x "$d/node.exe" ]; then
      echo "$d"
      return 0
    fi
  done

  # 2b. nvm keeps node in version subdirectories
  for d in "$HOME"/AppData/Roaming/nvm/*/; do
    if [ -d "$d" ] && [ -x "$d/node.exe" ]; then
      echo "${d%/}"
      return 0
    fi
  done

  # 2c. scoop
  if [ -x "$HOME/scoop/apps/nodejs/current/node.exe" ]; then
    echo "$HOME/scoop/apps/nodejs/current"
    return 0
  fi

  # 3. fall back to PATH
  local node_bin
  node_bin="$(command -v node 2>/dev/null || true)"
  if [ -n "$node_bin" ]; then
    dirname "$node_bin"
    return 0
  fi

  return 1
}

NODE_DIR="$(find_node_dir || true)"
if [ -n "$NODE_DIR" ]; then
  export PATH="$NODE_DIR:$PATH"
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "ERROR: npx not found." >&2
  echo "Install Node.js from https://nodejs.org, or drop a bundled Node into .freebuff/node/." >&2
  exit 1
fi

exec npx skills "$@"
