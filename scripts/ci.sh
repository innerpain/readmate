#!/usr/bin/env bash
# Local CI for ReadMate (D47). GitHub Actions is unusable here because `origin`
# points at someone else's repo, so the gate runs on the developer machine.
#
# Runs, in order, stopping at the first failure:
#   1) containerized pytest            (needs the local base image; see .env.example D50 note)
#   2) cd frontend && npx tsc --noEmit (typecheck only, no emit)
#   3) cd frontend && npm run build    (tsc --noEmit && vite build)
#
# Usage:
#   bash scripts/ci.sh            # all three steps
#   bash scripts/ci.sh pytest     # only step 1
#   bash scripts/ci.sh frontend   # only steps 2+3
#
# No secrets live in this file. It reads .env via docker compose, never prints it.
#
# Install as a pre-push hook (NOT done automatically - your git config is yours):
#   cp scripts/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
# or, to keep hooks versioned in-repo:
#   git config core.hooksPath .githooks

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Absolute host path for the -v mounts: Docker Desktop rejects relative paths
# ("mount path must be absolute") and MSYS path translation is disabled here.
REPO_NATIVE="$(pwd -W 2>/dev/null || pwd)"

SCOPE="${1:-all}"

log() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

run_pytest() {
  log "1/3 pytest (containerized)"
  # tests/ eval/ test_data/ pytest.ini are all .dockerignored, so they must be
  # mounted back in. HF_HUB_OFFLINE=1 keeps SentenceTransformer from probing
  # huggingface.co for optional configs during the run.
  docker compose run --rm --no-deps -e HF_HUB_OFFLINE=1 \
    -v "${REPO_NATIVE}/tests:/app/tests" \
    -v "${REPO_NATIVE}/eval:/app/eval" \
    -v "${REPO_NATIVE}/test_data:/app/test_data" \
    -v "${REPO_NATIVE}/pytest.ini:/app/pytest.ini" \
    api sh -lc 'pip install -q pytest pytest-asyncio && python -m pytest -q'
}

run_frontend() {
  log "2/3 frontend typecheck (npx tsc --noEmit)"
  ( cd frontend && npx tsc --noEmit )

  log "3/3 frontend build (npm run build)"
  ( cd frontend && npm run build )
}

case "$SCOPE" in
  all)      run_pytest; run_frontend ;;
  pytest)   run_pytest ;;
  frontend) run_frontend ;;
  *)        echo "usage: bash scripts/ci.sh [all|pytest|frontend]" >&2; exit 2 ;;
esac

log "CI OK"
