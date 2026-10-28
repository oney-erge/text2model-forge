#!/usr/bin/env bash
# Regenerate the hash-checked pip exports from uv.lock (see docs/packaging.md).
# CI, the security audit, the launchers and the container install from these
# files, so they must always match uv.lock.
set -euo pipefail
cd "$(dirname "$0")/.."

uv lock --check
uv export --frozen --no-emit-project --format requirements-txt --output-file requirements.lock --quiet
uv export --frozen --extra local-ai --no-emit-project --format requirements-txt --output-file requirements-local-ai.lock --quiet
uv export --frozen --extra dev --no-emit-project --format requirements-txt --output-file requirements-dev.lock --quiet
uv export --frozen --all-extras --no-emit-project --format requirements-txt --output-file requirements-all.lock --quiet
