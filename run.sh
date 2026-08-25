#!/usr/bin/env bash
# Text2Model Forge - one-command launcher on Linux and macOS.
#
# Usage:
#   ./run.sh                          install (if needed) + start Studio, core stack (fast, no model downloads)
#   ./run.sh doctor                   readiness report
#   ./run.sh install --ai-stack qwen --accept-sdxl-license --accept-hunyuan-license
#                                     full local AI stack; every option is text2model-forge.sh's
#
# This is a thin front door, not a second installer: every argument is
# forwarded to text2model-forge.sh, which does the real work and is already
# idempotent (re-run any time; it repairs what's missing and starts Studio)
# with its own bounded install retries. The only thing this script decides is
# the default AI stack -- "core" here, so a first run is fast and does not
# reach for a GPU or download anything -- and only when you have not already
# named one yourself.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

has_ai_stack=0
for arg in "$@"; do
    if [ "$arg" = "--ai-stack" ]; then
        has_ai_stack=1
        break
    fi
done

if [ "$has_ai_stack" -eq 1 ]; then
    exec ./text2model-forge.sh "$@"
else
    exec ./text2model-forge.sh --ai-stack core "$@"
fi
