# Text2Model Forge - one-command launcher on Windows.
#
# Usage:
#   .\run.ps1                    -> install (if needed) + start Studio, core stack (fast, no model downloads)
#   .\run.ps1 doctor              -> readiness report
#   .\run.ps1 -AiStack qwen -AcceptSdxlLicense -AcceptHunyuanLicense
#                                 -> full local AI stack; every option is text2model-forge.ps1's
#
# This is a thin front door, not a second installer: every argument is
# forwarded to text2model-forge.ps1, which does the real work and is already
# idempotent (re-run any time; it repairs what's missing and starts Studio)
# with its own bounded install retries. The only thing this script decides is
# the default AI stack -- "core" here, so a first run is fast and does not
# reach for a GPU or download anything -- and only when you have not already
# named one yourself.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$forwardArgs = [string[]]$args
if (-not ($forwardArgs -contains "-AiStack")) {
    $forwardArgs = @("-AiStack", "core") + $forwardArgs
}

& (Join-Path $PSScriptRoot "text2model-forge.ps1") @forwardArgs
exit $LASTEXITCODE
