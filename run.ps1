# Stable Windows entry point for native and Docker operation.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\scripts\install-utils.ps1
Initialize-Install -RepositoryRoot $PSScriptRoot -ProductName "Asset Forge"
trap { Write-InstallFailure $_; Exit-InstallLock; exit 1 }

$forwardArgs = [Collections.Generic.List[string]]::new()
$forwardArgs.AddRange([string[]]$args)
$action = if ($forwardArgs.Count -gt 0 -and $forwardArgs[0] -in @("run", "doctor", "repair", "docker", "stop", "logs")) {
    $value = $forwardArgs[0]
    $forwardArgs.RemoveAt(0)
    $value
} else { "run" }
$noBrowser = $forwardArgs -contains "--no-browser" -or $forwardArgs -contains "-NoBrowser"
$url = "http://127.0.0.1:8766"

function Wait-Studio {
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        try { Invoke-WebRequest -UseBasicParsing -Uri "$url/doctor" -TimeoutSec 2 | Out-Null; return $true }
        catch { Start-Sleep -Milliseconds 500 }
    }
    return $false
}
function Test-Studio {
    try { Invoke-WebRequest -UseBasicParsing -Uri "$url/doctor" -TimeoutSec 2 | Out-Null; return $true }
    catch { return $false }
}

if ($action -eq "run" -and (Test-Studio)) {
    Write-Host "Asset Forge Studio is already running at $url" -ForegroundColor Green
    if (-not $noBrowser) { Start-Process $url }
    exit 0
}

if ($action -in @("docker", "stop", "logs")) {
    $dockerReady = $false
    $dockerRunning = $false
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        docker info *> $null
        $dockerReady = $LASTEXITCODE -eq 0
        if ($dockerReady) {
            $container = docker compose ps --quiet studio 2>$null
            $dockerRunning = -not [string]::IsNullOrWhiteSpace(($container -join ""))
        }
    }
    if ($action -eq "stop" -and -not $dockerRunning) {
        Write-Host "No Asset Forge Docker service is running. For a native Studio session, press Ctrl+C in its terminal."
        exit 0
    }
    if ($action -eq "logs" -and -not $dockerRunning) {
        $nativeLogs = @(Get-ChildItem -LiteralPath .\runtime\logs -Filter *.log -File -ErrorAction SilentlyContinue)
        if ($nativeLogs.Count -eq 0) {
            Write-Host "Native Studio logs are shown in its terminal. No managed service logs exist yet."
            exit 0
        }
        Get-Content -LiteralPath $nativeLogs.FullName -Tail 100 -Wait
        exit 0
    }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "Docker is not installed." }
    if (-not $dockerReady) { throw "Docker is installed but its engine is not running." }
    if ($action -eq "stop") { docker compose down; exit $LASTEXITCODE }
    if ($action -eq "logs") { docker compose logs --follow; exit $LASTEXITCODE }
    Enter-InstallLock
    Assert-InstallFreeSpace -Path $PSScriptRoot -RequiredGB 3
    docker compose up --detach --build
    if ($LASTEXITCODE -ne 0) { throw "Docker Compose build or startup failed." }
    if (-not (Wait-Studio)) { docker compose logs studio; throw "Text2Model Forge did not become ready at $url." }
    Complete-Install
    Write-Host "Text2Model Forge is ready at $url" -ForegroundColor Green
    if (-not $noBrowser) { Start-Process $url }
    exit 0
}

if ($action -ne "run") { $forwardArgs.Insert(0, $action) }
if (-not ($forwardArgs -contains "-AiStack") -and -not ($forwardArgs -contains "--ai-stack")) {
    $insertAt = if ($action -eq "run") { 0 } else { 1 }
    $forwardArgs.Insert($insertAt, "core")
    $forwardArgs.Insert($insertAt, "-AiStack")
}
$argumentArray = $forwardArgs.ToArray()
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "text2model-forge.ps1") @argumentArray
exit $LASTEXITCODE
