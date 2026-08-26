[CmdletBinding()]
param(
    [ValidateSet("start", "install", "repair", "doctor")]
    [string]$Action = "start",

    [ValidateSet("auto", "qwen", "sdxl", "existing", "core")]
    [string]$AiStack = "auto",

    [ValidateSet("auto", "nvidia", "cpu")]
    [string]$Gpu = "auto",

    [string]$Workspace = "",
    [string]$RuntimeRoot = "",
    [string]$ReviewerModel = "qwen3-vl:8b-instruct",

    [ValidateRange(1, 5)]
    [int]$MaxAttempts = 3,

    [switch]$AcceptSdxlLicense,
    [switch]$AcceptHunyuanLicense,
    [switch]$SkipImageTo3D,
    [switch]$IncludeDev,
    [switch]$NonInteractive,
    [switch]$NoElevation,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "text2model-forge.ps1 is the native Windows launcher. On Linux or macOS run ./text2model-forge.sh."
}

$script:RepoRoot = [IO.Path]::GetFullPath($PSScriptRoot)
. (Join-Path $script:RepoRoot "scripts\install-utils.ps1")
Initialize-Install -RepositoryRoot $script:RepoRoot -ProductName "Asset Forge"
if (-not $Workspace) {
    $Workspace = Join-Path $env:USERPROFILE "Text2ModelForgeRuns"
}
if (-not $RuntimeRoot) {
    $RuntimeRoot = Join-Path $script:RepoRoot "runtime"
}
$Workspace = [IO.Path]::GetFullPath($Workspace)
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$script:VenvPython = Join-Path $script:RepoRoot ".venv\Scripts\python.exe"
$script:ComfyRoot = Join-Path $RuntimeRoot "ComfyUI_windows_portable"
$script:ComfyModels = Join-Path $script:ComfyRoot "ComfyUI\models"
$script:Step = 0
$script:TotalSteps = 9

$ComfyArchiveUrl = "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z"
$PythonInstallerUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
$OllamaInstallerUrl = "https://ollama.com/download/OllamaSetup.exe"
$SdxlUrl = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
$HunyuanUrl = "https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2_fp16.safetensors"
$PipVersion = "26.2.1"
$SetuptoolsVersion = "84.0.0"

function Write-Step {
    param([string]$Activity)
    $script:Step++
    $percent = [Math]::Min(100, [Math]::Round(($script:Step / $script:TotalSteps) * 100))
    Write-Progress -Id 1 -Activity "Text2Model Forge setup" -Status $Activity -PercentComplete $percent
    Write-Host "`n[$script:Step/$script:TotalSteps] $Activity" -ForegroundColor Cyan
}

function Invoke-WithRetry {
    param(
        [string]$Name,
        [scriptblock]$Operation,
        [int]$Attempts = $MaxAttempts
    )
    Invoke-InstallRetry -Label $Name -Operation $Operation -Attempts $Attempts
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [switch]$Elevated
    )
    if ($Elevated) {
        $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -Verb RunAs -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "$FilePath exited with code $($process.ExitCode)"
        }
        return
    }
    & $FilePath @ArgumentList | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath exited with code $LASTEXITCODE"
    }
}

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = @($env:Path, $machinePath, $userPath) -join ";"
}

function Test-Command {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-WingetInstall {
    param([string]$PackageId, [string]$Label)
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) { return $false }
    $common = @(
        "install", "--id", $PackageId, "--exact", "--source", "winget",
        "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity"
    )
    $boundedAttempts = [Math]::Min(2, $MaxAttempts)
    try {
        Invoke-WithRetry "$Label user-scope install" {
            Invoke-Native $winget.Source ($common + @("--scope", "user"))
        } $boundedAttempts
        return $true
    }
    catch {
        Write-Warning "$Label could not be installed in user scope: $($_.Exception.Message)"
    }
    try {
        Invoke-WithRetry "$Label standard install" {
            Invoke-Native $winget.Source $common
        } $boundedAttempts
        return $true
    }
    catch {
        Write-Warning "$Label standard install failed: $($_.Exception.Message)"
    }
    if (-not $NoElevation) {
        try {
            Write-Host "Requesting Windows elevation for the final $Label install method..." -ForegroundColor Yellow
            Invoke-Native $winget.Source $common -Elevated
            return $true
        }
        catch {
            Write-Warning "$Label elevated install failed or was declined: $($_.Exception.Message)"
        }
    }
    return $false
}

function Save-Download {
    param([string]$Url, [string]$Destination, [string]$Label)
    $directory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    if (Test-Path -LiteralPath $Destination) {
        Write-Host "$Label is already downloaded."
        return
    }
    $partial = "$Destination.partial"
    Invoke-WithRetry "$Label download" {
        if (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial -Force
        }
        if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
            $job = Start-BitsTransfer -Source $Url -Destination $partial -DisplayName $Label -Asynchronous
            try {
                while ($job.JobState -in @("Queued", "Connecting", "Transferring", "TransientError")) {
                    $job = Get-BitsTransfer -Id $job.Id
                    $downloadPercent = 0
                    if ($job.BytesTotal -gt 0) {
                        $downloadPercent = [Math]::Round(($job.BytesTransferred / $job.BytesTotal) * 100)
                    }
                    Write-Progress -Id 2 -ParentId 1 -Activity $Label -Status "$downloadPercent%" -PercentComplete $downloadPercent
                    if ($job.JobState -eq "TransientError") {
                        throw $job.ErrorDescription
                    }
                    Start-Sleep -Milliseconds 500
                }
                if ($job.JobState -ne "Transferred") {
                    throw "BITS ended in state $($job.JobState): $($job.ErrorDescription)"
                }
                Complete-BitsTransfer -BitsJob $job
            }
            catch {
                Remove-BitsTransfer -BitsJob $job -Confirm:$false -ErrorAction SilentlyContinue
                throw
            }
            finally {
                Write-Progress -Id 2 -ParentId 1 -Activity $Label -Completed
            }
        }
        else {
            Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
        }
        Move-Item -LiteralPath $partial -Destination $Destination -Force
    }
}

function Test-PythonExecutable {
    param([string]$Path)
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $version = & $Path -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return ($LASTEXITCODE -eq 0 -and [version]$version.Trim() -ge [version]"3.12")
    }
    catch { return $false }
}

function Find-Python {
    $candidates = New-Object System.Collections.Generic.List[string]
    if (Test-Command py) {
        $fromLauncher = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $fromLauncher) { $candidates.Add($fromLauncher.Trim()) }
    }
    foreach ($name in @("python", "python3")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) { $candidates.Add($command.Source) }
    }
    $localPrograms = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path -LiteralPath $localPrograms) {
        Get-ChildItem -LiteralPath $localPrograms -Filter python.exe -Recurse -ErrorAction SilentlyContinue |
            ForEach-Object { $candidates.Add($_.FullName) }
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-PythonExecutable $candidate) { return $candidate }
    }
    return $null
}

function Install-Python {
    if (Invoke-WingetInstall "Python.Python.3.12" "Python 3.12") {
        Refresh-ProcessPath
        $python = Find-Python
        if ($python) { return $python }
    }
    Write-Warning "WinGet methods did not produce Python 3.12; using Python.org's pinned per-user installer."
    $installer = Join-Path $RuntimeRoot "downloads\python-3.12.10-amd64.exe"
    Save-Download $PythonInstallerUrl $installer "Python 3.12.10"
    Invoke-WithRetry "Python.org per-user install" {
        Invoke-Native $installer @("/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_test=0", "Include_launcher=1")
    }
    Refresh-ProcessPath
    $python = Find-Python
    if (-not $python -and -not $NoElevation) {
        Write-Host "Requesting elevation for the final all-users Python install method..." -ForegroundColor Yellow
        Invoke-Native $installer @("/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_test=0", "Include_launcher=1") -Elevated
        Refresh-ProcessPath
        $python = Find-Python
    }
    if (-not $python) { throw "Python 3.12+ could not be installed or discovered." }
    return $python
}

function Ensure-Core {
    param([string]$SelectedStack)
    Write-Step "Checking Python and the Text2Model Forge environment"
    $systemPython = Find-Python
    if (-not $systemPython) { $systemPython = Install-Python }
    if (-not (Test-PythonExecutable $script:VenvPython)) {
        if (Test-Path -LiteralPath (Split-Path -Parent $script:VenvPython)) {
            Invoke-WithRetry "virtual environment repair" {
                Invoke-Native $systemPython @("-m", "venv", "--upgrade", (Join-Path $script:RepoRoot ".venv"))
            }
        }
        else {
            Invoke-WithRetry "virtual environment creation" {
                Invoke-Native $systemPython @("-m", "venv", (Join-Path $script:RepoRoot ".venv"))
            }
        }
    }
    $includeLocalAi = $SelectedStack -in @("qwen", "sdxl")
    $lockName = if ($includeLocalAi -and $IncludeDev) {
        "requirements-all.lock"
    }
    elseif ($includeLocalAi) {
        "requirements-local-ai.lock"
    }
    elseif ($IncludeDev) {
        "requirements-dev.lock"
    }
    else {
        "requirements.lock"
    }
    $lockPath = Join-Path $script:RepoRoot $lockName
    if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
        throw "The committed dependency lock is missing: $lockPath"
    }
    $projectHash = (Get-FileHash -Algorithm SHA256 (Join-Path $script:RepoRoot "pyproject.toml")).Hash
    $lockHash = (Get-FileHash -Algorithm SHA256 $lockPath).Hash
    $installFingerprint = "$projectHash|$lockHash|$lockName"
    $marker = Join-Path $script:RepoRoot ".venv\.text2model-forge-pyproject.sha256"
    $installedHash = if (Test-Path -LiteralPath $marker) { (Get-Content -Raw $marker).Trim() } else { "" }
    $importWorks = $false
    try {
        & $script:VenvPython -c "import text2model_forge; from text2model_forge import sprites" 2>$null
        $importWorks = $LASTEXITCODE -eq 0
    }
    catch { $importWorks = $false }
    if ($Action -eq "repair" -or -not $importWorks -or $installedHash -ne $installFingerprint) {
        Write-Host "Installing the editable project and its runtime dependencies..."
        Invoke-WithRetry "pip bootstrap" {
            Invoke-Native $script:VenvPython @(
                "-m", "pip", "install", "--upgrade",
                "pip==$PipVersion", "setuptools==$SetuptoolsVersion"
            )
        }
        Push-Location $script:RepoRoot
        try {
            Invoke-WithRetry "Text2Model Forge locked dependency install" {
                Invoke-Native $script:VenvPython @(
                    "-m", "pip", "install", "--require-hashes", "-r", $lockPath
                )
            }
            Invoke-WithRetry "Text2Model Forge editable install" {
                Invoke-Native $script:VenvPython @(
                    "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."
                )
            }
        }
        finally { Pop-Location }
        [IO.File]::WriteAllText($marker, $installFingerprint, (New-Object Text.UTF8Encoding($false)))
    }
    else {
        Write-Host "Core environment is current; no reinstall needed." -ForegroundColor Green
    }
}

function Find-Ollama {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    return $null
}

function Install-Ollama {
    $ollama = Find-Ollama
    if ($ollama) { return $ollama }
    if (Invoke-WingetInstall "Ollama.Ollama" "Ollama") {
        Refresh-ProcessPath
        $ollama = Find-Ollama
    }
    if (-not $ollama) {
        Write-Warning "WinGet methods did not install Ollama; opening the official per-user installer."
        $installer = Join-Path $RuntimeRoot "downloads\OllamaSetup.exe"
        Save-Download $OllamaInstallerUrl $installer "Ollama"
        Invoke-Native $installer @()
        Refresh-ProcessPath
        $ollama = Find-Ollama
    }
    if (-not $ollama) { throw "Ollama could not be installed or discovered." }
    return $ollama
}

function Test-Url {
    param([string]$Url, [int]$TimeoutSeconds = 2)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSeconds
        return $response.StatusCode -eq 200
    }
    catch { return $false }
}

function Wait-Url {
    param([string]$Url, [string]$Label, [int]$Seconds = 45)
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Url $Url) { return $true }
        Start-Sleep -Seconds 1
    }
    Write-Warning "$Label did not become ready at $Url within $Seconds seconds."
    return $false
}

function Start-Ollama {
    param([string]$Ollama)
    if (Test-Url "http://127.0.0.1:11434/api/tags") { return }
    $logRoot = Join-Path $RuntimeRoot "logs"
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    Start-Process -FilePath $Ollama -ArgumentList @("serve") -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot "ollama.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "ollama.stderr.log") | Out-Null
    [void](Wait-Url "http://127.0.0.1:11434/api/tags" "Ollama")
}

function Ensure-Reviewer {
    Write-Step "Installing and starting the local Qwen reviewer"
    $ollama = Install-Ollama
    Start-Ollama $ollama
    Invoke-WithRetry "reviewer model pull" {
        Invoke-Native $ollama @("pull", $ReviewerModel)
    }
}

function Resolve-SevenZip {
    $command = Get-Command 7z -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidate = Join-Path $env:ProgramFiles "7-Zip\7z.exe"
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    if (Invoke-WingetInstall "7zip.7zip" "7-Zip") {
        Refresh-ProcessPath
        $command = Get-Command 7z -ErrorAction SilentlyContinue
        if ($command) { return $command.Source }
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "7-Zip is required to extract the official ComfyUI portable archive."
}

function Ensure-ComfyUI {
    Write-Step "Checking the ComfyUI runtime"
    $comfyPython = Join-Path $script:ComfyRoot "python_embeded\python.exe"
    $comfyMain = Join-Path $script:ComfyRoot "ComfyUI\main.py"
    if ((Test-Path -LiteralPath $comfyPython) -and (Test-Path -LiteralPath $comfyMain)) {
        Write-Host "ComfyUI portable is already installed." -ForegroundColor Green
        return
    }
    $archive = Join-Path $RuntimeRoot "downloads\ComfyUI_windows_portable_nvidia.7z"
    Save-Download $ComfyArchiveUrl $archive "ComfyUI portable"
    $sevenZip = Resolve-SevenZip
    New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
    Invoke-WithRetry "ComfyUI extraction" {
        Invoke-Native $sevenZip @("x", $archive, "-o$RuntimeRoot", "-y")
    }
    if (-not ((Test-Path -LiteralPath $comfyPython) -and (Test-Path -LiteralPath $comfyMain))) {
        throw "The ComfyUI archive extracted without the expected portable folder at $script:ComfyRoot."
    }
}

function Start-ComfyUI {
    if (Test-Url "http://127.0.0.1:8188/system_stats") { return }
    $python = Join-Path $script:ComfyRoot "python_embeded\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Warning "No managed ComfyUI runtime is installed; expecting an existing service at port 8188."
        return
    }
    $logRoot = Join-Path $RuntimeRoot "logs"
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    $arguments = @("-s", "ComfyUI\main.py", "--listen", "127.0.0.1", "--port", "8188", "--windows-standalone-build")
    if ($Gpu -in @("auto", "nvidia")) {
        $arguments += @("--normalvram", "--reserve-vram", "0.75", "--preview-method", "none")
    }
    elseif ($Gpu -eq "cpu") { $arguments += "--cpu" }
    Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $script:ComfyRoot `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logRoot "comfyui.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "comfyui.stderr.log") | Out-Null
    [void](Wait-Url "http://127.0.0.1:8188/system_stats" "ComfyUI" 90)
}

function Confirm-License {
    param([string]$Name, [string]$Url, [bool]$AlreadyAccepted)
    if ($AlreadyAccepted) { return $true }
    if ($NonInteractive) {
        Write-Warning "$Name was not downloaded. Review $Url and rerun with its acceptance switch."
        return $false
    }
    $answer = Read-Host "$Name has separate terms at $Url. Type YES to accept them and download the model"
    return $answer -ceq "YES"
}

function Ensure-AiModels {
    param([string]$SelectedStack)
    Write-Step "Installing selected text-to-2D and image-to-3D models"
    New-Item -ItemType Directory -Force -Path $script:ComfyModels | Out-Null
    if ($SelectedStack -eq "qwen") {
        Invoke-WithRetry "Qwen Image 2512 model install" {
            Invoke-Native $script:VenvPython @(
                (Join-Path $script:RepoRoot "resources\adapters\install_qwen_image_edit_models.py"),
                "--models-root", $script:ComfyModels, "--profile", "image-2512"
            )
        }
    }
    $sdxlLicense = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md"
    if (Confirm-License "SDXL 1.0" $sdxlLicense ([bool]$AcceptSdxlLicense)) {
        $checkpoint = Join-Path $script:ComfyModels "checkpoints\sd_xl_base_1.0.safetensors"
        Save-Download $SdxlUrl $checkpoint "SDXL 1.0 checkpoint"
    }
    else {
        Write-Warning "SDXL was skipped. SDXL concept generation and the D8 ComfyUI surface pass need an installed checkpoint."
    }
    if (-not $SkipImageTo3D) {
        $hunyuanLicense = "https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE"
        if (Confirm-License "Hunyuan3D-2" $hunyuanLicense ([bool]$AcceptHunyuanLicense)) {
            $hunyuan = Join-Path $script:ComfyModels "checkpoints\hunyuan3d-dit-v2_fp16.safetensors"
            Save-Download $HunyuanUrl $hunyuan "Hunyuan3D-2 checkpoint"
        }
        else {
            Write-Warning "Hunyuan3D was skipped. D2 needs this checkpoint or a separately configured 3D worker."
        }
    }
}

function Find-Blender {
    $command = Get-Command blender -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $root = Join-Path $env:ProgramFiles "Blender Foundation"
    if (Test-Path -LiteralPath $root) {
        $candidate = Get-ChildItem -LiteralPath $root -Filter blender.exe -Recurse -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }
    return $null
}

function Ensure-Blender {
    Write-Step "Checking Blender"
    $blender = Find-Blender
    if (-not $blender) {
        if (-not (Invoke-WingetInstall "BlenderFoundation.Blender" "Blender")) {
            Write-Warning "Blender was not installed. Studio starts, but Blender-backed D3-D10 stages remain unavailable."
            return $null
        }
        Refresh-ProcessPath
        $blender = Find-Blender
    }
    if ($blender) { Write-Host "Blender: $blender" -ForegroundColor Green }
    return $blender
}

function ConvertTo-TomlString {
    param([string]$Value)
    $portable = $Value.Replace("\", "/").Replace('"', '\"')
    return '"' + $portable + '"'
}

function Ensure-LocalConfig {
    param([string]$SelectedStack, [string]$Blender)
    Write-Step "Checking machine-local configuration"
    $path = Join-Path $script:RepoRoot "config.local.toml"
    if (Test-Path -LiteralPath $path) {
        Write-Host "Keeping the existing gitignored config.local.toml unchanged." -ForegroundColor Green
        return
    }
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Generated once by text2model-forge.ps1. Machine-local and gitignored.")
    $lines.Add("schema_version = 1")
    $lines.Add("workspace_root = $(ConvertTo-TomlString $Workspace)")
    if ($SelectedStack -in @("qwen", "sdxl")) {
        $backend = if ($SelectedStack -eq "qwen") { "qwen_image_2512" } else { "sdxl" }
        $lines.Add("")
        $lines.Add("[studio_defaults]")
        $lines.Add("model = $(ConvertTo-TomlString $ReviewerModel)")
        $lines.Add('localdeploy_url = "http://127.0.0.1:11434/v1"')
        $lines.Add('comfy_url = "http://127.0.0.1:8188"')
        $lines.Add("concept_backend = $(ConvertTo-TomlString $backend)")
        $lines.Add('checkpoint = "sd_xl_base_1.0.safetensors"')
        $lines.Add('style_lora = ""')
        $lines.Add('spec_strategy = "chunked"')
        $lines.Add("llm_timeout_seconds = 600")
    }
    if ($Blender) {
        $adapter = Join-Path $script:RepoRoot "resources\adapters\blender_worker.py"
        $lines.Add("")
        $lines.Add("[workers.blender]")
        $lines.Add("command_prefix = [")
        $lines.Add("  $(ConvertTo-TomlString $Blender),")
        foreach ($argument in @("--background", "--factory-startup", "--offline-mode", "--python-exit-code", "23", "--python", $adapter, "--")) {
            $lines.Add("  $(ConvertTo-TomlString $argument),")
        }
        $lines.Add("]")
        $lines.Add("[workers.blender.environment]")
    }
    $canonical = Join-Path $script:RepoRoot "resources\adapters\canonical_short_biped_worker.py"
    $lines.Add("")
    $lines.Add('[workers."canonical.short_biped"]')
    $lines.Add("command_prefix = [$(ConvertTo-TomlString $script:VenvPython), $(ConvertTo-TomlString $canonical)]")
    $lines.Add('[workers."canonical.short_biped".environment]')
    [IO.File]::WriteAllText($path, (($lines -join "`n") + "`n"), (New-Object Text.UTF8Encoding($false)))
    Write-Host "Created $path. Existing files are never overwritten by this launcher." -ForegroundColor Green
}

function Resolve-AiStack {
    if ($AiStack -ne "auto") { return $AiStack }
    if ($NonInteractive) {
        if ((Find-Ollama) -or (Test-Path -LiteralPath $script:ComfyRoot)) { return "existing" }
        return "core"
    }
    Write-Host "`nChoose the local AI setup:" -ForegroundColor Cyan
    Write-Host "  1. Full Qwen stack (recommended): Qwen reviewer + Qwen Image + SDXL surface model + Hunyuan3D"
    Write-Host "  2. Full SDXL stack: Qwen reviewer + SDXL + Hunyuan3D"
    Write-Host "  3. Use services/models already installed on this machine"
    Write-Host "  4. Core Studio only (demo/control plane; no live generation)"
    $choice = Read-Host "Selection [1]"
    switch ($choice) {
        "2" { return "sdxl" }
        "3" { return "existing" }
        "4" { return "core" }
        default { return "qwen" }
    }
}

function Show-Doctor {
    Write-Host "`nText2Model Forge doctor" -ForegroundColor Cyan
    $checks = @(
        [pscustomobject]@{ Component = "Python 3.12+"; Ready = [bool](Find-Python); Detail = (Find-Python) },
        [pscustomobject]@{ Component = "Project environment"; Ready = (Test-PythonExecutable $script:VenvPython); Detail = $script:VenvPython },
        [pscustomobject]@{ Component = "Machine config"; Ready = (Test-Path -LiteralPath (Join-Path $script:RepoRoot "config.local.toml")); Detail = (Join-Path $script:RepoRoot "config.local.toml") },
        [pscustomobject]@{ Component = "Ollama reviewer"; Ready = (Test-Url "http://127.0.0.1:11434/api/tags"); Detail = "http://127.0.0.1:11434" },
        [pscustomobject]@{ Component = "ComfyUI"; Ready = (Test-Url "http://127.0.0.1:8188/system_stats"); Detail = "http://127.0.0.1:8188" },
        [pscustomobject]@{ Component = "Blender"; Ready = [bool](Find-Blender); Detail = (Find-Blender) }
    )
    $checks | Format-Table -AutoSize | Out-Host
    if (Test-PythonExecutable $script:VenvPython) {
        Write-Host "`nTyped worker preflight:" -ForegroundColor Cyan
        try {
            $workerJson = (& $script:VenvPython -m text2model_forge workers) -join "`n"
            $workerReport = $workerJson | ConvertFrom-Json
            $workerRows = foreach ($property in $workerReport.PSObject.Properties) {
                $item = $property.Value
                $detail = if ($item.executable) {
                    [string]$item.executable
                }
                elseif ($item.health_error) {
                    [string]$item.health_error
                }
                else {
                    [string]($item.blockers | Select-Object -First 1)
                }
                [pscustomobject]@{
                    Worker = $property.Name
                    Ready = [bool]$item.ready
                    State = [string]$item.declared_lifecycle
                    Detail = $detail
                }
            }
            $workerRows | Format-Table -AutoSize -Wrap | Out-Host
        }
        catch {
            Write-Warning "Worker preflight could not be summarized: $($_.Exception.Message)"
        }
    }
    return -not ($checks.Ready -contains $false)
}

try {
    if ($Action -eq "doctor") {
        [void](Show-Doctor)
        exit 0
    }

    $selectedStack = Resolve-AiStack
    $fullStack = $selectedStack -in @("qwen", "sdxl")
    Enter-InstallLock
    Assert-InstallFreeSpace -Path $script:RepoRoot -RequiredGB $(if ($fullStack) { 25 } else { 3 })
    $installOnly = $Action -in @("install", "repair")
    $script:TotalSteps = if ($fullStack) { if ($installOnly) { 8 } else { 9 } } else { if ($installOnly) { 5 } else { 6 } }
    Write-Host "Text2Model Forge action: $Action; AI stack: $selectedStack; workspace: $Workspace" -ForegroundColor White
    Ensure-Core $selectedStack
    New-Item -ItemType Directory -Force -Path $Workspace | Out-Null

    $blender = $null
    if ($selectedStack -in @("qwen", "sdxl")) {
        Ensure-Reviewer
        Ensure-ComfyUI
        Ensure-AiModels $selectedStack
        $blender = Ensure-Blender
    }
    elseif ($selectedStack -eq "existing") {
        Write-Step "Using existing local AI services"
        $blender = Find-Blender
    }
    else {
        Write-Step "Skipping optional local AI installation"
        $blender = Find-Blender
    }
    Ensure-LocalConfig $selectedStack $blender

    Write-Step "Running the deterministic offline smoke test"
    $smokeBase = [IO.Path]::GetFullPath((Join-Path $RuntimeRoot "launcher-smoke"))
    $smokeRoot = [IO.Path]::GetFullPath((Join-Path $smokeBase ([guid]::NewGuid().ToString("N"))))
    New-Item -ItemType Directory -Force -Path $smokeRoot | Out-Null
    try {
        Invoke-Native $script:VenvPython @("-m", "text2model_forge", "demo", "--workspace", $smokeRoot, "--run-id", "launcher.demo.v1")
    }
    finally {
        $safePrefix = $smokeBase.TrimEnd("\") + "\"
        if ($smokeRoot.StartsWith($safePrefix, [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $smokeRoot)) {
            Remove-Item -LiteralPath $smokeRoot -Recurse -Force
        }
    }

    if ($Action -eq "install" -or $Action -eq "repair") {
        Write-Step "Installation complete"
        [void](Show-Doctor)
        Write-Progress -Id 1 -Activity "Text2Model Forge setup" -Completed
        Write-Host "`nInstall complete. Run .\text2model-forge.ps1 to start Studio." -ForegroundColor Green
        Complete-Install
        exit 0
    }

    Complete-Install
    Write-Step "Starting local AI services"
    if ($selectedStack -in @("qwen", "sdxl", "existing")) {
        $ollama = Find-Ollama
        if ($ollama) { Start-Ollama $ollama }
        Start-ComfyUI
        if (-not $NoBrowser -and (Test-Url "http://127.0.0.1:8188/system_stats")) {
            Start-Process "http://127.0.0.1:8188" | Out-Null
        }
    }

    Write-Step "Starting Text2Model Forge Studio"
    Write-Progress -Id 1 -Activity "Text2Model Forge setup" -Completed
    $studioArguments = @("-m", "text2model_forge", "studio", "--workspace", $Workspace)
    if (-not $NoBrowser) { $studioArguments += "--open-browser" }
    Invoke-Native $script:VenvPython $studioArguments
}
catch {
    Write-InstallFailure $_
    Exit-InstallLock
    Write-Progress -Id 1 -Activity "Text2Model Forge setup" -Completed
    Write-Host "`nText2Model Forge could not finish: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Run .\text2model-forge.ps1 -Action doctor for a readiness report." -ForegroundColor Yellow
    exit 1
}
