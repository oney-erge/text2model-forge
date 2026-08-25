from pathlib import Path
import shutil
import subprocess
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = ROOT / "text2model-forge.ps1"
SHELL = ROOT / "text2model-forge.sh"
RUN_PS1 = ROOT / "run.ps1"
RUN_SH = ROOT / "run.sh"


def test_windows_launcher_exposes_bounded_install_start_and_ai_choices() -> None:
    source = POWERSHELL.read_text(encoding="utf-8")
    assert '"start", "install", "repair", "doctor"' in source
    assert '"auto", "qwen", "sdxl", "existing", "core"' in source
    assert "Invoke-WithRetry" in source
    assert '"--scope", "user"' in source
    assert "-Verb RunAs" in source
    assert "$NoElevation" in source
    assert "Write-Progress" in source
    assert "local-ai" in source
    assert '"requirements-all.lock"' in source
    assert '"--require-hashes"' in source
    assert '"--no-build-isolation"' in source
    assert "qwen_image_edit_models.py" in source
    assert "hunyuan3d-dit-v2_fp16.safetensors" in source
    assert "sd_xl_base_1.0.safetensors" in source
    assert "Keeping the existing gitignored config.local.toml unchanged" in source


def test_unix_launcher_has_native_linux_macos_install_start_and_repair_paths() -> None:
    source = SHELL.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/env bash")
    assert "Linux|Darwin" in source
    assert "start|install|repair|doctor" in source
    assert "auto|qwen|sdxl|existing|core" in source
    assert "with_retry" in source
    assert "UV_NO_MODIFY_PATH=1" in source
    assert 'UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${RUNTIME_ROOT}/python}"' in source
    assert "run_admin" in source and "sudo -n" in source
    assert "--no-elevation" in source
    assert "ComfyUI.git" in source
    assert "download.pytorch.org/whl/cu130" in source
    assert "qwen_image_edit_models.py" in source
    assert 'lock_name="requirements-all.lock"' in source
    assert "--require-hashes" in source
    assert "--no-build-isolation" in source
    assert "hunyuan3d-dit-v2_fp16.safetensors" in source
    assert "Keeping the existing gitignored config.local.toml unchanged" in source
    assert not (ROOT / "text2model-forge.cmd").exists()


def test_dependency_manifests_share_pyproject_as_the_source_of_truth() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "huggingface-hub>=0.27,<2" in project["project"]["optional-dependencies"]["local-ai"]
    assert (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()[-1] == "-e ."
    assert (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()[-1] == "-e .[dev]"
    assert (
        ROOT / "requirements-local-ai.txt"
    ).read_text(encoding="utf-8").splitlines()[-1] == "-e .[local-ai]"
    for filename in (
        "requirements.lock",
        "requirements-dev.lock",
        "requirements-local-ai.lock",
        "requirements-all.lock",
    ):
        lock = (ROOT / filename).read_text(encoding="utf-8")
        assert "--hash=sha256:" in lock
        assert "--no-emit-project" in lock


def test_docker_setup_is_local_only_persistent_and_uses_typed_config() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    nvidia = (ROOT / "compose.nvidia.yaml").read_text(encoding="utf-8")
    config = tomllib.loads((ROOT / "docker" / "config.local.toml").read_text(encoding="utf-8"))

    assert "USER text2model" in dockerfile
    assert "--require-hashes -r requirements.lock" in dockerfile
    assert '"--allow-non-loopback"' in dockerfile
    assert '127.0.0.1:${TEXT2MODEL_FORGE_PORT:-8766}:8766' in compose
    assert "workspace:/workspace" in compose
    assert 'profiles: ["models"]' in compose
    assert "condition: service_healthy" in compose
    assert "capabilities: [gpu]" in nvidia
    assert config["workspace_root"] == "/workspace"
    assert config["studio_defaults"]["localdeploy_url"] == "http://ollama:11434/v1"
    assert config["studio_defaults"]["comfy_url"] == "http://host.docker.internal:8188"


def test_run_wrapper_scripts_forward_to_the_real_launchers_with_a_fast_default_stack() -> None:
    """run.ps1/run.sh exist so this repo is discoverable the same way as the
    account's other repos (MetaScout, Agentarium both use run.ps1/run.sh).
    They must not duplicate installer logic -- they are a thin front door
    that defaults to the fast core stack and delegates everything else to
    the real, already-tested launcher."""
    ps1 = RUN_PS1.read_text(encoding="utf-8")
    assert 'Join-Path $PSScriptRoot "text2model-forge.ps1"' in ps1
    assert '$forwardArgs.Insert($insertAt, "-AiStack")' in ps1
    assert '$forwardArgs.Insert($insertAt, "core")' in ps1
    assert "-contains \"-AiStack\"" in ps1
    assert "$forwardArgs" in ps1
    assert "docker compose up --detach --build" in ps1

    sh = RUN_SH.read_text(encoding="utf-8")
    assert sh.startswith("#!/usr/bin/env bash")
    assert "exec ./text2model-forge.sh" in sh
    assert "--ai-stack core" in sh
    assert '"$arg" = --ai-stack' in sh
    assert "docker compose up --detach --build" in sh


def test_run_ps1_wrapper_parses_when_powershell_is_available() -> None:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is not installed on this test host")
    path = str(RUN_PS1).replace("'", "''")
    command = (
        "$tokens=$null; $errors=$null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{path}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )
    subprocess.run(
        [executable, "-NoLogo", "-NoProfile", "-Command", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_run_sh_wrapper_parses_when_a_real_bash_is_available() -> None:
    candidates = [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
    ]
    discovered = shutil.which("bash")
    if discovered:
        candidates.append(Path(discovered))
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        pytest.skip("Bash is not installed on this test host")
    subprocess.run(
        [str(executable), "-n", str(RUN_SH)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_powershell_launcher_parses_when_powershell_is_available() -> None:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is not installed on this test host")
    path = str(POWERSHELL).replace("'", "''")
    command = (
        "$tokens=$null; $errors=$null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{path}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )
    subprocess.run(
        [executable, "-NoLogo", "-NoProfile", "-Command", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_unix_launcher_parses_when_a_real_bash_is_available() -> None:
    candidates = [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
    ]
    discovered = shutil.which("bash")
    if discovered:
        candidates.append(Path(discovered))
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        pytest.skip("Bash is not installed on this test host")
    subprocess.run(
        [str(executable), "-n", str(SHELL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
