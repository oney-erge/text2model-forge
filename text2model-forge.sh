#!/usr/bin/env bash
# Native Linux/macOS installer, repair tool, doctor, and launcher for Text2Model Forge.

set -Eeuo pipefail
umask 022

REPO_ROOT="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${REPO_ROOT}/scripts/install-utils.sh"
install_init "$REPO_ROOT" "Asset Forge"
install_enable_traps
ACTION="start"
AI_STACK="auto"
GPU="auto"
WORKSPACE="${TEXT2MODEL_FORGE_WORKSPACE:-${HOME}/Text2ModelForgeRuns}"
RUNTIME_ROOT="${TEXT2MODEL_FORGE_RUNTIME_ROOT:-${REPO_ROOT}/runtime}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${RUNTIME_ROOT}/python}"
REVIEWER_MODEL="qwen3-vl:8b-instruct"
MAX_ATTEMPTS=3
ACCEPT_SDXL_LICENSE=0
ACCEPT_HUNYUAN_LICENSE=0
SKIP_IMAGE_TO_3D=0
INCLUDE_DEV=0
NON_INTERACTIVE=0
NO_ELEVATION=0
NO_BROWSER=0
STEP=0
TOTAL_STEPS=9
PYTHON_BIN=""
UV_BIN=""
BLENDER_BIN=""

VENV_ROOT="${REPO_ROOT}/.venv"
VENV_PYTHON="${VENV_ROOT}/bin/python"
COMFY_ROOT="${RUNTIME_ROOT}/ComfyUI"
COMFY_PYTHON="${COMFY_ROOT}/.venv/bin/python"
COMFY_MODELS="${COMFY_ROOT}/models"
UV_VERSION="0.11.32"
PIP_VERSION="26.2.1"
SETUPTOOLS_VERSION="84.0.0"
UV_INSTALLER_URL="https://astral.sh/uv/${UV_VERSION}/install.sh"
OLLAMA_INSTALLER_URL="https://ollama.com/install.sh"
SDXL_URL="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
HUNYUAN_URL="https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/hunyuan3d-dit-v2_fp16.safetensors"

usage() {
    cat <<'EOF'
Text2Model Forge setup and launcher for Linux and macOS

Usage: ./text2model-forge.sh [start|install|repair|doctor] [options]

Options:
  --action ACTION             start, install, repair, or doctor
  --ai-stack STACK            auto, qwen, sdxl, existing, or core
  --gpu MODE                  auto, nvidia, or cpu
  --workspace PATH            run workspace (default: ~/Text2ModelForgeRuns)
  --runtime-root PATH         managed tools/models root (default: ./runtime)
  --reviewer-model MODEL      Ollama reviewer model
  --max-attempts N            bounded retry count, 1-5 (default: 3)
  --accept-sdxl-license       accept SDXL's separate model terms
  --accept-hunyuan-license    accept Hunyuan3D-2's separate model terms
  --skip-image-to-3d          do not download Hunyuan3D-2
  --include-dev               install test/development dependencies
  --non-interactive           never prompt; skipped licenses need explicit flags
  --no-elevation              never invoke sudo
  --no-browser                do not open ComfyUI or Studio browser tabs
  -h, --help                  show this help

Running start is idempotent: it installs or repairs missing pieces, then starts.
EOF
}

die() {
    printf '\nText2Model Forge could not finish: %s\n' "$*" >&2
    printf 'Run ./text2model-forge.sh doctor for a readiness report.\n' >&2
    exit 1
}

on_error() {
    local exit_code=$?
    local line_number=${1:-unknown}
    printf '\nText2Model Forge stopped at line %s (exit %s).\n' "$line_number" "$exit_code" >&2
    printf 'Run ./text2model-forge.sh doctor for a readiness report.\n' >&2
    install_note "failure exit=$exit_code line=$line_number"
    install_unlock
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

need_value() {
    [[ $# -ge 2 && -n ${2:-} ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        start|install|repair|doctor) ACTION="$1"; shift ;;
        --action) need_value "$@"; ACTION="$2"; shift 2 ;;
        --ai-stack) need_value "$@"; AI_STACK="$2"; shift 2 ;;
        --gpu) need_value "$@"; GPU="$2"; shift 2 ;;
        --workspace) need_value "$@"; WORKSPACE="$2"; shift 2 ;;
        --runtime-root) need_value "$@"; RUNTIME_ROOT="$2"; shift 2 ;;
        --reviewer-model) need_value "$@"; REVIEWER_MODEL="$2"; shift 2 ;;
        --max-attempts) need_value "$@"; MAX_ATTEMPTS="$2"; shift 2 ;;
        --accept-sdxl-license) ACCEPT_SDXL_LICENSE=1; shift ;;
        --accept-hunyuan-license) ACCEPT_HUNYUAN_LICENSE=1; shift ;;
        --skip-image-to-3d) SKIP_IMAGE_TO_3D=1; shift ;;
        --include-dev) INCLUDE_DEV=1; shift ;;
        --non-interactive) NON_INTERACTIVE=1; shift ;;
        --no-elevation) NO_ELEVATION=1; shift ;;
        --no-browser) NO_BROWSER=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$ACTION" in start|install|repair|doctor) ;; *) die "invalid action: $ACTION" ;; esac
case "$AI_STACK" in auto|qwen|sdxl|existing|core) ;; *) die "invalid AI stack: $AI_STACK" ;; esac
case "$GPU" in auto|nvidia|cpu) ;; *) die "invalid GPU mode: $GPU" ;; esac
[[ "$MAX_ATTEMPTS" =~ ^[1-5]$ ]] || die "--max-attempts must be an integer from 1 through 5"

OS_NAME="$(uname -s)"
case "$OS_NAME" in
    Linux|Darwin) ;;
    *) die "this launcher supports Linux and macOS; on Windows run .\\text2model-forge.ps1" ;;
esac

# Re-resolve paths after argument parsing. mkdir + cd avoids a dependency on realpath.
mkdir -p -- "$WORKSPACE" "$RUNTIME_ROOT"
WORKSPACE="$(CDPATH='' cd -- "$WORKSPACE" && pwd -P)"
RUNTIME_ROOT="$(CDPATH='' cd -- "$RUNTIME_ROOT" && pwd -P)"
COMFY_ROOT="${RUNTIME_ROOT}/ComfyUI"
COMFY_PYTHON="${COMFY_ROOT}/.venv/bin/python"
COMFY_MODELS="${COMFY_ROOT}/models"

write_step() {
    local activity=$1
    local percent filled empty bar=""
    STEP=$((STEP + 1))
    percent=$((STEP * 100 / TOTAL_STEPS))
    (( percent > 100 )) && percent=100
    filled=$((percent / 5))
    empty=$((20 - filled))
    while (( filled-- > 0 )); do bar+="#"; done
    while (( empty-- > 0 )); do bar+="-"; done
    printf '\n[%s] %3s%%  (%s/%s) %s\n' "$bar" "$percent" "$STEP" "$TOTAL_STEPS" "$activity"
}

with_retry() {
    local label=$1
    shift
    INSTALL_RETRY_ATTEMPTS="$MAX_ATTEMPTS" install_retry "$label" "$@"
}

download_file() {
    local url=$1 destination=$2 label=$3
    if [[ -f "$destination" ]]; then
        printf '%s is already downloaded.\n' "$label"
        return 0
    fi
    install_download "$url" "$destination" "$label"
}

python_is_compatible() {
    [[ -x ${1:-} ]] || return 1
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1
}

find_python() {
    local name candidate managed_uv
    for name in python3.13 python3.12 python3 python; do
        candidate="$(command -v "$name" 2>/dev/null || true)"
        if python_is_compatible "$candidate"; then
            PYTHON_BIN="$candidate"
            return 0
        fi
    done
    managed_uv="${RUNTIME_ROOT}/bin/uv"
    if [[ -x "$managed_uv" ]]; then
        candidate="$($managed_uv python find 3.12 2>/dev/null || true)"
        if python_is_compatible "$candidate"; then
            PYTHON_BIN="$candidate"
            UV_BIN="$managed_uv"
            return 0
        fi
    fi
    return 1
}

ensure_uv() {
    local candidate installer
    candidate="$(command -v uv 2>/dev/null || true)"
    if [[ -x "$candidate" ]]; then
        UV_BIN="$candidate"
        return 0
    fi
    candidate="${RUNTIME_ROOT}/bin/uv"
    if [[ -x "$candidate" ]]; then
        UV_BIN="$candidate"
        return 0
    fi
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        printf 'No downloader was found; trying the operating-system package manager for curl.\n' >&2
        install_system_package "curl" curl curl curl curl || return 1
    fi
    installer="${RUNTIME_ROOT}/downloads/uv-${UV_VERSION}-install.sh"
    if ! download_file "$UV_INSTALLER_URL" "$installer" "uv ${UV_VERSION} installer"; then
        return 1
    fi
    if ! with_retry "per-user uv install" env \
        UV_INSTALL_DIR="${RUNTIME_ROOT}/bin" UV_NO_MODIFY_PATH=1 sh "$installer"; then
        return 1
    fi
    [[ -x "$candidate" ]] || return 1
    UV_BIN="$candidate"
}

ensure_python() {
    if find_python; then
        return 0
    fi
    printf 'Python 3.12+ was not found; trying the pinned per-user uv bootstrap.\n'
    if ensure_uv; then
        if with_retry "Python 3.12 install" "$UV_BIN" python install 3.12; then
            PYTHON_BIN="$($UV_BIN python find 3.12)"
            python_is_compatible "$PYTHON_BIN" && return 0
        fi
    fi
    printf 'The per-user Python bootstrap failed; trying the operating-system package manager.\n' >&2
    install_system_package "Python 3.12" python@3.12 python3.12 python3 python || \
        install_system_package "Python 3" python python3 python3 python || return 1
    find_python
}

run_admin() {
    if [[ $(id -u) -eq 0 ]]; then
        "$@"
    elif (( NO_ELEVATION )); then
        return 1
    elif command -v sudo >/dev/null 2>&1; then
        if (( NON_INTERACTIVE )); then
            sudo -n "$@"
        else
            sudo "$@"
        fi
    else
        return 1
    fi
}

install_system_package() {
    local label=$1 brew_name=$2 apt_name=$3 dnf_name=$4 pacman_name=$5
    if command -v brew >/dev/null 2>&1 && [[ -n "$brew_name" ]]; then
        with_retry "$label Homebrew install" brew install "$brew_name" && return 0
    fi
    if command -v apt-get >/dev/null 2>&1 && [[ -n "$apt_name" ]]; then
        with_retry "$label apt metadata refresh" run_admin apt-get update && \
            with_retry "$label apt install" run_admin apt-get install -y "$apt_name" && return 0
    fi
    if command -v dnf >/dev/null 2>&1 && [[ -n "$dnf_name" ]]; then
        with_retry "$label dnf install" run_admin dnf install -y "$dnf_name" && return 0
    fi
    if command -v pacman >/dev/null 2>&1 && [[ -n "$pacman_name" ]]; then
        with_retry "$label pacman install" run_admin pacman -Sy --needed --noconfirm "$pacman_name" && return 0
    fi
    return 1
}

ensure_core() {
    local selected_stack=$1 lock_name lock_path project_hash lock_hash fingerprint marker installed="" import_works=0
    write_step "Checking Python and the Text2Model Forge environment"
    ensure_python || die "Python 3.12+ could not be installed or discovered"

    if ! python_is_compatible "$VENV_PYTHON"; then
        if ! with_retry "native virtual environment creation" \
            "$PYTHON_BIN" -m venv --clear "$VENV_ROOT"; then
            if ensure_uv; then
                with_retry "uv virtual environment creation" \
                    "$UV_BIN" venv --clear --python "$PYTHON_BIN" "$VENV_ROOT"
            else
                install_system_package "Python venv support" "" python3-venv python3 "" || \
                    die "Python's venv module failed and neither the per-user nor administrator fallback succeeded"
                with_retry "virtual environment creation after venv package install" \
                    "$PYTHON_BIN" -m venv --clear "$VENV_ROOT"
            fi
        fi
    fi

    if [[ "$selected_stack" == qwen || "$selected_stack" == sdxl ]]; then
        if (( INCLUDE_DEV )); then
            lock_name="requirements-all.lock"
        else
            lock_name="requirements-local-ai.lock"
        fi
    elif (( INCLUDE_DEV )); then
        lock_name="requirements-dev.lock"
    else
        lock_name="requirements.lock"
    fi
    lock_path="${REPO_ROOT}/${lock_name}"
    [[ -f "$lock_path" ]] || die "the committed dependency lock is missing: ${lock_path}"
    project_hash="$($VENV_PYTHON -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${REPO_ROOT}/pyproject.toml")"
    lock_hash="$($VENV_PYTHON -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$lock_path")"
    fingerprint="${project_hash}|${lock_hash}|${lock_name}"
    marker="${VENV_ROOT}/.text2model-forge-pyproject.sha256"
    [[ -f "$marker" ]] && installed="$(<"$marker")"
    "$VENV_PYTHON" -c 'import text2model_forge; from text2model_forge import sprites' >/dev/null 2>&1 && import_works=1

    if [[ "$ACTION" == repair || $import_works -eq 0 || "$installed" != "$fingerprint" ]]; then
        printf 'Installing the editable project and its locked dependencies...\n'
        with_retry "pip bootstrap" "$VENV_PYTHON" -m pip install --upgrade \
            "pip==${PIP_VERSION}" "setuptools==${SETUPTOOLS_VERSION}"
        (
            cd -- "$REPO_ROOT"
            with_retry "Text2Model Forge locked dependency install" "$VENV_PYTHON" -m pip install \
                --require-hashes -r "$lock_path"
            with_retry "Text2Model Forge editable install" "$VENV_PYTHON" -m pip install \
                --no-deps --no-build-isolation -e .
        )
        printf '%s' "$fingerprint" > "$marker"
    else
        printf 'Core environment is current; no reinstall needed.\n'
    fi
}

url_ready() {
    local url=$1
    if command -v curl >/dev/null 2>&1; then
        curl --fail --silent --show-error --max-time 2 --output /dev/null "$url" 2>/dev/null
    elif command -v wget >/dev/null 2>&1; then
        wget --quiet --timeout=2 --output-document=/dev/null "$url" 2>/dev/null
    else
        return 1
    fi
}

wait_url() {
    local url=$1 label=$2 seconds=${3:-45} elapsed=0
    while (( elapsed < seconds )); do
        url_ready "$url" && return 0
        sleep 1
        elapsed=$((elapsed + 1))
        printf '\rWaiting for %s... %ss/%ss' "$label" "$elapsed" "$seconds"
    done
    printf '\n%s did not become ready at %s within %ss.\n' "$label" "$url" "$seconds" >&2
    return 1
}

find_ollama() {
    local candidate
    candidate="$(command -v ollama 2>/dev/null || true)"
    [[ -x "$candidate" ]] && { printf '%s' "$candidate"; return 0; }
    for candidate in \
        "${RUNTIME_ROOT}/bin/ollama" \
        "/Applications/Ollama.app/Contents/Resources/ollama"; do
        [[ -x "$candidate" ]] && { printf '%s' "$candidate"; return 0; }
    done
    return 1
}

install_ollama() {
    local installer ollama_bin=""
    ollama_bin="$(find_ollama || true)"
    [[ -n "$ollama_bin" ]] && { printf '%s' "$ollama_bin"; return 0; }

    if command -v brew >/dev/null 2>&1; then
        with_retry "Ollama Homebrew install" brew install ollama >&2 || true
        ollama_bin="$(find_ollama || true)"
        [[ -n "$ollama_bin" ]] && { printf '%s' "$ollama_bin"; return 0; }
    fi
    if [[ "$OS_NAME" == Linux && $NO_ELEVATION -eq 1 ]]; then
        return 1
    fi
    installer="${RUNTIME_ROOT}/downloads/ollama-install.sh"
    download_file "$OLLAMA_INSTALLER_URL" "$installer" "official Ollama installer" >&2
    with_retry "official Ollama install" sh "$installer" >&2
    find_ollama
}

start_ollama() {
    local ollama_bin=$1
    url_ready "http://127.0.0.1:11434/api/tags" && return 0
    mkdir -p -- "${RUNTIME_ROOT}/logs"
    if [[ "$OS_NAME" == Darwin && -d /Applications/Ollama.app ]]; then
        open -a Ollama --args hidden >/dev/null 2>&1 || true
    elif command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet ollama 2>/dev/null; then
        :
    else
        nohup "$ollama_bin" serve >"${RUNTIME_ROOT}/logs/ollama.stdout.log" \
            2>"${RUNTIME_ROOT}/logs/ollama.stderr.log" </dev/null &
    fi
    wait_url "http://127.0.0.1:11434/api/tags" "Ollama" 45
}

ensure_reviewer() {
    local ollama_bin
    write_step "Installing and starting the local Qwen reviewer"
    ollama_bin="$(install_ollama)" || die "Ollama needs administrator access on this Linux host; rerun without --no-elevation, install Ollama yourself, or use Docker Compose"
    start_ollama "$ollama_bin"
    with_retry "reviewer model pull" "$ollama_bin" pull "$REVIEWER_MODEL"
}

ensure_git() {
    command -v git >/dev/null 2>&1 && return 0
    install_system_package "Git" git git git git || return 1
    command -v git >/dev/null 2>&1
}

detect_comfy_gpu() {
    if [[ "$GPU" == cpu ]]; then
        printf 'cpu'
    elif [[ "$GPU" == nvidia ]]; then
        printf 'nvidia'
    elif [[ "$OS_NAME" == Darwin && $(uname -m) == arm64 ]]; then
        printf 'apple'
    elif command -v nvidia-smi >/dev/null 2>&1; then
        printf 'nvidia'
    else
        printf 'cpu'
    fi
}

clone_comfy() {
    git clone --filter=blob:none https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"
}

ensure_comfyui() {
    local gpu_mode requirements_hash marker installed=""
    write_step "Checking the isolated ComfyUI runtime"
    if [[ ! -f "${COMFY_ROOT}/main.py" ]]; then
        ensure_git || die "Git is required to install ComfyUI and could not be installed"
        mkdir -p -- "$(dirname -- "$COMFY_ROOT")"
        with_retry "ComfyUI clone" clone_comfy
    fi
    if ! python_is_compatible "$COMFY_PYTHON"; then
        with_retry "ComfyUI virtual environment creation" "$PYTHON_BIN" -m venv --clear "${COMFY_ROOT}/.venv"
    fi
    gpu_mode="$(detect_comfy_gpu)"
    requirements_hash="$($VENV_PYTHON -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "${COMFY_ROOT}/requirements.txt")"
    marker="${COMFY_ROOT}/.venv/.text2model-forge-requirements.sha256"
    [[ -f "$marker" ]] && installed="$(<"$marker")"
    if [[ "$ACTION" == repair || "$installed" != "${requirements_hash}|${gpu_mode}" ]] || \
        ! "$COMFY_PYTHON" -c 'import torch' >/dev/null 2>&1; then
        with_retry "ComfyUI pip bootstrap" "$COMFY_PYTHON" -m pip install --upgrade pip
        case "$gpu_mode" in
            nvidia)
                with_retry "PyTorch CUDA install" "$COMFY_PYTHON" -m pip install \
                    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
                ;;
            cpu)
                with_retry "PyTorch CPU install" "$COMFY_PYTHON" -m pip install \
                    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
                ;;
            apple)
                with_retry "PyTorch Apple install" "$COMFY_PYTHON" -m pip install torch torchvision torchaudio
                ;;
        esac
        with_retry "ComfyUI dependency install" "$COMFY_PYTHON" -m pip install -r "${COMFY_ROOT}/requirements.txt"
        printf '%s' "${requirements_hash}|${gpu_mode}" > "$marker"
    else
        printf 'ComfyUI environment is current; no reinstall needed.\n'
    fi
}

start_comfyui() {
    local gpu_mode args=()
    url_ready "http://127.0.0.1:8188/system_stats" && return 0
    if [[ ! -x "$COMFY_PYTHON" || ! -f "${COMFY_ROOT}/main.py" ]]; then
        printf 'No managed ComfyUI runtime is installed; expecting an existing service on port 8188.\n' >&2
        return 1
    fi
    mkdir -p -- "${RUNTIME_ROOT}/logs"
    args=("${COMFY_ROOT}/main.py" --listen 127.0.0.1 --port 8188)
    gpu_mode="$(detect_comfy_gpu)"
    [[ "$gpu_mode" == nvidia ]] && args+=(--normalvram --reserve-vram 0.75 --preview-method none)
    [[ "$gpu_mode" == cpu ]] && args+=(--cpu)
    nohup "$COMFY_PYTHON" "${args[@]}" >"${RUNTIME_ROOT}/logs/comfyui.stdout.log" \
        2>"${RUNTIME_ROOT}/logs/comfyui.stderr.log" </dev/null &
    wait_url "http://127.0.0.1:8188/system_stats" "ComfyUI" 90
}

confirm_license() {
    local name=$1 url=$2 accepted=$3 answer
    (( accepted )) && return 0
    if (( NON_INTERACTIVE )); then
        printf '%s was not downloaded. Review %s and rerun with its acceptance flag.\n' "$name" "$url" >&2
        return 1
    fi
    printf '%s has separate terms at %s.\nType YES to accept them and download the model: ' "$name" "$url"
    read -r answer
    [[ "$answer" == YES ]]
}

ensure_ai_models() {
    local selected_stack=$1 checkpoint hunyuan
    write_step "Installing selected text-to-2D and image-to-3D models"
    mkdir -p -- "$COMFY_MODELS"
    if [[ "$selected_stack" == qwen ]]; then
        with_retry "Qwen Image 2512 model install" "$VENV_PYTHON" \
            "${REPO_ROOT}/resources/adapters/install_qwen_image_edit_models.py" \
            --models-root "$COMFY_MODELS" --profile image-2512
    fi
    if confirm_license "SDXL 1.0" \
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md" \
        "$ACCEPT_SDXL_LICENSE"; then
        checkpoint="${COMFY_MODELS}/checkpoints/sd_xl_base_1.0.safetensors"
        download_file "$SDXL_URL" "$checkpoint" "SDXL 1.0 checkpoint"
    else
        printf 'SDXL was skipped; SDXL concept generation and the D8 surface pass need an installed checkpoint.\n' >&2
    fi
    if (( ! SKIP_IMAGE_TO_3D )); then
        if confirm_license "Hunyuan3D-2" \
            "https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE" \
            "$ACCEPT_HUNYUAN_LICENSE"; then
            hunyuan="${COMFY_MODELS}/checkpoints/hunyuan3d-dit-v2_fp16.safetensors"
            download_file "$HUNYUAN_URL" "$hunyuan" "Hunyuan3D-2 checkpoint"
        else
            printf 'Hunyuan3D was skipped; D2 needs it or a separately configured 3D worker.\n' >&2
        fi
    fi
}

find_blender() {
    local candidate
    candidate="$(command -v blender 2>/dev/null || true)"
    [[ -x "$candidate" ]] && { printf '%s' "$candidate"; return 0; }
    candidate="/Applications/Blender.app/Contents/MacOS/Blender"
    [[ -x "$candidate" ]] && { printf '%s' "$candidate"; return 0; }
    return 1
}

ensure_blender() {
    write_step "Checking Blender"
    BLENDER_BIN="$(find_blender || true)"
    if [[ -z "$BLENDER_BIN" ]]; then
        if [[ "$OS_NAME" == Darwin && -n $(command -v brew 2>/dev/null || true) ]]; then
            with_retry "Blender Homebrew install" brew install --cask blender || true
        else
            install_system_package "Blender" "" blender blender blender || true
        fi
        BLENDER_BIN="$(find_blender || true)"
    fi
    if [[ -n "$BLENDER_BIN" ]]; then
        printf 'Blender: %s\n' "$BLENDER_BIN"
    else
        printf 'Blender was not installed; Blender-backed stages remain unavailable.\n' >&2
    fi
}

toml_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

ensure_local_config() {
    local selected_stack=$1 config_path="${REPO_ROOT}/config.local.toml" backend canonical adapter
    write_step "Checking machine-local configuration"
    if [[ -f "$config_path" ]]; then
        printf 'Keeping the existing gitignored config.local.toml unchanged.\n'
        return 0
    fi
    canonical="${REPO_ROOT}/resources/adapters/canonical_short_biped_worker.py"
    {
        printf '# Generated once by text2model-forge.sh. Machine-local and gitignored.\n'
        printf 'schema_version = 1\n'
        printf 'workspace_root = "%s"\n' "$(toml_escape "$WORKSPACE")"
        if [[ "$selected_stack" == qwen || "$selected_stack" == sdxl ]]; then
            [[ "$selected_stack" == qwen ]] && backend=qwen_image_2512 || backend=sdxl
            printf '\n[studio_defaults]\n'
            printf 'model = "%s"\n' "$(toml_escape "$REVIEWER_MODEL")"
            printf 'localdeploy_url = "http://127.0.0.1:11434/v1"\n'
            printf 'comfy_url = "http://127.0.0.1:8188"\n'
            printf 'concept_backend = "%s"\n' "$backend"
            printf 'checkpoint = "sd_xl_base_1.0.safetensors"\n'
            printf 'style_lora = ""\n'
            printf 'spec_strategy = "chunked"\n'
            printf 'llm_timeout_seconds = 600\n'
        fi
        if [[ -n "$BLENDER_BIN" ]]; then
            adapter="${REPO_ROOT}/resources/adapters/blender_worker.py"
            printf '\n[workers.blender]\ncommand_prefix = [\n'
            printf '  "%s",\n' "$(toml_escape "$BLENDER_BIN")"
            for argument in --background --factory-startup --offline-mode --python-exit-code 23 --python "$adapter" --; do
                printf '  "%s",\n' "$(toml_escape "$argument")"
            done
            printf ']\n[workers.blender.environment]\n'
        fi
        printf '\n[workers."canonical.short_biped"]\n'
        printf 'command_prefix = ["%s", "%s"]\n' \
            "$(toml_escape "$VENV_PYTHON")" "$(toml_escape "$canonical")"
        printf '[workers."canonical.short_biped".environment]\n'
    } > "$config_path"
    printf 'Created %s. Existing files are never overwritten by this launcher.\n' "$config_path"
}

resolve_ai_stack() {
    local choice
    [[ "$AI_STACK" != auto ]] && { printf '%s' "$AI_STACK"; return 0; }
    if (( NON_INTERACTIVE )); then
        if find_ollama >/dev/null 2>&1 || [[ -f "${COMFY_ROOT}/main.py" ]]; then
            printf 'existing'
        else
            printf 'core'
        fi
        return 0
    fi
    printf '\nChoose the local AI setup:\n' >&2
    printf '  1. Full Qwen stack (recommended): reviewer + Qwen Image + SDXL + Hunyuan3D + Blender\n' >&2
    printf '  2. Full SDXL stack: reviewer + SDXL + Hunyuan3D + Blender\n' >&2
    printf '  3. Use services/models already installed on this machine\n' >&2
    printf '  4. Core Studio only (demo/control plane; no live generation)\n' >&2
    printf 'Selection [1]: ' >&2
    read -r choice
    case "$choice" in 2) printf sdxl ;; 3) printf existing ;; 4) printf core ;; *) printf qwen ;; esac
}

doctor_row() {
    local component=$1 ready=$2 detail=$3 status=NO
    (( ready )) && status=YES
    printf '%-24s %-5s %s\n' "$component" "$status" "$detail"
}

show_doctor() {
    local found_python="" found_ollama="" found_blender="" ready
    printf '\nText2Model Forge doctor\n%-24s %-5s %s\n' COMPONENT READY DETAIL
    find_python && found_python="$PYTHON_BIN" || true
    doctor_row "Python 3.12+" "$([[ -n "$found_python" ]] && echo 1 || echo 0)" "${found_python:-not found}"
    doctor_row "Project environment" "$(python_is_compatible "$VENV_PYTHON" && echo 1 || echo 0)" "$VENV_PYTHON"
    doctor_row "Machine config" "$([[ -f "${REPO_ROOT}/config.local.toml" ]] && echo 1 || echo 0)" "${REPO_ROOT}/config.local.toml"
    found_ollama="$(find_ollama || true)"
    url_ready "http://127.0.0.1:11434/api/tags" && ready=1 || ready=0
    doctor_row "Ollama reviewer" "$ready" "${found_ollama:-http://127.0.0.1:11434}"
    url_ready "http://127.0.0.1:8188/system_stats" && ready=1 || ready=0
    doctor_row "ComfyUI" "$ready" "http://127.0.0.1:8188"
    found_blender="$(find_blender || true)"
    doctor_row "Blender" "$([[ -n "$found_blender" ]] && echo 1 || echo 0)" "${found_blender:-not found}"
    if python_is_compatible "$VENV_PYTHON"; then
        printf '\nTyped worker preflight:\n'
        "$VENV_PYTHON" -m text2model_forge workers || true
    fi
}

open_url() {
    local url=$1
    (( NO_BROWSER )) && return 0
    if [[ "$OS_NAME" == Darwin ]]; then
        open "$url" >/dev/null 2>&1 &
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 &
    elif command -v gio >/dev/null 2>&1; then
        gio open "$url" >/dev/null 2>&1 &
    fi
}

run_smoke() {
    local smoke_base="${RUNTIME_ROOT}/launcher-smoke" smoke_root result=0
    write_step "Running the deterministic offline smoke test"
    mkdir -p -- "$smoke_base"
    smoke_root="$(mktemp -d "${smoke_base}/run.XXXXXX")"
    "$VENV_PYTHON" -m text2model_forge demo --workspace "$smoke_root" --run-id launcher.demo.v1 || result=$?
    case "$smoke_root" in
        "$smoke_base"/*) rm -rf -- "$smoke_root" ;;
        *) die "refusing to clean an unexpected smoke-test path: $smoke_root" ;;
    esac
    (( result == 0 )) || return "$result"
}

main() {
    local selected_stack full_stack=0 install_only=0 ollama_bin="" studio_args=()
    if [[ "$ACTION" == doctor ]]; then
        show_doctor
        return 0
    fi
    selected_stack="$(resolve_ai_stack)"
    [[ "$selected_stack" == qwen || "$selected_stack" == sdxl ]] && full_stack=1
    install_lock
    if (( full_stack )); then install_require_space "$REPO_ROOT" 25; else install_require_space "$REPO_ROOT" 3; fi
    [[ "$ACTION" == install || "$ACTION" == repair ]] && install_only=1
    if (( full_stack )); then
        (( install_only )) && TOTAL_STEPS=8 || TOTAL_STEPS=9
    else
        (( install_only )) && TOTAL_STEPS=5 || TOTAL_STEPS=6
    fi
    printf 'Text2Model Forge action: %s; AI stack: %s; workspace: %s\n' "$ACTION" "$selected_stack" "$WORKSPACE"

    ensure_core "$selected_stack"
    if (( full_stack )); then
        ensure_reviewer
        ensure_comfyui
        ensure_ai_models "$selected_stack"
        ensure_blender
    elif [[ "$selected_stack" == existing ]]; then
        write_step "Using existing local AI services"
        BLENDER_BIN="$(find_blender || true)"
    else
        write_step "Skipping optional local AI installation"
        BLENDER_BIN="$(find_blender || true)"
    fi
    ensure_local_config "$selected_stack"
    run_smoke

    if (( install_only )); then
        write_step "Installation complete"
        show_doctor
        printf '\nInstall complete. Run ./text2model-forge.sh to start Studio.\n'
        install_complete
        return 0
    fi

    install_complete
    write_step "Starting local AI services"
    if [[ "$selected_stack" == qwen || "$selected_stack" == sdxl || "$selected_stack" == existing ]]; then
        ollama_bin="$(find_ollama || true)"
        [[ -n "$ollama_bin" ]] && start_ollama "$ollama_bin" || true
        start_comfyui || true
        url_ready "http://127.0.0.1:8188/system_stats" && open_url "http://127.0.0.1:8188" || true
    fi

    write_step "Starting Text2Model Forge Studio"
    studio_args=(-m text2model_forge studio --workspace "$WORKSPACE")
    (( ! NO_BROWSER )) && studio_args+=(--open-browser)
    exec "$VENV_PYTHON" "${studio_args[@]}"
}

main
