#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./scripts/install-utils.sh
install_init "$PWD" "Asset Forge"
install_enable_traps
action=run
case "${1:-}" in run|doctor|repair|docker|stop|logs) action=$1; shift ;; esac
url=http://127.0.0.1:8766

health_check() {
    if command -v curl >/dev/null 2>&1; then curl -fsS --max-time 2 "$url/doctor" >/dev/null
    elif command -v wget >/dev/null 2>&1; then wget -qO- --timeout=2 "$url/doctor" >/dev/null
    else return 1; fi
}
wait_ready() { for _ in $(seq 1 120); do health_check 2>/dev/null && return; sleep 0.5; done; return 1; }
open_url() { command -v open >/dev/null 2>&1 && open "$url" || command -v xdg-open >/dev/null 2>&1 && xdg-open "$url" || true; }

case "$action" in
    docker|stop|logs)
        docker_ready=0; docker_running=0
        command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && docker_ready=1
        [ "$docker_ready" -eq 1 ] && [ -n "$(docker compose ps --quiet studio 2>/dev/null)" ] && docker_running=1
        if [ "$action" = stop ] && [ "$docker_running" -eq 0 ]; then
            echo "No Asset Forge Docker service is running. For a native Studio session, press Ctrl+C in its terminal."
            exit 0
        fi
        if [ "$action" = logs ] && [ "$docker_running" -eq 0 ]; then
            if compgen -G 'runtime/logs/*.log' >/dev/null; then exec tail -n 100 -F runtime/logs/*.log; fi
            echo "Native Studio logs are shown in its terminal. No managed service logs exist yet."
            exit 0
        fi
        command -v docker >/dev/null 2>&1 || { echo "Docker is not installed." >&2; exit 1; }
        [ "$docker_ready" -eq 1 ] || { echo "Docker is installed but its engine is not running." >&2; exit 1; }
        [ "$action" = stop ] && exec docker compose down
        [ "$action" = logs ] && exec docker compose logs --follow
        install_lock
        install_require_space "$PWD" 3
        docker compose up --detach --build
        wait_ready || { docker compose logs studio; echo "Text2Model Forge did not become ready at $url." >&2; exit 1; }
        install_complete
        echo "Text2Model Forge is ready at $url"
        case " $* " in *" --no-browser "*) ;; *) open_url ;; esac
        exit 0 ;;
esac

args=("$@")
[ "$action" != run ] && args=("$action" "${args[@]}")
has_ai_stack=0
for arg in "${args[@]}"; do [ "$arg" = --ai-stack ] && has_ai_stack=1; done
[ "$has_ai_stack" -eq 1 ] && exec ./text2model-forge.sh "${args[@]}"
exec ./text2model-forge.sh --ai-stack core "${args[@]}"
