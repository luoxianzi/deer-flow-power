#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-info}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_COMPOSE_FILE="$REPO_ROOT/docker/docker-compose-dev.yaml"
OVERRIDE_COMPOSE_FILE="$REPO_ROOT/docker/docker-compose.minimal-backup.yaml"
COMPOSE_PROJECT="deer-flow-dev"
MINIMAL_PORT="${MINIMAL_PORT:-2226}"
DEER_FLOW_ROOT="${DEER_FLOW_ROOT:-$REPO_ROOT}"

export MINIMAL_PORT
export DEER_FLOW_ROOT

compose_cmd() {
  docker compose -p "$COMPOSE_PROJECT" -f "$BASE_COMPOSE_FILE" -f "$OVERRIDE_COMPOSE_FILE" "$@"
}

ensure_runtime() {
  if [ ! -f "$REPO_ROOT/config.yaml" ]; then
    echo "Missing config.yaml: $REPO_ROOT/config.yaml" >&2
    exit 1
  fi

  if [ ! -f "$REPO_ROOT/extensions_config.json" ]; then
    echo "Missing extensions_config.json: $REPO_ROOT/extensions_config.json" >&2
    exit 1
  fi
}

start() {
  ensure_runtime
  if ! compose_cmd up -d --build --remove-orphans frontend gateway langgraph nginx; then
    echo "Build step failed; retrying with existing local images..." >&2
    compose_cmd up -d --remove-orphans frontend gateway langgraph nginx
  fi
  info
}

stop() {
  ensure_runtime
  compose_cmd down
}

health() {
  ensure_runtime
  echo "Homepage:"
  curl -fsSI "http://127.0.0.1:${MINIMAL_PORT}" | sed -n '1,5p'
  echo
  echo "Models:"
  curl -fsS "http://127.0.0.1:${MINIMAL_PORT}/api/models"
}

info() {
  ensure_runtime
  cat <<EOF
Config: $REPO_ROOT/config.yaml
Extensions: $REPO_ROOT/extensions_config.json
Runtime home: $REPO_ROOT/backend/.deer-flow
LangGraph state dir: $REPO_ROOT/backend/.langgraph_api
Port: http://127.0.0.1:${MINIMAL_PORT}
Purpose: original minimal profile backup
EOF
}

case "$CMD" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  health)
    health
    ;;
  info)
    info
    ;;
  *)
    echo "Usage: $0 {start|stop|health|info}" >&2
    exit 1
    ;;
esac
