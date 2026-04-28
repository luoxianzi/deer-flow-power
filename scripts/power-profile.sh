#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-info}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Load repo .env early so API_POOL_DIRECT_EGRESS / API_POOL_PROXY_URL etc. apply before tunnel setup.
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env" || true
  set +a
fi
COMPOSE_FILE="$REPO_ROOT/docker/docker-compose.power.yaml"
COMPOSE_PROJECT="deer-flow-power"
COMPOSE_PARALLEL_LIMIT="${COMPOSE_PARALLEL_LIMIT:-1}"

POWER_PORT="${POWER_PORT:-2026}"
BACKUP_MINIMAL_PORT="${BACKUP_MINIMAL_PORT:-2226}"
DEER_FLOW_HOME="${DEER_FLOW_HOME:-$REPO_ROOT/backend/.deer-flow-power}"
DEER_FLOW_LANGGRAPH_API_DIR="${DEER_FLOW_LANGGRAPH_API_DIR:-$REPO_ROOT/backend/.langgraph_api.power}"
DEER_FLOW_CONFIG_PATH="${DEER_FLOW_CONFIG_PATH:-$REPO_ROOT/config.power.yaml}"
DEER_FLOW_EXTENSIONS_CONFIG_PATH="${DEER_FLOW_EXTENSIONS_CONFIG_PATH:-$REPO_ROOT/extensions_config.power.json}"
DEER_FLOW_DOCKER_SOCKET="${DEER_FLOW_DOCKER_SOCKET:-/var/run/docker.sock}"
DEER_FLOW_REPO_ROOT="$REPO_ROOT"
DEER_FLOW_SANDBOX_BIND_HOST="${DEER_FLOW_SANDBOX_BIND_HOST:-127.0.0.1}"
NGINX_CONF="${NGINX_CONF:-nginx.conf}"
POWER_GATEWAY_WORKERS="${POWER_GATEWAY_WORKERS:-1}"
POWER_LANGGRAPH_JOBS_PER_WORKER="${POWER_LANGGRAPH_JOBS_PER_WORKER:-4}"
POWER_LANGGRAPH_BG_JOB_ISOLATED_LOOPS="${POWER_LANGGRAPH_BG_JOB_ISOLATED_LOOPS:-true}"

POWER_WORKSPACES_PATH="${POWER_WORKSPACES_PATH:-$HOME/workspaces}"
POWER_DOCUMENTS_PATH="${POWER_DOCUMENTS_PATH:-$HOME/Documents}"
POWER_DOWNLOADS_PATH="${POWER_DOWNLOADS_PATH:-$HOME/Downloads}"
POWER_PROJECT_ROOT_PATH="${POWER_PROJECT_ROOT_PATH:-$HOME/Downloads/omniai-whatsapp-v2}"
POWER_MOS1_PROJECT_PATH="${POWER_MOS1_PROJECT_PATH:-$HOME/Downloads/MOS1.0-main}"
POWER_OMS_PROJECT_PATH="${POWER_OMS_PROJECT_PATH:-$HOME/Downloads/Oms-main}"
POWER_WEBSITE_PROJECT_PATH="${POWER_WEBSITE_PROJECT_PATH:-$HOME/Downloads/website}"
POWER_PRIVATE_TMP_PATH="${POWER_PRIVATE_TMP_PATH:-/private/tmp}"
POWER_GITCONFIG_PATH="${POWER_GITCONFIG_PATH:-$HOME/.gitconfig}"
POWER_SSH_SOURCE_DIR_PATH="${POWER_SSH_SOURCE_DIR_PATH:-$HOME/.ssh}"
POWER_SSH_SOURCE_CONFIG_PATH="${POWER_SSH_SOURCE_CONFIG_PATH:-$POWER_SSH_SOURCE_DIR_PATH/config}"
POWER_SSH_SOURCE_KNOWN_HOSTS_PATH="${POWER_SSH_SOURCE_KNOWN_HOSTS_PATH:-$POWER_SSH_SOURCE_DIR_PATH/known_hosts}"
POWER_SSH_DIR_PATH="${POWER_SSH_DIR_PATH:-$DEER_FLOW_HOME/.ssh-runtime}"
POWER_SSH_CONFIG_PATH="${POWER_SSH_CONFIG_PATH:-$POWER_SSH_DIR_PATH/config}"
POWER_SSH_KNOWN_HOSTS_PATH="${POWER_SSH_KNOWN_HOSTS_PATH:-$POWER_SSH_DIR_PATH/known_hosts}"
POWER_SSH_SYSTEM_CONFIG_PATH="${POWER_SSH_SYSTEM_CONFIG_PATH:-$DEER_FLOW_HOME/.ssh-system-config}"
POWER_SSH_SYSTEM_KNOWN_HOSTS_PATH="${POWER_SSH_SYSTEM_KNOWN_HOSTS_PATH:-$DEER_FLOW_HOME/.ssh_known_hosts}"
POWER_SANDBOX_SSH_HOOK_SCRIPT_PATH="${POWER_SANDBOX_SSH_HOOK_SCRIPT_PATH:-$DEER_FLOW_HOME/sandbox-setup-ssh.sh}"
POWER_SANDBOX_RUN_HOOK_PRE_SERVICES="${POWER_SANDBOX_RUN_HOOK_PRE_SERVICES:-}"
POWER_SSH_AUTH_SOCK="${POWER_SSH_AUTH_SOCK:-${SSH_AUTH_SOCK:-}}"
POWER_SSH_AGENT_MODE="${POWER_SSH_AGENT_MODE:-forwarded}"
API_POOL_SSH_TUNNEL_HOST="${API_POOL_SSH_TUNNEL_HOST:-omniai-sg}"
API_POOL_SSH_TUNNEL_BIND_HOST="${API_POOL_SSH_TUNNEL_BIND_HOST:-127.0.0.1}"
API_POOL_SSH_TUNNEL_PORT="${API_POOL_SSH_TUNNEL_PORT:-17890}"
API_POOL_SSH_TUNNEL_CONTROL_PATH="${API_POOL_SSH_TUNNEL_CONTROL_PATH:-/private/tmp/deer-flow-api-pool-ssh.sock}"
API_POOL_SSH_TUNNEL_WATCH_PID_FILE="${API_POOL_SSH_TUNNEL_WATCH_PID_FILE:-$DEER_FLOW_HOME/api-pool-tunnel-watch.pid}"
API_POOL_SSH_TUNNEL_WATCH_LOG_PATH="${API_POOL_SSH_TUNNEL_WATCH_LOG_PATH:-$DEER_FLOW_HOME/api-pool-tunnel-watch.log}"
API_POOL_SSH_TUNNEL_WATCH_LABEL="${API_POOL_SSH_TUNNEL_WATCH_LABEL:-com.deerflow.api-pool-tunnel}"
API_POOL_SSH_TUNNEL_WATCH_PLIST_PATH="${API_POOL_SSH_TUNNEL_WATCH_PLIST_PATH:-$DEER_FLOW_HOME/api-pool-tunnel-watch.plist}"
API_POOL_PROXY_URL="${API_POOL_PROXY_URL:-}"
SSH_CHECK_TARGET="${2:-}"
POWER_CLAUDE_VM_CLI_PLACEHOLDER="$REPO_ROOT/docker/claude-vm-placeholder.sh"
if [ -z "${POWER_CLAUDE_VM_CLI_PATH:-}" ]; then
  POWER_CLAUDE_VM_CLI_PATH="$(
    find "$HOME/Library/Application Support/Claude/claude-code-vm" -maxdepth 5 -type f -name claude 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
fi
# Docker cannot bind-mount a directory onto /usr/local/bin/claude (expects a file). Invalid .env paths
# or app updates that expose `claude` as a directory would keep LangGraph down and nginx on 502.
if [ ! -f "$POWER_CLAUDE_VM_CLI_PATH" ]; then
  if [ -n "${POWER_CLAUDE_VM_CLI_PATH:-}" ]; then
    echo "POWER_CLAUDE_VM_CLI_PATH is missing or not a regular file: $POWER_CLAUDE_VM_CLI_PATH" >&2
    echo "Falling back to $POWER_CLAUDE_VM_CLI_PLACEHOLDER so LangGraph can start; fix the path for a real VM CLI." >&2
  fi
  POWER_CLAUDE_VM_CLI_PATH="$POWER_CLAUDE_VM_CLI_PLACEHOLDER"
fi

if [ -z "$POWER_SANDBOX_RUN_HOOK_PRE_SERVICES" ]; then
  POWER_SANDBOX_RUN_HOOK_PRE_SERVICES='bash /opt/deerflow-hooks/setup_ssh.sh'
fi

export POWER_PORT
export BACKUP_MINIMAL_PORT
export COMPOSE_PARALLEL_LIMIT
export DEER_FLOW_HOME
export DEER_FLOW_LANGGRAPH_API_DIR
export DEER_FLOW_CONFIG_PATH
export DEER_FLOW_EXTENSIONS_CONFIG_PATH
export DEER_FLOW_DOCKER_SOCKET
export DEER_FLOW_REPO_ROOT
export DEER_FLOW_SANDBOX_BIND_HOST
export NGINX_CONF
export POWER_GATEWAY_WORKERS
export POWER_LANGGRAPH_JOBS_PER_WORKER
export POWER_LANGGRAPH_BG_JOB_ISOLATED_LOOPS
export POWER_WORKSPACES_PATH
export POWER_DOCUMENTS_PATH
export POWER_DOWNLOADS_PATH
export POWER_PROJECT_ROOT_PATH
export POWER_MOS1_PROJECT_PATH
export POWER_OMS_PROJECT_PATH
export POWER_WEBSITE_PROJECT_PATH
export POWER_PRIVATE_TMP_PATH
export POWER_GITCONFIG_PATH
export POWER_SSH_SOURCE_DIR_PATH
export POWER_SSH_SOURCE_CONFIG_PATH
export POWER_SSH_SOURCE_KNOWN_HOSTS_PATH
export POWER_SSH_DIR_PATH
export POWER_SSH_CONFIG_PATH
export POWER_SSH_KNOWN_HOSTS_PATH
export POWER_SSH_SYSTEM_CONFIG_PATH
export POWER_SSH_SYSTEM_KNOWN_HOSTS_PATH
export POWER_SANDBOX_SSH_HOOK_SCRIPT_PATH
export POWER_SANDBOX_RUN_HOOK_PRE_SERVICES
export POWER_SSH_AUTH_SOCK
export POWER_SSH_AGENT_MODE
export API_POOL_SSH_TUNNEL_HOST
export API_POOL_SSH_TUNNEL_BIND_HOST
export API_POOL_SSH_TUNNEL_PORT
export API_POOL_SSH_TUNNEL_CONTROL_PATH
export API_POOL_SSH_TUNNEL_WATCH_PID_FILE
export API_POOL_SSH_TUNNEL_WATCH_LOG_PATH
export API_POOL_SSH_TUNNEL_WATCH_LABEL
export API_POOL_SSH_TUNNEL_WATCH_PLIST_PATH
export API_POOL_PROXY_URL
export POWER_CLAUDE_VM_CLI_PATH

compose_cmd() {
  docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" "$@"
}

build_services_sequentially() {
  local services=(frontend gateway langgraph)
  local service

  for service in "${services[@]}"; do
    echo "Building $service..."
    compose_cmd build "$service"
  done
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-60}"
  local delay="${4:-2}"
  local i

  for ((i = 1; i <= attempts; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done

  echo "Timed out waiting for $label at $url" >&2
  return 1
}

wait_for_container_http() {
  local container="$1"
  local url="$2"
  local label="$3"
  local attempts="${4:-60}"
  local delay="${5:-2}"
  local i

  for ((i = 1; i <= attempts; i++)); do
    if docker exec "$container" sh -lc "curl -fsS '$url' >/dev/null" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done

  echo "Timed out waiting for $label in $container at $url" >&2
  return 1
}

ensure_path() {
  local target="$1"
  local label="$2"
  if [ ! -e "$target" ]; then
    echo "Missing required $label: $target" >&2
    exit 1
  fi
}

resolve_ssh_agent_socket() {
  local candidate="${POWER_SSH_AUTH_SOCK:-}"
  local launchctl_sock=""

  if [ -z "$candidate" ] && command -v launchctl >/dev/null 2>&1; then
    launchctl_sock="$(launchctl getenv SSH_AUTH_SOCK 2>/dev/null || true)"
    if [ -n "$launchctl_sock" ]; then
      candidate="$launchctl_sock"
    fi
  fi

  if [ -n "$candidate" ] && [ -S "$candidate" ]; then
    POWER_SSH_AUTH_SOCK="$candidate"
    POWER_SSH_AGENT_MODE="forwarded"
    export POWER_SSH_AUTH_SOCK POWER_SSH_AGENT_MODE
    return
  fi

  POWER_SSH_AUTH_SOCK=""
  POWER_SSH_AGENT_MODE="unavailable"
  export POWER_SSH_AUTH_SOCK POWER_SSH_AGENT_MODE
}

build_runtime_ssh_dir() {
  ensure_path "$POWER_SSH_SOURCE_DIR_PATH" "ssh source directory"

  mkdir -p "$POWER_SSH_DIR_PATH"
  chmod 700 "$POWER_SSH_DIR_PATH"

  if command -v rsync >/dev/null 2>&1; then
    rsync \
      -a \
      --delete \
      --exclude '.DS_Store' \
      --exclude '*.sock' \
      --exclude 'control*' \
      "$POWER_SSH_SOURCE_DIR_PATH"/ "$POWER_SSH_DIR_PATH"/
  else
    find "$POWER_SSH_DIR_PATH" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
    cp -R "$POWER_SSH_SOURCE_DIR_PATH"/. "$POWER_SSH_DIR_PATH"/
  fi

  if [ -f "$POWER_SSH_SOURCE_CONFIG_PATH" ]; then
    {
      printf 'IgnoreUnknown UseKeychain\n'
      cat "$POWER_SSH_SOURCE_CONFIG_PATH"
    } > "$POWER_SSH_CONFIG_PATH"
  else
    : > "$POWER_SSH_CONFIG_PATH"
  fi

  chmod 600 "$POWER_SSH_CONFIG_PATH"

  if [ -f "$POWER_SSH_SOURCE_CONFIG_PATH" ]; then
    {
      printf 'IgnoreUnknown UseKeychain\n'
      awk '
        /^[[:space:]]*AddKeysToAgent[[:space:]]+/ { next }
        /^[[:space:]]*UseKeychain[[:space:]]+/ { next }
        /^[[:space:]]*IdentityFile[[:space:]]+/ { next }
        { print }
      ' "$POWER_SSH_SOURCE_CONFIG_PATH"
    } > "$POWER_SSH_SYSTEM_CONFIG_PATH"
  else
    : > "$POWER_SSH_SYSTEM_CONFIG_PATH"
  fi

  chmod 644 "$POWER_SSH_SYSTEM_CONFIG_PATH"

  if [ -f "$POWER_SSH_SOURCE_KNOWN_HOSTS_PATH" ]; then
    cp "$POWER_SSH_SOURCE_KNOWN_HOSTS_PATH" "$POWER_SSH_SYSTEM_KNOWN_HOSTS_PATH"
    chmod 644 "$POWER_SSH_SYSTEM_KNOWN_HOSTS_PATH"
  else
    : > "$POWER_SSH_SYSTEM_KNOWN_HOSTS_PATH"
    chmod 644 "$POWER_SSH_SYSTEM_KNOWN_HOSTS_PATH"
  fi

  cat > "$POWER_SANDBOX_SSH_HOOK_SCRIPT_PATH" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

mkdir -p /home/gem/.ssh
cp /etc/ssh/ssh_config.d/99-deerflow-power.conf /home/gem/.ssh/config
cp /etc/ssh/ssh_known_hosts /home/gem/.ssh/known_hosts
for src in /root/.ssh/id_* /root/.ssh/*.pub; do
  [ -f "$src" ] || continue
  cp "$src" /home/gem/.ssh/
done
chown -R gem:gem /home/gem/.ssh
chmod 700 /home/gem/.ssh
find /home/gem/.ssh -maxdepth 1 -type f \( -name "id_*" ! -name "*.pub" -o -name "config" \) -exec chmod 600 {} \;
find /home/gem/.ssh -maxdepth 1 -type f \( -name "*.pub" -o -name "known_hosts" \) -exec chmod 644 {} \;
EOF
  chmod 755 "$POWER_SANDBOX_SSH_HOOK_SCRIPT_PATH"
}

ensure_common_runtime() {
  ensure_path "$DEER_FLOW_CONFIG_PATH" "power config"
  ensure_path "$DEER_FLOW_EXTENSIONS_CONFIG_PATH" "power extensions config"
  mkdir -p "$DEER_FLOW_HOME" "$DEER_FLOW_LANGGRAPH_API_DIR"
}

ensure_start_runtime() {
  ensure_common_runtime
  ensure_path "$POWER_WORKSPACES_PATH" "workspaces mount"
  ensure_path "$POWER_DOCUMENTS_PATH" "Documents mount"
  ensure_path "$POWER_DOWNLOADS_PATH" "Downloads mount"
  ensure_path "$POWER_PROJECT_ROOT_PATH" "project root mount"
  ensure_path "$POWER_MOS1_PROJECT_PATH" "MOS1.0-main project mount"
  ensure_path "$POWER_OMS_PROJECT_PATH" "Oms-main project mount"
  ensure_path "$POWER_WEBSITE_PROJECT_PATH" "website project mount"
  ensure_path "$POWER_PRIVATE_TMP_PATH" "/private/tmp mount"
  ensure_path "$POWER_GITCONFIG_PATH" ".gitconfig mount"
  build_runtime_ssh_dir
  resolve_ssh_agent_socket
  ensure_path "$POWER_CLAUDE_VM_CLI_PATH" "Claude Code VM CLI mount"
  if [ ! -S "$DEER_FLOW_DOCKER_SOCKET" ]; then
    echo "Docker socket not found: $DEER_FLOW_DOCKER_SOCKET" >&2
    exit 1
  fi

  local secret_file="$DEER_FLOW_HOME/.better-auth-secret"
  if [ -z "${BETTER_AUTH_SECRET:-}" ]; then
    if [ -f "$secret_file" ]; then
      BETTER_AUTH_SECRET="$(cat "$secret_file")"
    else
      BETTER_AUTH_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
      printf '%s\n' "$BETTER_AUTH_SECRET" > "$secret_file"
      chmod 600 "$secret_file"
    fi
  fi
  export BETTER_AUTH_SECRET
  ensure_api_pool_proxy_runtime
}

sync_claude_credentials_if_available() {
  local bridge_script="$REPO_ROOT/scripts/claude-bridge.sh"

  if [ ! -x "$bridge_script" ]; then
    return 0
  fi

  if "$bridge_script" export >/dev/null 2>&1; then
    echo "Claude credentials synced from local Claude Code."
  else
    echo "Warning: failed to sync Claude credentials from local Claude Code." >&2
  fi
}

api_pool_tunnel_is_healthy() {
  if [ -S "$API_POOL_SSH_TUNNEL_CONTROL_PATH" ] && ssh -S "$API_POOL_SSH_TUNNEL_CONTROL_PATH" -O check "$API_POOL_SSH_TUNNEL_HOST" >/dev/null 2>&1; then
    return 0
  fi

  if lsof -nP -iTCP:"$API_POOL_SSH_TUNNEL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    return 0
  fi

  return 1
}

api_pool_tunnel_watch_is_running() {
  if command -v launchctl >/dev/null 2>&1; then
    if launchctl print "gui/$(id -u)/$API_POOL_SSH_TUNNEL_WATCH_LABEL" >/dev/null 2>&1; then
      return 0
    fi
  fi

  if [ ! -f "$API_POOL_SSH_TUNNEL_WATCH_PID_FILE" ]; then
    return 1
  fi

  local pid=""
  pid="$(cat "$API_POOL_SSH_TUNNEL_WATCH_PID_FILE" 2>/dev/null || true)"
  if [ -z "$pid" ]; then
    return 1
  fi

  if kill -0 "$pid" >/dev/null 2>&1; then
    return 0
  fi

  rm -f "$API_POOL_SSH_TUNNEL_WATCH_PID_FILE"
  return 1
}

write_api_pool_tunnel_watch_plist() {
  mkdir -p "$DEER_FLOW_HOME"
  cat > "$API_POOL_SSH_TUNNEL_WATCH_PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${API_POOL_SSH_TUNNEL_WATCH_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${REPO_ROOT}/scripts/power-profile.sh</string>
    <string>tunnel-watch</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${API_POOL_SSH_TUNNEL_WATCH_LOG_PATH}</string>
  <key>StandardErrorPath</key>
  <string>${API_POOL_SSH_TUNNEL_WATCH_LOG_PATH}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key>
    <string>${HOME}</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
EOF
}

start_api_pool_tunnel_watch() {
  if [ -z "$API_POOL_SSH_TUNNEL_HOST" ]; then
    return 0
  fi

  mkdir -p "$DEER_FLOW_HOME"

  if api_pool_tunnel_watch_is_running; then
    return 0
  fi

  if command -v launchctl >/dev/null 2>&1; then
    write_api_pool_tunnel_watch_plist
    launchctl bootout "gui/$(id -u)/$API_POOL_SSH_TUNNEL_WATCH_LABEL" >/dev/null 2>&1 || true
    launchctl bootstrap "gui/$(id -u)" "$API_POOL_SSH_TUNNEL_WATCH_PLIST_PATH"
    launchctl kickstart -k "gui/$(id -u)/$API_POOL_SSH_TUNNEL_WATCH_LABEL" >/dev/null 2>&1 || true
    return 0
  fi

  nohup "$0" tunnel-watch >>"$API_POOL_SSH_TUNNEL_WATCH_LOG_PATH" 2>&1 &
  printf '%s\n' "$!" > "$API_POOL_SSH_TUNNEL_WATCH_PID_FILE"
}

stop_api_pool_tunnel_watch() {
  if command -v launchctl >/dev/null 2>&1; then
    launchctl bootout "gui/$(id -u)/$API_POOL_SSH_TUNNEL_WATCH_LABEL" >/dev/null 2>&1 || true
  fi

  if ! api_pool_tunnel_watch_is_running; then
    rm -f "$API_POOL_SSH_TUNNEL_WATCH_PID_FILE"
    return 0
  fi

  local pid=""
  pid="$(cat "$API_POOL_SSH_TUNNEL_WATCH_PID_FILE" 2>/dev/null || true)"
  if [ -n "$pid" ]; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$API_POOL_SSH_TUNNEL_WATCH_PID_FILE"
}

ensure_api_pool_proxy_runtime() {
  # When true, API pool HTTP(S) uses container egress directly (no SSH SOCKS). Some gateways
  # (e.g. aiapis.help) time out via omniai-sg SOCKS but work from the host/Docker bridge path.
  case "${API_POOL_DIRECT_EGRESS:-}" in
  1 | true | True | TRUE | yes | Yes | YES | on | On | ON)
    API_POOL_PROXY_URL=""
    export API_POOL_PROXY_URL
    return 0
    ;;
  esac

  if [ -n "$API_POOL_PROXY_URL" ]; then
    export API_POOL_PROXY_URL
    return
  fi

  if [ -z "$API_POOL_SSH_TUNNEL_HOST" ]; then
    return
  fi

  if ! api_pool_tunnel_is_healthy; then
    rm -f "$API_POOL_SSH_TUNNEL_CONTROL_PATH"
    ssh \
      -fN \
      -M \
      -S "$API_POOL_SSH_TUNNEL_CONTROL_PATH" \
      -o BatchMode=yes \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -D "${API_POOL_SSH_TUNNEL_BIND_HOST}:${API_POOL_SSH_TUNNEL_PORT}" \
      "$API_POOL_SSH_TUNNEL_HOST"
  fi

  if ! api_pool_tunnel_is_healthy; then
    echo "Failed to establish API pool SSH SOCKS tunnel via ${API_POOL_SSH_TUNNEL_HOST}:${API_POOL_SSH_TUNNEL_PORT}" >&2
    exit 1
  fi

  API_POOL_PROXY_URL="socks5://host.docker.internal:${API_POOL_SSH_TUNNEL_PORT}"
  export API_POOL_PROXY_URL
}

stop_api_pool_proxy_runtime() {
  if [ -S "$API_POOL_SSH_TUNNEL_CONTROL_PATH" ]; then
    ssh -S "$API_POOL_SSH_TUNNEL_CONTROL_PATH" -O exit "$API_POOL_SSH_TUNNEL_HOST" >/dev/null 2>&1 || true
    rm -f "$API_POOL_SSH_TUNNEL_CONTROL_PATH"
  fi
}

tunnel_watch() {
  ensure_common_runtime

  while true; do
    resolve_ssh_agent_socket
    if ! api_pool_tunnel_is_healthy; then
      rm -f "$API_POOL_SSH_TUNNEL_CONTROL_PATH"
      ssh \
        -fN \
        -M \
        -S "$API_POOL_SSH_TUNNEL_CONTROL_PATH" \
        -o BatchMode=yes \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -D "${API_POOL_SSH_TUNNEL_BIND_HOST}:${API_POOL_SSH_TUNNEL_PORT}" \
        "$API_POOL_SSH_TUNNEL_HOST" >/dev/null 2>&1 || true
    fi
    sleep 1
  done
}

print_mounts() {
  cat <<EOF
/mnt/host/workspaces <= $POWER_WORKSPACES_PATH (rw)
/mnt/host/Documents <= $POWER_DOCUMENTS_PATH (ro)
/mnt/host/Downloads <= $POWER_DOWNLOADS_PATH (ro)
/mnt/host/omniai-whatsapp-v2 <= $POWER_PROJECT_ROOT_PATH (rw)
/mnt/host/MOS1.0-main <= $POWER_MOS1_PROJECT_PATH (rw)
/mnt/host/Oms-main <= $POWER_OMS_PROJECT_PATH (rw)
/mnt/projects/website <= $POWER_WEBSITE_PROJECT_PATH (rw)
/private/tmp <= $POWER_PRIVATE_TMP_PATH ($( [ "$POWER_SSH_AGENT_MODE" = "forwarded" ] && printf 'rw; host ssh-agent path available' || printf 'rw; host ssh-agent path mounted but no live socket detected' ))
/root/.gitconfig <= $POWER_GITCONFIG_PATH (ro)
/root/.ssh <= $POWER_SSH_DIR_PATH (ro; runtime copy from $POWER_SSH_SOURCE_DIR_PATH with Linux-safe config)
/etc/ssh/ssh_config.d/99-deerflow-power.conf <= $POWER_SSH_SYSTEM_CONFIG_PATH (ro; sandbox-user host aliases)
/etc/ssh/ssh_known_hosts <= $POWER_SSH_SYSTEM_KNOWN_HOSTS_PATH (ro; sandbox-user host keys)
/usr/local/bin/claude <= $POWER_CLAUDE_VM_CLI_PATH (ro)
EOF
}

print_models() {
  if curl -fsS "http://127.0.0.1:${POWER_PORT}/api/models" >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:${POWER_PORT}/api/models"
  else
    awk '
      /^models:/ {in_models=1; next}
      in_models && /^tool_groups:/ {exit}
      in_models && /^[[:space:]]*- name:/ {name=$3}
      in_models && /^[[:space:]]*display_name:/ {
        sub(/^[[:space:]]*display_name:[[:space:]]*/, "", $0)
        printf "%s\t%s\n", name, $0
      }
    ' "$DEER_FLOW_CONFIG_PATH"
  fi
}

start() {
  ensure_start_runtime
  start_api_pool_tunnel_watch
  sync_claude_credentials_if_available
  build_services_sequentially
  compose_cmd up -d --remove-orphans gateway
  compose_cmd up -d --remove-orphans langgraph
  compose_cmd up -d --remove-orphans frontend
  wait_for_container_http "deer-flow-power-gateway" "http://127.0.0.1:8001/api/models" "gateway" 45 2
  wait_for_container_http "deer-flow-power-langgraph" "http://127.0.0.1:2024/docs" "langgraph" 45 2
  compose_cmd up -d --remove-orphans nginx
  wait_for_http "http://127.0.0.1:${POWER_PORT}" "power profile" 60 2
  info
}

stop() {
  ensure_common_runtime
  compose_cmd down
  stop_api_pool_tunnel_watch
  stop_api_pool_proxy_runtime
}

health() {
  ensure_start_runtime
  echo "Homepage:"
  curl -fsSI "http://127.0.0.1:${POWER_PORT}" | sed -n '1,5p'
  echo
  echo "Models:"
  curl -fsS "http://127.0.0.1:${POWER_PORT}/api/models"
}

ssh_check() {
  ensure_start_runtime

  local target="$SSH_CHECK_TARGET"
  if [ -z "$target" ] && [ -f "$POWER_SSH_CONFIG_PATH" ]; then
    target="$(awk '/^Host / && $2 != "*" {print $2; exit}' "$POWER_SSH_CONFIG_PATH" 2>/dev/null || true)"
  fi

  echo "SSH agent mode: $POWER_SSH_AGENT_MODE"
  echo "SSH auth sock: ${POWER_SSH_AUTH_SOCK:-<unset>}"
  echo "SSH dir mount: $POWER_SSH_DIR_PATH"
  echo
  echo "Container SSH files:"
  docker exec deer-flow-power-langgraph sh -lc 'printf "SSH_AUTH_SOCK=%s\n" "$SSH_AUTH_SOCK"; test -d /root/.ssh && echo SSH_DIR_OK || echo SSH_DIR_MISSING; find /root/.ssh -maxdepth 1 -type f \( -name "id_*" ! -name "*.pub" \) -printf "%f\n" | sort'
  echo

  if [ -z "$target" ]; then
    echo "No SSH host alias found. Usage: $0 ssh-check <host-alias>" >&2
    return 1
  fi

  echo "Resolved SSH target: $target"
  docker exec deer-flow-power-langgraph sh -lc "ssh -G '$target' | sed -n '1,20p'"
  echo
  echo "Connectivity test:"
  docker exec deer-flow-power-langgraph sh -lc "ssh -o BatchMode=yes -o ConnectTimeout=10 '$target' hostname"
}

info() {
  ensure_common_runtime
  local displayed_api_pool_proxy="${API_POOL_PROXY_URL:-}"
  if [ -z "$displayed_api_pool_proxy" ] && api_pool_tunnel_is_healthy; then
    displayed_api_pool_proxy="socks5://host.docker.internal:${API_POOL_SSH_TUNNEL_PORT}"
  fi
  echo "Config: $DEER_FLOW_CONFIG_PATH"
  echo "Extensions: $DEER_FLOW_EXTENSIONS_CONFIG_PATH"
  echo "Runtime home: $DEER_FLOW_HOME"
  echo "LangGraph state dir: $DEER_FLOW_LANGGRAPH_API_DIR"
  echo "Port: http://127.0.0.1:${POWER_PORT}"
  echo "Minimal backup entry: http://127.0.0.1:${BACKUP_MINIMAL_PORT}"
  echo "Sandbox bind host: $DEER_FLOW_SANDBOX_BIND_HOST"
  echo "LangGraph jobs per worker: $POWER_LANGGRAPH_JOBS_PER_WORKER"
  echo "LangGraph isolated loops: $POWER_LANGGRAPH_BG_JOB_ISOLATED_LOOPS"
  echo "API pool proxy: ${displayed_api_pool_proxy:-<none>}"
  echo "API pool SSH tunnel: ${API_POOL_SSH_TUNNEL_HOST}:${API_POOL_SSH_TUNNEL_PORT}"
  echo "API pool SSH tunnel watch: $(api_pool_tunnel_watch_is_running && printf 'running' || printf 'stopped')"
  echo
  echo "Mounts:"
  print_mounts
  echo
  echo "Models:"
  print_models
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
  ssh-check)
    ssh_check
    ;;
  tunnel-watch)
    tunnel_watch
    ;;
  *)
    echo "Usage: $0 {start|stop|health|info|ssh-check [host-alias]|tunnel-watch}" >&2
    exit 1
    ;;
esac
