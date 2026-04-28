#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-status}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLAUDE_ROOT="${HOME}/Library/Application Support/Claude/claude-code"
CLAUDE_EMAIL="${CLAUDE_EMAIL:-}"
CLAUDE_EXPORT_PATH="${CLAUDE_EXPORT_PATH:-$HOME/.claude/.credentials.json}"

find_claude_cli() {
  local cli_path
  cli_path="$(
    find "$CLAUDE_ROOT" -maxdepth 6 -type f -path '*/claude.app/Contents/MacOS/claude' 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"
  if [ -z "$cli_path" ]; then
    echo "Claude Code CLI not found under: $CLAUDE_ROOT" >&2
    exit 1
  fi
  printf '%s\n' "$cli_path"
}

CLAUDE_CLI="$(find_claude_cli)"

status() {
  "$CLAUDE_CLI" auth status
}

login() {
  if [ -n "$CLAUDE_EMAIL" ]; then
    "$CLAUDE_CLI" auth login --claudeai --email "$CLAUDE_EMAIL"
  else
    "$CLAUDE_CLI" auth login --claudeai
  fi
}

export_credentials() {
  mkdir -p "$(dirname "$CLAUDE_EXPORT_PATH")"
  python3 "$REPO_ROOT/scripts/export_claude_code_oauth.py" --write-credentials "$CLAUDE_EXPORT_PATH"
  chmod 600 "$CLAUDE_EXPORT_PATH"
  echo "Exported Claude Code OAuth credentials to $CLAUDE_EXPORT_PATH"
}

sync() {
  export_credentials
  echo
  echo "Claude model is configured in config.power.yaml."
  echo "DeerFlow power entry: http://localhost:2026"
}

path() {
  printf '%s\n' "$CLAUDE_CLI"
}

case "$CMD" in
  status)
    status
    ;;
  login)
    login
    ;;
  export)
    export_credentials
    ;;
  sync)
    sync
    ;;
  path)
    path
    ;;
  *)
    echo "Usage: $0 {status|login|export|sync|path}" >&2
    exit 1
    ;;
esac
