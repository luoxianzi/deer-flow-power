#!/usr/bin/env bash
# Sync the Claude Code OAuth access token that the local Claude Desktop app is
# currently using into ~/.claude/.credentials.json so that DeerFlow (and any
# other CLI tool that reads the credentials file) can pick it up.
#
# Why this script exists
# ----------------------
# Claude Desktop on macOS holds the live OAuth access token in memory and
# injects it into child processes (Claude Code sessions) as the
# CLAUDE_CODE_OAUTH_TOKEN environment variable. It does NOT always write the
# rotated token back to the on-disk credentials file, so a plain `make dev`
# launched from Terminal sees a stale (possibly expired) token.
#
# DeerFlow's `credential_loader.py` already knows how to perform a
# standards-compliant OAuth2 refresh_token exchange when the stored token is
# expired. Unfortunately Anthropic's OAuth also invalidates the refresh_token
# once it has been unused past the access-token expiry for too long, so
# refresh can fail with `invalid_grant` after a few days of disuse.
#
# This script is the escape hatch: it grabs the live token from the Claude
# Desktop environment and persists it to disk so DeerFlow can load it on its
# next start.
#
# Usage
# -----
# Run this from a *Claude Code* terminal (any shell where Claude Desktop has
# injected CLAUDE_CODE_OAUTH_TOKEN). A regular Terminal.app shell will NOT
# have the variable set and the script will abort.
#
#     bash scripts/sync-claude-token.sh
#
# Options (environment overrides):
#   CLAUDE_TOKEN_TTL_MINUTES   How long we pretend the token is valid for.
#                              Default 45. The value is only used for DeerFlow's
#                              own "is this expired?" check — it does not
#                              extend the real token lifetime.
#   CLAUDE_CRED_PATH           Credential file path. Default ~/.claude/.credentials.json
#
set -euo pipefail

# Default 8 hours: matches the typical Claude Desktop access-token lifetime so
# DeerFlow doesn't spuriously treat a still-live token as expired. If the real
# token rotates early and the API returns 401, the provider's
# _reload_oauth_credential_from_disk auto-retry will pick up the new value on
# the next request.
TTL_MINUTES="${CLAUDE_TOKEN_TTL_MINUTES:-480}"
CRED_PATH="${CLAUDE_CRED_PATH:-$HOME/.claude/.credentials.json}"

if [[ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
  cat >&2 <<'EOF'
sync-claude-token: CLAUDE_CODE_OAUTH_TOKEN is not set.

This script must be run from a shell where Claude Desktop has injected the
OAuth token — i.e. from inside a Claude Code session (the "Terminal" that
Claude Desktop opens for you, or any subprocess started from it).

Running from a plain Terminal.app/iTerm shell will not see the variable.
EOF
  exit 1
fi

if [[ ! -f "$CRED_PATH" ]]; then
  echo "sync-claude-token: credentials file not found at $CRED_PATH" >&2
  echo "Launch Claude Desktop at least once before running this script." >&2
  exit 1
fi

python3 - "$CRED_PATH" "$TTL_MINUTES" <<'PY'
import json
import os
import shutil
import sys
import time
from pathlib import Path

cred_path = Path(sys.argv[1])
ttl_minutes = int(sys.argv[2])
token = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]

data = json.loads(cred_path.read_text())
backup = cred_path.with_name(cred_path.name + f".bak-{int(time.time())}")
shutil.copy2(cred_path, backup)

oauth = data.setdefault("claudeAiOauth", {})
oauth["accessToken"] = token
oauth["expiresAt"] = int((time.time() + ttl_minutes * 60) * 1000)

cred_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
os.chmod(cred_path, 0o600)

print(f"✓ wrote fresh access token to {cred_path}")
print(f"  expiresAt = {oauth['expiresAt']} "
      f"({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(oauth['expiresAt']/1000))})")
print(f"  backup kept at {backup}")
PY
