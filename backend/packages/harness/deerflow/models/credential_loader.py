"""Auto-load credentials from Claude Code CLI and Codex CLI.

Implements two credential strategies:
  1. Claude Code OAuth token from explicit env vars or an exported credentials file
     - Uses Authorization: Bearer header (NOT x-api-key)
     - Requires anthropic-beta: oauth-2025-04-20,claude-code-20250219
     - Supports $CLAUDE_CODE_OAUTH_TOKEN, $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR, and $ANTHROPIC_AUTH_TOKEN
     - Override path with $CLAUDE_CODE_CREDENTIALS_PATH
     - If the on-disk accessToken is expired, the file-based loader will attempt a
       standards-compliant OAuth2 refresh_token exchange against the Claude Code
       public-client token endpoint, write the rotated credentials back atomically,
       and emit a structured log. Refresh can be disabled with
       $DEER_FLOW_DISABLE_CLAUDE_OAUTH_REFRESH=1. Endpoint and client_id are
       overridable via $CLAUDE_CODE_OAUTH_TOKEN_URL and $CLAUDE_CODE_OAUTH_CLIENT_ID.
  2. Codex CLI token from ~/.codex/auth.json
     - Uses chatgpt.com/backend-api/codex/responses endpoint
     - Supports both legacy top-level tokens and current nested tokens shape
     - Override path with $CODEX_AUTH_PATH
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Required beta headers for Claude Code OAuth tokens
OAUTH_ANTHROPIC_BETAS = "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14"

# Claude Code OAuth (public-client, authorization_code + refresh_token).
# Source of truth: https://claude.ai/oauth/claude-code-client-metadata
# client_id is a URL (per RFC 7591 dynamic client registration). Overridable for
# sandboxed/enterprise deployments that proxy the upstream endpoints.
_DEFAULT_CLAUDE_CODE_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_DEFAULT_CLAUDE_CODE_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_REFRESH_HTTP_TIMEOUT_SECONDS = 15.0


def is_oauth_token(token: str) -> bool:
    """Check if a token is a Claude Code OAuth token (not a standard API key)."""
    return isinstance(token, str) and "sk-ant-oat" in token


@dataclass
class ClaudeCodeCredential:
    """Claude Code CLI OAuth credential."""

    access_token: str
    refresh_token: str = ""
    expires_at: int = 0
    source: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False
        return time.time() * 1000 > self.expires_at - 60_000  # 1 min buffer


@dataclass
class CodexCliCredential:
    """Codex CLI credential."""

    access_token: str
    account_id: str = ""
    source: str = ""


def _resolve_credential_path(env_var: str, default_relative_path: str) -> Path:
    configured_path = os.getenv(env_var)
    if configured_path:
        return Path(configured_path).expanduser()
    return _home_dir() / default_relative_path


def _home_dir() -> Path:
    home = os.getenv("HOME")
    if home:
        return Path(home).expanduser()
    return Path.home()


def _load_json_file(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        logger.debug(f"{label} not found: {path}")
        return None
    if path.is_dir():
        logger.warning(f"{label} path is a directory, expected a file: {path}")
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {label}: {e}")
        return None


def _read_secret_from_file_descriptor(env_var: str) -> str | None:
    fd_value = os.getenv(env_var)
    if not fd_value:
        return None

    try:
        fd = int(fd_value)
    except ValueError:
        logger.warning(f"{env_var} must be an integer file descriptor, got: {fd_value}")
        return None

    try:
        secret = os.read(fd, 1024 * 1024).decode().strip()
    except OSError as e:
        logger.warning(f"Failed to read {env_var}: {e}")
        return None

    return secret or None


def _credential_from_direct_token(access_token: str, source: str) -> ClaudeCodeCredential | None:
    token = access_token.strip()
    if not token:
        return None
    return ClaudeCodeCredential(access_token=token, source=source)


def _iter_claude_code_credential_paths() -> list[Path]:
    paths: list[Path] = []
    override_path = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
    if override_path:
        paths.append(Path(override_path).expanduser())

    default_path = _home_dir() / ".claude/.credentials.json"
    if not paths or paths[-1] != default_path:
        paths.append(default_path)

    return paths


def _refresh_disabled() -> bool:
    return os.getenv("DEER_FLOW_DISABLE_CLAUDE_OAUTH_REFRESH", "").strip().lower() in {"1", "true", "yes", "on"}


def _refresh_endpoint() -> str:
    return os.getenv("CLAUDE_CODE_OAUTH_TOKEN_URL", "").strip() or _DEFAULT_CLAUDE_CODE_OAUTH_TOKEN_URL


def _refresh_client_id() -> str:
    return os.getenv("CLAUDE_CODE_OAUTH_CLIENT_ID", "").strip() or _DEFAULT_CLAUDE_CODE_OAUTH_CLIENT_ID


def _refresh_claude_code_access_token(refresh_token: str) -> dict[str, Any] | None:
    """Exchange a refresh_token for a fresh access_token via OAuth2 RFC 6749 §6.

    Returns the parsed JSON response (dict with at least `access_token`) on success,
    or None on any failure. No exceptions bubble up — the caller falls back to the
    existing "token expired" behavior when this returns None.
    """
    if not refresh_token:
        return None

    try:
        import httpx  # type: ignore
    except ImportError:
        logger.debug("httpx unavailable; skipping Claude OAuth auto-refresh")
        return None

    endpoint = _refresh_endpoint()
    client_id = _refresh_client_id()
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }

    # Try form-urlencoded first (OAuth2 RFC 6749 §3.2 spec) and fall back to JSON
    # in case Anthropic ever enforces the body encoding they document in binary logs.
    attempts: list[tuple[str, dict[str, Any]]] = [
        ("form", {"data": payload, "headers": {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}}),
        ("json", {"json": payload, "headers": {"Accept": "application/json"}}),
    ]

    response = None
    last_status: int | None = None
    last_body: str = ""
    try:
        with httpx.Client(timeout=_REFRESH_HTTP_TIMEOUT_SECONDS) as client:
            for encoding_name, kwargs in attempts:
                response = client.post(endpoint, **kwargs)
                last_status = response.status_code
                last_body = response.text[:500] if response.text else ""
                if response.status_code == 200:
                    break
                logger.debug(
                    "Claude OAuth refresh %s-encoded attempt returned HTTP %s: %s",
                    encoding_name,
                    response.status_code,
                    last_body,
                )
    except (httpx.HTTPError, OSError) as exc:
        logger.warning("Claude OAuth refresh network error against %s: %s", endpoint, exc)
        return None

    if response is None or response.status_code != 200:
        logger.warning(
            "Claude OAuth refresh failed: HTTP %s from %s. Body: %s",
            last_status,
            endpoint,
            last_body,
        )
        return None

    try:
        body = response.json()
    except ValueError:
        logger.warning("Claude OAuth refresh returned non-JSON body; giving up")
        return None

    if not isinstance(body, dict) or not body.get("access_token"):
        logger.warning("Claude OAuth refresh response missing access_token; body keys=%s", list(body.keys()) if isinstance(body, dict) else type(body).__name__)
        return None

    return body


def _merge_refreshed_credentials(existing: dict[str, Any], refreshed: dict[str, Any]) -> dict[str, Any]:
    """Merge a fresh token response into the on-disk credentials container.

    Preserves any fields we don't manage (scopes, subscriptionType, rateLimitTier,
    etc.) so Claude Desktop/CLI continue to observe the full shape they expect.
    """
    updated = dict(existing) if isinstance(existing, dict) else {}
    oauth = dict(updated.get("claudeAiOauth") or {})

    oauth["accessToken"] = refreshed["access_token"]
    new_refresh = refreshed.get("refresh_token")
    if isinstance(new_refresh, str) and new_refresh:
        oauth["refreshToken"] = new_refresh

    expires_in = refreshed.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        # Claude Desktop stores expiresAt in ms since epoch.
        oauth["expiresAt"] = int((time.time() + float(expires_in)) * 1000)

    updated["claudeAiOauth"] = oauth
    return updated


def _atomic_write_json(path: Path, data: dict[str, Any]) -> bool:
    """Write `data` to `path` atomically, preserving permissions. Returns True on success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Inherit existing file mode (Claude Desktop writes 0600).
        original_mode: int | None = None
        if path.exists():
            try:
                original_mode = path.stat().st_mode & 0o777
            except OSError:
                original_mode = None

        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=str(path.parent),
            prefix=".",
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)

        if original_mode is not None:
            os.chmod(tmp_path, original_mode)
        else:
            os.chmod(tmp_path, 0o600)

        os.replace(tmp_path, path)
        return True
    except OSError as exc:
        logger.warning("Failed to write refreshed Claude credentials back to %s: %s", path, exc)
        return False


def _extract_claude_code_credential(
    data: dict[str, Any],
    source: str,
    *,
    persist_path: Path | None = None,
) -> ClaudeCodeCredential | None:
    oauth = data.get("claudeAiOauth", {})
    access_token = oauth.get("accessToken", "")
    if not access_token:
        logger.debug("Claude Code credentials container exists but no accessToken found")
        return None

    cred = ClaudeCodeCredential(
        access_token=access_token,
        refresh_token=oauth.get("refreshToken", ""),
        expires_at=oauth.get("expiresAt", 0),
        source=source,
    )

    if not cred.is_expired:
        return cred

    # Token is expired. Attempt a standards-compliant refresh_token exchange.
    if _refresh_disabled():
        logger.warning(
            "Claude Code OAuth token is expired and auto-refresh is disabled "
            "(DEER_FLOW_DISABLE_CLAUDE_OAUTH_REFRESH=1). Launch Claude Desktop or Claude Code "
            "to refresh, then retry."
        )
        return None

    if not cred.refresh_token:
        logger.warning(
            "Claude Code OAuth token is expired and no refreshToken is stored; "
            "open Claude Desktop or run Claude Code to re-authenticate."
        )
        return None

    logger.info("Claude Code OAuth token expired; attempting refresh against %s", _refresh_endpoint())
    refreshed = _refresh_claude_code_access_token(cred.refresh_token)
    if refreshed is None:
        logger.warning("Claude Code OAuth refresh did not succeed; caller will see no credential.")
        return None

    # Persist rotated token if we were loaded from disk.
    if persist_path is not None:
        merged = _merge_refreshed_credentials(data, refreshed)
        if _atomic_write_json(persist_path, merged):
            logger.info(
                "Persisted refreshed Claude Code credentials to %s (expiresAt=%s)",
                persist_path,
                merged.get("claudeAiOauth", {}).get("expiresAt"),
            )

    new_oauth = refreshed
    expires_in = new_oauth.get("expires_in")
    new_expires_at = 0
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        new_expires_at = int((time.time() + float(expires_in)) * 1000)

    return ClaudeCodeCredential(
        access_token=new_oauth["access_token"],
        refresh_token=new_oauth.get("refresh_token") or cred.refresh_token,
        expires_at=new_expires_at,
        source=f"{source}-refreshed",
    )


def load_claude_code_credential() -> ClaudeCodeCredential | None:
    """Load OAuth credential from explicit Claude Code handoff sources.

    Lookup order:
      1. $CLAUDE_CODE_OAUTH_TOKEN or $ANTHROPIC_AUTH_TOKEN
      2. $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR
      3. $CLAUDE_CODE_CREDENTIALS_PATH
      4. ~/.claude/.credentials.json

    Exported credentials files contain:
    {
      "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-...",
        "refreshToken": "sk-ant-ort01-...",
        "expiresAt": 1773430695128,
        "scopes": ["user:inference", ...],
        ...
      }
    }
    """
    direct_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    if direct_token:
        cred = _credential_from_direct_token(direct_token, "claude-cli-env")
        if cred:
            logger.info("Loaded Claude Code OAuth credential from environment")
        return cred

    fd_token = _read_secret_from_file_descriptor("CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR")
    if fd_token:
        cred = _credential_from_direct_token(fd_token, "claude-cli-fd")
        if cred:
            logger.info("Loaded Claude Code OAuth credential from file descriptor")
        return cred

    override_path = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
    override_path_obj = Path(override_path).expanduser() if override_path else None
    for cred_path in _iter_claude_code_credential_paths():
        data = _load_json_file(cred_path, "Claude Code credentials")
        if data is None:
            continue
        cred = _extract_claude_code_credential(data, "claude-cli-file", persist_path=cred_path)
        if cred:
            source_label = "override path" if override_path_obj is not None and cred_path == override_path_obj else "plaintext file"
            logger.info(f"Loaded Claude Code OAuth credential from {source_label} (expires_at={cred.expires_at})")
            return cred

    return None


def load_codex_cli_credential(*, emit_log: bool = True) -> CodexCliCredential | None:
    """Load credential from Codex CLI (~/.codex/auth.json)."""
    cred_path = _resolve_credential_path("CODEX_AUTH_PATH", ".codex/auth.json")
    data = _load_json_file(cred_path, "Codex CLI credentials")
    if data is None:
        return None
    tokens = data.get("tokens", {})
    if not isinstance(tokens, dict):
        tokens = {}

    access_token = (
        data.get("access_token")
        or data.get("token")
        or tokens.get("access_token")
        or tokens.get("accessToken")
        or ""
    )
    if isinstance(access_token, str):
        access_token = access_token.strip()
    account_id = (
        data.get("account_id")
        or data.get("accountId")
        or tokens.get("account_id")
        or tokens.get("accountId")
        or ""
    )
    if isinstance(account_id, str):
        account_id = account_id.strip()
    if not access_token:
        logger.debug("Codex CLI credentials file exists but no token found")
        return None

    auth_mode = data.get("auth_mode")
    if auth_mode and str(auth_mode) != "chatgpt":
        logger.warning(
            "Codex auth.json auth_mode=%s — DeerFlow CodexChatModel expects ChatGPT OAuth (auth_mode=chatgpt). "
            "If requests fail after login, re-run `codex login` or check CODEX_AUTH_PATH.",
            auth_mode,
        )

    if emit_log:
        logger.info("Loaded Codex CLI credential")
    return CodexCliCredential(
        access_token=access_token,
        account_id=account_id,
        source="codex-cli",
    )
