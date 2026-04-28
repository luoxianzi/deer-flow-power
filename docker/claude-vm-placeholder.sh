#!/bin/sh
# Placeholder bind-mount target for /usr/local/bin/claude in LangGraph when no real
# Claude Code VM binary is available. Keeps Docker from failing on bad host paths.
printf '%s\n' "claude-vm-placeholder: set POWER_CLAUDE_VM_CLI_PATH to your Claude Code VM claude executable." >&2
exit 127
