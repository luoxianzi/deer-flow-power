#!/usr/bin/env sh
# Deer-Flow CLI inside power-stack containers (PYTHONPATH matches compose services).
export PYTHONPATH=/app/backend/packages/harness:/app/backend
cd /app/backend && exec uv run python -m deerflow "$@"
