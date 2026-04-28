"""Allow `python -m deerflow <subcommand>` (same as `deerflow` console script when installed)."""

from __future__ import annotations

import sys

from deerflow.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
