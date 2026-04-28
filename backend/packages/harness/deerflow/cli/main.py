"""Console entry: `deerflow doctor`, `deerflow doctor tools`, `deerflow repair tools`."""

from __future__ import annotations

import argparse
import json
import sys


def _print_tool_diagnostics() -> None:
    from deerflow.sandbox.sandbox_provider import get_sandbox_provider

    provider = get_sandbox_provider()
    diag = getattr(provider, "tool_diagnostics_snapshot", None)
    if callable(diag):
        data = diag()
    else:
        from deerflow.community.aio_sandbox.tool_runtime_diagnostics import snapshot

        data = snapshot()
        data["note"] = "Current sandbox provider has no extended tool snapshot; showing global diagnostics only."
    print(json.dumps(data, indent=2, default=str))


def _cmd_doctor_general() -> int:
    from deerflow.config import get_app_config

    cfg = get_app_config()
    print("Deer-Flow doctor")
    print(f"  sandbox.use: {cfg.sandbox.use}")
    print("Tool runtime (sandbox HTTP bridge):")
    _print_tool_diagnostics()
    return 0


def _cmd_doctor_tools() -> int:
    _print_tool_diagnostics()
    return 0


def _cmd_repair_tools() -> int:
    from deerflow.community.aio_sandbox.tool_runtime_diagnostics import record_recovery, snapshot
    from deerflow.sandbox.sandbox_provider import reset_sandbox_provider, shutdown_sandbox_provider

    print("Shutting down sandbox provider and clearing singleton (next tool use will recreate sandboxes).")
    shutdown_sandbox_provider()
    reset_sandbox_provider()
    record_recovery("cli_repair_tools")
    print(json.dumps(snapshot(), indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deerflow")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Show environment and tool-runtime diagnostics")
    doc_sub = p_doc.add_subparsers(dest="doctor_sub", required=False)
    doc_sub.add_parser("tools", help="Only print tool-bridge JSON diagnostics")
    p_doc.set_defaults(func=None, doctor_sub=None)

    p_rt = sub.add_parser("repair", help="Repair local tool runtime (reset sandbox provider)")
    rep_sub = p_rt.add_subparsers(dest="repair_sub", required=True)
    rep_sub.add_parser("tools", help="Reset sandbox singleton / stale HTTP bridge handles")

    args = parser.parse_args(argv)

    if args.command == "doctor":
        if getattr(args, "doctor_sub", None) == "tools":
            return _cmd_doctor_tools()
        return _cmd_doctor_general()

    if args.command == "repair" and args.repair_sub == "tools":
        return _cmd_repair_tools()

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
