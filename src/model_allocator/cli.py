"""Command-line interface for model-allocator V1B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from model_allocator.resolver import ResolutionError, Resolver
from model_allocator.validator import Validator
from model_allocator.adapters import opencode
from model_allocator.adapters import ollama as ollama_adapter
from model_allocator.renderer import render_tmux_shell_string


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_WARNING = 2
EXIT_USAGE = 64


def _default_config_dir() -> str:
    """Default config directory: model-allocator repo root if detectable, else cwd."""
    # cli.py lives at src/model_allocator/cli.py; the repo root is three parents up.
    repo_root = Path(__file__).resolve().parent.parent.parent
    if (repo_root / "models.yaml").exists():
        return str(repo_root)
    return str(Path.cwd())


def _config_dir(args: argparse.Namespace) -> str:
    return args.config_dir if hasattr(args, "config_dir") and args.config_dir else _default_config_dir()


def cmd_resolve(args: argparse.Namespace) -> int:
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        result = resolver.resolve_role_client(args.role, args.client)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps(result, indent=2, default=str))
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    validator = Validator(config_dir=_config_dir(args))
    result = validator.validate(args.alias, args.client)
    print(validator.format_output(result))
    if result["validation_status"] == "ERROR":
        return EXIT_ERROR
    if result["validation_status"] == "WARNING":
        return EXIT_WARNING
    return EXIT_OK


def cmd_list(args: argparse.Namespace) -> int:
    resolver = Resolver(config_dir=_config_dir(args))
    validator = Validator(resolver=resolver)
    aliases = resolver.list_aliases()
    entries = []
    for alias_name in aliases:
        result = validator.validate(alias_name, args.client)
        if args.only_ok and result["validation_status"] != "OK":
            continue
        entries.append({
            "alias": alias_name,
            "status": result["validation_status"],
            "backend": result.get("resolved_backend"),
            "real_model": result.get("resolved_real_model"),
        })
    print(json.dumps(entries, indent=2, default=str))
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_alias(args.alias)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    backend = resolved.get("backend")
    if backend != "ollama":
        print(f"ERROR: status command for backend '{backend}' is not implemented in V1B", file=sys.stderr)
        return EXIT_ERROR
    api_base = ollama_adapter.OllamaAdapter.api_base_from_profile(resolved)
    adapter = ollama_adapter.OllamaAdapter(api_base=api_base, real_model=resolved.get("real_model", ""))
    report = {
        "alias": args.alias,
        "backend": backend,
        "api_base": api_base,
        "reachable": adapter.is_api_reachable(),
        "model_available": adapter.is_model_available(),
        "runtime": adapter.runtime_status(),
    }
    print(json.dumps(report, indent=2, default=str))
    if not report["reachable"]["reachable"]:
        return EXIT_WARNING
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    """Print the tmux-safe shell string for starting a client against an alias."""
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_role_client(args.role, args.client)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if resolved.get("backend") != "ollama":
        print(f"ERROR: run command for backend '{resolved.get('backend')}' is not implemented in V1B", file=sys.stderr)
        return EXIT_ERROR

    clients = resolved.get("clients", {})
    if not clients.get(args.client):
        print(f"ERROR: client '{args.client}' is not supported by alias '{resolved.get('alias')}'", file=sys.stderr)
        return EXIT_ERROR

    if args.client == "opencode":
        config_dir = resolved.get("config_dir") or args.role
        try:
            command_object = opencode.build_opencode_command(resolved, config_dir)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_ERROR
    else:
        print(f"ERROR: run command for client '{args.client}' is not implemented in V1B", file=sys.stderr)
        return EXIT_ERROR

    shell_string = render_tmux_shell_string(command_object)
    print(shell_string)
    return EXIT_OK


def cmd_start(args: argparse.Namespace) -> int:
    """Warm up the backend runtime for an alias."""
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_alias(args.alias)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if resolved.get("backend") != "ollama":
        print(f"ERROR: start command for backend '{resolved.get('backend')}' is not implemented in V1B", file=sys.stderr)
        return EXIT_ERROR

    api_base = ollama_adapter.OllamaAdapter.api_base_from_profile(resolved)
    adapter = ollama_adapter.OllamaAdapter(
        api_base=api_base,
        real_model=resolved.get("real_model", ""),
        context=resolved.get("context"),
    )
    result = adapter.start_model()
    print(json.dumps(result, indent=2, default=str))
    if result["started"]:
        return EXIT_OK
    return EXIT_WARNING


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop the backend runtime for an alias."""
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_alias(args.alias)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if resolved.get("backend") != "ollama":
        print(f"ERROR: stop command for backend '{resolved.get('backend')}' is not implemented in V1B", file=sys.stderr)
        return EXIT_ERROR

    api_base = ollama_adapter.OllamaAdapter.api_base_from_profile(resolved)
    adapter = ollama_adapter.OllamaAdapter(
        api_base=api_base,
        real_model=resolved.get("real_model", ""),
        context=resolved.get("context"),
    )
    result = adapter.stop_model(timeout=args.timeout)
    print(json.dumps(result, indent=2, default=str))
    if result["stopped"]:
        return EXIT_OK
    return EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-allocator",
        description="DPMtF Model Allocator — V1B First bridgeV002 Pilot",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory containing allocator config files (default: model-allocator repo root)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="Resolve the effective alias/backend/model for a role/client")
    p_resolve.add_argument("--role", required=True, help="Role key")
    p_resolve.add_argument("--client", required=True, help="Client key (e.g. opencode)")
    p_resolve.set_defaults(func=cmd_resolve)

    p_validate = sub.add_parser("validate", help="Check whether an alias is usable for a client")
    p_validate.add_argument("--alias", required=True, help="Logical alias name")
    p_validate.add_argument("--client", required=True, help="Client key (e.g. opencode)")
    p_validate.set_defaults(func=cmd_validate)

    p_list = sub.add_parser("list", help="List configured aliases")
    p_list.add_argument("--only-ok", action="store_true", help="Show only aliases that validate as OK")
    p_list.add_argument("--client", default="opencode", help="Client key to validate against")
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="Report backend/runtime status for an alias")
    p_status.add_argument("--alias", required=True, help="Logical alias name")
    p_status.set_defaults(func=cmd_status)

    p_run = sub.add_parser("run", help="Render the tmux-safe shell string for a role/client")
    p_run.add_argument("--role", required=True, help="Role key")
    p_run.add_argument("--client", required=True, help="Client key (e.g. opencode)")
    p_run.set_defaults(func=cmd_run)

    p_start = sub.add_parser("start", help="Warm up the backend runtime for an alias")
    p_start.add_argument("--alias", required=True, help="Logical alias name")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop the backend runtime for an alias")
    p_stop.add_argument("--alias", required=True, help="Logical alias name")
    p_stop.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for the stop command")
    p_stop.set_defaults(func=cmd_stop)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
