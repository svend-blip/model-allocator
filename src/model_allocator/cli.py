"""Command-line interface for model-allocator V1A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from model_allocator.resolver import ResolutionError, Resolver
from model_allocator.validator import Validator
from model_allocator.adapters import ollama as ollama_adapter


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_WARNING = 2
EXIT_USAGE = 64


def _config_dir(args: argparse.Namespace) -> str | None:
    return args.config_dir if hasattr(args, "config_dir") and args.config_dir else None


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
        print(f"ERROR: status command for backend '{backend}' is not implemented in V1A", file=sys.stderr)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-allocator",
        description="DPMtF Model Allocator — V1A Minimal Allocator Proof",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory containing allocator config files (default: current directory)",
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
