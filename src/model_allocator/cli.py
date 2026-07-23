"""Command-line interface for model-allocator V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from model_allocator.resolver import ResolutionError, Resolver
from model_allocator.validator import Validator
from model_allocator import config_writer
from model_allocator.doctor_cli import cmd_doctor
from model_allocator.adapters import claude_code
from model_allocator.adapters import llama_cpp as llama_cpp_adapter
from model_allocator.adapters import opencode
from model_allocator.adapters import ollama as ollama_adapter
from model_allocator.adapters import openai_compatible as openai_adapter
from model_allocator.adapters import onyx as onyx_adapter
from model_allocator.adapters import headless as headless_adapter
from model_allocator.renderer import render_tmux_shell_string


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_WARNING = 2
EXIT_USAGE = 64


def _default_config_dir() -> str:
    """Default config directory: model-allocator repo root if detectable, else cwd."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    if (repo_root / "models.yaml").exists():
        return str(repo_root)
    return str(Path.cwd())


def _config_dir(args: argparse.Namespace) -> str:
    return args.config_dir if hasattr(args, "config_dir") and args.config_dir else _default_config_dir()


def _get_backend_adapter(resolved: dict):
    backend = resolved.get("backend")
    if backend == "ollama":
        api_base = ollama_adapter.OllamaAdapter.api_base_from_profile(resolved)
        return ollama_adapter.OllamaAdapter(
            api_base=api_base,
            real_model=resolved.get("real_model", ""),
            context=resolved.get("context"),
        )
    if backend == "openai_compatible":
        return openai_adapter.OpenAICompatibleAdapter(
            api_base=openai_adapter.OpenAICompatibleAdapter.api_base_from_profile(resolved),
            api_key_env=resolved.get("api_key_env", ""),
        )
    if backend == "llama_cpp":
        return llama_cpp_adapter.LlamaCppAdapter(resolved)
    if backend == "onyx":
        return onyx_adapter.OnyxAdapter.from_resolved(resolved)
    raise ValueError(f"Unsupported backend: {backend}")


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
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
    else:
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

    try:
        adapter = _get_backend_adapter(resolved)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    backend = resolved.get("backend")
    report = {"alias": args.alias, "backend": backend}
    if backend == "ollama":
        report.update({
            "api_base": adapter.api_base,
            "reachable": adapter.is_api_reachable(),
            "model_available": adapter.is_model_available(),
            "runtime": adapter.runtime_status(),
        })
    elif backend == "openai_compatible":
        report.update(adapter.status())
    elif backend == "llama_cpp":
        report.update(adapter.status())

    print(json.dumps(report, indent=2, default=str))
    if backend == "ollama" and not report["reachable"]["reachable"]:
        return EXIT_WARNING
    if backend in ("openai_compatible", "llama_cpp") and not report.get("running", False):
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

    clients = resolved.get("clients", {})
    if not clients.get(args.client):
        print(f"ERROR: client '{args.client}' is not supported by alias '{resolved.get('alias')}'", file=sys.stderr)
        return EXIT_ERROR

    try:
        # Apply CLI overrides before building command
        if getattr(args, "max_output_tokens", None) is not None:
            resolved["max_output_tokens"] = args.max_output_tokens

        if args.client == "opencode":
            config_dir = resolved.get("config_dir") or args.role
            command_object = opencode.build_opencode_command(resolved, config_dir)
        elif args.client == "claude-code":
            command_object = claude_code.build_claude_code_command(resolved)
        elif args.client == "headless":
            command_object = headless_adapter.build_headless_command(resolved, args.role)
        else:
            print(f"ERROR: run command for client '{args.client}' is not implemented in V2", file=sys.stderr)
            return EXIT_ERROR
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    shell_string = render_tmux_shell_string(command_object)
    print(shell_string)
    return EXIT_OK


def cmd_start(args: argparse.Namespace) -> int:
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_alias(args.alias)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        adapter = _get_backend_adapter(resolved)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    backend = resolved.get("backend")
    if backend == "ollama":
        result = adapter.start_model()
    elif backend == "openai_compatible":
        result = adapter.start()
    elif backend == "llama_cpp":
        result = adapter.start(timeout=args.timeout)
    else:
        print(f"ERROR: start not implemented for backend '{backend}'", file=sys.stderr)
        return EXIT_ERROR

    print(json.dumps(result, indent=2, default=str))
    if result.get("started"):
        return EXIT_OK
    return EXIT_WARNING


def cmd_stop(args: argparse.Namespace) -> int:
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_alias(args.alias)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        adapter = _get_backend_adapter(resolved)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    backend = resolved.get("backend")
    if backend == "ollama":
        result = adapter.stop_model(timeout=args.timeout)
    elif backend == "openai_compatible":
        result = adapter.stop()
    elif backend == "llama_cpp":
        result = adapter.stop(timeout=args.timeout)
    else:
        print(f"ERROR: stop not implemented for backend '{backend}'", file=sys.stderr)
        return EXIT_ERROR

    print(json.dumps(result, indent=2, default=str))
    if result.get("stopped"):
        return EXIT_OK
    return EXIT_ERROR


def cmd_unload(args: argparse.Namespace) -> int:
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_alias(args.alias)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        adapter = _get_backend_adapter(resolved)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    backend = resolved.get("backend")
    if backend == "ollama":
        result = adapter.stop_model(timeout=args.timeout)
    elif backend == "openai_compatible":
        result = adapter.unload()
    elif backend == "llama_cpp":
        result = adapter.unload(timeout=args.timeout)
    else:
        print(f"ERROR: unload not implemented for backend '{backend}'", file=sys.stderr)
        return EXIT_ERROR

    print(json.dumps(result, indent=2, default=str))
    if result.get("stopped") or result.get("unloaded"):
        return EXIT_OK
    return EXIT_ERROR


def cmd_preflight(args: argparse.Namespace) -> int:
    """Resolve + validate + start + status in one shot."""
    resolver = Resolver(config_dir=_config_dir(args))
    validator = Validator(resolver=resolver)
    try:
        resolved = resolver.resolve_role_client(args.role, args.client)
    except ResolutionError as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        return EXIT_ERROR

    alias = resolved.get("alias", "")
    backend = resolved.get("backend", "")
    validation = validator.validate(alias, args.client)
    if validation["validation_status"] == "ERROR":
        print(f"NOT READY: validation failed: {'; '.join(validation['errors'])}", file=sys.stderr)
        return EXIT_ERROR

    try:
        adapter = _get_backend_adapter(resolved)
    except ValueError as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if backend == "ollama":
        start_result = adapter.start_model()
    elif backend == "openai_compatible":
        start_result = adapter.start()
    elif backend == "llama_cpp":
        start_result = adapter.start(timeout=args.start_timeout)
    else:
        print(f"NOT READY: backend '{backend}' not supported", file=sys.stderr)
        return EXIT_ERROR

    if not start_result.get("started"):
        print(f"NOT READY: start failed: {start_result.get('error')}", file=sys.stderr)
        return EXIT_ERROR

    print(f"READY: {alias} ({backend}) for client {args.client}")
    return EXIT_OK


def cmd_env(args: argparse.Namespace) -> int:
    """Print export statements for Claude Code environment variables."""
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_role_client(args.role, args.client)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        command_object = claude_code.build_claude_code_command(resolved)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    for key, value in command_object["env"].items():
        print(f"export {key}={value}")
    return EXIT_OK


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically using a temp file + rename in the same directory."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _merge_opencode_config(existing: dict, rendered: dict, provider_key: str) -> dict:
    """Merge rendered config into existing opencode.json.

    Preserves existing top-level keys (permission, mcp, $schema, other providers,
    etc.) while setting/updating the top-level ``model`` field and the provider
    block for the role's backend.
    """
    merged = dict(existing)
    if "model" in rendered:
        merged["model"] = rendered["model"]

    rendered_provider = rendered.get("provider", {})
    if rendered_provider:
        merged.setdefault("provider", {})
        merged["provider"] = dict(merged["provider"])
        for provider_name, provider_config in rendered_provider.items():
            merged["provider"][provider_name] = provider_config

    return merged


def cmd_render_config(args: argparse.Namespace) -> int:
    """Emit opencode.json content for a role/client."""
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_role_client(args.role, args.client)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.client != "opencode":
        print(f"ERROR: render-config only supports client 'opencode', got '{args.client}'", file=sys.stderr)
        return EXIT_ERROR

    config = opencode.build_opencode_config(resolved)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        provider_key = resolved.get("opencode_provider_name") or resolved.get("provider", "")

        if output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"ERROR: existing opencode.json is not valid JSON: {exc}", file=sys.stderr)
                return EXIT_ERROR
            config = _merge_opencode_config(existing, config, provider_key)

        _atomic_write_json(output_path, config)
        print(f"Config written to {output_path}")
        return EXIT_OK

    print(json.dumps(config, indent=2, default=str))
    return EXIT_OK


def cmd_config_show(args: argparse.Namespace) -> int:
    raw = config_writer.load_raw(_config_dir(args))
    print(json.dumps(raw, indent=2, default=str))
    return EXIT_OK


def _parse_json_arg(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise config_writer.ConfigWriteError(f"invalid --json payload: {exc}")
    if not isinstance(value, dict):
        raise config_writer.ConfigWriteError("--json payload must be a JSON object")
    return value


def _config_write(args: argparse.Namespace, action) -> int:
    try:
        action()
    except config_writer.ConfigWriteError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return EXIT_ERROR
    print(json.dumps({"ok": True, "name": args.name}))
    return EXIT_OK


def cmd_config_set_alias(args: argparse.Namespace) -> int:
    try:
        definition = _parse_json_arg(args.json)
    except config_writer.ConfigWriteError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return EXIT_ERROR
    return _config_write(args, lambda: config_writer.set_alias(_config_dir(args), args.name, definition))


def cmd_config_delete_alias(args: argparse.Namespace) -> int:
    return _config_write(args, lambda: config_writer.delete_alias(_config_dir(args), args.name))


def cmd_config_set_role(args: argparse.Namespace) -> int:
    try:
        definition = _parse_json_arg(args.json)
    except config_writer.ConfigWriteError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return EXIT_ERROR
    return _config_write(args, lambda: config_writer.set_role(_config_dir(args), args.name, definition))


def cmd_config_delete_role(args: argparse.Namespace) -> int:
    return _config_write(args, lambda: config_writer.delete_role(_config_dir(args), args.name))


def cmd_invoke(args: argparse.Namespace) -> int:
    """One-shot invocation of an invoke-capable alias (InvokeResult JSON).

    The prompt comes from --prompt or stdin. Capability is declared as
    data on the runtime profile (capabilities: [invoke]) — providers
    without it are rejected here, not inside adapters.
    """
    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_alias(args.alias)
    except ResolutionError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1
    capabilities = resolved.get("capabilities") or []
    if "invoke" not in capabilities:
        print(json.dumps({
            "status": "error",
            "error": (
                f"Alias '{args.alias}' (backend "
                f"'{resolved.get('backend')}') does not declare the "
                "'invoke' capability"
            ),
        }))
        return 1
    prompt = args.prompt
    if prompt is None or prompt == "-":
        prompt = sys.stdin.read()
    if not prompt or not prompt.strip():
        print(json.dumps({"status": "error", "error": "Empty prompt"}))
        return 1
    adapter = _get_backend_adapter(resolved)
    kwargs = {}
    if getattr(args, "timeout", None):
        kwargs["timeout"] = args.timeout
    if getattr(args, "persona", None) is not None:
        kwargs["persona_id"] = args.persona
    result = adapter.invoke(prompt, **kwargs)
    result.setdefault("metadata", {})["alias"] = args.alias
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "ok" else 1


def cmd_headless(args: argparse.Namespace) -> int:
    """Run the headless client loop (stdin prompts -> invoke -> stdout)."""
    from model_allocator import headless as headless_runner

    try:
        return headless_runner.main_from_args(
            args, lambda: Resolver(config_dir=_config_dir(args))
        )
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR


def cmd_mcp_serve(args: argparse.Namespace) -> int:
    """Serve onyx-mcp tools over MCP streamable-http (optional extra)."""
    from model_allocator import mcp_server

    try:
        return mcp_server.serve(args.alias, host=args.host, port=args.port)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-allocator",
        description="DPMtF Model Allocator — V2",
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
    p_validate.add_argument("--json", action="store_true", help="Output structured JSON instead of text")
    p_validate.set_defaults(func=cmd_validate)

    p_list = sub.add_parser("list", help="List configured aliases")
    p_list.add_argument("--only-ok", action="store_true", help="Show only aliases that validate as OK")
    p_list.add_argument("--client", default="opencode", help="Client key to validate against")
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="Report backend/runtime status for an alias")
    p_status.add_argument("--alias", required=True, help="Logical alias name")
    p_status.set_defaults(func=cmd_status)

    p_invoke = sub.add_parser("invoke", help="One-shot invocation of an invoke-capable alias (InvokeResult JSON)")
    p_invoke.add_argument("--alias", required=True, help="Logical alias name")
    p_invoke.add_argument("--prompt", default=None, help="Prompt text ('-' or omitted = read stdin)")
    p_invoke.add_argument("--timeout", type=int, default=None, help="Invoke timeout in seconds")
    p_invoke.add_argument("--persona", type=int, default=None, help="Provider persona/assistant override")
    p_invoke.set_defaults(func=cmd_invoke)

    p_headless = sub.add_parser("headless", help="Run the headless client loop in a tmux session (stdin prompts -> invoke)")
    p_headless.add_argument("--alias", required=True, help="Invoke-capable alias")
    p_headless.add_argument("--output-dir", default=None, help="Directory for per-invoke InvokeResult JSON files")
    p_headless.add_argument("--idle-seconds", type=float, default=2.0, help="Idle gap that ends a pasted prompt (default 2.0)")
    p_headless.add_argument("--timeout", type=int, default=None, help="Per-invoke timeout in seconds")
    p_headless.add_argument("--persona", type=int, default=None, help="Provider persona/assistant override")
    p_headless.set_defaults(func=cmd_headless)

    p_mcp = sub.add_parser("mcp-serve", help="Serve onyx_answer/onyx_status MCP tools for an invoke-capable alias")
    p_mcp.add_argument("--alias", default="company-knowledge", help="Invoke-capable alias (default company-knowledge)")
    p_mcp.add_argument("--host", default="127.0.0.1")
    p_mcp.add_argument("--port", type=int, default=9164)
    p_mcp.set_defaults(func=cmd_mcp_serve)

    p_run = sub.add_parser("run", help="Render the tmux-safe shell string for a role/client")
    p_run.add_argument("--role", required=True, help="Role key")
    p_run.add_argument("--client", required=True, help="Client key (e.g. opencode, claude-code)")
    p_run.add_argument("--max-output-tokens", type=int, default=None, help="Override max_output_tokens for Claude Code roles")
    p_run.set_defaults(func=cmd_run)

    p_start = sub.add_parser("start", help="Warm up the backend runtime for an alias")
    p_start.add_argument("--alias", required=True, help="Logical alias name")
    p_start.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="Stop the backend runtime for an alias")
    p_stop.add_argument("--alias", required=True, help="Logical alias name")
    p_stop.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    p_stop.set_defaults(func=cmd_stop)

    p_unload = sub.add_parser("unload", help="Free model memory for an alias")
    p_unload.add_argument("--alias", required=True, help="Logical alias name")
    p_unload.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    p_unload.set_defaults(func=cmd_unload)

    p_preflight = sub.add_parser("preflight", help="Resolve + validate + start + reachability check")
    p_preflight.add_argument("--role", required=True, help="Role key")
    p_preflight.add_argument("--client", required=True, help="Client key")
    p_preflight.add_argument("--start-timeout", type=int, default=120, help="Start timeout in seconds")
    p_preflight.set_defaults(func=cmd_preflight)

    p_env = sub.add_parser("env", help="Print Claude Code environment variables as export statements")
    p_env.add_argument("--role", required=True, help="Role key")
    p_env.add_argument("--client", required=True, help="Client key (claude-code)")
    p_env.set_defaults(func=cmd_env)

    p_render = sub.add_parser("render-config", help="Emit opencode.json content for a role/client")
    p_render.add_argument("--role", required=True, help="Role key")
    p_render.add_argument("--client", required=True, help="Client key (opencode)")
    p_render.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write opencode.json atomically (merges with existing file)",
    )
    p_render.set_defaults(func=cmd_render_config)

    p_config = sub.add_parser("config", help="Read/write allocator config (aliases, roles)")
    config_sub = p_config.add_subparsers(dest="config_command", required=True)

    c_show = config_sub.add_parser("show", help="Print full config (aliases, roles, profiles) as JSON")
    c_show.set_defaults(func=cmd_config_show)

    c_set_alias = config_sub.add_parser("set-alias", help="Create/update an alias")
    c_set_alias.add_argument("--name", required=True, help="Alias name")
    c_set_alias.add_argument("--json", required=True, help="Alias definition as a JSON object")
    c_set_alias.set_defaults(func=cmd_config_set_alias)

    c_del_alias = config_sub.add_parser("delete-alias", help="Delete an alias")
    c_del_alias.add_argument("--name", required=True, help="Alias name")
    c_del_alias.set_defaults(func=cmd_config_delete_alias)

    c_set_role = config_sub.add_parser("set-role", help="Create/update a role")
    c_set_role.add_argument("--name", required=True, help="Role name")
    c_set_role.add_argument("--json", required=True, help="Role definition as a JSON object")
    c_set_role.set_defaults(func=cmd_config_set_role)

    c_del_role = config_sub.add_parser("delete-role", help="Delete a role")
    c_del_role.add_argument("--name", required=True, help="Role name")
    c_del_role.set_defaults(func=cmd_config_delete_role)

    p_doctor = sub.add_parser("doctor", help="Validate config files for errors and warnings")
    p_doctor.add_argument("--json", action="store_true", help="Output JSON instead of text")
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
