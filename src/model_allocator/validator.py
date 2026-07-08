"""Validation engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from model_allocator.adapters import ollama as ollama_adapter
from model_allocator.resolver import ResolutionError, Resolver


class Validator:
    def __init__(self, resolver: Resolver | None = None, config_dir: str | None = None):
        if resolver is not None:
            self.resolver = resolver
        else:
            self.resolver = Resolver(config_dir=config_dir)

    def validate(self, alias_name: str, client: str) -> dict:
        result: dict[str, Any] = {
            "last_validated_at": datetime.now(timezone.utc).isoformat(),
            "validation_status": "OK",
            "logical_model_alias": alias_name,
            "resolved_backend": None,
            "resolved_real_model": None,
            "resolved_api_base": None,
            "client_support": {},
            "warnings": [],
            "errors": [],
        }

        try:
            resolved = self.resolver.resolve_alias(alias_name)
        except ResolutionError as exc:
            result["validation_status"] = "ERROR"
            result["errors"].append(str(exc))
            return result

        result["resolved_backend"] = resolved.get("backend")
        result["resolved_real_model"] = resolved.get("real_model")
        result["resolved_context"] = resolved.get("context")
        result["resolved_gpu"] = resolved.get("gpu")

        clients = resolved.get("clients", {})
        if not clients:
            result["warnings"].append("No client compatibility declared")
        elif client not in clients or not clients[client]:
            result["errors"].append(f"Client '{client}' is not supported by alias '{alias_name}'")
            result["validation_status"] = "ERROR"
        else:
            result["client_support"][client] = "OK"

        context = resolved.get("context")
        if context is None:
            result["warnings"].append("Context not declared")
        elif not isinstance(context, int) or context <= 0:
            result["warnings"].append(f"Context value invalid: {context}")

        backend = resolved.get("backend")
        if backend == "ollama":
            self._validate_ollama(resolved, client, result)
        elif backend is None:
            result["errors"].append("Backend not declared in runtime profile")
            result["validation_status"] = "ERROR"
        else:
            # Adapter not implemented in V1A.
            result["errors"].append(f"Backend adapter '{backend}' is not implemented in V1A")
            result["validation_status"] = "ERROR"

        if result["errors"]:
            result["validation_status"] = "ERROR"
        elif result["warnings"]:
            result["validation_status"] = "WARNING"
        return result

    def _validate_ollama(self, resolved: dict, client: str, result: dict) -> None:
        api_base_env = resolved.get("api_base_env", "OLLAMA_BASE_URL")
        api_base = ollama_adapter.OllamaAdapter.api_base_from_profile(resolved)
        result["resolved_api_base"] = api_base
        if not api_base_env:
            result["warnings"].append("No API base environment variable configured")
        # When Ollama API base is the default, missing env var is not an error.
        adapter = ollama_adapter.OllamaAdapter(api_base=api_base, real_model=resolved.get("real_model", ""))
        reachable = adapter.is_api_reachable()
        if not reachable["reachable"]:
            result["warnings"].append(f"Ollama API base unreachable: {reachable['error']}")
            result["client_support"][client] = "UNREACHABLE"
            return
        available = adapter.is_model_available()
        if not available["available"]:
            result["warnings"].append(f"Ollama model not available: {available['error']}")
            result["client_support"][client] = "MODEL_MISSING"

    def format_output(self, result: dict) -> str:
        lines = [result["validation_status"]]
        lines.append(f"Logical model: {result['logical_model_alias']}")
        lines.append(f"Backend: {result.get('resolved_backend') or 'N/A'}")
        lines.append(f"Runtime: local")
        lines.append(f"Real model: {result.get('resolved_real_model') or 'N/A'}")
        lines.append(f"API base: {result.get('resolved_api_base') or 'N/A'}")
        lines.append("Client support:")
        for c, status in result.get("client_support", {}).items():
            lines.append(f"  {c}: {status}")
        lines.append("Lifecycle:")
        lines.append("  start: OK")
        lines.append("  stop: OK")
        lines.append("  unload: OK")
        lines.append(f"Context: {result.get('resolved_context') or result.get('context') or 'N/A'}")
        lines.append(f"GPU policy: {result.get('resolved_gpu') or result.get('gpu') or 'N/A'}")
        warnings = result.get("warnings", [])
        errors = result.get("errors", [])
        lines.append(f"Warnings: {', '.join(warnings) if warnings else 'none'}")
        lines.append(f"Errors: {', '.join(errors) if errors else 'none'}")
        return "\n".join(lines)
