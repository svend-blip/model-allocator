"""Validation engine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import os
import shutil

from model_allocator.adapters.freetoken import (
    FreeTokenAdapter, FreeTokenAdapterError,
)
from model_allocator.adapters import llama_cpp as llama_cpp_adapter
from model_allocator.adapters import ollama as ollama_adapter
from model_allocator.adapters import openai_compatible as openai_adapter
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

        self._check_client_backend_compatibility(resolved, client, result)

        backend = resolved.get("backend")
        if backend == "ollama":
            self._validate_ollama(resolved, client, result)
        elif backend == "openai_compatible":
            self._validate_openai_compatible(resolved, client, result)
        elif backend == "llama_cpp":
            self._validate_llama_cpp(resolved, client, result)
        elif backend == "sglang":
            self._validate_sglang(resolved, client, result)
        elif backend == "freetoken":
            self._validate_freetoken(resolved, client, result)
        elif backend == "external":
            self._validate_external(resolved, client, result)
        elif backend == "onyx":
            self._validate_onyx(resolved, client, result)
        elif backend == "anthropic":
            self._validate_anthropic(resolved, client, result)
        elif backend is None:
            result["errors"].append("Backend not declared in runtime profile")
            result["validation_status"] = "ERROR"
        else:
            result["errors"].append(f"Backend adapter '{backend}' is not implemented")
            result["validation_status"] = "ERROR"

        if result["errors"]:
            result["validation_status"] = "ERROR"
        elif result["warnings"]:
            result["validation_status"] = "WARNING"
        return result

    def _check_client_backend_compatibility(self, resolved: dict, client: str, result: dict) -> None:
        backend = resolved.get("backend")
        provider = resolved.get("provider", "")
        if client == "claude-code":
            if backend == "ollama":
                return
            if backend == "openai_compatible" and provider != "minimax":
                return
            if backend == "anthropic":
                return
            # llama.cpp serves an Anthropic-shaped endpoint that the
            # claude_code adapter has handled since it was written: its
            # llama_cpp branch sets ANTHROPIC_BASE_URL to the server and
            # unsets ANTHROPIC_API_KEY. Commit 8dee242 recorded it proven by
            # a live two-role run. This rule said otherwise for months, and
            # only `laguna-local` ever showed the resulting error — the four
            # other llama.cpp aliases that declare claude-code list opencode
            # first, and the UI validates against the FIRST enabled client,
            # so dict ordering hid the same condition on all of them.
            if backend == "llama_cpp":
                return
            result["errors"].append(
                f"Client 'claude-code' is incompatible with backend '{backend}' (provider '{provider}')"
            )
            result["validation_status"] = "ERROR"
            return
        if backend == "onyx" and client not in ("headless",):
            result["errors"].append(
                f"Client '{client}' is not supported for the onyx backend (use 'headless')"
            )
            result["validation_status"] = "ERROR"
            return
        # Locally served backends speak OpenAI Chat Completions, so any client
        # that can be pointed at a base URL works. Pi is the second such
        # client: it takes --provider/--model per invocation and reads the
        # endpoint from a custom provider in its models.json.
        if backend == "llama_cpp" and client not in ("opencode", "pi"):
            result["errors"].append(f"Client '{client}' is not supported for llama.cpp backend")
            result["validation_status"] = "ERROR"
            return
        if backend == "sglang" and client not in ("opencode", "pi"):
            result["errors"].append(f"Client '{client}' is not supported for sglang backend")
            result["validation_status"] = "ERROR"
            return
        if backend == "anthropic" and client != "claude-code":
            result["errors"].append(
                f"Client '{client}' is incompatible with backend 'anthropic'"
            )
            result["validation_status"] = "ERROR"

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

    def _validate_openai_compatible(self, resolved: dict, client: str, result: dict) -> None:
        api_base_env = resolved.get("api_base_env")
        api_key_env = resolved.get("api_key_env")
        if not api_base_env:
            result["warnings"].append("No API base environment variable configured")
        if not api_key_env:
            result["warnings"].append("No API key environment variable configured")

        api_base = openai_adapter.OpenAICompatibleAdapter.api_base_from_profile(resolved)
        result["resolved_api_base"] = api_base
        adapter = openai_adapter.OpenAICompatibleAdapter(api_base=api_base, api_key_env=api_key_env or "")

        credentials = adapter.are_credentials_present()
        if not credentials["present"]:
            result["warnings"].append(credentials["error"])

        reachable = adapter.is_api_reachable()
        if not reachable["reachable"]:
            result["warnings"].append(f"Cloud API base unreachable: {reachable['error']}")

        if not credentials["present"] or not reachable["reachable"]:
            result["client_support"][client] = "UNREACHABLE"

    def _validate_onyx(self, resolved: dict, client: str, result: dict) -> None:
        from model_allocator.adapters import onyx as onyx_adapter

        adapter = onyx_adapter.OnyxAdapter.from_resolved(resolved)
        status = adapter.status()
        if not status["reachable"]:
            result["errors"].append(
                f"ONYX endpoint unreachable: {status.get('error')}"
            )
        if not status["credentials_present"]:
            result["errors"].append(
                "ONYX credentials missing (api key or email/password env vars)"
            )
        if "invoke" not in (resolved.get("capabilities") or []):
            result["warnings"].append(
                "Profile does not declare the 'invoke' capability"
            )

    def _validate_anthropic(self, resolved: dict, client: str, result: dict) -> None:
        from model_allocator.adapters import anthropic as anthropic_adapter

        if resolved.get("credentials", "api_key") == "subscription":
            # Subscription mode deliberately has NO API key: the session uses
            # Claude Code's own login. Checking for ANTHROPIC_API_KEY here would
            # report the intended configuration as NO_CREDENTIALS, so check for
            # the login instead.
            login = Path.home() / ".claude" / ".credentials.json"
            if not login.is_file():
                result["warnings"].append(
                    "Profile uses credentials: subscription but no Claude Code "
                    f"login was found at {login}; run 'claude login'"
                )
                result["client_support"][client] = "NO_CREDENTIALS"
            return

        api_key_env = resolved.get("api_key_env", "ANTHROPIC_API_KEY")
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env=api_key_env)
        credentials = adapter.are_credentials_present()
        if not credentials["present"]:
            result["warnings"].append(credentials["error"])
            result["client_support"][client] = "NO_CREDENTIALS"

    def _validate_llama_cpp(self, resolved: dict, client: str, result: dict) -> None:
        try:
            adapter = llama_cpp_adapter.LlamaCppAdapter(resolved)
            server_bin = adapter.server_bin()
            if not os.path.isfile(server_bin):
                result["warnings"].append(f"llama-server binary not found: {server_bin}")
            model_path = adapter.model_path()
            if not os.path.isfile(model_path):
                result["warnings"].append(f"Model file not found: {model_path}")
            status = adapter.status()
            if status["running"]:
                result["client_support"][client] = "OK"
            else:
                result["warnings"].append(f"llama.cpp server not running on port {adapter.port}: {status['error']}")
                result["client_support"][client] = "NOT_RUNNING"
        except llama_cpp_adapter.LlamaCppAdapterError as exc:
            result["warnings"].append(str(exc))
            result["client_support"][client] = "UNREACHABLE"

    def _validate_sglang(self, resolved: dict, client: str, result: dict) -> None:
        """Probe the live server first; fall back to static launch checks.

        The static checks (model_path, venv) only matter for STARTING the
        server. When the health endpoint answers, the alias is usable and
        the status is OK — before this probe existed, every SGLang alias sat
        on WARNING and Start could never change it, because nothing here
        ever looked at the running server.
        """
        port = resolved.get("port") or resolved.get("default_port")
        if port:
            from model_allocator.adapters import sglang as sglang_adapter
            try:
                status = sglang_adapter.SGLangAdapter(resolved).status()
            except Exception as exc:
                status = {"running": False, "error": str(exc)}
            if status.get("running"):
                result["client_support"][client] = "OK"
                return
            result["warnings"].append(
                f"SGLang server not running on port {port} "
                f"({status.get('error') or 'health endpoint unreachable'}) — "
                "Start the alias to reach OK")
            result["client_support"][client] = "NOT_RUNNING"
        else:
            result["client_support"][client] = "OK"

        model_path = resolved.get("model_path", "")
        if not model_path:
            result["warnings"].append("model_path not configured for SGLang alias")
        venv = resolved.get("venv", "")
        if venv and not os.path.isdir(venv):
            result["warnings"].append(f"SGLang venv not found: {venv}")

    def _validate_freetoken(self, resolved: dict, client: str, result: dict) -> None:
        """Check what can be checked without starting a 27B model.

        Everything here is static or cheap: the model reference is present,
        the binary resolves, and the KV budget the profile asks for is
        declared. Nothing touches the GPU and nothing waits on a load — a
        validation that costs two minutes and 30 GB is a validation nobody
        runs.
        """
        model_path = resolved.get("model_path", "")
        if not model_path:
            result["errors"].append(
                "model_path not configured — FreeToken needs a local "
                "checkpoint directory or a Hugging Face repo ID")
        # Deliberately NOT os.path.exists(): a Hugging Face repo id is a
        # first-class model reference and rejecting it would fail exactly the
        # configuration this machine qualified.

        executable = resolved.get("executable") or resolved.get("server_bin_path") or ""
        executable_env = resolved.get("executable_env") or ""
        if executable:
            if not (os.path.isfile(executable) and os.access(executable, os.X_OK)):
                result["errors"].append(
                    f"FreeToken executable not found or not executable: {executable}")
        elif executable_env:
            # The profile pins a specific install through a named variable
            # (the Flash-Next qualification lives in its own venv). An unset
            # variable is a configuration error to name, not a cue to go
            # looking for some other `ft` on PATH.
            result["errors"].append(
                f"FreeToken executable unresolved: the runtime profile "
                f"expands `executable` from ${executable_env}, which is not "
                f"set. Export it to the qualified venv's `ft` — no PATH "
                f"fallback, a different ft is an unqualified runtime")
        elif not shutil.which("ft"):
            result["warnings"].append(
                "FreeToken executable unresolved: no `executable` on the "
                "runtime profile and no `ft` on PATH. The qualified install "
                "is a project-local venv, so PATH will not find it — set "
                "FREETOKEN_BIN or the profile's executable")

        if not resolved.get("num_tokens") and not resolved.get("num_pages"):
            result["warnings"].append(
                "No KV budget declared (num_tokens): FreeToken will size it "
                "from leftover VRAM, which on a full card lands far below the "
                "model's context — measured at 14303 tokens against an "
                "advertised 262144")

        # Checkpoint integrity, when the cache can answer. Filesystem reads
        # only — an index and a stat per shard — so it is cheap enough for
        # `list`. A warning rather than an error: validation says whether the
        # alias is well-formed, and start() is where an incomplete cache
        # refuses, with the full diagnostics.
        try:
            preflight = FreeTokenAdapter(resolved).checkpoint_preflight()
        except FreeTokenAdapterError as exc:
            preflight = {"checked": False, "ok": True, "error": str(exc)}
        if preflight.get("checked") and not preflight.get("ok"):
            result["warnings"].append(
                f"checkpoint preflight: {preflight.get('error')}")

        result["client_support"][client] = "OK"

    def _validate_external(self, resolved: dict, client: str, result: dict) -> None:
        """The `external` backend owns nothing, by design.

        A client like freebuff manages its own model and runtime; the profile
        exists only because every alias must name one. There is no adapter to
        implement and nothing to start, stop or reach, so the generic
        "adapter is not implemented" error was reporting a deliberate design
        choice as a fault.
        """
        result["client_support"][client] = "OK (client-managed runtime)"

    def format_output(self, result: dict) -> str:
        lines = [result["validation_status"]]
        lines.append(f"Logical model: {result['logical_model_alias']}")
        lines.append(f"Backend: {result.get('resolved_backend') or 'N/A'}")
        lines.append(f"Real model: {result.get('resolved_real_model') or 'N/A'}")
        lines.append(f"API base: {result.get('resolved_api_base') or 'N/A'}")
        lines.append("Client support:")
        for c, status in result.get("client_support", {}).items():
            lines.append(f"  {c}: {status}")
        lines.append(f"Context: {result.get('resolved_context') or result.get('context') or 'N/A'}")
        lines.append(f"GPU policy: {result.get('resolved_gpu') or result.get('gpu') or 'N/A'}")
        warnings = result.get("warnings", [])
        errors = result.get("errors", [])
        lines.append(f"Warnings: {', '.join(warnings) if warnings else 'none'}")
        lines.append(f"Errors: {', '.join(errors) if errors else 'none'}")
        return "\n".join(lines)
