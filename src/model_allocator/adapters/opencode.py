"""OpenCode client adapter for model-allocator."""

from __future__ import annotations

import os
import shutil
from typing import Any


def _resolve_opencode_bin() -> str:
    opencode_bin = os.environ.get("OPENCODE_BIN", "opencode")
    if os.path.isabs(opencode_bin):
        if not (os.path.isfile(opencode_bin) and os.access(opencode_bin, os.X_OK)):
            raise ValueError(f"OpenCode binary not found: {opencode_bin}")
        return opencode_bin
    if shutil.which(opencode_bin) is None:
        raise ValueError(f"OpenCode binary not found on PATH: {opencode_bin}")
    return opencode_bin


def _config_env(config_dir: str) -> dict[str, str]:
    config_base = os.environ.get("OPENCODE_ROLES_CONFIG_BASE", "$HOME/.config/opencode-roles")
    full_config_dir = f"{config_base}/{config_dir}"
    return {
        "OPENCODE_CONFIG_DIR": full_config_dir,
        "OPENCODE_CONFIG": f"{full_config_dir}/opencode.json",
    }


def _ollama_v1_mode(resolved: dict) -> bool:
    """True when the alias opts into OpenCode's openai-compatible provider
    against Ollama's /v1 endpoint instead of the built-in ollama provider.

    OpenCode's built-in ollama provider fails to deliver structured tool
    calls for some models (live incident 2026-07-27: qwen3-coder emitted
    its native XML tool syntax as plain text and no tool ever executed).
    Ollama's OpenAI-compatible /v1 endpoint returns structured tool_calls
    for the same models, so routing through @ai-sdk/openai-compatible
    fixes tool calling without touching the model or server.
    """
    return resolved.get("opencode_ollama_mode") == "openai_compatible"


def _ollama_v1_provider_name(resolved: dict) -> str:
    return resolved.get("opencode_provider_name") or "ollama-v1"


def _model_arg(resolved: dict) -> str:
    backend = resolved.get("backend")
    real_model = resolved.get("real_model", "")
    provider = resolved.get("provider", "")

    if backend == "ollama":
        if _ollama_v1_mode(resolved):
            return f"{_ollama_v1_provider_name(resolved)}/{real_model}"
        return f"ollama/{real_model}"
    if backend == "openai_compatible":
        if provider == "openrouter":
            return f"openrouter/{real_model}"
        if provider == "minimax":
            return real_model
        return real_model
    if backend == "llama_cpp":
        provider_name = resolved.get("opencode_provider_name") or provider or "llama-local"
        model_id = resolved.get("opencode_model_id") or real_model or "model"
        return f"{provider_name}/{model_id}"
    return real_model


def build_opencode_command(resolved: dict, config_dir: str) -> dict[str, Any]:
    """Build an OpenCode command object equivalent to command_builder.

    Supports Ollama, OpenAI-compatible cloud (OpenRouter, Minimax), and
    llama.cpp backends. Paths are never hardcoded.

    NOTE: The --model flag is NOT included — OpenCode's TUI silently ignores
    it on session resumption (live incident: Kimi/OpenRouter instead of
    qwen3-coder/Ollama). The model is set via the `model` field in
    opencode.json, which must be refreshed by `run` before emitting the
    shell string (see cli.py cmd_run auto-refresh logic).
    """
    real_model = resolved.get("real_model", "")
    if not real_model:
        raise ValueError("Resolved alias is missing real_model")

    opencode_bin = _resolve_opencode_bin()

    return {
        "env": _config_env(config_dir),
        "argv": [opencode_bin],
    }


def build_opencode_config(resolved: dict) -> dict[str, Any]:
    """Emit an opencode.json content dict for the resolved alias."""
    backend = resolved.get("backend")
    provider = resolved.get("provider", "")

    # Top-level model field is required by the OpenCode TUI for model selection.
    model_field = _model_arg(resolved)

    if backend == "llama_cpp":
        provider_name = resolved.get("opencode_provider_name") or provider or "llama-local"
        model_id = resolved.get("opencode_model_id") or resolved.get("real_model") or "model"
        host = resolved.get("host", "127.0.0.1")
        port = resolved.get("port", resolved.get("default_port", 8080))
        return {
            "model": model_field,
            "provider": {
                provider_name: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": provider_name,
                    "options": {"baseURL": f"http://{host}:{port}/v1"},
                    "models": {
                        model_id: {
                            "name": resolved.get("display_name") or model_id,
                        },
                    },
                },
            },
        }

    if backend == "openai_compatible":
        provider_name = resolved.get("opencode_provider_name") or provider or "openai-compatible"
        model_id = resolved.get("opencode_model_id") or resolved.get("real_model") or "model"
        api_base = resolved.get("default_api_base", "")
        api_base_env = resolved.get("api_base_env")
        if api_base_env:
            api_base = os.environ.get(api_base_env, "") or api_base
        base_url = f"{api_base.rstrip('/')}/v1" if api_base else ""
        return {
            "model": model_field,
            "provider": {
                provider_name: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": provider_name,
                    "options": {"baseURL": base_url},
                    "models": {
                        model_id: {
                            "name": resolved.get("display_name") or model_id,
                        },
                    },
                },
            },
        }

    if backend == "ollama":
        api_base = resolved.get("default_api_base", "http://127.0.0.1:11434")
        api_base_env = resolved.get("api_base_env")
        if api_base_env:
            api_base = os.environ.get(api_base_env, "") or api_base
        model_id = resolved.get("real_model") or "model"
        model_entry: dict[str, Any] = {}
        context = resolved.get("context")
        if context:
            model_entry["limit"] = {
                "context": int(context),
                "output": min(int(context), 65536),
            }
        if _ollama_v1_mode(resolved):
            provider_name = _ollama_v1_provider_name(resolved)
            model_entry["name"] = resolved.get("display_name") or model_id
            return {
                "model": model_field,
                "provider": {
                    provider_name: {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": provider_name,
                        "options": {"baseURL": f"{api_base.rstrip('/')}/v1"},
                        "models": {
                            model_id: model_entry,
                        },
                    },
                },
            }
        return {
            "model": model_field,
            "provider": {
                "ollama": {
                    "models": {
                        model_id: model_entry,
                    },
                },
            },
        }

    return {}
