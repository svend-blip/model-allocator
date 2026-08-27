"""OpenCode client adapter for model-allocator."""

from __future__ import annotations

import os
import shutil

from model_allocator.adapters.llama_cpp import served_model_id as _llamacpp_name
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


def _config_env(config_dir: str, config_path: str | None = None) -> dict[str, str]:
    """Where OpenCode reads its configuration.

    Normally the role's shared directory under OPENCODE_ROLES_CONFIG_BASE,
    which `run` refreshes on the way past.

    ``config_path`` names a file the CALLER owns and has already written.
    The allocator points OpenCode at it and refreshes nothing: the caller
    built that file for one run and is entitled to have it be the one read.

    DPMtF-LightWorker needs this. It renders a per-execution config, merges
    its machine's provider endpoint and a permission block confining the
    role to that execution's worktree, validates, and publishes -- and then
    the command named the allocator's shared file instead, so the whole
    sequence produced something nobody read. The role ran with no
    confinement and no endpoint it could reach.
    """
    if config_path:
        directory = os.path.dirname(config_path) or "."
        return {
            "OPENCODE_CONFIG_DIR": directory,
            "OPENCODE_CONFIG": config_path,
        }
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


def _client_context(resolved: dict):
    """The context window a CLIENT should be told, or ``None`` when unknown.

    An alias carries two different numbers and only one of them is a promise.
    ``context`` is what the MODEL supports; ``num_tokens`` is what the runtime
    was actually given room to hold, which on a VRAM-bound card is routinely
    far smaller. Telling a client the first when the second is the truth is
    not a rounding error — it is the difference between a client that manages
    its window and one that walks into a wall.

    Run 002 was blocked exactly there. freetoken-qwen36-35b-a3b advertises
    262144 and holds 49152. Qwen Code, which has no way to be told anything,
    filled the real budget while reporting 4.9% of an imagined one; the server
    then clamped its reply to 31 tokens and the stream died with no visible
    progress. OpenCode CAN be told — but it was being told 262144 too, so
    moving a role here without this would have relocated the failure rather
    than removed it.

    Returns the smaller of the two whenever both are present, so a runtime
    that genuinely holds the full window is unaffected.
    """
    context = resolved.get("context")
    budget = resolved.get("num_tokens")
    values = [int(v) for v in (context, budget) if v]
    return min(values) if values else None


def _freetoken_provider_name(resolved: dict) -> str:
    return resolved.get("opencode_provider_name") or "freetoken-local"


def _freetoken_model_id(resolved: dict) -> str:
    """The id OpenCode uses for a FreeToken model.

    FreeToken serves under the name the runtime loaded, which is what
    ``/v1/models`` reports and what the completions endpoint expects, so the
    same value keys the provider's model block and suffixes the top-level
    ``model`` field. Both call sites read it from here so they cannot drift.
    """
    return (
        resolved.get("opencode_model_id")
        or resolved.get("served_model_name")
        or resolved.get("real_model")
        or "model"
    )


def _openai_provider_name(resolved: dict) -> str:
    """Provider key for an openai_compatible alias.

    The same value keys the provider block in opencode.json and prefixes the
    top-level `model` field, so both call sites must agree.
    """
    return (
        resolved.get("opencode_provider_name")
        or resolved.get("provider", "")
        or "openai-compatible"
    )


def _model_arg(resolved: dict) -> str:
    backend = resolved.get("backend")
    real_model = resolved.get("real_model", "")
    provider = resolved.get("provider", "")

    if backend == "ollama":
        if _ollama_v1_mode(resolved):
            return f"{_ollama_v1_provider_name(resolved)}/{real_model}"
        return f"ollama/{real_model}"
    if backend == "openai_compatible":
        # build_opencode_config declares a custom provider block keyed by the
        # provider name, so the model reference must be provider-qualified —
        # a bare model id resolves against no provider and OpenCode falls back
        # to its own default model.
        return f"{_openai_provider_name(resolved)}/{real_model}"
    if backend == "llama_cpp":
        provider_name = resolved.get("opencode_provider_name") or provider or "llama-local"
        model_id = _llamacpp_name(resolved) or "model"
        return f"{provider_name}/{model_id}"
    if backend == "sglang":
        provider_name = resolved.get("opencode_provider_name") or provider or "sglang-local"
        model_id = resolved.get("opencode_model_id") or resolved.get("served_model_name") or real_model or "model"
        return f"{provider_name}/{model_id}"
    if backend == "freetoken":
        return f"{_freetoken_provider_name(resolved)}/{_freetoken_model_id(resolved)}"
    return real_model


def build_opencode_command(
    resolved: dict, config_dir: str, config_path: str | None = None
) -> dict[str, Any]:
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
        "env": _config_env(config_dir, config_path),
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
        # One function names the model for both the server's --alias and this
        # config, so the two cannot drift apart again.
        model_id = _llamacpp_name(resolved) or "model"
        host = resolved.get("host", "127.0.0.1")
        port = resolved.get("port", resolved.get("default_port", 8080))
        model_entry: dict[str, Any] = {
            "name": resolved.get("display_name") or model_id,
        }
        # This branch emitted no limit at all, so a llama.cpp role never told
        # OpenCode its context window and the client fell back to whatever it
        # assumes for an unknown model. The whole point of configuring a
        # window is that the client knows it.
        context = _client_context(resolved)
        if context:
            model_entry["limit"] = {
                "context": int(context),
                "output": int(
                    resolved.get("max_output_tokens") or min(int(context) // 2, 8192)
                ),
            }
        return {
            "model": model_field,
            "provider": {
                provider_name: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": provider_name,
                    "options": {"baseURL": f"http://{host}:{port}/v1"},
                    "models": {model_id: model_entry},
                },
            },
        }

    if backend == "freetoken":
        # FreeToken had no branch here at all, and an unhandled backend falls
        # through to `return {}` — an empty config, which OpenCode reads as
        # "no provider configured" and answers by silently using its own
        # default model. A role would have launched, run, and produced work
        # against a model nobody chose.
        provider_name = _freetoken_provider_name(resolved)
        model_id = _freetoken_model_id(resolved)
        host = resolved.get("host", resolved.get("default_host", "127.0.0.1"))
        port = resolved.get("port", resolved.get("default_port", 8088))
        model_entry: dict[str, Any] = {
            "name": resolved.get("display_name") or model_id,
        }
        context = _client_context(resolved)
        if context:
            model_entry["limit"] = {
                "context": int(context),
                "output": int(
                    resolved.get("max_output_tokens") or min(int(context) // 2, 8192)
                ),
            }
        return {
            "model": model_field,
            "provider": {
                provider_name: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": provider_name,
                    "options": {
                        "baseURL": f"http://{host}:{port}/v1",
                        # FreeToken authenticates nothing on loopback, but the
                        # OpenAI-compatible SDK will not call without a key.
                        "apiKey": "dummy",
                    },
                    "models": {model_id: model_entry},
                },
            },
        }

    if backend == "sglang":
        provider_name = resolved.get("opencode_provider_name") or provider or "sglang-local"
        model_id = resolved.get("opencode_model_id") or resolved.get("served_model_name") or resolved.get("real_model") or "model"
        host = resolved.get("host", resolved.get("default_host", "127.0.0.1"))
        port = resolved.get("port", resolved.get("default_port", 30000))
        model_entry: dict[str, Any] = {
            "name": resolved.get("display_name") or model_id,
        }
        context = _client_context(resolved)
        if context:
            model_entry["limit"] = {
                "context": int(context),
                "output": int(resolved.get("max_output_tokens") or min(int(context) // 2, 8192)),
            }
        return {
            "model": model_field,
            "provider": {
                provider_name: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": provider_name,
                    "options": {
                        "baseURL": f"http://{host}:{port}/v1",
                        "apiKey": "dummy",
                    },
                    "models": {model_id: model_entry},
                },
            },
        }

    if backend == "openai_compatible":
        provider_name = _openai_provider_name(resolved)
        model_id = resolved.get("opencode_model_id") or resolved.get("real_model") or "model"
        api_base = resolved.get("default_api_base", "")
        api_base_env = resolved.get("api_base_env")
        if api_base_env:
            api_base = os.environ.get(api_base_env, "") or api_base
        base_url = f"{api_base.rstrip('/')}/v1" if api_base else ""
        options: dict[str, Any] = {"baseURL": base_url}
        # OpenCode only resolves credentials on its own for providers it knows
        # natively; a custom @ai-sdk/openai-compatible block authenticates with
        # nothing unless the key is named here. {env:VAR} keeps the secret out
        # of the rendered file — OpenCode expands it at startup.
        api_key_env = resolved.get("api_key_env")
        if api_key_env:
            options["apiKey"] = "{env:" + api_key_env + "}"
        model_entry: dict[str, Any] = {"name": resolved.get("display_name") or model_id}
        context = _client_context(resolved)
        if context:
            model_entry["limit"] = {
                "context": int(context),
                "output": int(resolved.get("max_output_tokens") or min(int(context) // 2, 8192)),
            }
        return {
            "model": model_field,
            "provider": {
                provider_name: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": provider_name,
                    "options": options,
                    "models": {
                        model_id: model_entry,
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
        context = _client_context(resolved)
        if context:
            model_entry["limit"] = {
                "context": int(context),
                # `max_output_tokens`, like every other backend. This branch
                # alone hardcoded output to the whole context, so an ollama
                # role's output budget always equalled its window and there
                # was no headroom for the system prompt or the work by any
                # accounting. Setting max_output_tokens in models.yaml had no
                # effect at all -- it looked like a deliberate config choice
                # and was the adapter.
                "output": int(
                    resolved.get("max_output_tokens") or min(int(context) // 2, 8192)
                ),
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
