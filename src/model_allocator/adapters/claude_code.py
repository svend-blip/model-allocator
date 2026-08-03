"""Claude Code client adapter for model-allocator."""

from __future__ import annotations

import os
import shutil
from typing import Any


def build_claude_code_command(resolved: dict) -> dict[str, Any]:
    """Build a Claude Code command object for an Ollama or cloud alias.

    Mirrors command_builder.build_claude_ollama_command and
    build_claude_openrouter_command:
      - argv: [claude_bin, *extra_args, "--model", <real_model>]
      - env: ANTHROPIC_BASE_URL = endpoint
             ANTHROPIC_AUTH_TOKEN = "ollama" for Ollama, or $<API_KEY_ENV> for cloud
             ANTHROPIC_API_KEY = "" for cloud (prevents direct Anthropic fallback)
             CLAUDE_CODE_MAX_OUTPUT_TOKENS (when configured)
             CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1 (when configured)
             MODEL_ALLOCATOR_ACTIVE_MODEL (for debugging)

    Minimax and other non-Anthropic-compatible cloud backends are rejected.
    """
    real_model = resolved.get("real_model", "")
    if not real_model:
        raise ValueError("Resolved alias is missing real_model")

    backend = resolved.get("backend")
    if backend not in ("ollama", "openai_compatible", "anthropic"):
        raise ValueError(f"Backend '{backend}' is not supported by the Claude Code adapter")

    provider = resolved.get("provider", "")
    if backend == "openai_compatible" and provider == "minimax":
        raise ValueError("Minimax does not expose an Anthropic-compatible endpoint for Claude Code")

    # Resolve binary: explicit env > config field > PATH
    claude_bin = os.environ.get("CLAUDE_BIN", "")
    if not claude_bin:
        claude_bin = resolved.get("claude_binary", "claude")
    if os.path.isabs(claude_bin):
        if not (os.path.isfile(claude_bin) and os.access(claude_bin, os.X_OK)):
            raise ValueError(f"Claude binary not found: {claude_bin}")
    else:
        resolved_bin = shutil.which(claude_bin)
        if resolved_bin is None:
            raise ValueError(f"Claude binary not found on PATH: {claude_bin}")
        claude_bin = resolved_bin

    env: dict[str, str] = {}
    if backend == "ollama":
        api_base_env = resolved.get("api_base_env", "OLLAMA_BASE_URL")
        endpoint = os.environ.get(api_base_env, "") or resolved.get("default_api_base", "http://127.0.0.1:11434")
        env["ANTHROPIC_BASE_URL"] = endpoint
        env["ANTHROPIC_AUTH_TOKEN"] = "ollama"
        env["ANTHROPIC_API_KEY"] = ""
    elif backend == "anthropic":
        if resolved.get("credentials", "api_key") == "subscription":
            # Use Claude Code's OWN login (Max/Pro subscription). All three
            # variables are BLANKED rather than omitted: an inherited
            # ANTHROPIC_API_KEY would silently move the session onto API
            # billing, and an inherited ANTHROPIC_BASE_URL would redirect it to
            # a local endpoint entirely. Empty means absent to Claude Code —
            # the same assumption the cloud branch below already relies on when
            # it blanks ANTHROPIC_API_KEY to prevent an Anthropic fallback.
            env["ANTHROPIC_API_KEY"] = ""
            env["ANTHROPIC_BASE_URL"] = ""
            env["ANTHROPIC_AUTH_TOKEN"] = ""
        else:
            api_key_env = resolved.get("api_key_env", "ANTHROPIC_API_KEY")
            env["ANTHROPIC_API_KEY"] = f"${api_key_env}"
            env["ANTHROPIC_BASE_URL"] = ""
            env["ANTHROPIC_AUTH_TOKEN"] = ""
    else:
        api_base_env = resolved.get("api_base_env")
        api_key_env = resolved.get("api_key_env")
        if not api_base_env:
            raise ValueError("Cloud alias missing api_base_env")
        if not api_key_env:
            raise ValueError("Cloud alias missing api_key_env")
        endpoint = os.environ.get(api_base_env, "")
        if not endpoint:
            endpoint = resolved.get("default_api_base", "")
        if not endpoint:
            raise ValueError(f"Cloud API base environment variable '{api_base_env}' is not set")
        env["ANTHROPIC_BASE_URL"] = endpoint
        env["ANTHROPIC_AUTH_TOKEN"] = f"${api_key_env}"
        env["ANTHROPIC_API_KEY"] = ""

    # Max output tokens (from config or CLI override applied by caller)
    max_tokens = resolved.get("max_output_tokens")
    if max_tokens:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)

    # Disable adaptive thinking
    if resolved.get("disable_adaptive_thinking"):
        env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"

    # Allocator-active model metadata for debugging
    alias = resolved.get("alias")
    if alias:
        env["MODEL_ALLOCATOR_ACTIVE_MODEL"] = alias

    # Extra args (e.g. --bare) — come BEFORE --model
    extra_args = resolved.get("claude_extra_args", [])
    if not isinstance(extra_args, list):
        extra_args = []

    return {
        "env": env,
        "argv": [claude_bin, *extra_args, "--model", real_model],
    }
