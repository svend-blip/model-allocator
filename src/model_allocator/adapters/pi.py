"""Pi coding-agent client adapter for model-allocator.

Pi (`@earendil-works/pi-coding-agent`) is a third code frontend alongside
Claude Code and OpenCode. It is a *frontend only*: which model runs, and
whether its server is up, stays owned by the allocator. Pi is told the
endpoint and the model id and nothing else.

Two things make Pi cheaper to wire than OpenCode was.

**The model is chosen by flag, not by file.** `--provider` and `--model` are
honoured on every invocation, so there is no per-role config directory to
keep in step and no equivalent of the defect that made OpenCode ignore
`--model` on session resumption. One shared `~/.pi/agent/models.json`
declares the custom providers; the role's own model comes from argv.

**Cloud providers Pi ships with are used as they are.** MiniMax and
OpenRouter are built-in and maintained upstream, complete with model
metadata, so this adapter declares nothing for them and lets Pi's own
integration handle the wire format. That is the whole reason a Pi/MiniMax
role is worth measuring against an OpenCode/MiniMax one: OpenCode reaches
MiniMax through a generic `@ai-sdk/openai-compatible` block, and Pi does not.

Local servers are the opposite case and need declaring: a custom provider
with `api: "openai-completions"` pointing at the allocator's own endpoint.
Pi does have a first-class llama.cpp integration, but it drives llama.cpp's
*router* mode — a server started without `--model`, discovering GGUFs from a
directory and loading them on demand. That contradicts the allocator, which
starts single-model servers with per-alias flags (`--n-cpu-moe`, KV cache
types, context). The generic custom-provider path keeps model lifecycle
where it belongs.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from model_allocator.adapters.llama_cpp import served_model_id as _llamacpp_name

# Backends whose servers the allocator runs itself, and which therefore have
# to be declared to Pi as custom OpenAI-compatible providers.
_LOCAL_BACKENDS = ("llama_cpp", "sglang", "ollama")

# Pi ships providers for these and keeps their model metadata current; the
# adapter must not shadow them with a custom block.
_PI_BUILTIN_PROVIDERS = ("minimax", "minimax-cn", "openrouter", "anthropic",
                         "openai", "google", "deepseek", "groq", "xai",
                         "mistral", "cerebras", "zai")


class PiAdapterError(Exception):
    pass


def _resolve_pi_bin() -> str:
    pi_bin = os.environ.get("PI_BIN", "pi")
    if os.path.isabs(pi_bin):
        if not (os.path.isfile(pi_bin) and os.access(pi_bin, os.X_OK)):
            raise PiAdapterError(f"Pi binary not found: {pi_bin}")
        return pi_bin
    resolved = shutil.which(pi_bin)
    if resolved is None:
        raise PiAdapterError(f"Pi binary not found on PATH: {pi_bin}")
    return resolved


def pi_agent_dir() -> str:
    """Where Pi keeps auth.json, settings.json and models.json.

    Shared, not per-role. Isolating it per role would also isolate
    `auth.json`, and every cloud role would lose the credentials Pi resolves
    from it. The model is a flag here, so roles do not need separate files
    to differ.
    """
    return os.path.expanduser(
        os.environ.get("PI_CODING_AGENT_DIR", "~/.pi/agent")
    )


def is_builtin_provider(resolved: dict) -> bool:
    """True when Pi already ships the provider this alias resolves to."""
    if resolved.get("backend") in _LOCAL_BACKENDS:
        return False
    provider = (resolved.get("pi_provider_name")
                or resolved.get("provider", "")).lower()
    return provider in _PI_BUILTIN_PROVIDERS


def pi_provider_name(resolved: dict) -> str:
    """The name Pi will be given via --provider.

    For a built-in provider this must match Pi's own key exactly; for a local
    server it is ours to choose, and it keys the models.json block.
    """
    explicit = resolved.get("pi_provider_name")
    if explicit:
        return str(explicit)
    provider = resolved.get("provider", "")
    if provider:
        return str(provider)
    backend = resolved.get("backend")
    if backend == "llama_cpp":
        return "llama-local"
    if backend == "sglang":
        return "sglang-local"
    if backend == "ollama":
        return "ollama"
    raise PiAdapterError(
        f"Cannot determine a Pi provider name for backend {backend!r}"
    )


def pi_model_id(resolved: dict) -> str:
    """The model id Pi requests, which must be what the server serves.

    For llama.cpp this is the same helper the server's own `--alias` uses,
    so the two cannot drift — the mismatch that only shows on the first
    request, after preflight has passed.
    """
    if resolved.get("backend") == "llama_cpp":
        served = _llamacpp_name(resolved)
        if served:
            return served
    model_id = resolved.get("pi_model_id") or resolved.get("real_model") or ""
    if not model_id:
        raise PiAdapterError("Resolved alias is missing real_model")
    return str(model_id)


def _endpoint(resolved: dict) -> str:
    backend = resolved.get("backend")
    host = resolved.get("host", "127.0.0.1")
    if backend in ("llama_cpp", "sglang"):
        port = resolved.get("port", resolved.get("default_port"))
        if port is None:
            raise PiAdapterError(
                f"No port resolved for {backend} alias "
                f"{resolved.get('alias')!r}"
            )
        return f"http://{host}:{port}/v1"
    if backend == "ollama":
        base = (resolved.get("api_base")
                or resolved.get("default_api_base")
                or "http://127.0.0.1:11434")
        return f"{base.rstrip('/')}/v1"
    raise PiAdapterError(f"No local endpoint for backend {backend!r}")


def build_pi_models_json(resolved: dict) -> dict[str, Any] | None:
    """The `providers` fragment declaring this alias to Pi.

    Returns None when Pi already knows the provider, which is the signal to
    the caller that nothing should be written.

    `supportsDeveloperRole` and `supportsReasoningEffort` are switched off
    for local servers on Pi's own advice: llama.cpp, vLLM, SGLang and Ollama
    do not all understand the `developer` role that reasoning-capable models
    otherwise get, and a server that rejects it fails the request rather than
    degrading. `apiKey` is a placeholder because these servers ignore it —
    but Pi treats a model without auth as unavailable and hides it from
    `--model`, so it cannot simply be omitted.
    """
    if is_builtin_provider(resolved):
        return None

    model_entry: dict[str, Any] = {"id": pi_model_id(resolved)}
    context = resolved.get("context")
    if context:
        model_entry["contextWindow"] = int(context)
        model_entry["maxTokens"] = int(
            resolved.get("max_output_tokens") or min(int(context) // 2, 16384)
        )
    if resolved.get("reasoning") in ("on", "auto", True):
        model_entry["reasoning"] = True
    display = resolved.get("display_name")
    if display:
        model_entry["name"] = str(display)

    return {
        pi_provider_name(resolved): {
            "baseUrl": _endpoint(resolved),
            "api": "openai-completions",
            "apiKey": resolved.get("pi_api_key") or "local",
            "compat": {
                "supportsDeveloperRole": False,
                "supportsReasoningEffort": False,
            },
            "models": [model_entry],
        }
    }


def build_pi_command(resolved: dict) -> dict[str, Any]:
    """Build the Pi command object: {"env": {...}, "argv": [...]}.

    `--no-session` is the default and is what makes Pi worth having in a
    handoff chain: the process saves nothing, so a role that is restarted
    starts genuinely empty rather than resuming whatever it was doing when
    the previous handoff ended. Set `pi_no_session: false` on an alias that
    needs continuity across restarts.

    `--tools` is how a role's file-and-shell permissions stop being purely a
    matter of prompt compliance. An alias that sets `pi_tools` gets exactly
    those tools and no others.
    """
    argv = [_resolve_pi_bin(),
            "--provider", pi_provider_name(resolved),
            "--model", pi_model_id(resolved)]

    if resolved.get("pi_no_session", True):
        argv.append("--no-session")

    thinking = resolved.get("pi_thinking") or resolved.get("thinking_level")
    if thinking:
        argv += ["--thinking", str(thinking)]

    tools = resolved.get("pi_tools")
    if tools:
        argv += ["--tools", ",".join(tools) if isinstance(tools, list) else str(tools)]

    for skill in resolved.get("pi_skills") or []:
        argv += ["--skill", os.path.expanduser(str(skill))]

    for extra in resolved.get("pi_extra_args") or []:
        argv.append(str(extra))

    env: dict[str, str] = {
        # Debug aid, and the same breadcrumb the other adapters leave.
        "MODEL_ALLOCATOR_ACTIVE_MODEL": pi_model_id(resolved),
    }
    # Only set when an alias overrides it; otherwise Pi's own default keeps
    # auth.json and the model catalogue where the user's `pi` already looks.
    agent_dir = resolved.get("pi_agent_dir")
    if agent_dir:
        env["PI_CODING_AGENT_DIR"] = os.path.expanduser(str(agent_dir))

    return {"env": env, "argv": argv}
