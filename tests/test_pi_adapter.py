"""Pi is a frontend, and these tests exist to keep it one.

The failure they guard against is the adapter quietly becoming a second
source of truth about models: shadowing a provider Pi maintains upstream,
or naming a model something the server does not serve.
"""

import pytest

from model_allocator.adapters import pi


GLM = {
    "alias": "glm-air-derestricted-local",
    "backend": "llama_cpp",
    "real_model": "glm-4.5-air-derestricted",
    "display_name": "GLM-4.5-Air-Derestricted (IQ4_XS)",
    "context": 65536,
    "max_output_tokens": 16384,
    "port": 8080,
    "host": "127.0.0.1",
}

MINIMAX = {
    "alias": "cloud_minimax",
    "backend": "openai_compatible",
    "provider": "minimax",
    "real_model": "MiniMax-M3",
    "context": 1000000,
    "max_output_tokens": 65536,
}


def test_builtin_provider_gets_no_models_json():
    """Pi maintains MiniMax's metadata; a custom block would shadow it."""
    assert pi.is_builtin_provider(MINIMAX) is True
    assert pi.build_pi_models_json(MINIMAX) is None


def test_local_server_is_declared_with_its_endpoint():
    fragment = pi.build_pi_models_json(GLM)
    assert list(fragment) == ["llama-local"]
    provider = fragment["llama-local"]
    assert provider["baseUrl"] == "http://127.0.0.1:8080/v1"
    assert provider["api"] == "openai-completions"
    # Pi hides models it believes have no auth, so a keyless local server
    # still needs a placeholder or it never appears in --model.
    assert provider["apiKey"]
    # Pi's own advice for llama.cpp/vLLM/SGLang/Ollama: these servers do not
    # all accept the `developer` role, and one that does not fails the
    # request rather than degrading.
    assert provider["compat"] == {
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
    }
    assert provider["models"][0]["id"] == "glm-4.5-air-derestricted"
    assert provider["models"][0]["contextWindow"] == 65536


def test_llamacpp_model_id_matches_the_servers_alias():
    """The id Pi requests and the server's --alias come from one helper.

    They were assembled in different places once, and the mismatch only
    shows on the first request, after preflight has passed.
    """
    from model_allocator.adapters.llama_cpp import served_model_id
    assert pi.pi_model_id(GLM) == served_model_id(GLM)


def test_command_names_provider_and_model_explicitly(monkeypatch):
    """Pi honours --model per invocation; nothing may rely on a config file.

    OpenCode does not, which is why its roles need a refreshed config on
    every run. Losing these flags would reintroduce that whole class of bug.
    """
    monkeypatch.setattr(pi, "_resolve_pi_bin", lambda: "/usr/bin/pi")
    argv = pi.build_pi_command(MINIMAX)["argv"]
    assert argv[argv.index("--provider") + 1] == "minimax"
    assert argv[argv.index("--model") + 1] == "MiniMax-M3"


def test_sessions_are_ephemeral_by_default(monkeypatch):
    """A restarted role must start empty, not resume the last handoff."""
    monkeypatch.setattr(pi, "_resolve_pi_bin", lambda: "/usr/bin/pi")
    assert "--no-session" in pi.build_pi_command(GLM)["argv"]
    keep = dict(GLM, pi_no_session=False)
    assert "--no-session" not in pi.build_pi_command(keep)["argv"]


def test_tool_allowlist_is_passed_through(monkeypatch):
    """`pi_tools` is governance the client enforces, not a prompt request."""
    monkeypatch.setattr(pi, "_resolve_pi_bin", lambda: "/usr/bin/pi")
    reviewer = dict(GLM, pi_tools=["read", "grep", "bash"])
    argv = pi.build_pi_command(reviewer)["argv"]
    assert argv[argv.index("--tools") + 1] == "read,grep,bash"


def test_unknown_backend_is_refused_rather_than_guessed():
    with pytest.raises(pi.PiAdapterError):
        pi.pi_provider_name({"backend": "onyx"})
