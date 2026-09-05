"""Remote AI-PC model endpoint support (SCOPE addendum 2026-09-05).

A remote AI-PC model endpoint is just an ``openai_compatible`` runtime
profile whose base URL points at another machine (LAN/Tailscale). The
existing OpenAICompatibleAdapter already resolves the base and validates
reachability; these tests pin the alias->endpoint resolution and the
``is_model_available`` where-practical check the addendum adds.
"""

import json

from model_allocator.adapters.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAICompatibleAdapterError,
)


REMOTE_PROFILE = {
    "backend": "openai_compatible",
    "api_base_env": "REMOTE_AI_PC_API_BASE",
    "default_api_base": "http://remote-ai-pc:8090/v1",
    "api_key_env": "REMOTE_AI_PC_API_KEY",
    "provider": "remote_ai_pc",
}


def test_alias_resolves_to_the_fixed_remote_default_base(monkeypatch):
    """One alias -> one fixed endpoint: the default_api_base is used when no env override."""
    monkeypatch.delenv("REMOTE_AI_PC_API_BASE", raising=False)
    base = OpenAICompatibleAdapter.api_base_from_profile(REMOTE_PROFILE)
    assert base == "http://remote-ai-pc:8090/v1"


def test_env_override_wins_over_default(monkeypatch):
    """Deployments can point the same alias at a different machine via the env var."""
    monkeypatch.setenv("REMOTE_AI_PC_API_BASE", "http://100.73.166.28:8090/v1")
    base = OpenAICompatibleAdapter.api_base_from_profile(REMOTE_PROFILE)
    assert base == "http://100.73.166.28:8090/v1"


def _adapter():
    return OpenAICompatibleAdapter(api_base="http://remote-ai-pc:8090/v1")


def test_model_available_when_listed(monkeypatch):
    adapter = _adapter()

    def fake_request(path, method="GET", timeout=5):
        assert path in ("/models", "/v1/models")
        body = json.dumps({"data": [{"id": "Qwen3.8-Flash-Next"}, {"id": "other"}]})
        return {"status_code": 200, "body": body, "error": None}

    monkeypatch.setattr(adapter, "_request", fake_request)
    result = adapter.is_model_available("Qwen3.8-Flash-Next")
    assert result["available"] is True
    assert result["error"] is None


def test_model_not_available_when_absent(monkeypatch):
    adapter = _adapter()

    def fake_request(path, method="GET", timeout=5):
        return {"status_code": 200, "body": json.dumps({"data": [{"id": "other"}]}), "error": None}

    monkeypatch.setattr(adapter, "_request", fake_request)
    result = adapter.is_model_available("Qwen3.8-Flash-Next")
    assert result["available"] is False
    assert "not listed" in (result["error"] or "")


def test_model_available_falls_through_to_v1_models(monkeypatch):
    """Base without /v1: /models 404s, /v1/models carries the list."""
    adapter = OpenAICompatibleAdapter(api_base="http://remote-ai-pc:8090")
    calls = []

    def fake_request(path, method="GET", timeout=5):
        calls.append(path)
        if path == "/models":
            return {"status_code": 404, "body": "", "error": "Not Found"}
        return {"status_code": 200, "body": json.dumps({"data": [{"id": "m"}]}), "error": None}

    monkeypatch.setattr(adapter, "_request", fake_request)
    result = adapter.is_model_available("m")
    assert result["available"] is True
    assert calls == ["/models", "/v1/models"]


def test_unreachable_endpoint_is_a_clear_failure_not_a_reroute(monkeypatch):
    """A stopped remote model must produce an explicit failure, never silent rerouting."""
    adapter = _adapter()

    def fake_request(path, method="GET", timeout=5):
        raise OpenAICompatibleAdapterError("API base unreachable: [Errno 111] Connection refused")

    monkeypatch.setattr(adapter, "_request", fake_request)

    availability = adapter.is_model_available("Qwen3.8-Flash-Next")
    assert availability["available"] is False
    assert "unreachable" in (availability["error"] or "")

    reach = adapter.is_api_reachable()
    assert reach["reachable"] is False
    assert reach["error"]


def test_no_model_configured_is_reported(monkeypatch):
    adapter = _adapter()
    result = adapter.is_model_available("")
    assert result["available"] is False
    assert "no model" in (result["error"] or "")
