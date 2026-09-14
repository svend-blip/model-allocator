"""The Lenovo P620 RTX 3080 FreeToken node as a remote backend (2026-09-14).

Profile `remote_freetoken_p620_rtx3080` and alias `p620-qwen36`: the node
serves Qwen3.6-35B-A3B-NVFP4 through FreeToken 0.1.2 over its Tailscale
HTTPS endpoint, 32768 context, --max-running-requests 1. The allocator only
routes to it (and can send one chat request through `invoke`). Everything
here is deterministic: HTTP is a fake `_request`; the node is never
contacted. Live checks are the operator's `validate`/`status`/`invoke`.
"""
import json
import threading
import time
from pathlib import Path

import pytest
import yaml

from model_allocator import schema
from model_allocator.adapters import openai_compatible as oa
from model_allocator.adapters.openai_compatible import OpenAICompatibleAdapter
from model_allocator.cli import _get_backend_adapter
from model_allocator.resolver import Resolver
from model_allocator.validator import Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
ALIAS = "p620-qwen36"
PROFILE = "remote_freetoken_p620_rtx3080"
MODEL = "Qwen3.6-35B-A3B-NVFP4"
BASE_URL = "https://omarchy-1.tail8c8e74.ts.net/v1"
CONTEXT = 32768


def _load(name: str) -> dict:
    return yaml.safe_load((REPO_ROOT / name).read_text())


def _resolved() -> dict:
    return Resolver(config_dir=str(REPO_ROOT)).resolve_alias(ALIAS)


# --- 1. the profile and alias parse and validate ---------------------------

@pytest.mark.parametrize("profiles_file,models_file", [
    ("runtime_profiles.yaml", "models.yaml"),
    ("runtime_profiles.example.yaml", "models.example.yaml"),
])
def test_profile_and_alias_parse_and_validate_cleanly(profiles_file, models_file):
    profiles = _load(profiles_file)["runtime_profiles"]
    models = _load(models_file)["models"]
    assert profiles[PROFILE]["backend"] == "openai_compatible"
    assert profiles[PROFILE]["default_api_base"] == BASE_URL
    assert profiles[PROFILE]["max_running_requests"] == 1
    assert "invoke" in profiles[PROFILE]["capabilities"]
    assert "api_key_env" not in profiles[PROFILE]  # tailnet-restricted, no auth
    issues = schema.validate_alias(ALIAS, models[ALIAS], profiles)
    assert [i for i in issues if i.level == "error"] == []
    assert [i for i in issues if "unknown" in i.message.lower()] == []


# --- 2-5. the alias resolves to the right backend, model, URL, context ------

def test_alias_resolves_to_the_p620_node(monkeypatch):
    resolved = _resolved()
    assert resolved["backend"] == "openai_compatible"
    assert resolved["runtime_profile"] == PROFILE
    assert resolved["real_model"] == MODEL
    assert resolved["context"] == CONTEXT
    assert resolved["enable_thinking"] is False
    assert resolved["lifecycle_policy"] == "persistent"
    assert OpenAICompatibleAdapter.api_base_from_profile(resolved) == BASE_URL


def test_existing_p620_alias_keeps_its_profile_and_is_capped_at_32768():
    resolved = Resolver(config_dir=str(REPO_ROOT)).resolve_alias("p620-qwen36-35b-a3b")
    assert resolved["runtime_profile"] == "p620_freetoken_remote"
    assert resolved["real_model"] == MODEL
    assert resolved["context"] == CONTEXT


# --- 6. single-worker declaration reaches the adapter ------------------------

def test_backend_is_treated_as_single_worker():
    adapter = _get_backend_adapter(_resolved())
    assert isinstance(adapter, OpenAICompatibleAdapter)
    assert adapter.api_base == BASE_URL
    assert adapter.real_model == MODEL
    assert adapter.single_worker is True
    assert adapter.provider == "freetoken-remote"


def test_invoke_sends_one_openai_chat_request_with_thinking_off(monkeypatch):
    adapter = _get_backend_adapter(_resolved())
    seen = {}

    def fake_request(path, method="GET", timeout=5, body=None, headers=None):
        seen.update(path=path, method=method, body=json.loads(body), headers=headers)
        reply = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                 "usage": {"total_tokens": 7}}
        return {"status_code": 200, "body": json.dumps(reply), "error": None}

    monkeypatch.setattr(adapter, "_request", fake_request)
    result = adapter.invoke("Say OK.")
    assert result["status"] == "ok"
    assert result["text"] == "OK"
    assert result["provider"] == "freetoken-remote"
    assert seen["path"] == "/chat/completions" and seen["method"] == "POST"
    assert seen["body"]["model"] == MODEL
    assert seen["body"]["messages"] == [{"role": "user", "content": "Say OK."}]
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "Authorization" not in seen["headers"]  # no auth on the tailnet node
    assert result["metadata"]["finish_reason"] == "stop"


def test_alias_without_the_field_sends_no_thinking_override():
    adapter = OpenAICompatibleAdapter(api_base=BASE_URL, real_model=MODEL)
    assert "chat_template_kwargs" not in adapter.build_chat_payload("hi")


# --- 7. queued invocations never run concurrently on this node ---------------

def test_queued_invocations_are_serialized_on_the_single_worker_node(monkeypatch):
    adapter = _get_backend_adapter(_resolved())
    state = {"in_flight": 0, "max_in_flight": 0, "order": []}
    guard = threading.Lock()

    def slow_request(path, method="GET", timeout=5, body=None, headers=None):
        with guard:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        time.sleep(0.05)
        with guard:
            state["in_flight"] -= 1
            state["order"].append(json.loads(body)["messages"][0]["content"])
        reply = {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}
        return {"status_code": 200, "body": json.dumps(reply), "error": None}

    monkeypatch.setattr(adapter, "_request", slow_request)
    threads = [threading.Thread(target=adapter.invoke, args=(f"job {i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert state["max_in_flight"] == 1  # queued, never concurrent
    assert sorted(state["order"]) == [f"job {i}" for i in range(4)]  # nothing dropped


def test_a_multi_worker_endpoint_is_not_serialized(monkeypatch):
    adapter = OpenAICompatibleAdapter(api_base="http://pool.example/v1", real_model="m", max_running_requests=4)
    assert adapter.single_worker is False
    state = {"in_flight": 0, "max_in_flight": 0}
    guard = threading.Lock()

    def slow_request(path, method="GET", timeout=5, body=None, headers=None):
        with guard:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        time.sleep(0.05)
        with guard:
            state["in_flight"] -= 1
        return {"status_code": 200, "body": json.dumps({"choices": [{"message": {"content": "x"}}]}), "error": None}

    monkeypatch.setattr(adapter, "_request", slow_request)
    threads = [threading.Thread(target=adapter.invoke, args=("j",)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert state["max_in_flight"] > 1


# --- 8. an unreachable node is handled cleanly ---------------------------------

def test_unreachable_node_is_reported_offline_not_raised(monkeypatch):
    adapter = _get_backend_adapter(_resolved())

    def down(path, method="GET", timeout=5, body=None, headers=None):
        raise oa.OpenAICompatibleAdapterError("API base unreachable: connection refused")

    monkeypatch.setattr(adapter, "_request", down)
    assert adapter.is_api_reachable()["reachable"] is False
    assert adapter.status()["reachable"] is False
    assert adapter.is_model_available(MODEL)["available"] is False
    result = adapter.invoke("hello")
    assert result["status"] == "error"
    assert "unreachable" in result["error"]
    assert result["text"] == ""


def test_offline_node_validates_as_unreachable_warning_not_error(monkeypatch):
    def down(self):
        return {"reachable": False, "status_code": None, "error": "API base unreachable: timed out"}

    monkeypatch.setattr(OpenAICompatibleAdapter, "is_api_reachable", down)
    result = Validator(config_dir=str(REPO_ROOT)).validate(ALIAS, "simple-harness")
    assert result["validation_status"] != "ERROR"
    assert result["client_support"]["simple-harness"] == "UNREACHABLE"
    assert result["resolved_api_base"] == BASE_URL


def test_online_node_validates_ok_for_every_client(monkeypatch):
    def up(self):
        return {"reachable": True, "status_code": 404, "error": None}

    monkeypatch.setattr(OpenAICompatibleAdapter, "is_api_reachable", up)
    for client in ("opencode", "claude-code", "simple-harness"):
        result = Validator(config_dir=str(REPO_ROOT)).validate(ALIAS, client)
        assert result["validation_status"] != "ERROR", (client, result["errors"])
        assert result["client_support"][client] == "OK", (client, result)


def test_model_identity_is_checked_at_v1_models(monkeypatch):
    adapter = OpenAICompatibleAdapter(api_base=BASE_URL)

    def fake_request(path, method="GET", timeout=5, body=None, headers=None):
        return {"status_code": 200, "body": json.dumps({"data": [{"id": MODEL}]}), "error": None}

    monkeypatch.setattr(adapter, "_request", fake_request)
    assert adapter.is_model_available(MODEL)["available"] is True
    assert adapter.is_model_available("Qwen3.8-Flash-Next")["available"] is False


# --- 9. nothing else moved ---------------------------------------------------------

def test_every_other_profile_and_alias_still_validates_without_errors():
    profiles = _load("runtime_profiles.yaml")["runtime_profiles"]
    doc = _load("models.yaml")
    models = doc["models"]
    for name, definition in models.items():
        issues = schema.validate_alias(
            name, definition, profiles,
            doc.get("runtime_instances"), doc.get("inference_profiles"),
        )
        assert [i for i in issues if i.level == "error"] == [], name
    assert "p620_freetoken_remote" in profiles
    assert profiles["p620_freetoken_remote"]["default_api_base"] == "http://100.92.72.93:1919/v1"
