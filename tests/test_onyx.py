"""ONYX adapter tests (V5A) — fixtures only, zero network.

Covers: cookie-login auth flow, API-key auth, invoke response
normalization (answer + citations), upstream error propagation,
unreachable handling, status(), and the OPTIONALITY GUARANTEE: with the
onyx profile configured but the instance down/absent, every non-onyx
alias keeps resolving and the onyx alias fails cleanly.
"""

import json
import os
import unittest
from unittest import mock

from model_allocator.adapters.onyx import OnyxAdapter, OnyxAdapterError
from model_allocator.invoke_result import INVOKE_RESULT_VERSION
from model_allocator.resolver import Resolver

CONFIG = {
    "models": {
        "company-knowledge": {
            "runtime_profile": "local_onyx",
            "persona_id": 7,
            "lifecycle_policy": "cloud_noop",
            "clients": {"headless": True},
        },
        "plain-ollama": {
            "runtime_profile": "local_ollama",
            "real_model": "qwen:latest",
            "lifecycle_policy": "persistent",
            "clients": {"opencode": True},
        },
    },
    "runtime_profiles": {
        "local_onyx": {
            "backend": "onyx",
            "default_api_base": "http://127.0.0.1:9162",
            "api_key_env": "TEST_ONYX_API_KEY",
            "email_env": "TEST_ONYX_EMAIL",
            "password_env": "TEST_ONYX_PASSWORD",
            "capabilities": ["invoke"],
        },
        "local_ollama": {
            "backend": "ollama",
            "default_api_base": "http://127.0.0.1:11434",
        },
    },
    "roles": {},
}

CHAT_RESPONSE = {
    "answer": "The stop distance is 3.56% [1].",
    "answer_citationless": "The stop distance is 3.56%.",
    "pre_answer_reasoning": None,
    "tool_calls": [],
    "top_documents": [
        {
            "document_id": "doc-1",
            "semantic_identifier": "GATES.md",
            "link": "https://example/gates",
            "source_type": "file",
        }
    ],
    "citation_info": [{"citation_num": 1, "document_id": "doc-1"}],
    "message_id": 42,
    "chat_session_id": "abc-123",
    "error_msg": None,
}


class FakeHttp:
    """Scriptable http_client: records calls, returns queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, method, headers, body, timeout):
        self.calls.append(
            {"url": url, "method": method, "headers": headers, "body": body}
        )
        if not self.responses:
            raise OnyxAdapterError("ONYX unreachable: no scripted response")
        return self.responses.pop(0)


def login_ok():
    return (204, "", {"Set-Cookie": "fastapi_users_token=tok123; HttpOnly"})


def chat_ok(payload=None):
    return (200, json.dumps(payload or CHAT_RESPONSE), {})


class OnyxInvokeTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(
            os.environ,
            {
                "TEST_ONYX_EMAIL": "a@b.c",
                "TEST_ONYX_PASSWORD": "pw",
                "TEST_ONYX_API_KEY": "",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def adapter(self, http):
        return OnyxAdapter(
            api_base="http://127.0.0.1:9162",
            api_key_env="TEST_ONYX_API_KEY",
            email_env="TEST_ONYX_EMAIL",
            password_env="TEST_ONYX_PASSWORD",
            persona_id=7,
            http_client=http,
        )

    def test_invoke_login_flow_and_normalization(self):
        http = FakeHttp([login_ok(), chat_ok()])
        result = self.adapter(http).invoke("What is the stop distance?")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["invoke_result_version"], INVOKE_RESULT_VERSION)
        self.assertIn("3.56%", result["text"])
        self.assertEqual(len(result["citations"]), 1)
        citation = result["citations"][0]
        self.assertEqual(citation["title"], "GATES.md")
        self.assertEqual(citation["citation_number"], 1)
        self.assertEqual(result["metadata"]["persona_id"], 7)
        self.assertEqual(result["metadata"]["chat_session_id"], "abc-123")
        # First call was the login, second the chat with the cookie
        self.assertIn("/auth/login", http.calls[0]["url"])
        self.assertIn("fastapi_users_token=tok123", http.calls[1]["headers"]["Cookie"])
        body = json.loads(http.calls[1]["body"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["chat_session_info"]["persona_id"], 7)

    def test_invoke_api_key_skips_login(self):
        with mock.patch.dict(os.environ, {"TEST_ONYX_API_KEY": "sk-onyx"}):
            http = FakeHttp([chat_ok()])
            result = self.adapter(http).invoke("hi")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(
            http.calls[0]["headers"]["Authorization"], "Bearer sk-onyx"
        )

    def test_invoke_persona_override(self):
        http = FakeHttp([login_ok(), chat_ok()])
        self.adapter(http).invoke("hi", persona_id=99)
        body = json.loads(http.calls[1]["body"])
        self.assertEqual(body["chat_session_info"]["persona_id"], 99)

    def test_upstream_error_msg_propagates(self):
        payload = dict(CHAT_RESPONSE, answer="", error_msg="model returned nothing")
        http = FakeHttp([login_ok(), chat_ok(payload)])
        result = self.adapter(http).invoke("hi")
        self.assertEqual(result["status"], "error")
        self.assertIn("model returned nothing", result["error"])

    def test_unreachable_returns_error_envelope(self):
        http = FakeHttp([])  # every call raises
        result = self.adapter(http).invoke("hi")
        self.assertEqual(result["status"], "error")
        self.assertIn("unreachable", result["error"])

    def test_missing_credentials_is_clean_error(self):
        with mock.patch.dict(
            os.environ,
            {"TEST_ONYX_EMAIL": "", "TEST_ONYX_PASSWORD": "", "TEST_ONYX_API_KEY": ""},
        ):
            result = self.adapter(FakeHttp([])).invoke("hi")
        self.assertEqual(result["status"], "error")
        self.assertIn("credentials", result["error"].lower())

    def test_status_reports_reachability_and_credentials(self):
        http = FakeHttp([(200, '{"success":true}', {})])
        status = self.adapter(http).status()
        self.assertTrue(status["reachable"])
        self.assertTrue(status["credentials_present"])
        self.assertIsNone(status["error"])

    def test_lifecycle_noops(self):
        http = FakeHttp([(200, "", {})])
        adapter = self.adapter(http)
        self.assertTrue(adapter.start()["started"])
        self.assertTrue(adapter.stop()["stopped"])


class OptionalityTests(unittest.TestCase):
    """ONYX configured-but-down must never affect other providers."""

    def test_non_onyx_alias_resolves_with_onyx_absent(self):
        resolver = Resolver(config=CONFIG)
        resolved = resolver.resolve_alias("plain-ollama")
        self.assertEqual(resolved["backend"], "ollama")
        self.assertEqual(resolved["real_model"], "qwen:latest")

    def test_onyx_alias_resolves_fields_through_generic_merge(self):
        resolver = Resolver(config=CONFIG)
        resolved = resolver.resolve_alias("company-knowledge")
        self.assertEqual(resolved["backend"], "onyx")
        self.assertEqual(resolved["persona_id"], 7)
        self.assertEqual(resolved["capabilities"], ["invoke"])
        # No onyx import happened as a side effect of resolving; the
        # adapter is only constructed on explicit invoke/validate.

    def test_onyx_down_fails_cleanly_not_loudly(self):
        adapter = OnyxAdapter(
            api_base="http://127.0.0.1:1",  # nothing listens here
            email_env="TEST_ONYX_EMAIL",
            password_env="TEST_ONYX_PASSWORD",
        )
        status = adapter.status()
        self.assertFalse(status["reachable"])
        self.assertIsNotNone(status["error"])


if __name__ == "__main__":
    unittest.main()
