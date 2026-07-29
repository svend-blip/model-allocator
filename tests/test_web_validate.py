"""Regression tests for the web UI's model-validation endpoint.

The endpoint layer had no tests, which is how this bug shipped: both the
Validate button (``static/js/app.js``) and the endpoint's fallback hardcoded
``client="opencode"``. Every alias that does not enable opencode was therefore
reported as ERROR — 10 of the 19 aliases in the live ``models.yaml``, including
the Anthropic ones (fable5, opus5, sonnet5), the claude-code-only Ollama roles
(imple01-claude, coder-*, trend-local, learn-local) and the headless-only
``company-knowledge``. The model list never showed those errors, because it
validates against the alias's own first enabled client, so the table said OK
while the button said Error.

The contract these tests lock:

* no client named  -> validate against the clients the ALIAS declares
* client named     -> validate exactly that client (CLI parity, so a genuine
                      incompatibility is still reported)
* no enabled client-> ERROR naming the alias, not a silent OK
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from model_allocator.web import app as web_app


CONFIG = {
    "models": {
        # claude-code only, Anthropic backend — the shape that regressed.
        "opus5": {
            "runtime_profile": "cloud_anthropic",
            "real_model": "claude-opus-5",
            "context": 200000,
            "lifecycle_policy": "cloud_noop",
            "clients": {"claude-code": True},
        },
        # opencode only — the shape that always worked.
        "review01-local": {
            "runtime_profile": "local_ollama_cuda0",
            "real_model": "qwen3:30b",
            "context": 131072,
            "lifecycle_policy": "stop_after_step",
            "clients": {"opencode": True},
        },
        # Explicitly disabled everywhere.
        "orphan": {
            "runtime_profile": "local_ollama_cuda0",
            "real_model": "qwen3:30b",
            "context": 131072,
            "lifecycle_policy": "stop_after_step",
            "clients": {"opencode": False, "claude-code": False},
        },
    },
    "runtime_profiles": {
        "cloud_anthropic": {
            "backend": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "provider": "anthropic",
        },
        "local_ollama_cuda0": {
            "backend": "ollama",
            "api_base_env": "OLLAMA_BASE_URL",
            "default_api_base": "http://127.0.0.1:11434",
            "gpu": "cuda0",
        },
    },
    "roles": {},
}


class WebValidateEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="alloc-web-")
        cfg_dir = Path(self._tmp.name)
        (cfg_dir / "models.yaml").write_text(
            yaml.safe_dump({"models": CONFIG["models"]}), encoding="utf-8"
        )
        (cfg_dir / "runtime_profiles.yaml").write_text(
            yaml.safe_dump({"runtime_profiles": CONFIG["runtime_profiles"]}),
            encoding="utf-8",
        )
        (cfg_dir / "roles.yaml").write_text(
            yaml.safe_dump({"roles": CONFIG["roles"]}), encoding="utf-8"
        )
        self._previous_config_dir = web_app.CONFIG_DIR
        web_app.CONFIG_DIR = cfg_dir
        self.client = TestClient(web_app.app)

    def tearDown(self) -> None:
        web_app.CONFIG_DIR = self._previous_config_dir
        self._tmp.cleanup()

    def _validate(self, alias: str, body: dict | None = None) -> dict:
        return self.client.post(
            f"/api/models/{alias}/validate", json=body if body is not None else {}
        ).json()

    def test_claude_code_only_alias_is_not_reported_as_error(self):
        """THE REGRESSION: opus5 supports claude-code, so it must validate OK.

        Before the fix this returned ERROR with "Client 'opencode' is
        incompatible with backend 'anthropic'" — a correct statement about a
        client the alias never claimed to support.
        """
        result = self._validate("opus5")
        self.assertEqual(result["validation_status"], "OK", result.get("errors"))
        self.assertEqual(result["validated_clients"], ["claude-code"])
        self.assertEqual(result["errors"], [])

    def test_opencode_only_alias_still_validates(self):
        result = self._validate("review01-local")
        self.assertEqual(result["validated_clients"], ["opencode"])
        self.assertNotIn("claude-code", result["validated_clients"])

    def test_explicit_client_is_honoured_and_still_reports_incompatibility(self):
        """CLI parity: asking specifically about opencode must still say no.

        The fix must not paper over real incompatibilities — it must stop
        ASSUMING the question. This is the assertion that separates the two.
        """
        result = self._validate("opus5", {"client": "opencode"})
        self.assertEqual(result["validation_status"], "ERROR")
        self.assertTrue(
            any("opencode" in message for message in result["errors"]),
            result["errors"],
        )

    def test_alias_with_no_enabled_client_reports_error_naming_the_alias(self):
        result = self._validate("orphan")
        self.assertEqual(result["validation_status"], "ERROR")
        self.assertEqual(result["validated_clients"], [])
        self.assertTrue(
            any("orphan" in message for message in result["errors"]), result["errors"]
        )

    def test_errors_are_prefixed_with_the_client_that_raised_them(self):
        """A multi-client aggregate is useless if you cannot tell who failed."""
        result = self._validate("opus5")
        for message in result["errors"] + result["warnings"]:
            self.assertRegex(message, r"^\[[^\]]+\] ")

    def test_model_list_status_agrees_with_the_button(self):
        """The table and the button must not disagree — that was the symptom."""
        listed = {m["alias"]: m for m in self.client.get("/api/models").json()["models"]}
        for alias in ("opus5", "review01-local"):
            self.assertEqual(
                listed[alias]["validation_status"],
                self._validate(alias)["validation_status"].lower(),
                f"{alias}: list status and validate status disagree",
            )


if __name__ == "__main__":
    unittest.main()
