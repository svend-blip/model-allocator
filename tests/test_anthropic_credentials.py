"""Tests for the cloud_anthropic ``credentials`` mode (subscription vs api_key).

Before this existed, the Claude Code adapter unconditionally injected
``ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY`` for every ``anthropic``-backend alias,
so every role session started through the allocator on fable5 / opus5 / sonnet5
billed API credits instead of using the Max subscription login — with no way to
express the other intent.

The contract these tests lock:

* ``credentials: subscription`` -> the three Anthropic variables are BLANKED,
  so neither an inherited API key nor an inherited base URL can redirect the
  session away from Claude Code's own login.
* ``credentials: api_key`` (and the absent default, for backwards
  compatibility) -> the key is passed through as ``$ANTHROPIC_API_KEY``.
* the validator does not report subscription mode as ``NO_CREDENTIALS`` just
  because no API key is in the environment.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from model_allocator.adapters import claude_code
from model_allocator import resolver, validator


BASE = {
    "alias": "sonnet5",
    "backend": "anthropic",
    "provider": "anthropic",
    "real_model": "claude-sonnet-5",
    "context": 200000,
    "api_key_env": "ANTHROPIC_API_KEY",
    "clients": {"claude-code": True},
}


class SubscriptionModeTests(unittest.TestCase):
    def _env(self, **overrides) -> dict:
        resolved = dict(BASE)
        resolved.update(overrides)
        return claude_code.build_claude_code_command(resolved)["env"]

    def test_subscription_blanks_all_three_anthropic_variables(self):
        env = self._env(credentials="subscription")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "")

    def test_subscription_does_not_pass_the_api_key_through(self):
        """THE REGRESSION: a '$ANTHROPIC_API_KEY' here means API billing."""
        env = self._env(credentials="subscription")
        self.assertNotIn("$", env["ANTHROPIC_API_KEY"])

    def test_subscription_blanks_base_url_so_an_inherited_one_cannot_redirect(self):
        """An inherited ANTHROPIC_BASE_URL (e.g. a local Ollama) must not win."""
        with patch.dict("os.environ", {"ANTHROPIC_BASE_URL": "http://localhost:11434"}):
            env = self._env(credentials="subscription")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "")

    def test_api_key_mode_passes_the_key_through(self):
        env = self._env(credentials="api_key")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "$ANTHROPIC_API_KEY")

    def test_absent_credentials_field_defaults_to_api_key(self):
        """Backwards compatibility: profiles that predate the field are unchanged."""
        env = self._env()
        self.assertEqual(env["ANTHROPIC_API_KEY"], "$ANTHROPIC_API_KEY")

    def test_custom_api_key_env_is_honoured_in_api_key_mode(self):
        env = self._env(credentials="api_key", api_key_env="MY_OTHER_KEY")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "$MY_OTHER_KEY")

    def test_model_flag_is_unchanged_by_the_credentials_mode(self):
        for mode in ("subscription", "api_key"):
            resolved = dict(BASE, credentials=mode)
            argv = claude_code.build_claude_code_command(resolved)["argv"]
            self.assertIn("--model", argv)
            self.assertIn("claude-sonnet-5", argv)


class SubscriptionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        config = {
            "models": {
                "sonnet5": {
                    "runtime_profile": "cloud_anthropic",
                    "real_model": "claude-sonnet-5",
                    "context": 200000,
                    "lifecycle_policy": "cloud_noop",
                    "clients": {"claude-code": True},
                },
            },
            "runtime_profiles": {
                "cloud_anthropic": {
                    "backend": "anthropic",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "provider": "anthropic",
                    "credentials": "subscription",
                },
            },
            "roles": {},
        }
        self.v = validator.Validator(resolver=resolver.Resolver(config=config))

    def test_subscription_is_not_reported_as_missing_credentials(self):
        """With no ANTHROPIC_API_KEY set at all, subscription mode is still fine."""
        with patch.dict("os.environ", {}, clear=True):
            result = self.v.validate("sonnet5", "claude-code")
        self.assertEqual(result["validation_status"], "OK", result)
        self.assertNotEqual(result["client_support"].get("claude-code"), "NO_CREDENTIALS")
        self.assertFalse(
            [w for w in result["warnings"] if "ANTHROPIC_API_KEY" in w and "not set" in w],
            result["warnings"],
        )


if __name__ == "__main__":
    unittest.main()
