"""`claude_env` on an alias reaches the claude-code client's environment,
and the adapter's own keys stay authoritative (2026-09-02, Qwen Cloud
Token Plan: one Anthropic-shaped endpoint, several model ids by role)."""
import unittest

from model_allocator.adapters.claude_code import build_claude_code_command


def _resolved(**extra):
    base = {
        "alias": "cloud_qwen38max-claude",
        "backend": "openai_compatible",
        "api_base_env": "DASHSCOPE_ANTHROPIC_BASE",
        "default_api_base": "https://token-plan.example/apps/anthropic",
        "api_key_env": "DASHSCOPE_API_KEY",
        "real_model": "qwen3.8-max",
        "context": 983616,
    }
    base.update(extra)
    return base


class ClaudeEnvPassthrough(unittest.TestCase):
    def test_declared_env_reaches_the_client(self):
        cmd = build_claude_code_command(_resolved(claude_env={
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.6-flash",
            "CLAUDE_CODE_SUBAGENT_MODEL": "qwen3.7-max",
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": 983616,
        }))
        env = cmd["env"]
        self.assertEqual(env["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "qwen3.6-flash")
        self.assertEqual(env["CLAUDE_CODE_SUBAGENT_MODEL"], "qwen3.7-max")
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "983616")
        self.assertEqual(cmd["argv"][-2:], ["--model", "qwen3.8-max"])

    def test_adapter_owned_keys_cannot_be_redirected(self):
        cmd = build_claude_code_command(_resolved(claude_env={
            "ANTHROPIC_BASE_URL": "https://elsewhere.example",
            "ANTHROPIC_AUTH_TOKEN": "leak",
            "ANTHROPIC_API_KEY": "leak",
        }))
        env = cmd["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "https://token-plan.example/apps/anthropic")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "$DASHSCOPE_API_KEY")
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertIn("ANTHROPIC_API_KEY", cmd["unset_env"])

    def test_absent_or_malformed_claude_env_is_ignored(self):
        self.assertNotIn("CLAUDE_CODE_SUBAGENT_MODEL", build_claude_code_command(_resolved())["env"])
        self.assertNotIn("CLAUDE_CODE_SUBAGENT_MODEL", build_claude_code_command(_resolved(claude_env="nope"))["env"])


if __name__ == "__main__":
    unittest.main()
