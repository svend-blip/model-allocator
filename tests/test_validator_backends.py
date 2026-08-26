"""The validator's backend coverage — the UI's status column, in effect.

`/api/models` reports each alias's status by calling `Validator.validate`
against its FIRST enabled client. A backend the validator does not know
reports ERROR even when its adapter is fully implemented, and the alias looks
broken in the UI while working perfectly from the CLI.

That is not hypothetical: `freetoken` shipped a complete adapter, was
registered in `schema.BACKENDS` and in `cli._get_backend_adapter`, and still
showed ERROR here because the validator keeps its own dispatch chain. Nine
places have to agree about a backend and one was missed.
"""
import os
import unittest
from unittest.mock import patch

from model_allocator.validator import Validator


class _StubResolver:
    def __init__(self, resolved):
        self._resolved = resolved

    def resolve_alias(self, name):
        return dict(self._resolved, alias=name)


def _validate(resolved, client="opencode"):
    return Validator(resolver=_StubResolver(resolved)).validate("a", client)


class TestFreeTokenIsKnownToTheValidator(unittest.TestCase):
    BASE = {
        "backend": "freetoken",
        "clients": {"opencode": True},
        "context": 262144,
        "model_path": "vrfai/Qwen3.8-27B-NVFP4",
        "executable": "/x/ft",
        "num_tokens": 49152,
    }

    def test_a_complete_profile_validates_clean(self):
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = _validate(self.BASE)
        self.assertEqual(result["validation_status"], "OK", result["errors"])
        self.assertNotIn("is not implemented", " ".join(result["errors"]))

    def test_a_missing_model_reference_is_an_error(self):
        resolved = dict(self.BASE, model_path="")
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = _validate(resolved)
        self.assertEqual(result["validation_status"], "ERROR")
        self.assertTrue(any("model_path" in e for e in result["errors"]))

    def test_a_hugging_face_repo_id_is_not_filesystem_checked(self):
        """Rejecting a repo id would fail the configuration this machine
        qualified — both shipped profiles name repos, not paths."""
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = _validate(dict(self.BASE, model_path="Qwen/Qwen3.6-35B-A3B"))
        self.assertEqual(result["validation_status"], "OK", result["errors"])

    def test_an_unresolvable_executable_is_an_error(self):
        with patch("os.path.isfile", return_value=False):
            result = _validate(self.BASE)
        self.assertEqual(result["validation_status"], "ERROR")
        self.assertTrue(any("executable" in e for e in result["errors"]))

    def test_no_executable_and_none_on_path_warns_about_the_venv(self):
        resolved = dict(self.BASE)
        resolved.pop("executable")
        with patch("model_allocator.validator.shutil.which", return_value=None):
            result = _validate(resolved)
        self.assertEqual(result["validation_status"], "WARNING")
        self.assertTrue(any("FREETOKEN_BIN" in w for w in result["warnings"]))

    def test_a_missing_kv_budget_warns(self):
        """The dense profile genuinely has no num_tokens and should say so.

        Measured in FT-6: without it FreeToken sizes the budget from leftover
        VRAM and lands on 14303 tokens against an advertised 262144, which a
        coding harness exceeds before reading a file. The warning is the
        truth about that profile, not a demand to change it — raising the
        budget on the dense model was measured to OOM in prefill.
        """
        resolved = dict(self.BASE)
        resolved.pop("num_tokens")
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = _validate(resolved)
        self.assertEqual(result["validation_status"], "WARNING")
        self.assertTrue(any("KV budget" in w for w in result["warnings"]))

    def test_a_declared_kv_budget_does_not_warn(self):
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = _validate(self.BASE)
        self.assertFalse(any("KV budget" in w for w in result["warnings"]))


class TestExternalBackendIsNotAFault(unittest.TestCase):
    def test_client_managed_runtime_validates(self):
        """`external` owns nothing by design — freebuff manages its own
        model and runtime, and the profile exists only because every alias
        must name one. Reporting that as an unimplemented adapter was
        reporting a design choice as a defect."""
        result = _validate({"backend": "external",
                            "clients": {"freebuff": True},
                            "context": 128000},
                           client="freebuff")
        self.assertEqual(result["validation_status"], "OK", result["errors"])
        self.assertIn("client-managed", result["client_support"]["freebuff"])


class TestClaudeCodeAgainstLlamaCpp(unittest.TestCase):
    """The adapter has always handled it; the compatibility rule had not."""

    def test_llama_cpp_is_compatible_with_claude_code(self):
        result = _validate({"backend": "llama_cpp",
                            "clients": {"claude-code": True},
                            "context": 262144,
                            "model_path": "/models/x.gguf",
                            "port": 8080},
                           client="claude-code")
        self.assertFalse(
            any("incompatible" in e for e in result["errors"]),
            "claude_code.build_claude_code_command accepts llama_cpp and sets "
            "ANTHROPIC_BASE_URL to the server; the validator must agree")

    def test_the_adapter_and_the_validator_agree_on_the_client_list(self):
        """Bound together so they cannot drift apart again.

        Only `laguna-local` ever surfaced the disagreement, because the UI
        validates against the FIRST enabled client and the four other
        llama.cpp aliases declaring claude-code list opencode first. Dict
        ordering hid the same condition on all of them.
        """
        from model_allocator.adapters import claude_code
        import inspect
        source = inspect.getsource(claude_code.build_claude_code_command)
        for backend in ("ollama", "openai_compatible", "anthropic", "llama_cpp"):
            self.assertIn(f'"{backend}"', source,
                          f"adapter no longer names {backend}")
            result = _validate({"backend": backend,
                                "clients": {"claude-code": True},
                                "context": 4096,
                                "model_path": "/m.gguf",
                                "real_model": "m"},
                               client="claude-code")
            self.assertFalse(
                any("incompatible" in e for e in result["errors"]),
                f"validator rejects {backend} that the adapter accepts")


if __name__ == "__main__":
    unittest.main()
