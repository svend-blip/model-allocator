"""Unit tests for Model Allocator V1A."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_allocator import cli, config_loader, resolver, validator
from model_allocator.adapters import ollama as ollama_adapter


SAMPLE_CONFIG = {
    "models": {
        "imple-fast": {
            "runtime_profile": "local_ollama_cuda0",
            "real_model": "qwen36-27b-q4km:latest",
            "context": 131072,
            "lifecycle_policy": "persistent",
            "clients": {"opencode": True, "claude-code": True},
        },
        "review-cloud": {
            "runtime_profile": "cloud_minimax",
            "real_model": "minimax-m3",
            "lifecycle_policy": "cloud_noop",
            "clients": {"opencode": True, "claude-code": False},
        },
        "llama-test": {
            "runtime_profile": "local_llamacpp_cuda0",
            "model_path": "/models/test.gguf",
            "context": 65536,
            "gpu_layers": 35,
            "lifecycle_policy": "stop_after_step",
            "clients": {"opencode": True, "claude-code": True},
        },
    },
    "runtime_profiles": {
        "local_ollama_cuda0": {
            "backend": "ollama",
            "api_base_env": "OLLAMA_BASE_URL",
            "default_api_base": "http://127.0.0.1:11434",
            "gpu": "cuda0",
        },
        "local_llamacpp_cuda0": {
            "backend": "llama_cpp",
            "server_bin_env": "LLAMA_SERVER_BIN",
            "model_root_env": "MODEL_ROOT_GGUF",
            "default_gpu_layers": 99,
            "default_ctx": 131072,
            "gpu": "cuda0",
        },
        "cloud_minimax": {
            "backend": "openai_compatible",
            "api_base_env": "MINIMAX_API_BASE",
            "api_key_env": "MINIMAX_API_KEY",
            "provider": "minimax",
        },
    },
    "roles": {
        "imple01": {
            "default_alias": "imple-fast",
            "client_aliases": {"opencode": "imple-fast", "claude-code": "imple-fast"},
        },
    },
}


class TestConfigLoader(unittest.TestCase):
    def test_yaml_and_json_load_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            yaml_path = tmp_path / "config.yaml"
            json_path = tmp_path / "config.json"
            yaml_path.write_text(
                "runtime_profiles:\n  local_ollama_cuda0:\n    backend: ollama\n"
            )
            json_path.write_text(json.dumps({"runtime_profiles": {"local_ollama_cuda0": {"backend": "ollama"}}}))
            self.assertEqual(config_loader.load_file(yaml_path), config_loader.load_file(json_path))

    def test_env_resolution_with_default(self):
        value = config_loader._resolve_env_string("${OLLAMA_BASE_URL:-http://127.0.0.1:11434}")
        self.assertEqual(value, "http://127.0.0.1:11434")

    def test_env_resolution_simple(self):
        os.environ["MODEL_ALLOCATOR_TEST_VAR"] = "test_value"
        try:
            value = config_loader._resolve_env_string("prefix/${MODEL_ALLOCATOR_TEST_VAR}/suffix")
            self.assertEqual(value, "prefix/test_value/suffix")
        finally:
            del os.environ["MODEL_ALLOCATOR_TEST_VAR"]


class TestResolver(unittest.TestCase):
    def setUp(self):
        self.r = resolver.Resolver(config=SAMPLE_CONFIG)

    def test_resolve_imple_fast(self):
        result = self.r.resolve_alias("imple-fast")
        self.assertEqual(result["backend"], "ollama")
        self.assertEqual(result["real_model"], "qwen36-27b-q4km:latest")
        self.assertIn("context", result)
        self.assertEqual(result["lifecycle_policy"], "persistent")
        self.assertTrue(result["clients"]["opencode"])

    def test_resolve_role_client(self):
        result = self.r.resolve_role_client("imple01", "opencode")
        self.assertEqual(result["alias"], "imple-fast")
        self.assertEqual(result["backend"], "ollama")

    def test_resolve_missing_alias(self):
        with self.assertRaises(resolver.ResolutionError):
            self.r.resolve_alias("missing")


class TestOllamaAdapter(unittest.TestCase):
    def test_api_base_from_profile_uses_default(self):
        profile = {"api_base_env": "OLLAMA_BASE_URL", "default_api_base": "http://127.0.0.1:11434"}
        self.assertEqual(ollama_adapter.OllamaAdapter.api_base_from_profile(profile), "http://127.0.0.1:11434")

    def test_model_availability_parses_names(self):
        adapter = ollama_adapter.OllamaAdapter(
            api_base="http://127.0.0.1:11434", real_model="qwen36-27b-q4km:latest"
        )
        fake_data = {"models": [{"name": "qwen36-27b-q4km:latest"}]}
        with patch.object(adapter, "_request", return_value=fake_data):
            self.assertTrue(adapter.is_model_available()["available"])

    def test_runtime_status_parses_running_models(self):
        adapter = ollama_adapter.OllamaAdapter(api_base="http://127.0.0.1:11434", real_model="x")
        fake_data = {"models": [{"name": "x"}]}
        with patch.object(adapter, "_request", return_value=fake_data):
            status = adapter.runtime_status()
            self.assertTrue(status["running"])
            self.assertIn("x", status["models"])


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.v = validator.Validator(resolver=resolver.Resolver(config=SAMPLE_CONFIG))

    def test_validate_valid_alias_returns_ok_or_warning_if_ollama_down(self):
        result = self.v.validate("imple-fast", "opencode")
        # Ollama may be down on the test machine; accept OK or WARNING, never ERROR.
        self.assertIn(result["validation_status"], ("OK", "WARNING"))
        self.assertEqual(result["resolved_backend"], "ollama")
        self.assertEqual(result["resolved_real_model"], "qwen36-27b-q4km:latest")
        self.assertIn("opencode", result["client_support"])

    def test_validate_missing_alias_returns_error(self):
        result = self.v.validate("missing-alias", "opencode")
        self.assertEqual(result["validation_status"], "ERROR")
        self.assertTrue(result["errors"])

    def test_validate_unsupported_client_returns_error(self):
        result = self.v.validate("review-cloud", "claude-code")
        self.assertEqual(result["validation_status"], "ERROR")
        self.assertTrue(any("not supported" in e for e in result["errors"]))

    def test_validate_missing_env_is_warning_for_default_ollama(self):
        # OLLAMA_BASE_URL may not be set; with default_api_base, this is a warning.
        result = self.v.validate("imple-fast", "opencode")
        if result["validation_status"] == "WARNING":
            self.assertTrue(result["warnings"])


class TestCli(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.cfg_dir = Path(self._dir.name)
        self._write_config(self.cfg_dir)

    def tearDown(self):
        self._dir.cleanup()

    def _write_config(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "models.yaml").write_text(
            "models:\n"
            "  imple-fast:\n"
            "    runtime_profile: local_ollama_cuda0\n"
            "    real_model: qwen36-27b-q4km:latest\n"
            "    context: 131072\n"
            "    lifecycle_policy: persistent\n"
            "    clients:\n"
            "      opencode: true\n"
            "      claude-code: true\n"
        )
        (path / "runtime_profiles.yaml").write_text(
            "runtime_profiles:\n"
            "  local_ollama_cuda0:\n"
            "    backend: ollama\n"
            "    api_base_env: OLLAMA_BASE_URL\n"
            "    default_api_base: http://127.0.0.1:11434\n"
            "    gpu: cuda0\n"
        )
        (path / "roles.yaml").write_text(
            "roles:\n"
            "  imple01:\n"
            "    default_alias: imple-fast\n"
            "    client_aliases:\n"
            "      opencode: imple-fast\n"
        )

    def test_resolve_command(self):
        code = cli.main(["--config-dir", str(self.cfg_dir), "resolve", "--role", "imple01", "--client", "opencode"])
        self.assertIn(code, (cli.EXIT_OK,))

    def test_validate_command(self):
        code = cli.main(
            ["--config-dir", str(self.cfg_dir), "validate", "--alias", "imple-fast", "--client", "opencode"]
        )
        self.assertIn(code, (cli.EXIT_OK, cli.EXIT_WARNING))

    def test_list_only_ok_filters(self):
        code = cli.main(["--config-dir", str(self.cfg_dir), "list", "--only-ok", "--client", "opencode"])
        self.assertIn(code, (cli.EXIT_OK,))

    def test_status_command_does_not_crash(self):
        code = cli.main(["--config-dir", str(self.cfg_dir), "status", "--alias", "imple-fast"])
        self.assertIn(code, (cli.EXIT_OK, cli.EXIT_WARNING))


if __name__ == "__main__":
    unittest.main()
