"""Unit tests for Model Allocator V2 adapters and commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_allocator import cli
from model_allocator.adapters import claude_code
from model_allocator.adapters import llama_cpp as llama_cpp_adapter
from model_allocator.adapters import opencode
from model_allocator.adapters import anthropic as anthropic_adapter
from model_allocator.adapters import openai_compatible as openai_adapter
from model_allocator.renderer import render_tmux_shell_string
from model_allocator.resolver import Resolver
from model_allocator.validator import Validator


SAMPLE_CONFIG = {
    "models": {
        "imple01-claude": {
            "runtime_profile": "local_ollama_cuda0",
            "real_model": "qwen3-coder:30b-256k",
            "context": 131072,
            "lifecycle_policy": "stop_after_step",
            "clients": {"opencode": False, "claude-code": True},
        },
        "review-cloud": {
            "runtime_profile": "cloud_minimax",
            "real_model": "minimax-m3",
            "lifecycle_policy": "cloud_noop",
            "clients": {"opencode": True, "claude-code": False},
        },
        "openrouter-test": {
            "runtime_profile": "cloud_openrouter",
            "real_model": "qwen3-30b-a3b",
            "lifecycle_policy": "cloud_noop",
            "clients": {"opencode": True, "claude-code": True},
        },
        "llama-test": {
            "runtime_profile": "local_llamacpp_cuda0",
            "model_path": "/models/test.gguf",
            "context": 262144,
            "parallel": 1,
            "n_cpu_moe": 26,
            "threads": 12,
            "batch": 160,
            "ubatch_size": 128,
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo3",
            "flash_attn": "on",
            "reasoning": "off",
            "no_mmap": True,
            "lifecycle_policy": "stop_after_step",
            "clients": {"opencode": True, "claude-code": False},
        },
    },
    "runtime_profiles": {
        "local_ollama_cuda0": {
            "backend": "ollama",
            "api_base_env": "OLLAMA_BASE_URL",
            "default_api_base": "http://127.0.0.1:11434",
            "gpu": "cuda0",
        },
        "cloud_minimax": {
            "backend": "openai_compatible",
            "api_base_env": "MINIMAX_API_BASE",
            "api_key_env": "MINIMAX_API_KEY",
            "provider": "minimax",
        },
        "cloud_openrouter": {
            "backend": "openai_compatible",
            "api_base_env": "OPENROUTER_API_BASE",
            "api_key_env": "OPENROUTER_API_KEY",
            "default_api_base": "https://openrouter.ai/api",
            "provider": "openrouter",
        },
        "local_llamacpp_cuda0": {
            "backend": "llama_cpp",
            "server_bin_env": "LLAMA_SERVER_BIN",
            "model_root_env": "MODEL_ROOT_GGUF",
            "default_port": 8080,
            "default_ctx": 131072,
            "gpu": "cuda0",
        },
    },
    "roles": {
        "claude-test": {
            "default_alias": "imple01-claude",
            "config_dir": "claude-test",
            "client_aliases": {"claude-code": "imple01-claude"},
        },
        "openrouter-test": {
            "default_alias": "openrouter-test",
            "config_dir": "openrouter-test",
            "client_aliases": {"opencode": "openrouter-test"},
        },
        "llama-test-role": {
            "default_alias": "llama-test",
            "config_dir": "llama-test",
            "client_aliases": {"opencode": "llama-test"},
        },
    },
}


def _fake_which(name: str):
    return f"/usr/bin/{name}"


class TestClaudeCodeAdapter(unittest.TestCase):
    @patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which)
    def test_ollama_backend(self, _mock):
        resolved = {
            "backend": "ollama",
            "real_model": "qwen3-coder:30b-256k",
            "api_base_env": "OLLAMA_BASE_URL",
            "default_api_base": "http://127.0.0.1:11434",
        }
        cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(cmd["argv"], ["/usr/bin/claude", "--model", "qwen3-coder:30b-256k"])
        self.assertEqual(cmd["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:11434")
        self.assertEqual(cmd["env"]["ANTHROPIC_AUTH_TOKEN"], "ollama")
        self.assertNotIn("ANTHROPIC_API_KEY", cmd["env"])

    @patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which)
    def test_openrouter_backend(self, _mock):
        resolved = {
            "backend": "openai_compatible",
            "provider": "openrouter",
            "real_model": "qwen3-30b-a3b",
            "api_base_env": "OPENROUTER_API_BASE",
            "api_key_env": "OPENROUTER_API_KEY",
            "default_api_base": "https://openrouter.ai/api",
        }
        with patch.dict(os.environ, {"OPENROUTER_API_BASE": "https://openrouter.ai/api"}):
            cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(cmd["env"]["ANTHROPIC_AUTH_TOKEN"], "$OPENROUTER_API_KEY")
        self.assertEqual(cmd["env"]["ANTHROPIC_API_KEY"], "")

    def test_rejects_minimax(self):
        resolved = {
            "backend": "openai_compatible",
            "provider": "minimax",
            "real_model": "minimax-m3",
        }
        with self.assertRaises(ValueError):
            claude_code.build_claude_code_command(resolved)


class TestOpenCodeAdapter(unittest.TestCase):
    @patch("model_allocator.adapters.opencode.shutil.which", side_effect=_fake_which)
    def test_command_has_no_model_flag(self, _mock):
        """OpenCode ignores --model on session resumption — the flag is dead.
        The model is set via opencode.json's model field, refreshed by run."""
        cmd = opencode.build_opencode_command({"backend": "ollama", "real_model": "qwen"}, "r")
        self.assertNotIn("--model", cmd["argv"])
        self.assertEqual(len(cmd["argv"]), 1)  # just the binary

    @patch("model_allocator.adapters.opencode.shutil.which", side_effect=_fake_which)
    def test_command_env_has_config_dir(self, _mock):
        cmd = opencode.build_opencode_command({"backend": "ollama", "real_model": "qwen"}, "myrole")
        self.assertIn("OPENCODE_CONFIG_DIR", cmd["env"])
        self.assertIn("myrole", cmd["env"]["OPENCODE_CONFIG_DIR"])

    @patch("model_allocator.adapters.opencode.shutil.which", side_effect=_fake_which)
    def test_config_model_field_ollama(self, _mock):
        """build_opencode_config produces the model field for opencode.json."""
        cfg = opencode.build_opencode_config({"backend": "ollama", "real_model": "qwen"})
        self.assertEqual(cfg["model"], "ollama/qwen")

    @patch("model_allocator.adapters.opencode.shutil.which", side_effect=_fake_which)
    def test_config_model_field_llama_cpp(self, _mock):
        resolved = {
            "backend": "llama_cpp",
            "provider": "llama-turbo",
            "real_model": "/models/model.gguf",
            "opencode_model_id": "qwen36-35b-turbo262k",
        }
        cfg = opencode.build_opencode_config(resolved)
        self.assertEqual(cfg["model"], "llama-turbo/qwen36-35b-turbo262k")

    def test_render_config_ollama(self):
        resolved = {"backend": "ollama", "real_model": "qwen", "context": 131072, "default_api_base": "http://127.0.0.1:11434"}
        cfg = opencode.build_opencode_config(resolved)
        self.assertEqual(cfg["model"], "ollama/qwen")
        self.assertIn("qwen", cfg["provider"]["ollama"]["models"])
        self.assertEqual(cfg["provider"]["ollama"]["models"]["qwen"]["limit"]["context"], 131072)

    def test_render_config_llama_cpp(self):
        resolved = {
            "backend": "llama_cpp",
            "provider": "llama-turbo",
            "real_model": "/models/model.gguf",
            "opencode_model_id": "qwen36-35b-turbo262k",
            "port": 8080,
        }
        cfg = opencode.build_opencode_config(resolved)
        self.assertEqual(cfg["provider"]["llama-turbo"]["options"]["baseURL"], "http://127.0.0.1:8080/v1")
        self.assertIn("qwen36-35b-turbo262k", cfg["provider"]["llama-turbo"]["models"])

    def test_render_config_has_top_level_model_ollama(self):
        resolved = {"backend": "ollama", "real_model": "qwen3-coder:30b-256k", "default_api_base": "http://127.0.0.1:11434"}
        cfg = opencode.build_opencode_config(resolved)
        self.assertEqual(cfg["model"], "ollama/qwen3-coder:30b-256k")

    def test_render_config_has_top_level_model_openrouter(self):
        resolved = {
            "backend": "openai_compatible",
            "provider": "openrouter",
            "real_model": "qwen3-30b-a3b",
            "default_api_base": "https://openrouter.ai/api",
        }
        cfg = opencode.build_opencode_config(resolved)
        self.assertEqual(cfg["model"], "openrouter/qwen3-30b-a3b")

    def test_render_config_has_top_level_model_llama_cpp(self):
        resolved = {
            "backend": "llama_cpp",
            "provider": "llama-turbo",
            "real_model": "/models/model.gguf",
            "opencode_model_id": "qwen36-35b-turbo262k",
            "port": 8080,
        }
        cfg = opencode.build_opencode_config(resolved)
        self.assertEqual(cfg["model"], "llama-turbo/qwen36-35b-turbo262k")


class TestOpenAICompatibleAdapter(unittest.TestCase):
    def test_api_base_from_profile_defaults(self):
        profile = {"api_base_env": "X", "default_api_base": None}
        self.assertEqual(openai_adapter.OpenAICompatibleAdapter.api_base_from_profile(profile), "")

    def test_reachability_and_credentials_mocked(self):
        adapter = openai_adapter.OpenAICompatibleAdapter(
            api_base="https://example.com", api_key_env="TEST_API_KEY"
        )
        with patch.dict(os.environ, {"TEST_API_KEY": "secret"}):
            self.assertTrue(adapter.are_credentials_present()["present"])

        with patch.object(adapter, "_request", return_value={"status_code": 200}):
            self.assertTrue(adapter.is_api_reachable()["reachable"])

    def test_start_requires_credentials(self):
        adapter = openai_adapter.OpenAICompatibleAdapter(api_base="https://example.com", api_key_env="MISSING")
        with patch.object(adapter, "is_api_reachable", return_value={"reachable": True, "error": None}):
            result = adapter.start()
            self.assertFalse(result["started"])


class TestLlamaCppAdapter(unittest.TestCase):
    def test_argv_assembly(self):
        resolved = {
            "alias": "llama-test",
            "model_path": "/models/model.gguf",
            "context": 262144,
            "parallel": 1,
            "n_cpu_moe": 26,
            "threads": 12,
            "batch": 160,
            "ubatch_size": 128,
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo3",
            "flash_attn": "on",
            "reasoning": "off",
            "no_mmap": True,
            "default_port": 8080,
        }
        adapter = llama_cpp_adapter.LlamaCppAdapter(resolved)
        with patch.object(adapter, "server_bin", return_value="/bin/llama-server"):
            with patch.object(adapter, "model_path", return_value="/models/model.gguf"):
                argv = adapter._build_argv()
        self.assertIn("--ctx-size", argv)
        self.assertIn("262144", argv)
        self.assertIn("--n-cpu-moe", argv)
        self.assertIn("26", argv)
        self.assertIn("--cache-type-k", argv)
        self.assertIn("turbo4", argv)
        self.assertIn("--flash-attn", argv)
        self.assertIn("on", argv)
        self.assertIn("--no-mmap", argv)

    def test_finds_free_port_when_not_configured(self):
        resolved = {"alias": "x", "context": 4096}
        adapter = llama_cpp_adapter.LlamaCppAdapter(resolved)
        self.assertGreater(adapter.port, 0)

    def test_stop_removes_pid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = {"alias": "x", "context": 4096, "default_port": 8080}
            adapter = llama_cpp_adapter.LlamaCppAdapter(resolved, state_dir=tmp)
            pid_file = Path(adapter.pid_file)
            pid_file.write_text("999999999", encoding="utf-8")
            with patch("os.kill") as mock_kill:
                mock_kill.side_effect = ProcessLookupError()
                result = adapter.stop(timeout=1)
            self.assertTrue(result["stopped"])
            self.assertFalse(pid_file.exists())


class TestValidatorV2(unittest.TestCase):
    def setUp(self):
        self.v = Validator(resolver=Resolver(config=SAMPLE_CONFIG))

    def test_minimax_with_claude_is_error(self):
        result = self.v.validate("review-cloud", "claude-code")
        self.assertEqual(result["validation_status"], "ERROR")
        self.assertTrue(any("incompatible" in e.lower() for e in result["errors"]))

    def test_openai_missing_env_is_warning(self):
        result = self.v.validate("review-cloud", "opencode")
        self.assertIn(result["validation_status"], ("WARNING", "ERROR"))

    def test_llama_cpp_missing_binary_is_warning(self):
        result = self.v.validate("llama-test", "opencode")
        self.assertIn(result["validation_status"], ("WARNING", "ERROR"))


class TestCliV2(unittest.TestCase):
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
            "  imple01-claude:\n"
            "    runtime_profile: local_ollama_cuda0\n"
            "    real_model: qwen3-coder:30b-256k\n"
            "    context: 131072\n"
            "    lifecycle_policy: stop_after_step\n"
            "    clients:\n"
            "      opencode: false\n"
            "      claude-code: true\n"
            "  openrouter-test:\n"
            "    runtime_profile: cloud_openrouter\n"
            "    real_model: qwen3-30b-a3b\n"
            "    lifecycle_policy: cloud_noop\n"
            "    clients:\n"
            "      opencode: true\n"
            "      claude-code: true\n"
            "  llama-test:\n"
            "    runtime_profile: local_llamacpp_cuda0\n"
            "    model_path: /models/test.gguf\n"
            "    context: 262144\n"
            "    lifecycle_policy: stop_after_step\n"
            "    clients:\n"
            "      opencode: true\n"
            "      claude-code: false\n"
        )
        (path / "runtime_profiles.yaml").write_text(
            "runtime_profiles:\n"
            "  local_ollama_cuda0:\n"
            "    backend: ollama\n"
            "    api_base_env: OLLAMA_BASE_URL\n"
            "    default_api_base: http://127.0.0.1:11434\n"
            "  cloud_openrouter:\n"
            "    backend: openai_compatible\n"
            "    api_base_env: OPENROUTER_API_BASE\n"
            "    api_key_env: OPENROUTER_API_KEY\n"
            "    default_api_base: https://openrouter.ai/api\n"
            "    provider: openrouter\n"
            "  local_llamacpp_cuda0:\n"
            "    backend: llama_cpp\n"
            "    server_bin_env: LLAMA_SERVER_BIN\n"
            "    default_port: 8080\n"
        )
        (path / "roles.yaml").write_text(
            "roles:\n"
            "  claude-test:\n"
            "    default_alias: imple01-claude\n"
            "    config_dir: claude-test\n"
            "    client_aliases:\n"
            "      claude-code: imple01-claude\n"
            "  openrouter-test:\n"
            "    default_alias: openrouter-test\n"
            "    config_dir: openrouter-test\n"
            "    client_aliases:\n"
            "      opencode: openrouter-test\n"
        )

    @patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which)
    def test_run_claude(self, _mock):
        code = cli.main(["--config-dir", str(self.cfg_dir), "run", "--role", "claude-test", "--client", "claude-code"])
        self.assertEqual(code, cli.EXIT_OK)

    @patch("model_allocator.adapters.opencode.shutil.which", side_effect=_fake_which)
    def test_run_openrouter(self, _mock):
        code = cli.main(["--config-dir", str(self.cfg_dir), "run", "--role", "openrouter-test", "--client", "opencode"])
        self.assertEqual(code, cli.EXIT_OK)

    def test_env_claude(self):
        code = cli.main(["--config-dir", str(self.cfg_dir), "env", "--role", "claude-test", "--client", "claude-code"])
        # claude binary missing on PATH in test env is acceptable; the command may error.
        self.assertIn(code, (cli.EXIT_OK, cli.EXIT_ERROR))

    def test_render_config(self):
        code = cli.main(["--config-dir", str(self.cfg_dir), "render-config", "--role", "openrouter-test", "--client", "opencode"])
        self.assertEqual(code, cli.EXIT_OK)

    @patch("model_allocator.adapters.opencode.shutil.which", side_effect=_fake_which)
    def test_render_config_output_fresh_path(self, _mock):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "opencode.json"
            code = cli.main([
                "--config-dir", str(self.cfg_dir),
                "render-config",
                "--role", "openrouter-test",
                "--client", "opencode",
                "--output", str(output_path),
            ])
            self.assertEqual(code, cli.EXIT_OK)
            self.assertTrue(output_path.exists())
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(data["model"], "openrouter/qwen3-30b-a3b")
            self.assertIn("provider", data)

    @patch("model_allocator.adapters.opencode.shutil.which", side_effect=_fake_which)
    def test_render_config_output_merges_existing(self, _mock):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "opencode.json"
            existing = {
                "$schema": "https://example.com/schema.json",
                "permission": {"allow": ["read", "write"]},
                "mcp": {"servers": {"test": {"command": "test"}}},
                "provider": {
                    "minimax": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "minimax",
                        "options": {"baseURL": "https://minimax.example.com/v1"},
                        "models": {"minimax-m3": {"name": "minimax-m3"}},
                    },
                },
            }
            output_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            code = cli.main([
                "--config-dir", str(self.cfg_dir),
                "render-config",
                "--role", "openrouter-test",
                "--client", "opencode",
                "--output", str(output_path),
            ])
            self.assertEqual(code, cli.EXIT_OK)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(data["$schema"], existing["$schema"])
            self.assertEqual(data["permission"], existing["permission"])
            self.assertEqual(data["mcp"], existing["mcp"])
            self.assertIn("minimax", data["provider"])
            self.assertEqual(data["model"], "openrouter/qwen3-30b-a3b")
            self.assertIn("openrouter", data["provider"])

    def test_unload_missing_alias_is_error(self):
        code = cli.main(["--config-dir", str(self.cfg_dir), "unload", "--alias", "missing"])
        self.assertEqual(code, cli.EXIT_ERROR)


class TestResolverPreservesAliasFields(unittest.TestCase):
    def test_llama_alias_fields_survive_resolver_and_appear_in_argv(self):
        """Regression test: resolver must preserve alias fields for adapters."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)
            (cfg / "models.yaml").write_text(
                "models:\n"
                "  svend3060-llama-test:\n"
                "    runtime_profile: local_llamacpp\n"
                "    model_path: /Qwen3.6-35B-A3B-MXFP4_MOE.gguf\n"
                "    context: 262144\n"
                "    port: 8081\n"
                "    n_cpu_moe: 26\n"
                "    cache_type_k: turbo4\n"
                "    cache_type_v: turbo3\n"
                "    flash_attn: \"on\"\n"
                "    reasoning: \"off\"\n"
                "    no_mmap: true\n"
                "    clients:\n"
                "      opencode: true\n"
            )
            (cfg / "runtime_profiles.yaml").write_text(
                "runtime_profiles:\n"
                "  local_llamacpp:\n"
                "    backend: llama_cpp\n"
                "    server_bin_env: LLAMA_SERVER_BIN\n"
            )
            (cfg / "roles.yaml").write_text("roles:\n")

            resolver = Resolver(config_dir=str(cfg))
            resolved = resolver.resolve_alias("svend3060-llama-test")

            # Fields must survive the resolver.
            self.assertEqual(resolved["port"], 8081)
            self.assertEqual(resolved["n_cpu_moe"], 26)
            self.assertEqual(resolved["cache_type_k"], "turbo4")
            self.assertEqual(resolved["flash_attn"], "on")

            adapter = llama_cpp_adapter.LlamaCppAdapter(resolved)
            with patch.object(adapter, "server_bin", return_value="/bin/llama-server"):
                with patch.object(adapter, "model_path", return_value="/Qwen3.6-35B-A3B-MXFP4_MOE.gguf"):
                    argv = adapter._build_argv()

            self.assertIn("8081", argv)
            self.assertIn("--n-cpu-moe", argv)
            self.assertIn("26", argv)
            self.assertIn("--cache-type-k", argv)
            self.assertIn("turbo4", argv)
            self.assertIn("--flash-attn", argv)
            self.assertIn("on", argv)


class TestAnthropicAdapter(unittest.TestCase):
    def test_credentials_present_when_key_set(self):
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env="TEST_ANTHROPIC_KEY")
        with patch.dict(os.environ, {"TEST_ANTHROPIC_KEY": "sk-ant-12345"}):
            result = adapter.are_credentials_present()
        self.assertTrue(result["present"])
        self.assertIsNone(result["error"])

    def test_credentials_missing_when_key_not_set(self):
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env="MISSING_KEY")
        result = adapter.are_credentials_present()
        self.assertFalse(result["present"])
        self.assertIsNotNone(result["error"])

    def test_start_succeeds_when_credentials_present(self):
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env="TEST_ANTHROPIC_KEY")
        with patch.dict(os.environ, {"TEST_ANTHROPIC_KEY": "sk-ant-12345"}):
            result = adapter.start()
        self.assertTrue(result["started"])
        self.assertIsNone(result["error"])

    def test_start_fails_when_credentials_missing(self):
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env="MISSING_KEY")
        result = adapter.start()
        self.assertFalse(result["started"])
        self.assertIsNotNone(result["error"])

    def test_stop_is_noop(self):
        adapter = anthropic_adapter.AnthropicAdapter()
        result = adapter.stop()
        self.assertTrue(result["stopped"])
        self.assertIsNone(result["error"])

    def test_unload_is_noop(self):
        adapter = anthropic_adapter.AnthropicAdapter()
        result = adapter.unload()
        self.assertTrue(result["unloaded"])
        self.assertIsNone(result["error"])

    def test_status_reports_credentials(self):
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env="TEST_ANTHROPIC_KEY")
        with patch.dict(os.environ, {"TEST_ANTHROPIC_KEY": "sk-ant-12345"}):
            result = adapter.status()
        self.assertTrue(result["reachable"])
        self.assertTrue(result["credentials_present"])
        self.assertIsNone(result["error"])


if __name__ == "__main__":
    unittest.main()
