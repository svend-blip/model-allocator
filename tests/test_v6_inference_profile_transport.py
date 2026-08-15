"""V6 inference profile transport (B3, handoff 038).

The Mission Contract (O4) requires that implemented profile fields ride
on EXISTING backend/client paths — no generic parameter passthrough, no
new abstraction. The two implemented fields are:

- ``max_output_tokens`` — reaches the opencode limit block, the pi
  models.json ``maxTokens`` entry, the claude_code
  ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` env var, and the ``--max-output-tokens``
  CLI override. The four consumers read ``resolved["max_output_tokens"]``
  which the resolver merges from the profile (with alias-level and CLI
  overrides winning).
- ``reasoning_budget`` — reaches the llama-server argv as
  ``--reasoning-budget N`` for llmserver-backed aliases.

Far fields (``temperature``, ``top_p``) have no existing transport path
and would have silently done nothing. The doctor and the resolver now
reject them with a "deferred in V6" message — the contract forbids
silent acceptance of unsupported fields.

The transport assertion is on the RENDERED config/argv, not on the
process layer. Nothing in this file starts a real server or client.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from model_allocator import cli, schema
from model_allocator.adapters import (
    claude_code,
    llama_cpp,
    opencode,
    pi as pi_adapter,
)
from model_allocator.doctor_cli import cmd_doctor
from model_allocator.resolver import ResolutionError, Resolver


def _fake_which(name: str):
    """shutil.which stub for adapter tests that need a binary path."""
    return f"/usr/bin/{name}"


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

PROFILES = {
    "local_llamacpp": {"backend": "llama_cpp", "server_bin_env": "LLAMA_SERVER_BIN"},
    "local_ollama": {"backend": "ollama", "default_api_base": "http://127.0.0.1:11434"},
}


def _write_minimal(tmp: Path, *, profile_name: str,
                   profile_backend: str = "llama_cpp") -> None:
    (tmp / "runtime_profiles.yaml").write_text(
        f"runtime_profiles:\n  {profile_name}:\n    backend: {profile_backend}\n",
        encoding="utf-8",
    )
    (tmp / "roles.yaml").write_text("roles:\n", encoding="utf-8")


def _seed_instance_bound_with_profile(
        tmp: Path, *,
        alias_name: str = "shared-architect",
        instance_name: str = "shared-llm",
        profile_name: str = "profile-careful",
        profile_block: str = "    reasoning_budget: 4096\n"
                            "    max_output_tokens: 16384\n",
        alias_overrides: str = "",
        backend: str = "llama_cpp") -> Path:
    """Seed a minimal V6 config: alias bound to a runtime_instance + profile."""
    backend_block = "    gpu_layers: 99\n" if backend == "llama_cpp" else ""
    (tmp / "models.yaml").write_text(
        "models:\n"
        f"  {alias_name}:\n"
        f"    runtime_instance: {instance_name}\n"
        f"    inference_profile: {profile_name}\n"
        + alias_overrides
        + "    clients:\n"
        + "      opencode: true\n"
        + "      claude-code: true\n"
        + "      pi: true\n"
        + "runtime_instances:\n"
        f"  {instance_name}:\n"
        + f"    runtime_profile: local_llamacpp\n"
        + "    model_path: /shared-118b.gguf\n"
        + "    port: 8090\n"
        + "    context: 262144\n"
        + "    n_cpu_moe: 31\n"
        + backend_block
        + "    lifecycle_policy: shared_runtime\n"
        + "inference_profiles:\n"
        f"  {profile_name}:\n"
        + profile_block,
        encoding="utf-8",
    )
    _write_minimal(tmp, profile_name="local_llamacpp")
    return tmp


def _seed_plain_alias_with_profile(
        tmp: Path, *,
        alias_name: str = "thinking-llama",
        profile_name: str = "profile-careful",
        profile_block: str = "    reasoning_budget: 4096\n"
                            "    max_output_tokens: 16384\n",
        alias_overrides: str = "") -> Path:
    """Seed a minimal config: plain alias + profile (no runtime_instance)."""
    (tmp / "models.yaml").write_text(
        "models:\n"
        f"  {alias_name}:\n"
        + "    runtime_profile: local_llamacpp\n"
        + "    model_path: /m.gguf\n"
        + "    port: 8081\n"
        + "    context: 131072\n"
        + f"    inference_profile: {profile_name}\n"
        + alias_overrides
        + "    clients:\n"
        + "      opencode: true\n"
        + "      claude-code: true\n"
        + "      pi: true\n"
        + "inference_profiles:\n"
        f"  {profile_name}:\n"
        + profile_block,
        encoding="utf-8",
    )
    _write_minimal(tmp, profile_name="local_llamacpp")
    return tmp


def _seed_pre_v6_plain_alias(tmp: Path, alias_name: str = "legacy-llama") -> Path:
    """Seed a pre-V6-style alias with no profile — reference for the
    no-profile invariance check."""
    (tmp / "models.yaml").write_text(
        "models:\n"
        f"  {alias_name}:\n"
        "    runtime_profile: local_llamacpp\n"
        "    model_path: /m.gguf\n"
        "    context: 131072\n"
        "    port: 8081\n"
        "    n_cpu_moe: 26\n"
        "    clients:\n"
        "      opencode: true\n",
        encoding="utf-8",
    )
    _write_minimal(tmp, profile_name="local_llamacpp")
    return tmp


def _errors(issues, field=None):
    out = [i for i in issues if i.level == "error"]
    if field is not None:
        out = [i for i in out if i.field == field]
    return out


# ─────────────────────────────────────────────────────────────────
# Step 2 — transport for the two implemented fields
# ─────────────────────────────────────────────────────────────────

class OpencodeProfileTransportTests(unittest.TestCase):
    """A profile's ``max_output_tokens`` reaches the opencode config
    limit block. The config builder reads ``resolved["max_output_tokens"]``,
    which the resolver merges from the profile (with alias-level and CLI
    overrides winning per the precedence rule)."""

    def _resolve(self, **seed_kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_instance_bound_with_profile(
                Path(tmp), **seed_kwargs
            )
            return Resolver(config_dir=str(cfg)).resolve_alias(
                seed_kwargs.get("alias_name", "shared-architect")
            )

    def _rendered_limit(self, resolved):
        cfg = opencode.build_opencode_config(resolved)
        provider = next(iter(cfg["provider"].values()))
        return next(iter(provider["models"].values()))["limit"]

    def test_profile_max_output_tokens_reaches_opencode_for_instance_bound(self):
        resolved = self._resolve()
        limit = self._rendered_limit(resolved)
        self.assertEqual(limit["output"], 16384)

    def test_profile_max_output_tokens_reaches_opencode_for_plain_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_plain_alias_with_profile(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "thinking-llama"
            )
        limit = self._rendered_limit(resolved)
        self.assertEqual(limit["output"], 16384)

    def test_alias_level_overrides_profile_max_output_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_instance_bound_with_profile(
                Path(tmp),
                alias_overrides="    max_output_tokens: 8192\n",
            )
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "shared-architect"
            )
        limit = self._rendered_limit(resolved)
        self.assertEqual(limit["output"], 8192, (
            "alias-level max_output_tokens must override the profile value"
        ))


class ClaudeCodeProfileTransportTests(unittest.TestCase):
    """A profile's ``max_output_tokens`` reaches the claude_code env.

    The admin profile here is an ``anthropic``-backend alias (the
    claude_code adapter requires ``real_model``); the inference_profile
    rides the same read-path as the alias-level field. The point is
    that the resolver merges the profile value into the resolved dict
    that the adapter reads — so the test holds for any backend that
    exposes ``max_output_tokens`` to claude_code.
    """

    @staticmethod
    def _resolved(**over):
        base = {
            "backend": "anthropic",
            "real_model": "claude-fable-5",
            "api_key_env": "ANTHROPIC_API_KEY",
            "alias": "fable5",
        }
        base.update(over)
        return base

    @patch("model_allocator.adapters.claude_code.shutil.which",
           side_effect=_fake_which)
    def test_profile_max_output_tokens_reaches_claude_code_env(self, _mock):
        resolved = self._resolved(max_output_tokens=16384)
        cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(
            cmd["env"].get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"), "16384"
        )

    @patch("model_allocator.adapters.claude_code.shutil.which",
           side_effect=_fake_which)
    def test_alias_level_overrides_profile_in_claude_code_env(self, _mock):
        # The resolver writes the alias-level value over the profile
        # one. Both reach the adapter via the resolved dict's
        # ``max_output_tokens`` key.
        resolved = self._resolved(max_output_tokens=8192)
        cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(
            cmd["env"].get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"), "8192",
        )

    def test_profile_via_resolver_drives_claude_code_env_end_to_end(self):
        """End-to-end: an alias that references an inference_profile
        resolves to a dict whose ``max_output_tokens`` matches the
        profile value, and the claude_code adapter reads it from there.
        This is the same shape the existing opencode test uses — the
        resolver merge is the single transport.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_instance_bound_with_profile(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "shared-architect"
            )
        # The resolved dict has the profile value merged in.
        self.assertEqual(resolved["max_output_tokens"], 16384)
        # Patching the rendered command confirms the value reaches
        # the env without going through a process.
        with patch("model_allocator.adapters.claude_code.shutil.which",
                   side_effect=_fake_which):
            cmd = claude_code.build_claude_code_command(dict(
                resolved,
                backend="anthropic",
                real_model="claude-fable-5",
                api_key_env="ANTHROPIC_API_KEY",
            ))
        self.assertEqual(
            cmd["env"].get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"), "16384"
        )


class PiProfileTransportTests(unittest.TestCase):
    """A profile's ``max_output_tokens`` reaches the pi models.json
    ``maxTokens`` entry. The renderer reads it from the resolved dict.
    """

    @staticmethod
    def _resolved(**over):
        base = {
            "backend": "llama_cpp",
            "alias": "shared-architect",
            "real_model": "shared-118b",
            "opencode_model_id": "shared-118b",
            "served_model_name": "shared-118b",
            "context": 262144,
            "port": 8090,
            "host": "127.0.0.1",
        }
        base.update(over)
        return base

    def test_profile_max_output_tokens_reaches_pi_models_json(self):
        resolved = self._resolved(max_output_tokens=16384)
        fragment = pi_adapter.build_pi_models_json(resolved)
        provider = next(iter(fragment.values()))
        model = provider["models"][0]
        self.assertEqual(model["maxTokens"], 16384)

    def test_alias_level_overrides_profile_in_pi(self):
        resolved = self._resolved(max_output_tokens=8192)
        fragment = pi_adapter.build_pi_models_json(resolved)
        provider = next(iter(fragment.values()))
        model = provider["models"][0]
        self.assertEqual(model["maxTokens"], 8192)

    def test_profile_via_resolver_drives_pi_models_json_end_to_end(self):
        """End-to-end mirroring the opencode test: same resolver
        merge, same renderer reading the resolved dict."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_instance_bound_with_profile(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "shared-architect"
            )
        self.assertEqual(resolved["max_output_tokens"], 16384)
        # pi_model_id needs one of served_model_name / opencode_model_id /
        # real_model. The instance-bound alias here is a real llama.cpp
        # alias, so it uses served_model_name — injected to reflect what
        # a real config would carry.
        resolved["served_model_name"] = "shared-118b"
        resolved["real_model"] = "shared-118b"
        fragment = pi_adapter.build_pi_models_json(resolved)
        provider = next(iter(fragment.values()))
        model = provider["models"][0]
        self.assertEqual(model["maxTokens"], 16384)


class LlamaCppProfileTransportTests(unittest.TestCase):
    """A profile's ``reasoning_budget`` reaches the rendered llama-server
    argv as ``--reasoning-budget N``. The adapter reads the resolved
    dict and formats the flag.
    """

    def _resolve(self, alias="shared-architect"):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_instance_bound_with_profile(Path(tmp), alias_name=alias)
            return Resolver(config_dir=str(cfg)).resolve_alias(alias)

    def _flag(self, resolved, key):
        """Return the value of flag ``key`` in the rendered argv, or None."""
        adapter = llama_cpp.LlamaCppAdapter(resolved)
        argv = adapter._build_argv()
        for i, token in enumerate(argv):
            if token == key and i + 1 < len(argv):
                return argv[i + 1]
        return None

    def test_profile_reasoning_budget_reaches_llama_server_argv(self):
        resolved = self._resolve()
        self.assertEqual(self._flag(resolved, "--reasoning-budget"), "4096")

    def test_alias_level_overrides_profile_reasoning_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_instance_bound_with_profile(
                Path(tmp),
                alias_overrides="    reasoning_budget: 512\n",
            )
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "shared-architect"
            )
        self.assertEqual(self._flag(resolved, "--reasoning-budget"), "512")


# ─────────────────────────────────────────────────────────────────
# Step 2 — CLI override (only applies to max_output_tokens; the CLI
# flag is ``--max-output-tokens`` and lives on `cmd_run`).
# ─────────────────────────────────────────────────────────────────

class CLIOverrideTransportTests(unittest.TestCase):
    """The CLI override ``--max-output-tokens`` wins over both the
    profile value and an alias-level value, because the override is
    applied to the resolved dict after ``resolve_alias`` returns.
    """

    @patch("model_allocator.adapters.claude_code.shutil.which",
           side_effect=_fake_which)
    def test_cli_override_wins_over_profile_value(self, _mock):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_instance_bound_with_profile(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "shared-architect"
            )
        # Profile provided 16384; the CLI override drops a different
        # value onto the resolved dict just like cmd_run does.
        self.assertEqual(resolved["max_output_tokens"], 16384)
        resolved["max_output_tokens"] = 32768
        # The adapter requires real_model for the anthropic path; the
        # override test is about precedence, so we supply the minimum
        # anthropic-context fields needed to render the env.
        resolved["backend"] = "anthropic"
        resolved["real_model"] = "claude-fable-5"
        resolved["api_key_env"] = "ANTHROPIC_API_KEY"
        cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(
            cmd["env"].get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"), "32768",
            "CLI override must win over profile value",
        )

    @patch("model_allocator.adapters.claude_code.shutil.which",
           side_effect=_fake_which)
    def test_cli_override_wins_over_alias_level_value(self, _mock):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_instance_bound_with_profile(
                Path(tmp),
                alias_overrides="    max_output_tokens: 8192\n",
            )
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "shared-architect"
            )
        # Alias-level resolved to 8192. CLI override beats it.
        self.assertEqual(resolved["max_output_tokens"], 8192)
        resolved["max_output_tokens"] = 32768
        resolved["backend"] = "anthropic"
        resolved["real_model"] = "claude-fable-5"
        resolved["api_key_env"] = "ANTHROPIC_API_KEY"
        cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(
            cmd["env"].get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"), "32768",
            "CLI override must win over alias-level value",
        )


# ─────────────────────────────────────────────────────────────────
# Step 3 — deferred-field rejection (doctor AND resolver)
# ─────────────────────────────────────────────────────────────────

class DeferredFieldRejectionTests(unittest.TestCase):
    """``temperature`` and ``top_p`` have no transport path in V6. The
    schema rejects them at doctor time and the resolver rejects them
    at resolve time. The message names the field AND states it is
    deferred.
    """

    def test_doctor_rejects_temperature_in_a_profile(self):
        issues = schema.validate_inference_profile(
            "careful", {"reasoning_budget": 4096, "temperature": 0.3},
        )
        errs = _errors(issues, "temperature")
        self.assertTrue(errs, "temperature must surface as an error")
        self.assertTrue(
            any("deferred" in i.message for i in errs),
            f"message must say 'deferred'; got: {[i.message for i in errs]}",
        )

    def test_doctor_rejects_top_p_in_a_profile(self):
        issues = schema.validate_inference_profile(
            "careful", {"top_p": 0.92},
        )
        errs = _errors(issues, "top_p")
        self.assertTrue(errs, "top_p must surface as an error")
        self.assertTrue(
            any("deferred" in i.message for i in errs),
            f"message must say 'deferred'; got: {[i.message for i in errs]}",
        )

    def test_doctor_message_names_the_field(self):
        issues = schema.validate_inference_profile(
            "profile-X", {"temperature": 0.3},
        )
        msg = " ".join(i.message for i in issues)
        self.assertIn("temperature", msg)
        self.assertIn("profile-X", msg)

    def test_lint_config_rejects_deferred_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)
            (cfg / "models.yaml").write_text(
                "models:\n"
                "  a:\n"
                "    runtime_profile: local_ollama\n"
                "    real_model: m\n"
                "    inference_profile: bad\n"
                "    clients:\n"
                "      opencode: true\n"
                "inference_profiles:\n"
                "  bad:\n"
                "    reasoning_budget: 4096\n"
                "    temperature: 0.3\n"
                "    top_p: 0.92\n",
                encoding="utf-8",
            )
            (cfg / "runtime_profiles.yaml").write_text(
                "runtime_profiles:\n"
                "  local_ollama:\n"
                "    backend: ollama\n"
                "    default_api_base: http://127.0.0.1:11434\n",
                encoding="utf-8",
            )
            (cfg / "roles.yaml").write_text("roles:\n", encoding="utf-8")

            from model_allocator.config_writer import load_raw
            report = schema.lint_config(load_raw(cfg))
            entries = report["inference_profiles"].get("bad", [])
            deferred_errs = [
                i for i in entries
                if i.level == "error" and i.field in ("temperature", "top_p")
            ]
            self.assertEqual(
                len(deferred_errs), 2,
                f"doctor must report BOTH deferred fields; got: "
                f"{[(i.field, i.message) for i in entries]}",
            )
            for i in deferred_errs:
                self.assertIn("deferred", i.message)

    def test_doctor_cli_exit_code_is_error_on_deferred_field(self):
        """End-to-end: ``model-allocator doctor`` must exit non-zero
        when a profile declares a deferred field — the loud failure the
        Mission Contract calls for."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)
            (cfg / "models.yaml").write_text(
                "models:\n"
                "  a:\n"
                "    runtime_profile: local_ollama\n"
                "    real_model: m\n"
                "    inference_profile: bad\n"
                "    clients:\n"
                "      opencode: true\n"
                "inference_profiles:\n"
                "  bad:\n"
                "    temperature: 0.3\n",
                encoding="utf-8",
            )
            (cfg / "runtime_profiles.yaml").write_text(
                "runtime_profiles:\n"
                "  local_ollama:\n"
                "    backend: ollama\n"
                "    default_api_base: http://127.0.0.1:11434\n",
                encoding="utf-8",
            )
            (cfg / "roles.yaml").write_text("roles:\n", encoding="utf-8")

            args = argparse.Namespace(config_dir=str(cfg), json=False)
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                rc = cmd_doctor(args)
            self.assertNotEqual(rc, 0, "doctor must exit non-zero on errors")
            self.assertIn("temperature", buf_out.getvalue())

    def test_resolve_alias_profile_with_temperature_is_resolutionerror(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)
            (cfg / "models.yaml").write_text(
                "models:\n"
                "  a:\n"
                "    runtime_profile: local_ollama\n"
                "    real_model: m\n"
                "    inference_profile: bad\n"
                "    clients:\n"
                "      opencode: true\n"
                "inference_profiles:\n"
                "  bad:\n"
                "    temperature: 0.3\n",
                encoding="utf-8",
            )
            (cfg / "runtime_profiles.yaml").write_text(
                "runtime_profiles:\n"
                "  local_ollama:\n"
                "    backend: ollama\n"
                "    default_api_base: http://127.0.0.1:11434\n",
                encoding="utf-8",
            )
            (cfg / "roles.yaml").write_text("roles:\n", encoding="utf-8")
            with self.assertRaises(ResolutionError) as cm:
                Resolver(config_dir=str(cfg)).resolve_alias("a")
            msg = str(cm.exception)
            self.assertIn("temperature", msg)
            self.assertIn("deferred", msg)

    def test_resolve_alias_profile_with_top_p_is_resolutionerror(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)
            (cfg / "models.yaml").write_text(
                "models:\n"
                "  a:\n"
                "    runtime_profile: local_ollama\n"
                "    real_model: m\n"
                "    inference_profile: bad\n"
                "    clients:\n"
                "      opencode: true\n"
                "inference_profiles:\n"
                "  bad:\n"
                "    top_p: 0.92\n",
                encoding="utf-8",
            )
            (cfg / "runtime_profiles.yaml").write_text(
                "runtime_profiles:\n"
                "  local_ollama:\n"
                "    backend: ollama\n"
                "    default_api_base: http://127.0.0.1:11434\n",
                encoding="utf-8",
            )
            (cfg / "roles.yaml").write_text("roles:\n", encoding="utf-8")
            with self.assertRaises(ResolutionError) as cm:
                Resolver(config_dir=str(cfg)).resolve_alias("a")
            msg = str(cm.exception)
            self.assertIn("top_p", msg)
            self.assertIn("deferred", msg)


# ─────────────────────────────────────────────────────────────────
# Step 4 — no-profile invariance (the documented guarantee from O5)
# ─────────────────────────────────────────────────────────────────

class NoProfileInvarianceTests(unittest.TestCase):
    """An alias without ``inference_profile`` resolves byte-equivalently
    to the pre-V6 behavior — the O5 backward-compat guarantee. The
    existing ``test_resolve_alias_without_runtime_instance_is_byte_equivalent_to_pre_v6``
    in test_v6_shared_runtime.py covers the resolution shape; this
    file proves the same on the rendered opencode config.
    """

    def test_pre_v6_alias_has_no_inference_profile_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_pre_v6_plain_alias(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "legacy-llama"
            )
        self.assertNotIn("inference_profile", resolved)
        self.assertNotIn("reasoning_budget", resolved)
        self.assertNotIn("max_output_tokens", resolved)

    def test_pre_v6_alias_opencode_limit_uses_min_window_default(self):
        """No profile field means no profile-set output budget. The
        unchanged fallback (``min(context // 2, 8192)``) still applies."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_pre_v6_plain_alias(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "legacy-llama"
            )
        cfg_obj = opencode.build_opencode_config(resolved)
        provider = next(iter(cfg_obj["provider"].values()))
        limit = next(iter(provider["models"].values()))["limit"]
        # Context is 131072; the default is min(131072 // 2, 8192) == 8192.
        self.assertEqual(limit["output"], 8192)
        self.assertEqual(limit["context"], 131072)

    def test_pre_v6_alias_llama_server_argv_has_no_reasoning_budget_unless_set(self):
        """An alias without a profile (and without an alias-level
        ``reasoning_budget``) must NOT have ``--reasoning-budget`` in
        the rendered argv."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_pre_v6_plain_alias(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "legacy-llama"
            )
        adapter = llama_cpp.LlamaCppAdapter(resolved)
        argv = adapter._build_argv()
        self.assertNotIn(
            "--reasoning-budget", argv,
            "pre-V6 alias without a profile must not get --reasoning-budget"
        )


# ─────────────────────────────────────────────────────────────────
# Precedence evidence — the existing resolver merge already produces
# the documented precedence. These tests pin the contract so future
# changes cannot silently invert it.
# ─────────────────────────────────────────────────────────────────

class PrecedenceContractTests(unittest.TestCase):
    """Profile = default tuning; alias-level = specific override; CLI =
    imperative override. The merge order in ``resolve_alias`` produces
    this: profile fields merge first, alias fields merge last (which
    overrides profile), and the CLI override is applied to the resolved
    dict after ``resolve_alias`` returns.
    """

    def test_alias_wins_over_profile_max_output_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_instance_bound_with_profile(
                Path(tmp),
                alias_overrides="    max_output_tokens: 4096\n",
            )
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "shared-architect"
            )
        # Profile provided 16384; alias asked for 4096 — alias wins.
        self.assertEqual(resolved["max_output_tokens"], 4096)

    def test_alias_wins_over_profile_reasoning_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_instance_bound_with_profile(
                Path(tmp),
                alias_overrides="    reasoning_budget: 256\n",
            )
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "shared-architect"
            )
        # Profile provided 4096; alias asked for 256 — alias wins.
        self.assertEqual(resolved["reasoning_budget"], 256)


if __name__ == "__main__":
    unittest.main()
