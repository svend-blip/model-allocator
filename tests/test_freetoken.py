"""Tests for the FreeToken runtime adapter.

Nothing here needs FreeToken installed, a GPU, CUDA, model weights or a
network: the subprocess boundary and the HTTP boundary are both injected. The
tests that do need the real runtime live behind the `freetoken_hardware`
marker and are not collected by an ordinary run.

The regression that matters most is test_qualified_qwen38_keeps_nvfp4_backend.
On the qualified card, dropping `--nvfp4-backend auto` costs roughly a factor
of fifteen in throughput while everything still appears to work, which is
exactly the kind of defect a configuration cleanup introduces silently.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from model_allocator import schema
from model_allocator.adapters import freetoken as ft
from model_allocator.adapters.freetoken import (
    FreeTokenAdapter, FreeTokenAdapterError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _qualified_alias(name: str) -> dict:
    """The committed alias definition, merged with its runtime profile.

    Read from the repository rather than restated here: a test that carries
    its own copy of the configuration passes happily while the shipped
    configuration is broken.
    """
    models = yaml.safe_load((REPO_ROOT / "models.yaml").read_text())
    profiles = yaml.safe_load((REPO_ROOT / "runtime_profiles.yaml").read_text())
    alias = dict(models["models"][name])
    profile = dict(profiles["runtime_profiles"][alias["runtime_profile"]])
    merged = {**profile, **alias, "alias": name}
    merged["executable"] = "/fake/venv/bin/ft"
    return merged


class _FakeResponse:
    """Minimal stand-in for what urlopen returns."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeRuntime:
    """A FreeToken server that exists only as a routing table.

    Enough of the real one to drive the whole lifecycle — loading into ready,
    model discovery, telemetry, and a failure state — without a process.
    """

    def __init__(self, state="ok", model="Qwen3.8-27B-NVFP4", stats_ok=True,
                 ready_after=None):
        self.state = state
        self.model = model
        self.stats_ok = stats_ok
        # Flip to serving after this many health checks, so a caller can watch
        # a real loading -> ready transition instead of a state that was
        # already settled before it looked.
        self.ready_after = ready_after
        self.health_checks = 0
        self.calls: list[str] = []

    def urlopen(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        self.calls.append(url)
        if url.endswith("/health"):
            self.health_checks += 1
            if (self.ready_after is not None
                    and self.health_checks >= self.ready_after):
                self.state = "ok"
            if self.state == "loading":
                return _FakeResponse({"status": "loading", "phase": "weights",
                                      "model": self.model})
            if self.state == "error":
                return _FakeResponse({"status": "error", "message": "CUDA OOM"})
            return _FakeResponse({"status": "ok", "model": self.model,
                                  "uptime_s": 12})
        if url.endswith("/v1/models"):
            return _FakeResponse({"data": [{"id": self.model}]})
        if url.endswith("/v1/stats"):
            if not self.stats_ok:
                raise OSError("stats endpoint exploded")
            return _FakeResponse({
                "model": {"name": self.model, "max_model_len": 262144,
                          "moe": False, "attn": "hybrid_linear"},
                "memory": {"vram_bytes": 30184308736},
                "kv": {"used": 0, "capacity": 14303},
            })
        if url.endswith("/openapi.json"):
            return _FakeResponse({"paths": {"/v1/chat/completions": {},
                                            "/v1/messages": {}}})
        raise OSError(f"no route for {url}")


class TestCommandConstruction(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()

    def _adapter(self, **overrides):
        resolved = {
            "alias": "ft-test",
            "model_path": "vrfai/Qwen3.8-27B-NVFP4",
            "executable": "/fake/venv/bin/ft",
            "port": 8088,
            "host": "127.0.0.1",
        }
        resolved.update(overrides)
        return FreeTokenAdapter(resolved, state_dir=self.state_dir)

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_base_command(self, *_):
        argv = self._adapter().build_argv()
        self.assertEqual(argv[1], "serve")
        self.assertIn("--model-path", argv)
        self.assertIn("vrfai/Qwen3.8-27B-NVFP4", argv)
        self.assertIn("--port", argv)
        self.assertIn("8088", argv)
        self.assertIn("--host", argv)
        self.assertIn("127.0.0.1", argv)

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_unset_options_emit_no_flags(self, *_):
        """An option nobody configured must not appear at all.

        FreeToken does substantial automatic backend selection, and emitting
        a flag with a guessed value takes that decision away from it.
        """
        argv = self._adapter().build_argv()
        for flag in ("--memory-ratio", "--nvfp4-backend", "--moe-backend",
                     "--kv-reserve-tokens", "--dtype", "--attention-backend",
                     "--moe-cache-auto", "--served-model-name"):
            self.assertNotIn(flag, argv)

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_gpu_name_becomes_index(self, *_):
        """`cuda0` is the allocator's device name; FreeToken wants `0`."""
        for value, expected in (("cuda0", "0"), ("cuda:1", "1"), ("0", "0"),
                                (2, "2")):
            argv = self._adapter(gpu=value).build_argv()
            self.assertEqual(argv[argv.index("--gpu") + 1], expected)

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_unreadable_gpu_is_an_error_not_a_guess(self, *_):
        with self.assertRaises(FreeTokenAdapterError):
            self._adapter(gpu="the big one").build_argv()

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_missing_model_is_refused(self, *_):
        with self.assertRaises(FreeTokenAdapterError):
            self._adapter(model_path="").build_argv()

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_huggingface_repo_id_is_not_filesystem_checked(self, *_):
        """A repo ID is a valid model reference and must not be path-validated."""
        argv = self._adapter(model_path="Qwen/Qwen3.6-35B-A3B").build_argv()
        self.assertIn("Qwen/Qwen3.6-35B-A3B", argv)

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_extra_args_are_appended(self, *_):
        argv = self._adapter(extra_args=["--page-size", "64"]).build_argv()
        self.assertEqual(argv[-2:], ["--page-size", "64"])

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_extra_args_may_not_move_the_runtime(self, *_):
        """The allocator tracks this process by host, port, GPU and model.

        An escape hatch that can rewrite those leaves the ownership records
        describing a server that is somewhere else.
        """
        for hostile in (["--port", "9999"], ["--host=0.0.0.0"], ["--gpu", "1"],
                        ["--model-path", "other/model"], ["--model", "x"]):
            with self.assertRaises(FreeTokenAdapterError):
                self._adapter(extra_args=hostile).build_argv()


class TestQualifiedProfiles(unittest.TestCase):
    """The two profiles that were measured working on the RTX 5090."""

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_qualified_qwen38_keeps_nvfp4_backend(self, *_):
        """MANDATORY REGRESSION.

        Measured on the qualified card: the default Triton NVFP4 path gave
        ~4.3 tokens/sec, `--nvfp4-backend auto` gave ~63. Both configurations
        start, serve and answer correctly, so nothing but this assertion
        stands between a tidy-up and a fifteenfold slowdown.
        """
        argv = FreeTokenAdapter(_qualified_alias("freetoken-qwen38-27b")).build_argv()
        self.assertIn("--nvfp4-backend", argv)
        self.assertEqual(argv[argv.index("--nvfp4-backend") + 1], "auto")

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_qualified_qwen38_reproduces_the_measured_command(self, *_):
        argv = FreeTokenAdapter(_qualified_alias("freetoken-qwen38-27b")).build_argv()
        pairs = {argv[i]: argv[i + 1] for i in range(len(argv) - 1)
                 if argv[i].startswith("--")}
        self.assertEqual(pairs["--model-path"], "vrfai/Qwen3.8-27B-NVFP4")
        self.assertEqual(pairs["--gpu"], "0")
        self.assertEqual(pairs["--port"], "8088")
        self.assertEqual(float(pairs["--memory-ratio"]), 0.90)
        self.assertEqual(pairs["--sampling-defaults"], "model")
        self.assertEqual(pairs["--reasoning-parser"], "auto")
        self.assertEqual(pairs["--tool-call-parser"], "auto")

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_dense_profile_gets_no_moe_flags(self, *_):
        """Qwen3.8 reports moe=false; MoE flags would configure nothing."""
        argv = FreeTokenAdapter(_qualified_alias("freetoken-qwen38-27b")).build_argv()
        for flag in ("--moe-backend", "--moe-cache-auto", "--kv-reserve-tokens",
                     "--moe-cache-size", "--moe-cache-rate"):
            self.assertNotIn(flag, argv)

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_qualified_qwen36_moe_command(self, *_):
        argv = FreeTokenAdapter(
            _qualified_alias("freetoken-qwen36-35b-a3b")).build_argv()
        self.assertEqual(argv[argv.index("--moe-backend") + 1], "auto")
        self.assertIn("--moe-cache-auto", argv)
        self.assertEqual(argv[argv.index("--kv-reserve-tokens") + 1], "16384")
        self.assertNotIn("--nvfp4-backend", argv)

    def test_qualified_aliases_validate(self):
        models = yaml.safe_load((REPO_ROOT / "models.yaml").read_text())
        profiles = yaml.safe_load((REPO_ROOT / "runtime_profiles.yaml").read_text())
        for name in ("freetoken-qwen38-27b", "freetoken-qwen36-35b-a3b"):
            issues = schema.validate_alias(
                name, models["models"][name], profiles["runtime_profiles"], {})
            errors = [i for i in issues if i.level == "error"]
            warnings = [i for i in issues if i.level == "warning"]
            self.assertEqual(errors, [], f"{name}: {[i.message for i in errors]}")
            self.assertEqual(warnings, [], f"{name}: {[i.message for i in warnings]}")

    def test_freetoken_alias_requires_a_model(self):
        profiles = {"p": {"backend": "freetoken"}}
        issues = schema.validate_alias("x", {"runtime_profile": "p"}, profiles, {})
        self.assertTrue(any(i.level == "error" and i.field == "model_path"
                            for i in issues))


class TestExecutableResolution(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()

    def test_explicit_executable_wins(self):
        adapter = FreeTokenAdapter(
            {"alias": "a", "port": 1, "executable": "/x/ft"},
            state_dir=self.state_dir)
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            self.assertEqual(adapter.resolve_executable(), "/x/ft")

    def test_configured_but_absent_executable_is_an_error(self):
        adapter = FreeTokenAdapter(
            {"alias": "a", "port": 1, "executable": "/x/ft"},
            state_dir=self.state_dir)
        with patch("os.path.isfile", return_value=False):
            with self.assertRaises(FreeTokenAdapterError):
                adapter.resolve_executable()

    def test_path_lookup_is_the_fallback(self):
        adapter = FreeTokenAdapter({"alias": "a", "port": 1},
                                   state_dir=self.state_dir)
        with patch.object(ft.shutil, "which", return_value="/usr/bin/ft"):
            self.assertEqual(adapter.resolve_executable(), "/usr/bin/ft")

    def test_absent_everywhere_names_the_venv_problem(self):
        adapter = FreeTokenAdapter({"alias": "a", "port": 1},
                                   state_dir=self.state_dir)
        with patch.object(ft.shutil, "which", return_value=None):
            with self.assertRaises(FreeTokenAdapterError) as caught:
                adapter.resolve_executable()
        self.assertIn("executable", str(caught.exception))

    def test_missing_port_is_refused_at_construction(self):
        with self.assertRaises(FreeTokenAdapterError):
            FreeTokenAdapter({"alias": "a"}, state_dir=self.state_dir)


class TestVersionPreflight(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.adapter = FreeTokenAdapter(
            {"alias": "a", "port": 8088, "executable": "/x/ft"},
            state_dir=self.state_dir)

    def _run(self, stdout, returncode=0):
        class R:
            pass
        result = R()
        result.stdout, result.stderr, result.returncode = stdout, "", returncode
        return result

    def test_parses_the_qualified_version(self):
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True), \
             patch.object(ft.subprocess, "run",
                          return_value=self._run("freetoken version 0.1.2\n")):
            report = self.adapter.version()
        self.assertTrue(report["ok"])
        self.assertEqual(report["version"], "0.1.2")
        self.assertTrue(report["qualified"])
        self.assertIsNone(report["warning"])

    def test_newer_version_warns_but_does_not_fail(self):
        """An unqualified runtime is reported, not refused.

        Whether a new FreeToken becomes the production default is a decision
        that deserves a qualification run — but a startup path that fails
        closed on a patch release makes that decision by accident.
        """
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True), \
             patch.object(ft.subprocess, "run",
                          return_value=self._run("freetoken version 0.2.0\n")):
            report = self.adapter.version()
        self.assertTrue(report["ok"])
        self.assertFalse(report["qualified"])
        self.assertIn("0.2.0", report["warning"])

    def test_unparseable_output_is_a_failure(self):
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True), \
             patch.object(ft.subprocess, "run",
                          return_value=self._run("command not found")):
            report = self.adapter.version()
        self.assertFalse(report["ok"])
        self.assertIsNone(report["version"])


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.adapter = FreeTokenAdapter(
            {"alias": "ft-test", "port": 8088, "host": "127.0.0.1",
             "model_path": "vrfai/Qwen3.8-27B-NVFP4", "executable": "/x/ft"},
            state_dir=self.state_dir)

    def _with_runtime(self, runtime, port_open=True):
        return (
            patch.object(ft.urllib.request, "urlopen", runtime.urlopen),
            patch.object(FreeTokenAdapter, "_port_open", return_value=port_open),
        )

    def test_ready_when_health_says_ok(self):
        runtime = _FakeRuntime(state="ok")
        http, port = self._with_runtime(runtime)
        with http, port, patch.object(FreeTokenAdapter, "_server_pids",
                                      return_value=[4242]):
            status = self.adapter.status()
        self.assertEqual(status["state"], "ready")
        self.assertTrue(status["running"])
        self.assertTrue(status["ready"])

    def test_loading_counts_as_running(self):
        """A loading server must never look startable.

        It is in the middle of claiming ~30 GB of a 32 GB card; a caller that
        auto-starts on "not running" would put a second one beside it.
        """
        runtime = _FakeRuntime(state="loading")
        http, port = self._with_runtime(runtime)
        with http, port, patch.object(FreeTokenAdapter, "_server_pids",
                                      return_value=[4242]):
            status = self.adapter.status()
        self.assertEqual(status["state"], "loading")
        self.assertTrue(status["running"])
        self.assertFalse(status["ready"])

    def test_live_process_without_a_port_is_starting_not_ready(self):
        runtime = _FakeRuntime()
        with patch.object(ft.urllib.request, "urlopen",
                          side_effect=OSError("refused")), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=False), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[]), \
             patch("os.kill", return_value=None):
            Path(self.adapter.pid_file).write_text("4242")
            status = self.adapter.status()
        self.assertEqual(status["state"], "starting")
        self.assertFalse(status["ready"])

    def test_fatal_health_reports_failed(self):
        runtime = _FakeRuntime(state="error")
        http, port = self._with_runtime(runtime)
        with http, port, patch.object(FreeTokenAdapter, "_server_pids",
                                      return_value=[4242]):
            status = self.adapter.status()
        self.assertEqual(status["state"], "failed")
        self.assertFalse(status["running"])

    def test_nothing_running_reports_stopped(self):
        with patch.object(ft.urllib.request, "urlopen",
                          side_effect=OSError("refused")), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=False), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[]):
            status = self.adapter.status()
        self.assertEqual(status["state"], "stopped")

    def test_model_discovery_confirms_the_expected_model(self):
        runtime = _FakeRuntime(model="Qwen3.8-27B-NVFP4")
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen):
            verification = self.adapter.verify_model()
        self.assertTrue(verification["ok"])
        self.assertEqual(verification["served"], "Qwen3.8-27B-NVFP4")

    def test_model_mismatch_is_explicit(self):
        runtime = _FakeRuntime(model="SomethingElse-7B")
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen):
            verification = self.adapter.verify_model()
        self.assertFalse(verification["ok"])
        self.assertEqual(verification["expected"], "Qwen3.8-27B-NVFP4")
        self.assertIn("SomethingElse-7B", verification["error"])

    def test_served_model_name_overrides_the_expectation(self):
        adapter = FreeTokenAdapter(
            {"alias": "a", "port": 8088, "model_path": "vrfai/X",
             "served_model_name": "house-name", "executable": "/x/ft"},
            state_dir=self.state_dir)
        self.assertEqual(adapter.expected_model_name(), "house-name")

    def test_stats_failure_does_not_condemn_the_runtime(self):
        """Telemetry is optional by contract.

        Health and the completion route decide whether inference works; a
        stats endpoint that falls over says nothing about either.
        """
        runtime = _FakeRuntime(state="ok", stats_ok=False)
        http, port = self._with_runtime(runtime)
        with http, port, patch.object(FreeTokenAdapter, "_server_pids",
                                      return_value=[4242]):
            status = self.adapter.status()
            telemetry = self.adapter.stats()
        self.assertFalse(telemetry["ok"])
        self.assertEqual(status["state"], "ready")
        self.assertTrue(status["running"])

    def test_stats_are_normalized_when_available(self):
        runtime = _FakeRuntime()
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen):
            telemetry = self.adapter.stats()
        self.assertTrue(telemetry["ok"])
        self.assertEqual(telemetry["context_max"], 262144)
        self.assertFalse(telemetry["moe"])
        self.assertEqual(telemetry["vram_bytes"], 30184308736)
        self.assertEqual(telemetry["kv_capacity"], 14303)

    def test_endpoint_is_the_harness_seam(self):
        runtime = _FakeRuntime()
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen):
            endpoint = self.adapter.endpoint()
        self.assertEqual(endpoint["provider"], "freetoken")
        self.assertEqual(endpoint["api_base"], "http://127.0.0.1:8088/v1")
        self.assertEqual(endpoint["model"], "Qwen3.8-27B-NVFP4")
        self.assertEqual(endpoint["context_length"], 262144)
        self.assertIn("openai", endpoint["api_compatibility"])

    def test_endpoint_reports_anthropic_only_when_the_route_exists(self):
        runtime = _FakeRuntime()
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen):
            self.assertIn("anthropic", self.adapter.endpoint()["api_compatibility"])

        class NoMessages(_FakeRuntime):
            def urlopen(self, request, timeout=None):
                url = request if isinstance(request, str) else request.full_url
                if url.endswith("/openapi.json"):
                    return _FakeResponse({"paths": {"/v1/chat/completions": {}}})
                return super().urlopen(request, timeout)

        with patch.object(ft.urllib.request, "urlopen", NoMessages().urlopen):
            self.assertNotIn("anthropic",
                             self.adapter.endpoint()["api_compatibility"])


class TestPortOwnership(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.adapter = FreeTokenAdapter(
            {"alias": "ft-test", "port": 8088, "host": "127.0.0.1",
             "model_path": "vrfai/Qwen3.8-27B-NVFP4", "executable": "/x/ft"},
            state_dir=self.state_dir)

    def test_free_port_is_free(self):
        with patch.object(FreeTokenAdapter, "_port_open", return_value=False):
            self.assertFalse(self.adapter.inspect_port()["occupied"])

    def test_matching_freetoken_is_reusable(self):
        """Reuse saves a multi-minute model load between steps."""
        runtime = _FakeRuntime(model="Qwen3.8-27B-NVFP4")
        Path(self.adapter.pid_file).write_text("4242")
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[4242]):
            report = self.adapter.inspect_port()
        self.assertTrue(report["reusable"])
        self.assertEqual(report["kind"], "allocator_owned_freetoken")

    def test_freetoken_serving_another_model_is_not_reusable(self):
        runtime = _FakeRuntime(model="Qwen3.6-35B-A3B")
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[]):
            report = self.adapter.inspect_port()
        self.assertFalse(report["reusable"])
        self.assertEqual(report["kind"], "external_freetoken")

    def test_foreign_service_on_the_port_is_left_alone(self):
        with patch.object(ft.urllib.request, "urlopen",
                          side_effect=OSError("not http")), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True):
            report = self.adapter.inspect_port()
        self.assertEqual(report["kind"], "unknown_process")
        self.assertFalse(report["reusable"])

    def test_start_refuses_an_occupied_port_instead_of_overwriting(self):
        runtime = _FakeRuntime(model="Qwen3.6-35B-A3B")
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[]), \
             patch.object(ft.subprocess, "Popen") as popen:
            result = self.adapter.start(timeout=1)
        self.assertFalse(result["started"])
        self.assertIn("8088", result["error"])
        popen.assert_not_called()

    def test_start_reuses_a_matching_runtime_without_spawning(self):
        runtime = _FakeRuntime(model="Qwen3.8-27B-NVFP4")
        Path(self.adapter.pid_file).write_text("4242")
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[4242]), \
             patch.object(ft.subprocess, "Popen") as popen:
            result = self.adapter.start(timeout=1)
        self.assertTrue(result["started"])
        self.assertTrue(result["reused"])
        popen.assert_not_called()

    def test_stop_is_not_reported_done_while_the_port_answers(self):
        """A stop that deletes its own bookkeeping and claims success is how a
        live server becomes invisible to the allocator while holding the card.
        """
        Path(self.adapter.pid_file).write_text("4242")
        with patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[]), \
             patch.object(FreeTokenAdapter, "_terminate", return_value=True), \
             patch.object(ft.urllib.request, "urlopen",
                          side_effect=OSError("no drain")):
            result = self.adapter.stop(timeout=1)
        self.assertFalse(result["stopped"])
        self.assertTrue(Path(self.adapter.pid_file).exists(),
                        "pid file must survive an unconfirmed stop")

    def test_stop_confirmed_by_the_port_clears_the_records(self):
        Path(self.adapter.pid_file).write_text("4242")
        with patch.object(FreeTokenAdapter, "_port_open", return_value=False), \
             patch.object(FreeTokenAdapter, "_terminate", return_value=True):
            result = self.adapter.stop(timeout=1)
        self.assertTrue(result["stopped"])
        self.assertFalse(Path(self.adapter.pid_file).exists())


class TestFakeRuntimeIntegration(unittest.TestCase):
    """The whole lifecycle against a runtime that is only a routing table."""

    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.adapter = FreeTokenAdapter(
            {"alias": "ft-test", "port": 8088, "host": "127.0.0.1",
             "model_path": "vrfai/Qwen3.8-27B-NVFP4", "executable": "/x/ft",
             "gpu": "cuda0", "memory_ratio": 0.90, "nvfp4_backend": "auto"},
            state_dir=self.state_dir)

    def test_start_waits_for_loading_to_finish(self):
        runtime = _FakeRuntime(state="loading", ready_after=3)

        class Process:
            pid = 4242

            def poll(self):
                return None

        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True), \
             patch.object(ft.subprocess, "run") as version_run, \
             patch.object(ft.subprocess, "Popen", return_value=Process()), \
             patch.object(ft.urllib.request, "urlopen", runtime.urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=False), \
             patch.object(ft.time, "sleep", return_value=None):
            version_run.return_value.stdout = "freetoken version 0.1.2"
            version_run.return_value.stderr = ""
            version_run.return_value.returncode = 0
            result = self.adapter.start(timeout=30)

        self.assertTrue(result["started"], result.get("error"))
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["model"], "Qwen3.8-27B-NVFP4")
        self.assertEqual(
            Path(self.adapter.pid_file).read_text(encoding="utf-8"), "4242")

    def test_process_dying_before_ready_is_reported_not_waited_out(self):
        class DeadProcess:
            pid = 4242
            returncode = 1

            def poll(self):
                return 1

        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True), \
             patch.object(ft.subprocess, "run") as version_run, \
             patch.object(ft.subprocess, "Popen", return_value=DeadProcess()), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=False), \
             patch.object(ft.time, "sleep", return_value=None):
            version_run.return_value.stdout = "freetoken version 0.1.2"
            version_run.return_value.stderr = ""
            version_run.return_value.returncode = 0
            result = self.adapter.start(timeout=30)

        self.assertFalse(result["started"])
        self.assertEqual(result["state"], "failed")
        self.assertIn("exited before becoming ready", result["error"])
        self.assertFalse(Path(self.adapter.pid_file).exists())

    def test_start_stops_early_on_a_fatal_health_status(self):
        runtime = _FakeRuntime(state="error")

        class Process:
            pid = 4242

            def poll(self):
                return None

        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True), \
             patch.object(ft.subprocess, "run") as version_run, \
             patch.object(ft.subprocess, "Popen", return_value=Process()), \
             patch.object(ft.urllib.request, "urlopen", runtime.urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=False), \
             patch.object(ft.time, "sleep", return_value=None):
            version_run.return_value.stdout = "freetoken version 0.1.2"
            version_run.return_value.stderr = ""
            version_run.return_value.returncode = 0
            result = self.adapter.start(timeout=300)

        self.assertFalse(result["started"])
        self.assertEqual(result["state"], "failed")
        self.assertIn("CUDA OOM", result["error"])


class TestBackendRegistration(unittest.TestCase):
    def test_freetoken_is_a_known_backend(self):
        self.assertIn("freetoken", schema.BACKENDS)

    def test_freetoken_is_a_local_server_backend(self):
        """It owns a GPU-resident process, so VRAM reclaim must reach it."""
        from model_allocator import cli
        self.assertIn("freetoken", cli.LOCAL_SERVER_BACKENDS)

    def test_adapter_factory_returns_the_freetoken_adapter(self):
        from model_allocator import cli
        adapter = cli._get_backend_adapter(
            {"backend": "freetoken", "alias": "a", "port": 8088,
             "model_path": "x/y"})
        self.assertIsInstance(adapter, FreeTokenAdapter)


if __name__ == "__main__":
    unittest.main()
