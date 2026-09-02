"""Tests for the `freetoken-qwen38-flash-next` alias and what it required.

Qwen3.8-Flash-Next-NVFP4 was qualified on the RTX 5090 against FreeToken
0.1.2 with the MINIMAL launch — model, host, port, nothing else — and its
own venv. Everything here is deterministic: the HTTP boundary is a routing
table, the subprocess boundary is patched, the Hugging Face cache is a
temporary directory built to shape. No FreeToken, no GPU, no network.

The regressions that matter most:

- test_launch_is_minimal: the alias must not inherit `--nvfp4-backend auto`
  from the 27B alias. It was qualified for that dense checkpoint; on this
  MoE one FreeToken chose its own backends, and that is what was measured.
- TestCheckpointPreflight: "the directory exists" is not proof. The same
  repo was found on the qualified machine in a second cache layout holding
  the index and none of its 206 shards.
"""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

from model_allocator import cli, schema
from model_allocator.adapters import freetoken as ft
from model_allocator.adapters.freetoken import (
    CONTEXT_CAPABILITY_MISMATCH, MODEL_CACHE_INCOMPLETE,
    MODEL_IDENTITY_MISMATCH, RESOURCE_CONFLICT, RUNTIME_NOT_READY,
    RUNTIME_START_FAILED, FreeTokenAdapter, FreeTokenAdapterError,
)
from model_allocator.config_loader import load_config
from model_allocator.resolver import Resolver
from model_allocator.validator import Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
ALIAS = "freetoken-qwen38-flash-next"
PROFILE = "freetoken_qwen38_cuda0"
MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
SERVED = "Qwen3.8-Flash-Next-NVFP4"
ENV_NAME = "FREETOKEN_QWEN38_BIN"
CONTEXT = 262144


def _shipped_alias() -> dict:
    """The committed alias merged with its profile, read from the repository
    so a broken shipped configuration cannot be masked by a test copy."""
    models = yaml.safe_load((REPO_ROOT / "models.yaml").read_text())
    profiles = yaml.safe_load((REPO_ROOT / "runtime_profiles.yaml").read_text())
    alias = dict(models["models"][ALIAS])
    profile = dict(profiles["runtime_profiles"][alias["runtime_profile"]])
    merged = {**profile, **alias, "alias": ALIAS, "backend": profile["backend"]}
    return merged


def _fake_executable(directory: str) -> str:
    path = os.path.join(directory, "ft")
    Path(path).write_text("#!/bin/sh\nexit 0\n")
    os.chmod(path, 0o755)
    return path


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Runtime:
    """A FreeToken 0.1.2 endpoint as a routing table.

    The health document is the one `ft ctl health` reported at
    qualification (status=ok model=… maintenance=serving version=0.1.2), and
    /v1/models carries max_model_len = context_length = 262144 as observed.
    """

    def __init__(self, state="ok", model=SERVED, context=CONTEXT,
                 ready_after=None, stats_ok=True, health_has_model=True):
        self.state = state
        self.model = model
        self.context = context
        self.ready_after = ready_after
        self.stats_ok = stats_ok
        self.health_has_model = health_has_model
        self.health_checks = 0
        self.calls: list[str] = []

    def urlopen(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        self.calls.append(url)
        if url.endswith("/health"):
            self.health_checks += 1
            if self.ready_after is not None and self.health_checks >= self.ready_after:
                self.state = "ok"
            if self.state == "loading":
                return _Response({"status": "loading", "phase": "expert banks",
                                  "model": self.model})
            if self.state == "error":
                return _Response({"status": "error", "message": "CUDA OOM"})
            doc = {"status": "ok", "maintenance": "serving", "version": "0.1.2"}
            if self.health_has_model:
                doc["model"] = self.model
            return _Response(doc)
        if url.endswith("/v1/models"):
            entry = {"id": self.model, "object": "model"}
            if self.context is not None:
                entry["max_model_len"] = self.context
                entry["context_length"] = self.context
            return _Response({"data": [entry]})
        if url.endswith("/v1/stats"):
            if not self.stats_ok:
                raise OSError("stats unavailable")
            # page_size here disagrees with the resolved runtime config on
            # purpose: stats are telemetry and must not be read as config.
            return _Response({"model": {"id": self.model, "ctx": 4096,
                                        "moe": True, "attn": "qsa_sparse"},
                              "kv": {"used_pages": 0, "total_pages": 1000,
                                     "page_size": 64},
                              "throughput": {"decode_tps": 0.0},
                              "requests": {"completed": 0}})
        if url.endswith("/openapi.json"):
            return _Response({"paths": {"/v1/chat/completions": {}}})
        raise OSError(f"no route for {url}")


class _Process:
    def __init__(self, pid=4242, exit_code=None):
        self.pid = pid
        self.returncode = exit_code

    def poll(self):
        return self.returncode


class _Clock:
    """A time.time() that advances a fixed step per call, so a wait loop
    reaches its timeout without waiting."""

    def __init__(self, step=1.0):
        self.now = 1000.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


# ------------------------------------------------------------ configuration


class TestAliasConfiguration(unittest.TestCase):
    def test_alias_parses_with_the_qualified_facts(self):
        alias = yaml.safe_load((REPO_ROOT / "models.yaml").read_text())["models"][ALIAS]
        self.assertEqual(alias["runtime_profile"], PROFILE)
        self.assertEqual(alias["model_path"], MODEL)
        self.assertEqual(alias["real_model"], SERVED)
        self.assertEqual(alias["context"], CONTEXT)
        self.assertEqual(alias["port"], 8090)
        self.assertEqual(alias["qualification"]["runtime_version"], "0.1.2")
        self.assertEqual(alias["qualification"]["gpu_class"], "RTX_5090")

    def test_alias_pins_no_backend_flags(self):
        """The auto-selected backends are recorded as facts, not as flags.

        The KV budget is different: it is not a backend choice but the one
        knob without which the minimal launch is unusable for its purpose.
        Measured 2026-09-02: minimal launch = 8256 tokens for prompt plus
        generation; with moe_cache_auto + kv_reserve_tokens 131072 the floor
        costs 3.10 GiB, generation stays ~46 tok/s and a 112k-token prompt
        prefills in 25 s. So the two budget keys are allowed and pinned.
        """
        alias = yaml.safe_load((REPO_ROOT / "models.yaml").read_text())["models"][ALIAS]
        for key in ("nvfp4_backend", "moe_backend", "attention_backend",
                    "moe_cache_size", "moe_cache_rate", "cache_type",
                    "num_tokens", "num_pages", "memory_ratio"):
            self.assertNotIn(key, alias, f"{key} must not be pinned")
        self.assertTrue(alias.get("moe_cache_auto") is True)
        self.assertEqual(alias.get("kv_reserve_tokens"), 131072)
        self.assertEqual(alias["qualification"]["auto_selected"]["attention_backend"],
                         "qsa_sparse")
        self.assertEqual(alias["qualification"]["auto_selected"]["nvfp4_backend"],
                         "triton")

    def test_reasoning_effort_is_a_per_request_field(self):
        """reasoning_effort is not launch configuration (never an argv flag);
        it is the per-request level the harness sends. Role use pins low."""
        alias = yaml.safe_load((REPO_ROOT / "models.yaml").read_text())["models"][ALIAS]
        self.assertEqual(alias.get("reasoning_effort"), "low")
        self.assertNotIn("reasoning_effort", dict(ft._VALUE_FLAGS))
        self.assertNotIn("reasoning_effort", dict(ft._BOOL_FLAGS))
        self.assertEqual(alias["qualification"]["reasoning_efforts"],
                         ["low", "medium", "xhigh"])

    def test_profile_uses_its_own_env_placeholder(self):
        raw = (REPO_ROOT / "runtime_profiles.yaml").read_text()
        profiles = yaml.safe_load(raw)["runtime_profiles"]
        profile = profiles[PROFILE]
        self.assertEqual(profile["backend"], "freetoken")
        self.assertEqual(profile["executable"], "${" + ENV_NAME + "}")
        self.assertEqual(profile["executable_env"], ENV_NAME)
        self.assertNotEqual(profile["executable"],
                            profiles["freetoken_cuda0"]["executable"],
                            "the qualified venv is distinct from FREETOKEN_BIN")
        self.assertNotIn("/home/", raw)

    def test_alias_and_profile_validate_without_warnings(self):
        models = yaml.safe_load((REPO_ROOT / "models.yaml").read_text())
        profiles = yaml.safe_load((REPO_ROOT / "runtime_profiles.yaml").read_text())
        issues = schema.validate_alias(ALIAS, models["models"][ALIAS],
                                       profiles["runtime_profiles"], {})
        self.assertEqual([i.message for i in issues], [])
        issues = schema.validate_profile(PROFILE, profiles["runtime_profiles"][PROFILE])
        self.assertEqual([i.message for i in issues], [])

    def test_alias_is_assigned_only_to_the_implementer(self):
        """Human decision 2026-09-02: 9000-implementer runs on Flash-Next."""
        roles = yaml.safe_load((REPO_ROOT / "roles.yaml").read_text())["roles"]
        for name, role in roles.items():
            uses = (role.get("default_alias") == ALIAS
                    or ALIAS in (role.get("client_aliases") or {}).values())
            self.assertEqual(uses, name == "9000-implementer", name)

    def test_resolver_resolves_the_alias(self):
        with patch.dict(os.environ, {ENV_NAME: "/qualified/venv/bin/ft"}):
            resolved = Resolver().resolve_alias(ALIAS)
        self.assertEqual(resolved["backend"], "freetoken")
        self.assertEqual(resolved["model_path"], MODEL)
        self.assertEqual(resolved["port"], 8090)
        self.assertEqual(resolved["default_host"], "127.0.0.1")
        self.assertEqual(resolved["gpu"], "cuda0")
        self.assertEqual(resolved["context"], CONTEXT)
        self.assertEqual(resolved["executable"], "/qualified/venv/bin/ft")
        self.assertEqual(resolved["qualified_runtime_version"], "0.1.2")

    def test_provider_selection_yields_the_freetoken_adapter(self):
        with patch.dict(os.environ, {ENV_NAME: "/qualified/venv/bin/ft"}):
            resolved = Resolver().resolve_alias(ALIAS)
        self.assertIsInstance(cli._get_backend_adapter(resolved), FreeTokenAdapter)

    def test_cli_list_shows_the_alias(self):
        """`list` validates every alias, and validating the ollama and cloud
        aliases probes their endpoints — so the shipped FreeToken aliases
        and profiles are copied into a config dir of their own. The alias
        text is still the committed one."""
        models = yaml.safe_load((REPO_ROOT / "models.yaml").read_text())["models"]
        profiles = yaml.safe_load(
            (REPO_ROOT / "runtime_profiles.yaml").read_text())["runtime_profiles"]
        config_dir = tempfile.mkdtemp()
        Path(config_dir, "models.yaml").write_text(yaml.safe_dump({"models": {
            name: alias for name, alias in models.items()
            if profiles.get(alias.get("runtime_profile"), {}).get("backend")
            == "freetoken"}}))
        Path(config_dir, "runtime_profiles.yaml").write_text(yaml.safe_dump(
            {"runtime_profiles": {name: p for name, p in profiles.items()
                                  if p["backend"] == "freetoken"}}))
        with patch.dict(os.environ, {ENV_NAME: "/qualified/venv/bin/ft"}), \
             patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            out = io.StringIO()
            with redirect_stdout(out):
                rc = cli.main(["--config-dir", config_dir, "list"])
        self.assertEqual(rc, 0)
        names = [entry["alias"] for entry in json.loads(out.getvalue())]
        self.assertIn(ALIAS, names)
        entry = next(e for e in json.loads(out.getvalue()) if e["alias"] == ALIAS)
        self.assertEqual(entry["backend"], "freetoken")
        self.assertEqual(entry["real_model"], SERVED)


# ------------------------------------------------------------------ launch


class TestLaunch(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.bin_dir = tempfile.mkdtemp()
        self.executable = _fake_executable(self.bin_dir)

    def _adapter(self, **overrides):
        resolved = _shipped_alias()
        resolved["executable"] = self.executable
        resolved.update(overrides)
        return FreeTokenAdapter(resolved, state_dir=self.state_dir)

    def test_launch_is_minimal(self):
        """MANDATORY REGRESSION: exactly the qualified command plus the KV budget.

        The two budget flags were added after measurement (see
        test_alias_pins_no_backend_flags); everything else stays minimal.

        `--nvfp4-backend auto` was measured load-bearing on the dense 27B
        alias and is NOT carried over: on this MoE checkpoint FreeToken
        chose qsa_sparse / offload / hybrid_radix / triton itself, and that
        is the configuration that was qualified.
        """
        argv = self._adapter().build_argv()
        self.assertEqual(argv, [
            self.executable, "serve",
            "--model-path", MODEL,
            "--host", "127.0.0.1",
            "--port", "8090",
            "--gpu", "0",
            "--kv-reserve-tokens", "131072",
            "--moe-cache-auto",
        ])

    def test_no_forced_backend_flags(self):
        argv = self._adapter().build_argv()
        for flag in ("--attention-backend", "--moe-backend", "--nvfp4-backend",
                     "--moe-cache-size", "--expert-load", "--memory-ratio",
                     "--num-tokens", "--reasoning-effort"):
            self.assertNotIn(flag, argv)

    def test_model_id_propagates(self):
        argv = self._adapter().build_argv()
        self.assertEqual(argv[argv.index("--model-path") + 1], MODEL)

    def test_served_model_identity(self):
        self.assertEqual(self._adapter().expected_model_name(), SERVED)

    def test_host_is_configurable(self):
        argv = self._adapter(host="0.0.0.0").build_argv()
        self.assertEqual(argv[argv.index("--host") + 1], "0.0.0.0")

    def test_port_is_configurable(self):
        adapter = self._adapter(port=8095)
        argv = adapter.build_argv()
        self.assertEqual(argv[argv.index("--port") + 1], "8095")
        self.assertEqual(adapter.port, 8095)

    def test_gpu_is_configurable(self):
        argv = self._adapter(gpu="cuda:1").build_argv()
        self.assertEqual(argv[argv.index("--gpu") + 1], "1")

    def test_context_capability_is_exposed(self):
        adapter = self._adapter()
        self.assertEqual(adapter.resolved["context"], CONTEXT)
        caps = adapter.capabilities()["capabilities"]
        self.assertEqual(caps["context_length"], CONTEXT)


class TestExecutable(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.bin_dir = tempfile.mkdtemp()
        self.executable = _fake_executable(self.bin_dir)

    def _resolved_profile(self):
        return load_config(REPO_ROOT)["runtime_profiles"][PROFILE]

    def test_env_placeholder_expands_to_the_qualified_venv(self):
        with patch.dict(os.environ, {ENV_NAME: self.executable}):
            profile = self._resolved_profile()
            adapter = FreeTokenAdapter({**profile, "alias": ALIAS, "port": 8090},
                                       state_dir=self.state_dir)
            self.assertEqual(profile["executable"], self.executable)
            self.assertEqual(adapter.resolve_executable(), self.executable)

    def test_missing_env_is_a_typed_error_naming_the_variable(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_NAME, None)
            profile = self._resolved_profile()
            self.assertEqual(profile["executable"], "")
            adapter = FreeTokenAdapter({**profile, "alias": ALIAS, "port": 8090},
                                       state_dir=self.state_dir)
            with self.assertRaises(FreeTokenAdapterError) as caught:
                adapter.resolve_executable()
        self.assertIn(ENV_NAME, str(caught.exception))

    def test_missing_env_does_not_fall_back_to_path(self):
        """A different `ft` on PATH is a different, unqualified runtime."""
        with patch.dict(os.environ, {}, clear=False), \
             patch.object(ft.shutil, "which", return_value="/usr/bin/ft"):
            os.environ.pop(ENV_NAME, None)
            adapter = FreeTokenAdapter(
                {"alias": ALIAS, "port": 8090, "executable": "",
                 "executable_env": ENV_NAME}, state_dir=self.state_dir)
            with self.assertRaises(FreeTokenAdapterError):
                adapter.resolve_executable()

    def test_env_pointing_at_nothing_is_a_typed_error(self):
        with patch.dict(os.environ, {ENV_NAME: "/nowhere/ft"}):
            adapter = FreeTokenAdapter(
                {"alias": ALIAS, "port": 8090, "executable": "",
                 "executable_env": ENV_NAME}, state_dir=self.state_dir)
            with self.assertRaises(FreeTokenAdapterError) as caught:
                adapter.resolve_executable()
        self.assertIn("/nowhere/ft", str(caught.exception))

    def test_validator_reports_the_unset_variable_as_an_error(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_NAME, None)
            result = Validator(resolver=Resolver()).validate(ALIAS, "opencode")
        self.assertEqual(result["validation_status"], "ERROR")
        self.assertTrue(any(ENV_NAME in e for e in result["errors"]), result["errors"])

    def test_other_providers_are_unaffected_by_the_missing_variable(self):
        """FreeToken stays optional: nothing else notices its env is unset."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_NAME, None)
            resolver = Resolver()
            others = 0
            for name in resolver.list_aliases():
                resolved = resolver.resolve_alias(name)
                if resolved["backend"] == "freetoken":
                    continue
                others += 1
                # Resolution and adapter construction must not raise. The
                # validator is not run here: for ollama and cloud aliases
                # it probes their endpoints, which is network.
                if resolved["backend"] in ("external",):
                    continue
                cli._get_backend_adapter(resolved)
            self.assertGreater(others, 0)


# --------------------------------------------------------------- readiness


class TestReadiness(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.bin_dir = tempfile.mkdtemp()
        self.executable = _fake_executable(self.bin_dir)

    def _adapter(self, **overrides):
        resolved = _shipped_alias()
        resolved["executable"] = self.executable
        resolved.update(overrides)
        return FreeTokenAdapter(resolved, state_dir=self.state_dir)

    def test_models_route_carries_the_context(self):
        with patch.object(ft.urllib.request, "urlopen", _Runtime().urlopen):
            discovered = self._adapter().models()
        self.assertEqual(discovered["models"], [SERVED])
        self.assertEqual(discovered["context"][SERVED], CONTEXT)

    def test_readiness_chain_succeeds_on_the_qualified_endpoint(self):
        with patch.object(ft.urllib.request, "urlopen", _Runtime().urlopen):
            readiness = self._adapter().readiness(alive=True)
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["stage"], "ready")
        self.assertEqual(readiness["served"], SERVED)
        self.assertEqual(readiness["context"], CONTEXT)
        self.assertIsNone(readiness["code"])

    def test_a_dead_process_is_not_ready(self):
        with patch.object(ft.urllib.request, "urlopen", _Runtime().urlopen):
            readiness = self._adapter().readiness(alive=False)
        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["stage"], "process")
        self.assertEqual(readiness["code"], RUNTIME_NOT_READY)

    def test_health_loading_is_not_ready(self):
        with patch.object(ft.urllib.request, "urlopen",
                          _Runtime(state="loading").urlopen):
            readiness = self._adapter().readiness(alive=True)
        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["stage"], "health")
        self.assertEqual(readiness["code"], RUNTIME_NOT_READY)

    def test_wrong_served_model_is_an_identity_mismatch(self):
        with patch.object(ft.urllib.request, "urlopen",
                          _Runtime(model="Qwen3.8-27B-NVFP4").urlopen):
            readiness = self._adapter().readiness(alive=True)
        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["stage"], "identity")
        self.assertEqual(readiness["code"], MODEL_IDENTITY_MISMATCH)
        self.assertIn("Qwen3.8-27B-NVFP4", readiness["error"])

    def test_smaller_live_context_is_a_capability_mismatch(self):
        with patch.object(ft.urllib.request, "urlopen",
                          _Runtime(context=131072).urlopen):
            readiness = self._adapter().readiness(alive=True)
        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["stage"], "context")
        self.assertEqual(readiness["code"], CONTEXT_CAPABILITY_MISMATCH)

    def test_larger_live_context_is_acceptable(self):
        with patch.object(ft.urllib.request, "urlopen",
                          _Runtime(context=CONTEXT * 2).urlopen):
            self.assertTrue(self._adapter().readiness(alive=True)["ready"])

    def test_unadvertised_context_is_unverified_not_refused(self):
        """The 27B and 35B aliases were qualified before /v1/models carried
        a figure; a missing one is reported, not treated as a defect."""
        with patch.object(ft.urllib.request, "urlopen",
                          _Runtime(context=None).urlopen):
            adapter = self._adapter()
            context = adapter.verify_context()
            readiness = adapter.readiness(alive=True)
        self.assertTrue(context["ok"])
        self.assertFalse(context["checked"])
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["context"], CONTEXT)

    def test_status_is_running_but_not_ready_on_a_mismatch(self):
        """Not READY merely because the process exists — but still running,
        so no caller auto-starts a second runtime onto the card."""
        with patch.object(ft.urllib.request, "urlopen",
                          _Runtime(model="other").urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[]):
            status = self._adapter().status()
        self.assertTrue(status["running"])
        self.assertFalse(status["ready"])
        self.assertEqual(status["state"], "model_mismatch")
        self.assertEqual(status["code"], MODEL_IDENTITY_MISMATCH)

    def test_readiness_does_not_depend_on_stats_or_throughput(self):
        with patch.object(ft.urllib.request, "urlopen",
                          _Runtime(stats_ok=False).urlopen):
            adapter = self._adapter()
            readiness = adapter.readiness(alive=True)
            endpoint = adapter.endpoint()
        self.assertTrue(readiness["ready"])
        self.assertEqual(endpoint["context_length"], CONTEXT)

    def test_endpoint_context_comes_from_models_not_stats(self):
        """/v1/stats says ctx 4096 and page_size 64 in the fake; both are
        telemetry, and the fake disagrees on purpose."""
        with patch.object(ft.urllib.request, "urlopen", _Runtime().urlopen):
            endpoint = self._adapter().endpoint()
        self.assertEqual(endpoint["context_length"], CONTEXT)
        self.assertEqual(endpoint["model"], SERVED)
        self.assertEqual(endpoint["api_base"], "http://127.0.0.1:8090/v1")


# ------------------------------------------------------------------- start


class TestStartLifecycle(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.bin_dir = tempfile.mkdtemp()
        self.executable = _fake_executable(self.bin_dir)
        self.cache = tempfile.mkdtemp()
        self.env = patch.dict(os.environ, {"HF_HUB_CACHE": self.cache})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _adapter(self, **overrides):
        resolved = _shipped_alias()
        resolved["executable"] = self.executable
        resolved.update(overrides)
        return FreeTokenAdapter(resolved, state_dir=self.state_dir)

    def _start(self, adapter, runtime, process, timeout=30, clock=None,
               port_open=False, log_writer=None):
        """Drive start() against a fake runtime; returns (result, popen)."""
        def popen(argv, **kwargs):
            if log_writer:
                log_writer(kwargs)
            return process

        patches = [
            patch.object(ft.subprocess, "run"),
            patch.object(ft.subprocess, "Popen", side_effect=popen),
            patch.object(ft.urllib.request, "urlopen", runtime.urlopen),
            patch.object(FreeTokenAdapter, "_port_open", return_value=port_open),
            patch.object(FreeTokenAdapter, "_nvidia_smi",
                         lambda self, args, timeout=10.0: None),
            patch.object(FreeTokenAdapter, "_shield_from_oomd",
                         return_value={"applied": False, "reason": "test"}),
            patch.object(ft.time, "sleep", return_value=None),
        ]
        if clock is not None:
            patches.append(patch.object(ft.time, "time", clock))
        started = [p.start() for p in patches]
        try:
            version_run = started[0]
            version_run.return_value.stdout = "freetoken version 0.1.2"
            version_run.return_value.stderr = ""
            version_run.return_value.returncode = 0
            result = adapter.start(timeout=timeout)
            return result, started[1]
        finally:
            for p in patches:
                p.stop()

    def test_start_reaches_ready_and_records_ownership(self):
        adapter = self._adapter()
        result, popen = self._start(adapter, _Runtime(state="loading", ready_after=3),
                                    _Process())
        self.assertTrue(result["started"], result.get("error"))
        self.assertEqual(result["state"], "ready")
        self.assertIsNone(result["code"])
        self.assertEqual(result["model"], SERVED)
        self.assertEqual(result["context"], CONTEXT)
        self.assertEqual(Path(adapter.pid_file).read_text(), "4242")
        recorded = json.loads(Path(adapter.fingerprint_file).read_text())
        self.assertEqual(recorded["digest"], result["fingerprint"])
        self.assertEqual(recorded["served_model"], SERVED)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[:2], [self.executable, "serve"])
        self.assertNotIn("--nvfp4-backend", argv)

    def test_start_times_out_as_not_ready(self):
        adapter = self._adapter()
        result, _ = self._start(adapter, _Runtime(state="loading"), _Process(),
                                timeout=10, clock=_Clock(step=2.0))
        self.assertFalse(result["started"])
        self.assertEqual(result["code"], RUNTIME_NOT_READY)
        self.assertEqual(result["state"], "loading")
        self.assertEqual(result["diagnostics"]["alias"], ALIAS)
        self.assertEqual(result["diagnostics"]["port"], 8090)

    def test_fatal_health_is_a_start_failure(self):
        result, _ = self._start(self._adapter(), _Runtime(state="error"), _Process())
        self.assertFalse(result["started"])
        self.assertEqual(result["code"], RUNTIME_START_FAILED)
        self.assertIn("CUDA OOM", result["error"])

    def test_wrong_served_model_after_start_is_an_identity_mismatch(self):
        result, _ = self._start(self._adapter(),
                                _Runtime(model="Qwen3.8-27B-NVFP4"), _Process())
        self.assertFalse(result["started"])
        self.assertEqual(result["state"], "model_mismatch")
        self.assertEqual(result["code"], MODEL_IDENTITY_MISMATCH)
        self.assertEqual(result["diagnostics"]["served_model_expected"], SERVED)

    def test_exit_before_ready_carries_the_log_not_an_opaque_crash(self):
        adapter = self._adapter()

        def write_log(kwargs):
            kwargs["stdout"].write(b"loading weights\nTraceback\n"
                                   b"FileNotFoundError: layer-00003.safetensors: "
                                   b"No such file or directory\n")

        result, _ = self._start(adapter, _Runtime(), _Process(exit_code=1),
                                log_writer=write_log)
        self.assertFalse(result["started"])
        self.assertEqual(result["code"], MODEL_CACHE_INCOMPLETE)
        self.assertIn("layer-00003.safetensors", result["log_tail"])
        self.assertFalse(Path(adapter.pid_file).exists())
        self.assertFalse(Path(adapter.fingerprint_file).exists())

    def test_plain_exit_is_a_start_failure_with_diagnostics(self):
        result, _ = self._start(self._adapter(), _Runtime(), _Process(exit_code=2))
        self.assertEqual(result["code"], RUNTIME_START_FAILED)
        self.assertEqual(result["diagnostics"]["runtime_state"], "exited")
        self.assertEqual(result["diagnostics"]["executable"], self.executable)
        self.assertEqual(result["diagnostics"]["version"], "0.1.2")

    def test_slow_expert_bank_build_is_surfaced_not_failed(self):
        """Both lines appeared on the qualified, working run."""
        adapter = self._adapter()

        def write_log(kwargs):
            kwargs["stdout"].write(
                b"expert banks: low free RAM -> serial build\n"
                b"expert banks: slow path (serial build)\n")

        result, _ = self._start(adapter, _Runtime(state="loading", ready_after=4),
                                _Process(), log_writer=write_log)
        self.assertTrue(result["started"], result.get("error"))
        self.assertTrue(result["initialisation"]["slow_path"])
        self.assertEqual(len(result["initialisation"]["notes"]), 2)

    def test_gpu_refusal_is_a_resource_conflict(self):
        adapter = self._adapter()

        def smi(_self, args, timeout=10.0):
            joined = " ".join(args)
            if "memory.free" in joined:
                return "2700"
            return "4242, 28365, ft"

        with patch.object(FreeTokenAdapter, "_nvidia_smi", smi), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[]), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=False), \
             patch.object(ft.subprocess, "run") as version_run, \
             patch.object(ft.subprocess, "Popen") as popen:
            version_run.return_value.stdout = "freetoken version 0.1.2"
            version_run.return_value.stderr = ""
            version_run.return_value.returncode = 0
            result = adapter.start(timeout=1)
        self.assertFalse(result["started"])
        self.assertEqual(result["code"], RESOURCE_CONFLICT)
        self.assertEqual(result["state"], "gpu_unavailable")
        popen.assert_not_called()


class TestOwnership(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.bin_dir = tempfile.mkdtemp()
        self.executable = _fake_executable(self.bin_dir)
        self.cache = tempfile.mkdtemp()
        self.env = patch.dict(os.environ, {"HF_HUB_CACHE": self.cache})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _adapter(self, **overrides):
        resolved = _shipped_alias()
        resolved["executable"] = self.executable
        resolved.update(overrides)
        return FreeTokenAdapter(resolved, state_dir=self.state_dir)

    def _own(self, adapter, fingerprint=None):
        Path(adapter.pid_file).write_text("4242")
        adapter._record_fingerprint(fingerprint or adapter.runtime_fingerprint())

    def test_owned_matching_runtime_is_reused(self):
        adapter = self._adapter()
        self._own(adapter)
        with patch.object(ft.urllib.request, "urlopen", _Runtime().urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[4242]), \
             patch.object(FreeTokenAdapter, "_shield_from_oomd",
                          return_value={"applied": False, "reason": "test"}), \
             patch.object(ft.subprocess, "Popen") as popen:
            report = adapter.inspect_port()
            result = adapter.start(timeout=1)
        self.assertEqual(report["kind"], "allocator_owned_freetoken")
        self.assertTrue(report["reusable"])
        self.assertTrue(result["reused"])
        popen.assert_not_called()

    def test_owned_runtime_from_another_configuration_is_not_reused(self):
        """Same served name on the same port, but started with different
        launch arguments: a pid file proves we started something, not that
        it is what this alias would start."""
        adapter = self._adapter()
        other = self._adapter(gpu="cuda:1").runtime_fingerprint()
        self._own(adapter, fingerprint=other)
        with patch.object(ft.urllib.request, "urlopen", _Runtime().urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[4242]), \
             patch.object(ft.subprocess, "Popen") as popen, \
             patch("os.kill") as kill:
            report = adapter.inspect_port()
            result = adapter.start(timeout=1)
        self.assertTrue(report["fingerprint_mismatch"])
        self.assertFalse(report["reusable"])
        self.assertFalse(result["started"])
        self.assertEqual(result["code"], RESOURCE_CONFLICT)
        self.assertIn("fingerprint", result["error"])
        popen.assert_not_called()
        kill.assert_not_called()

    def test_records_without_a_fingerprint_fall_back_to_identity(self):
        adapter = self._adapter()
        Path(adapter.pid_file).write_text("4242")
        with patch.object(ft.urllib.request, "urlopen", _Runtime().urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[4242]):
            report = adapter.inspect_port()
        self.assertTrue(report["reusable"])
        self.assertFalse(report["fingerprint_mismatch"])

    def test_external_freetoken_serving_another_model_is_refused_not_killed(self):
        adapter = self._adapter()
        with patch.object(ft.urllib.request, "urlopen",
                          _Runtime(model="Qwen3.6-35B-A3B").urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[]), \
             patch.object(ft.subprocess, "Popen") as popen, \
             patch("os.kill") as kill:
            result = adapter.start(timeout=1)
        self.assertFalse(result["started"])
        self.assertEqual(result["code"], RESOURCE_CONFLICT)
        self.assertEqual(result["occupant"]["kind"], "external_freetoken")
        popen.assert_not_called()
        kill.assert_not_called()

    def test_llama_server_on_the_shared_port_is_incompatible_and_untouched(self):
        """Port 8090 is also laguna-shared-118b's. llama-server answers
        /health with {"status": "ok"} and no model — that is not ours."""
        adapter = self._adapter()
        with patch.object(ft.urllib.request, "urlopen",
                          _Runtime(health_has_model=False).urlopen), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(ft.subprocess, "Popen") as popen, \
             patch("os.kill") as kill:
            report = adapter.inspect_port()
            result = adapter.start(timeout=1)
        self.assertEqual(report["kind"], "incompatible_service")
        self.assertFalse(report["reusable"])
        self.assertEqual(result["code"], RESOURCE_CONFLICT)
        popen.assert_not_called()
        kill.assert_not_called()

    def test_stop_acts_only_on_the_recorded_pid_and_clears_records(self):
        adapter = self._adapter()
        self._own(adapter)
        terminated = []

        def terminate(_self, pid, timeout):
            terminated.append(pid)
            return True

        with patch.object(FreeTokenAdapter, "_port_open", return_value=False), \
             patch.object(FreeTokenAdapter, "_terminate", terminate):
            result = adapter.stop(timeout=1)
        self.assertTrue(result["stopped"])
        self.assertEqual(terminated, [4242])
        self.assertFalse(Path(adapter.pid_file).exists())
        self.assertFalse(Path(adapter.fingerprint_file).exists())

    def test_unconfirmed_stop_keeps_the_records(self):
        adapter = self._adapter()
        self._own(adapter)
        with patch.object(FreeTokenAdapter, "_port_open", return_value=True), \
             patch.object(FreeTokenAdapter, "_server_pids", return_value=[]), \
             patch.object(FreeTokenAdapter, "_terminate", return_value=True), \
             patch.object(ft.urllib.request, "urlopen", side_effect=OSError("x")), \
             patch.object(ft.time, "sleep", return_value=None), \
             patch.object(ft.time, "time", _Clock()):
            result = adapter.stop(timeout=1)
        self.assertFalse(result["stopped"])
        self.assertTrue(Path(adapter.fingerprint_file).exists())


class TestFingerprint(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.bin_dir = tempfile.mkdtemp()
        self.executable = _fake_executable(self.bin_dir)
        self.cache = tempfile.mkdtemp()
        self.env = patch.dict(os.environ, {"HF_HUB_CACHE": self.cache})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _adapter(self, **overrides):
        resolved = _shipped_alias()
        resolved["executable"] = self.executable
        resolved.update(overrides)
        return FreeTokenAdapter(resolved, state_dir=self.state_dir)

    def test_is_deterministic(self):
        self.assertEqual(self._adapter().runtime_fingerprint()["digest"],
                         self._adapter().runtime_fingerprint()["digest"])

    def test_covers_runtime_identity(self):
        fp = self._adapter().runtime_fingerprint(version="0.1.2")
        self.assertEqual(fp["provider"], "freetoken")
        self.assertEqual(fp["model"], MODEL)
        self.assertEqual(fp["served_model"], SERVED)
        self.assertEqual(fp["gpu"], 0)
        self.assertEqual(fp["runtime_profile"], PROFILE)
        self.assertEqual(fp["port"], 8090)
        self.assertEqual(fp["executable"], self.executable)
        self.assertEqual(fp["version"], "0.1.2")
        self.assertEqual(fp["launch_args"],
                         ["--model-path", MODEL, "--host", "127.0.0.1",
                          "--port", "8090", "--gpu", "0",
                          "--kv-reserve-tokens", "131072", "--moe-cache-auto"])

    def test_changes_with_material_launch_properties(self):
        base = self._adapter().runtime_fingerprint()["digest"]
        self.assertNotEqual(base, self._adapter(port=8091).runtime_fingerprint()["digest"])
        self.assertNotEqual(base, self._adapter(gpu="cuda1").runtime_fingerprint()["digest"])
        self.assertNotEqual(base, self._adapter(
            model_path="vrfai/Qwen3.8-27B-NVFP4").runtime_fingerprint()["digest"])

    def test_excludes_request_level_reasoning_effort(self):
        base = self._adapter().runtime_fingerprint()["digest"]
        self.assertEqual(base, self._adapter(
            reasoning_effort="none").runtime_fingerprint()["digest"])


# --------------------------------------------------------------- checkpoint


class TestCheckpointPreflight(unittest.TestCase):
    """Built to the hub cache's real shape: `models--Org--Name/refs/main`
    names a revision, `snapshots/<rev>/` holds symlinks into `blobs/`, and a
    download in progress is a blob with an .incomplete suffix."""

    REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
    SHARDS = ("layer-00000-experts-0000-0127.safetensors",
              "layer-00000-experts-0128-0255.safetensors",
              "embed.safetensors")

    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.root = tempfile.mkdtemp()
        self.hub = os.path.join(self.root, "hub")
        os.makedirs(self.hub)
        self.env = patch.dict(os.environ, {"HF_HUB_CACHE": self.hub})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _adapter(self, model=MODEL):
        return FreeTokenAdapter(
            {"alias": ALIAS, "port": 8090, "model_path": model,
             "executable": "/x/ft", "gpu": "cuda0", "context": CONTEXT},
            state_dir=self.state_dir)

    def _entry(self, cache=None):
        return os.path.join(cache or self.hub, "models--RadixArk--Qwen3.8-Flash-Next-NVFP4")

    def _build(self, cache=None, shards=None, index=True, weight_map=None):
        """A complete checkpoint unless told otherwise; returns snapshot dir."""
        entry = self._entry(cache)
        blobs = os.path.join(entry, "blobs")
        snapshot = os.path.join(entry, "snapshots", self.REVISION)
        os.makedirs(blobs)
        os.makedirs(snapshot)
        os.makedirs(os.path.join(entry, "refs"))
        Path(entry, "refs", "main").write_text(self.REVISION + "\n")
        shards = self.SHARDS if shards is None else shards
        for i, name in enumerate(shards):
            blob = os.path.join(blobs, f"sha{i}")
            Path(blob).write_bytes(b"weights")
            os.symlink(os.path.relpath(blob, snapshot), os.path.join(snapshot, name))
        if index:
            if weight_map is None:
                weight_map = {f"model.layers.{i}.w": name
                              for i, name in enumerate(self.SHARDS)}
                weight_map["model.layers.9.w"] = self.SHARDS[0]  # duplicate ref
            Path(snapshot, ft.SAFETENSORS_INDEX).write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map}))
        return snapshot

    def _snapshot_state(self, snapshot):
        entry = os.path.dirname(os.path.dirname(snapshot))
        return sorted(
            (os.path.relpath(os.path.join(d, f), entry),
             os.path.islink(os.path.join(d, f)))
            for d, _, files in os.walk(entry) for f in files)

    def test_complete_checkpoint_passes(self):
        snapshot = self._build()
        report = self._adapter().checkpoint_preflight()
        self.assertTrue(report["ok"], report["error"])
        self.assertTrue(report["checked"])
        self.assertEqual(report["snapshot"], snapshot)
        self.assertEqual(report["revision"], self.REVISION)
        self.assertEqual(report["shards"], 3, "unique files, not weight_map entries")
        self.assertEqual(report["missing"], [])

    def test_missing_index_fails(self):
        self._build(index=False)
        report = self._adapter().checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertEqual(report["code"], MODEL_CACHE_INCOMPLETE)
        self.assertEqual(report["missing"], [ft.SAFETENSORS_INDEX])

    def test_malformed_index_fails(self):
        snapshot = self._build()
        Path(snapshot, ft.SAFETENSORS_INDEX).write_text("{not json")
        report = self._adapter().checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertEqual(report["code"], MODEL_CACHE_INCOMPLETE)
        self.assertIn("malformed", report["error"])

    def test_index_without_a_weight_map_is_malformed(self):
        snapshot = self._build()
        Path(snapshot, ft.SAFETENSORS_INDEX).write_text(json.dumps({"metadata": {}}))
        report = self._adapter().checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertIn("malformed", report["error"])

    def test_missing_referenced_shard_fails(self):
        """The index names a shard the snapshot never got — the layout that
        was actually found beside the complete one on the qualified machine."""
        self._build(shards=self.SHARDS[:2])
        report = self._adapter().checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertEqual(report["code"], MODEL_CACHE_INCOMPLETE)
        self.assertEqual(report["missing"], ["embed.safetensors"])
        self.assertIn("embed.safetensors", report["error"])

    def test_broken_referenced_symlink_fails(self):
        snapshot = self._build()
        blob = os.path.realpath(os.path.join(snapshot, self.SHARDS[1]))
        os.unlink(blob)
        report = self._adapter().checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertEqual(report["broken_symlinks"], [self.SHARDS[1]])

    def test_incomplete_download_fails(self):
        snapshot = self._build()
        blob = os.path.realpath(os.path.join(snapshot, self.SHARDS[2]))
        os.rename(blob, blob + ".incomplete")
        report = self._adapter().checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertEqual(report["incomplete"], [self.SHARDS[2]])
        self.assertIn("partial", report["error"])

    def test_directory_existence_alone_is_not_proof(self):
        os.makedirs(os.path.join(self._entry(), "snapshots", self.REVISION))
        os.makedirs(os.path.join(self._entry(), "refs"))
        Path(self._entry(), "refs", "main").write_text(self.REVISION)
        adapter = self._adapter()
        self.assertTrue(adapter.model_is_cached(), "presence is still presence")
        report = adapter.checkpoint_preflight()
        self.assertFalse(report["ok"], "but presence is not completeness")
        self.assertTrue(report["checked"])
        self.assertFalse(adapter.capabilities()["capabilities"]["checkpoint_complete"])

    def test_preflight_never_modifies_the_cache(self):
        for builder in (lambda: self._build(shards=self.SHARDS[:2]),
                        lambda: self._build(index=False)):
            snapshot = builder()
            before = self._snapshot_state(snapshot)
            self._adapter().checkpoint_preflight()
            self.assertEqual(self._snapshot_state(snapshot), before)
            import shutil
            shutil.rmtree(self._entry())

    def test_hub_layout_is_preferred_over_the_sibling(self):
        """Both layouts were seen; FreeToken resolved hub/. The sibling here
        is complete and the hub entry is not — the report must describe the
        one the runtime will actually load."""
        self._build(cache=self.root)
        self._build(shards=self.SHARDS[:1])
        report = self._adapter().checkpoint_preflight()
        self.assertEqual(report["cache_dir"], self.hub)
        self.assertFalse(report["ok"])

    def test_sibling_layout_is_found_when_hub_has_no_entry(self):
        self._build(cache=self.root)
        report = self._adapter().checkpoint_preflight()
        self.assertEqual(report["cache_dir"], self.root)
        self.assertTrue(report["ok"], report["error"])

    def test_uncached_repo_is_unchecked_not_refused(self):
        """Nothing to verify; whether to fetch is the runtime's decision."""
        report = self._adapter().checkpoint_preflight()
        self.assertTrue(report["ok"])
        self.assertFalse(report["checked"])
        self.assertIsNone(self._adapter().capabilities()["capabilities"]["checkpoint_complete"])

    def test_ambiguous_snapshots_are_not_guessed(self):
        self._build()
        os.unlink(os.path.join(self._entry(), "refs", "main"))
        os.makedirs(os.path.join(self._entry(), "snapshots", "deadbeef"))
        report = self._adapter().checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertIsNone(report["snapshot"])
        self.assertIn("cannot tell", report["error"])

    def test_local_directory_checkpoint_is_verified_too(self):
        local = tempfile.mkdtemp()
        Path(local, ft.SAFETENSORS_INDEX).write_text(
            json.dumps({"weight_map": {"a": "one.safetensors"}}))
        report = self._adapter(model=local).checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertEqual(report["missing"], ["one.safetensors"])
        Path(local, "one.safetensors").write_bytes(b"w")
        self.assertTrue(self._adapter(model=local).checkpoint_preflight()["ok"])

    def test_start_refuses_an_incomplete_checkpoint_before_launch(self):
        self._build(shards=self.SHARDS[:2])
        adapter = self._adapter()
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True), \
             patch.object(FreeTokenAdapter, "_port_open", return_value=False), \
             patch.object(FreeTokenAdapter, "_nvidia_smi",
                          lambda self, args, timeout=10.0: None), \
             patch.object(ft.subprocess, "run") as version_run, \
             patch.object(ft.subprocess, "Popen") as popen:
            version_run.return_value.stdout = "freetoken version 0.1.2"
            version_run.return_value.stderr = ""
            version_run.return_value.returncode = 0
            result = adapter.start(timeout=1)
        self.assertFalse(result["started"])
        self.assertEqual(result["code"], MODEL_CACHE_INCOMPLETE)
        self.assertEqual(result["diagnostics"]["missing_artifact"], ["embed.safetensors"])
        self.assertEqual(result["diagnostics"]["revision"], self.REVISION)
        self.assertEqual(result["diagnostics"]["model"], MODEL)
        popen.assert_not_called()

    def test_validator_surfaces_an_incomplete_checkpoint_as_a_warning(self):
        self._build(shards=self.SHARDS[:2])
        resolved = _shipped_alias()
        resolved["executable"] = "/x/ft"

        class Stub:
            def resolve_alias(self, name):
                return dict(resolved, alias=name)

        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True):
            result = Validator(resolver=Stub()).validate(ALIAS, "opencode")
        self.assertTrue(any("checkpoint preflight" in w for w in result["warnings"]),
                        result["warnings"])


class TestFreeTokenHardware(unittest.TestCase):
    """Placeholder for the checks that need the real runtime and the card.

    Deliberately skipped: starting Qwen3.8-Flash-Next loads ~100 GB of
    weights and claims the whole GPU, which is never something a test run
    should do by accident. Run by hand with FREETOKEN_HARDWARE_TESTS=1 on the
    qualified workstation, with the card free.
    """

    @unittest.skipUnless(os.environ.get("FREETOKEN_HARDWARE_TESTS") == "1",
                         "freetoken_hardware: needs the qualified card and runtime")
    def test_live_readiness_chain(self):  # pragma: no cover
        adapter = FreeTokenAdapter(Resolver().resolve_alias(ALIAS))
        self.assertTrue(adapter.readiness()["ready"])


if __name__ == "__main__":
    unittest.main()


class TestChildEnvCudaToolkit(unittest.TestCase):
    """The server process must find the toolkit that matches torch, not the
    apt `nvcc` wrapper: a systemd broker has no toolkit on PATH (2026-09-02)."""

    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.bin_dir = tempfile.mkdtemp()
        self.executable = _fake_executable(self.bin_dir)

    def _adapter(self):
        resolved = _shipped_alias()
        resolved["executable"] = self.executable
        return FreeTokenAdapter(resolved, state_dir=self.state_dir)

    def test_cuda_home_bin_is_prepended_after_the_venv_bin(self):
        adapter = self._adapter()
        with tempfile.TemporaryDirectory() as tmp:
            cuda_bin = Path(tmp) / "bin"
            cuda_bin.mkdir()
            (cuda_bin / "nvcc").write_text("")
            env = adapter.child_env({"PATH": "/usr/bin", "CUDA_HOME": tmp})
        parts = env["PATH"].split(os.pathsep)
        self.assertEqual(parts[0], os.path.dirname(adapter.resolve_executable()))
        self.assertEqual(parts[1], str(cuda_bin))
        self.assertEqual(parts[-1], "/usr/bin")

    def test_no_toolkit_means_no_change(self):
        adapter = self._adapter()
        with patch.object(ft.FreeTokenAdapter, "cuda_toolkit_bin", staticmethod(lambda env: "")):
            env = adapter.child_env({"PATH": "/usr/bin"})
        self.assertEqual(env["PATH"].split(os.pathsep)[1:], ["/usr/bin"])

    def test_cuda_home_without_nvcc_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotEqual(ft.FreeTokenAdapter.cuda_toolkit_bin({"CUDA_HOME": tmp}), os.path.join(tmp, "bin"))
