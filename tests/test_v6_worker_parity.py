"""V6 worker parity (TG7, handoff 039).

A LightWorker installation runs the model-allocator against its OWN
``models.yaml`` / ``runtime_profiles.yaml`` — Father config is not
present, no Father-specific paths are resolved, and no DPMtF-WebUI
imports are touched. The same V6 features (shared runtime, inference
profiles) must work on a worker just as they do on the Father's own
allocator copy.

The TG7 evidence is two-fold:

1. **No Father-specific paths or imports in src/model_allocator/.**
   ``grep -rn /home/svend src/model_allocator/ --include="*.py"``
   must return nothing. Verified outside this test file (see the
   ``<validation>`` block in the 039 handoff).
2. **The shared-runtime / inference-profile machinery runs against a
   worker-local config fixture.** That is what this module does: a
   self-contained temp config directory, generic paths (no
   ``/home/svend/...``), no DPMtF imports. The same fixture supplies
   a non-V6 alias so the pre-V6 regression guarantee is also tested
   on the worker style.

The V6 fixtures in test_v6_shared_runtime.py and the V6 worker
fixture here mirror each other — the worker one is a self-contained
re-creation of the same shapes, NOT a parameterised re-use of the
Father fixture. A worker machine does not have the Father's
``models.yaml`` and must not require it.

The process layer is mocked everywhere. No real llama-server is
spawned by any test in this module.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from model_allocator import cli, schema
from model_allocator.adapters import llama_cpp, opencode
from model_allocator.resolver import Resolver
from model_allocator.doctor_cli import cmd_doctor


# ─────────────────────────────────────────────────────────────────
# Worker-local config fixture
# ─────────────────────────────────────────────────────────────────

def _seed_worker_config(tmp: Path) -> Path:
    """Build a self-contained worker-style config directory.

    Path strings are deliberately generic (no ``/home/svend``, no
    Father-specific directories). The env vars referenced by the
    config are named but unbound — the loader does not require them
    to be set, and the resolver only needs them when the rendered
    command actually shells out (which the tests never do).
    """
    (tmp / "models.yaml").write_text(
        "models:\n"
        "  worker-shared-architect:\n"
        "    runtime_instance: worker-shared-llm\n"
        "    inference_profile: worker-profile-careful\n"
        "    clients:\n"
        "      opencode: true\n"
        "  worker-shared-reviewer:\n"
        "    runtime_instance: worker-shared-llm\n"
        "    inference_profile: worker-profile-fast\n"
        "    clients:\n"
        "      opencode: true\n"
        "  worker-legacy-llama:\n"
        "    runtime_profile: worker_local_llamacpp\n"
        "    model_path: ${WORKER_MODEL_ROOT_GGUF}/legacy.gguf\n"
        "    port: 9081\n"
        "    context: 131072\n"
        "    clients:\n"
        "      opencode: true\n"
        "runtime_instances:\n"
        "  worker-shared-llm:\n"
        "    runtime_profile: worker_local_llamacpp\n"
        "    model_path: ${WORKER_MODEL_ROOT_GGUF}/shared-118b.gguf\n"
        "    server_bin_path: ${WORKER_LLAMA_SERVER_BIN}\n"
        "    port: 9090\n"
        "    context: 262144\n"
        "    n_cpu_moe: 31\n"
        "    gpu_layers: 99\n"
        "    cache_type_k: q4_0\n"
        "    cache_type_v: q4_0\n"
        "    lifecycle_policy: shared_runtime\n"
        "inference_profiles:\n"
        "  worker-profile-careful:\n"
        "    reasoning_budget: 4096\n"
        "    max_output_tokens: 16384\n"
        "  worker-profile-fast:\n"
        "    reasoning_budget: 1024\n"
        "    max_output_tokens: 8192\n",
        encoding="utf-8",
    )
    (tmp / "runtime_profiles.yaml").write_text(
        "runtime_profiles:\n"
        "  worker_local_llamacpp:\n"
        "    backend: llama_cpp\n"
        "    server_bin_env: WORKER_LLAMA_SERVER_BIN\n"
        "    model_root_env: WORKER_MODEL_ROOT_GGUF\n",
        encoding="utf-8",
    )
    (tmp / "roles.yaml").write_text("roles:\n", encoding="utf-8")
    return tmp


# ─────────────────────────────────────────────────────────────────
# Step 2 — worker parity: shared runtime identity from the worker
# config (TG7). Process layer mocked.
# ─────────────────────────────────────────────────────────────────

class WorkerSharedRuntimeParityTests(unittest.TestCase):
    """Two aliases in the worker-style config resolve to the same
    recorded instance identity at the lifecycle layer (TG7)."""
    ...

    def _runtime_path_for(self, alias: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_worker_config(Path(tmp))
            return str(Path(tmp) / cfg.name) if hasattr(cfg, "name") else str(cfg)

    def test_two_worker_aliases_resolve_to_same_instance_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_worker_config(Path(tmp))
            resolver = Resolver(config_dir=str(cfg))
            a = resolver.resolve_alias("worker-shared-architect")
            b = resolver.resolve_alias("worker-shared-reviewer")
        # Both aliases point at the same worker-shared-llm instance.
        self.assertEqual(a["runtime_instance"], "worker-shared-llm")
        self.assertEqual(b["runtime_instance"], "worker-shared-llm")
        # Physical identity is taken from the instance (not the alias).
        self.assertEqual(a["model_path"], b["model_path"])
        self.assertEqual(a["port"], b["port"])
        self.assertEqual(a["context"], b["context"])
        self.assertEqual(a["port"], 9090)
        self.assertEqual(a["context"], 262144)
        # Each alias carries its own inference profile.
        self.assertEqual(a["inference_profile"], "worker-profile-careful")
        self.assertEqual(b["inference_profile"], "worker-profile-fast")
        self.assertEqual(a["reasoning_budget"], 4096)
        self.assertEqual(b["reasoning_budget"], 1024)


class WorkerInstanceLifecycleParityTests(unittest.TestCase):
    """The shared lifecycle (start-once / reuse-by-identity) works
    against the worker config when the process layer is mocked. This
    is the worker counterpart of test_v6_instance_lifecycle.py."""
    ...

    def _resolved(self, alias: str, **over) -> dict:
        base = {
            "alias": alias,
            "backend": "llama_cpp",
            "runtime_profile": "worker_local_llamacpp",
            "runtime_instance": "worker-shared-llm",
            "model_path": "/m.gguf",
            "server_bin_path": "/bin/sh",
            "port": 9090,
            "context": 262144,
            "host": "127.0.0.1",
        }
        base.update(over)
        return base

    @pytest.fixture(autouse=True)
    def _patch_layer(self, monkeypatch):
        alive = set()
        spawned = []
        kill_calls = []

        def fake_popen(argv, **kwargs):
            next_pid = 20000 + len(spawned)
            proc = MagicMock()
            proc.pid = next_pid
            proc.poll.return_value = None
            spawned.append({"pid": next_pid, "argv": list(argv)})
            return proc

        def fake_kill(pid, sig):
            kill_calls.append((int(pid), int(sig)))
            if sig == 0 and pid not in alive:
                raise ProcessLookupError(f"no such pid: {pid}")
            if sig != 0:
                alive.discard(int(pid))

        def fake_urlopen(url, **kwargs):
            return MagicMock()

        import tempfile as _tf
        state_dir = _tf.mkdtemp()
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr(os, "kill", fake_kill)
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(llama_cpp.LlamaCppAdapter, "_server_pids",
                            lambda self: [])
        monkeypatch.setattr(llama_cpp.LlamaCppAdapter, "_port_open",
                            lambda self: False)
        self._monkeypatch = monkeypatch
        self._state_dir = state_dir
        self._spawned = spawned
        self._alive = alive
        yield

    def test_worker_first_alias_start_spawns_one_server(self):
        import json
        adapter = llama_cpp.LlamaCppAdapter(
            self._resolved("worker-shared-architect"),
            state_dir=self._state_dir,
        )
        result = adapter.start(timeout=5)
        self.assertTrue(result["started"])
        self.assertFalse(result["reused"])
        self.assertEqual(len(self._spawned), 1)
        # State file is keyed by the worker instance name.
        state_path = os.path.join(
            self._state_dir,
            "model-allocator-instance-worker-shared-llm.json",
        )
        self.assertTrue(os.path.exists(state_path))
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        self.assertEqual(state["instance_name"], "worker-shared-llm")

    def test_worker_second_alias_start_reuses_recorded_identity(self):
        # First alias starts the server and records the state.
        first = llama_cpp.LlamaCppAdapter(
            self._resolved("worker-shared-architect"),
            state_dir=self._state_dir,
        )
        first.start(timeout=5)
        recorded_pid = self._spawned[0]["pid"]
        # The second alias must reuse the recorded PID — not respawn.
        # The monkeypatch auto-removes the pid from the alive set on
        # signal, so we register it alive fresh.
        self._alive.add(recorded_pid)
        self._spawned.clear()
        second = llama_cpp.LlamaCppAdapter(
            self._resolved("worker-shared-reviewer"),
            state_dir=self._state_dir,
        )
        result = second.start(timeout=5)
        self.assertTrue(result["started"])
        self.assertTrue(result["reused"])
        self.assertEqual(result["pid"], recorded_pid)
        self.assertEqual(len(self._spawned), 0,
                         "second alias must not respawn — reuse-by-identity")


# ─────────────────────────────────────────────────────────────────
# Step 2 — worker parity: inference profile transport (TG7).
# ─────────────────────────────────────────────────────────────────

class WorkerInferenceProfileTransportTests(unittest.TestCase):
    """A profile's ``reasoning_budget`` reaches the rendered
    llama-server argv from the worker fixture. The transport is the
    resolver merge + the existing adapter read-path — same code
    the Father fixture uses."""

    def test_worker_profile_reasoning_budget_reaches_llama_server_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_worker_config(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "worker-shared-architect"
            )
        adapter = llama_cpp.LlamaCppAdapter(resolved)
        argv = adapter._build_argv()
        # --reasoning-budget 4096 must appear in the rendered argv.
        self.assertIn("--reasoning-budget", argv)
        idx = argv.index("--reasoning-budget")
        self.assertEqual(argv[idx + 1], "4096")

    def test_worker_profile_max_output_tokens_reaches_opencode_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_worker_config(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "worker-shared-architect"
            )
        cfg_obj = opencode.build_opencode_config(resolved)
        provider = next(iter(cfg_obj["provider"].values()))
        model = next(iter(provider["models"].values()))
        # The profile's max_output_tokens rides the opencode limit block.
        self.assertEqual(model["limit"]["output"], 16384)


# ─────────────────────────────────────────────────────────────────
# Step 2 — worker parity: no-profile invariance from the worker
# fixture (TG7 — backward compatibility holds on the worker style).
# ─────────────────────────────────────────────────────────────────

class WorkerNoProfileInvarianceTests(unittest.TestCase):
    """An alias in the worker fixture without runtime_instance and
    without inference_profile resolves identically to the pre-V6
    behavior. This is the worker counterpart of the O5 backward-
    compat guarantee."""

    def test_worker_legacy_alias_resolves_without_v6_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_worker_config(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "worker-legacy-llama"
            )
        # The legacy alias carries its own runtime_profile and
        # physical identity — no V6 keys leak.
        self.assertNotIn("runtime_instance", resolved)
        self.assertNotIn("inference_profile", resolved)
        self.assertNotIn("reasoning_budget", resolved)
        self.assertNotIn("max_output_tokens", resolved)
        # Alias-level fields resolve as before.
        self.assertEqual(resolved["runtime_profile"], "worker_local_llamacpp")
        self.assertEqual(resolved["backend"], "llama_cpp")
        self.assertEqual(resolved["port"], 9081)
        self.assertEqual(resolved["context"], 131072)

    def test_worker_legacy_alias_llama_argv_has_no_reasoning_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_worker_config(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_alias(
                "worker-legacy-llama"
            )
        # Without a profile, --reasoning-budget must NOT be in the
        # rendered argv (the pre-V6 contract).
        argv = llama_cpp.LlamaCppAdapter(resolved)._build_argv()
        self.assertNotIn("--reasoning-budget", argv)


# ─────────────────────────────────────────────────────────────────
# Step 2 — worker parity: schema layer runs against the worker
# fixture (TG7). The doctor sees a clean worker config.
# ─────────────────────────────────────────────────────────────────

class WorkerDoctorTests(unittest.TestCase):
    """``model-allocator doctor`` runs against the worker-style
    config and reports a clean result. Proves the schema read-path
    has no Father-specific assumptions."""

    def test_worker_config_passes_doctor(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_worker_config(Path(tmp))
            import argparse
            args = argparse.Namespace(config_dir=str(cfg), json=True)
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cmd_doctor(args)
            self.assertEqual(rc, 0, f"doctor must exit 0 on a clean worker "
                                   f"config; got rc={rc}, output={buf.getvalue()}")
            report = buf.getvalue()
            self.assertIn('"errors": 0', report)
            self.assertIn('"warnings": 0', report)


if __name__ == "__main__":
    unittest.main()
