"""V6 instance lifecycle — process layer (B2, handoff 037).

The schema/validation tests live in test_v6_shared_runtime.py; this
file covers the start/stop/status rules on the recorded instance
identity, with every process-layer dependency mocked. No real
llama-server is started by any test in this file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_allocator import cli
from model_allocator.adapters.llama_cpp import LlamaCppAdapter
from model_allocator.resolver import Resolver


# ─────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────

def _instance_resolved(alias, instance_name="shared-llm", **over):
    base = {
        "alias": alias,
        "backend": "llama_cpp",
        "runtime_profile": "local_llamacpp",
        "runtime_instance": instance_name,
        "model_path": "/m.gguf",
        # Real path so the adapter's binary-existence check passes —
        # subprocess.Popen is mocked, no real binary is invoked.
        "server_bin_path": "/bin/sh",
        "port": 8090,
        "context": 262144,
        "host": "127.0.0.1",
    }
    base.update(over)
    return base


def _alias_resolved(alias, **over):
    base = {
        "alias": alias,
        "backend": "llama_cpp",
        "runtime_profile": "local_llamacpp",
        "model_path": "/m.gguf",
        "server_bin_path": "/bin/sh",
        "port": 8091,
        "context": 131072,
        "host": "127.0.0.1",
    }
    base.update(over)
    return base


@pytest.fixture
def state_dir(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    return str(d)


def _patch_process_layer(monkeypatch, *, alive_pids=None,
                         healthy=True, health_error=None,
                         port_open=False, server_pids=None):
    """Patch subprocess.Popen, os.kill, urlopen, and /proc scan.

    Defaults match a "clean machine" — port closed, no /proc listeners,
    no recorded state. Tests that exercise the ownership checks opt-in
    to ``port_open=True`` with explicit ``server_pids``.

    Returns a dict with lists of observed calls: ``spawned`` (Popen
    invocations), ``kill_calls`` (every ``os.kill``), ``urlopen_calls``.
    """
    alive = set(alive_pids or [])
    server_pids = list(server_pids or [])

    spawned = []

    def fake_popen(argv, **kwargs):
        next_pid = 10000 + len(spawned)
        proc = MagicMock()
        proc.pid = next_pid
        proc.poll.return_value = None
        spawned.append({"pid": next_pid, "argv": list(argv)})
        return proc

    kill_calls = []

    def fake_kill(pid, sig):
        kill_calls.append((int(pid), int(sig)))
        if sig == 0:
            if pid in alive:
                return
            raise ProcessLookupError(f"no such pid: {pid}")
        # Any real signal removes the pid from the alive set so a
        # subsequent probe (kill(pid, 0)) reflects the post-signal
        # state — the way a real OS would.
        alive.discard(int(pid))
        return

    urlopen_calls = []

    def fake_urlopen(url, **kwargs):
        urlopen_calls.append(url)
        if healthy:
            return MagicMock()
        raise Exception(health_error or "connection refused")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(os, "kill", fake_kill)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(LlamaCppAdapter, "_server_pids",
                        lambda self: list(server_pids))
    monkeypatch.setattr(LlamaCppAdapter, "_port_open",
                        lambda self: bool(port_open))

    return {"spawned": spawned, "kill_calls": kill_calls,
            "urlopen_calls": urlopen_calls}


def _write_instance_state(state_dir, *, instance_name="shared-llm",
                          pid=12345, port=8090, alias="shared-architect"):
    state_path = os.path.join(
        state_dir, f"model-allocator-instance-{instance_name}.json"
    )
    Path(state_path).write_text(json.dumps({
        "instance_name": instance_name,
        "pid": pid,
        "port": port,
        "started_by_alias": alias,
        "started_at": time.time(),
    }), encoding="utf-8")
    return state_path


def _real_kills(fakes):
    return [(p, s) for (p, s) in fakes["kill_calls"] if s != 0]


# ─────────────────────────────────────────────────────────────────
# Step 3 — start-once / reuse-by-identity
# ─────────────────────────────────────────────────────────────────

def test_first_alias_start_spawns_one_server_and_records_instance_state(
        monkeypatch, state_dir):
    fakes = _patch_process_layer(monkeypatch, healthy=True)
    adapter = LlamaCppAdapter(_instance_resolved("shared-architect"),
                              state_dir=state_dir)
    result = adapter.start(timeout=5)

    assert result["started"] is True
    assert result["reused"] is False
    assert result["pid"] == 10000
    assert result["port"] == 8090
    assert result["instance_name"] == "shared-llm"
    assert len(fakes["spawned"]) == 1, \
        f"first start must spawn exactly one server, got: {fakes['spawned']}"

    state_path = os.path.join(
        state_dir, "model-allocator-instance-shared-llm.json"
    )
    assert os.path.exists(state_path)
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    assert state["instance_name"] == "shared-llm"
    assert state["pid"] == 10000
    assert state["port"] == 8090
    assert state["started_by_alias"] == "shared-architect"


def test_second_alias_start_reuses_recorded_identity_without_spawning(
        monkeypatch, state_dir):
    _write_instance_state(state_dir, pid=12345, alias="shared-architect")
    fakes = _patch_process_layer(monkeypatch, alive_pids={12345},
                                 healthy=True)
    adapter = LlamaCppAdapter(_instance_resolved("shared-reviewer"),
                              state_dir=state_dir)
    result = adapter.start(timeout=5)

    assert result["started"] is True
    assert result["reused"] is True
    assert result["pid"] == 12345
    assert result["port"] == 8090
    assert result["instance_name"] == "shared-llm"
    assert len(fakes["spawned"]) == 0, \
        f"second start must NOT spawn — reuse is the recorded PID's job, " \
        f"but got: {fakes['spawned']}"
    # Only the alive-probe (signal 0) ever fired.
    assert _real_kills(fakes) == [], \
        f"reuse must never signal the recorded PID, but got: {fakes['kill_calls']}"


def test_reuse_uses_recorded_identity_not_port_or_model_path_equality(
        monkeypatch, state_dir):
    """The decision to reuse must come from the recorded state — not from
    'is the port open' or 'does a model_path match'. A different model on
    the same port with no recorded state must NOT be reused.
    """
    # Port is open, /proc shows an unrelated llama-server on it, and no
    # state was recorded. Reuse must NOT happen.
    fakes = _patch_process_layer(monkeypatch, alive_pids={55555},
                                 healthy=True, port_open=True,
                                 server_pids=[55555])
    adapter = LlamaCppAdapter(_instance_resolved("shared-architect"),
                              state_dir=state_dir)
    result = adapter.start(timeout=5)
    assert result["started"] is False
    assert "unmanaged llama-server" in result["error"]
    assert len(fakes["spawned"]) == 0
    assert _real_kills(fakes) == []


# ─────────────────────────────────────────────────────────────────
# Step 6 — status reports instance identity identically across aliases
# ─────────────────────────────────────────────────────────────────

def test_status_via_alias_a_alias_b_and_instance_name_report_same_identity(
        monkeypatch, state_dir):
    _write_instance_state(state_dir, pid=12345, alias="shared-architect")
    _patch_process_layer(monkeypatch, alive_pids={12345}, healthy=True)

    a = LlamaCppAdapter(_instance_resolved("shared-architect"),
                        state_dir=state_dir).status()
    b = LlamaCppAdapter(_instance_resolved("shared-reviewer"),
                        state_dir=state_dir).status()
    inst = LlamaCppAdapter(_instance_resolved("shared-llm"),
                           state_dir=state_dir).status()

    for label, report in (("alias A", a), ("alias B", b),
                          ("instance", inst)):
        assert report["instance_name"] == "shared-llm", label
        assert report["pid"] == 12345, label
        assert report["port"] == 8090, label
        assert report["instance_managed"] is True, label
        assert report["running"] is True, label


# ─────────────────────────────────────────────────────────────────
# Step 5 — alias-level stop/unload is a no-op on a shared_runtime instance
# ─────────────────────────────────────────────────────────────────

def test_alias_level_stop_is_noop_for_instance_bound_alias(
        monkeypatch, state_dir):
    state_path = _write_instance_state(state_dir, pid=12345,
                                       alias="shared-architect")
    fakes = _patch_process_layer(monkeypatch, alive_pids={12345},
                                 healthy=True)
    adapter = LlamaCppAdapter(_instance_resolved("shared-architect"),
                              state_dir=state_dir)
    result = adapter.stop(timeout=5)

    assert result["stopped"] is True
    assert result["skipped"] is True
    assert result["instance_name"] == "shared-llm"
    assert result["pid"] == 12345
    assert _real_kills(fakes) == [], \
        f"alias-level stop must NOT signal the recorded PID, " \
        f"got: {fakes['kill_calls']}"
    # State preserved — the instance is still managed.
    assert os.path.exists(state_path), \
        "alias-level stop must not delete the instance state"


def test_alias_level_unload_is_noop_for_instance_bound_alias(
        monkeypatch, state_dir):
    state_path = _write_instance_state(state_dir, pid=12345,
                                       alias="shared-architect")
    fakes = _patch_process_layer(monkeypatch, alive_pids={12345},
                                 healthy=True)
    adapter = LlamaCppAdapter(_instance_resolved("shared-architect"),
                              state_dir=state_dir)
    result = adapter.unload(timeout=5)

    assert result["stopped"] is True
    assert result["skipped"] is True
    assert _real_kills(fakes) == [], \
        f"alias-level unload must NOT signal the recorded PID, " \
        f"got: {fakes['kill_calls']}"
    assert os.path.exists(state_path)


# ─────────────────────────────────────────────────────────────────
# Step 6 — explicit instance-level stop kills and removes the record
# ─────────────────────────────────────────────────────────────────

def test_instance_level_stop_kills_recorded_pid_and_removes_state(
        monkeypatch, state_dir):
    state_path = _write_instance_state(state_dir, pid=12345,
                                       alias="shared-architect")
    fakes = _patch_process_layer(monkeypatch, alive_pids={12345},
                                 healthy=True)
    adapter = LlamaCppAdapter(_instance_resolved("shared-llm"),
                              state_dir=state_dir)
    result = adapter.stop_instance(timeout=5)

    assert result["stopped"] is True
    assert result["pid"] == 12345
    assert result["instance_name"] == "shared-llm"
    real_kills = [(p, s) for (p, s) in fakes["kill_calls"] if s != 0]
    assert any(p == 12345 for (p, s) in real_kills), \
        f"instance-level stop must signal the recorded PID, " \
        f"got: {fakes['kill_calls']}"
    assert not os.path.exists(state_path), \
        "instance-level stop must remove the state record"


def test_instance_level_stop_with_no_recorded_state_is_safe_no_kill(
        monkeypatch, state_dir):
    """No state recorded — nothing to kill. Must NOT fall back to a
    port-scan kill of whatever happens to be on the port.
    """
    fakes = _patch_process_layer(monkeypatch, alive_pids={12345},
                                 healthy=True, port_open=True,
                                 server_pids=[12345])
    adapter = LlamaCppAdapter(_instance_resolved("shared-llm"),
                              state_dir=state_dir)
    result = adapter.stop_instance(timeout=5)
    assert result["stopped"] is True
    assert result["skipped"] is True
    assert result["pid"] is None
    assert _real_kills(fakes) == [], \
        f"stop_instance without recorded state must NOT kill anything, " \
        f"got: {fakes['kill_calls']}"


# ─────────────────────────────────────────────────────────────────
# Step 4 — managed-only ownership
# ─────────────────────────────────────────────────────────────────

def test_foreign_unrecorded_llama_server_on_instance_port_is_not_signalled(
        monkeypatch, state_dir):
    """Port is open, /proc shows an unrecorded llama-server, no state.

    The allocator must refuse, NOT signal the foreign pid, NOT spawn,
    and NOT adopt.
    """
    fakes = _patch_process_layer(monkeypatch, alive_pids={77777},
                                 healthy=True, port_open=True,
                                 server_pids=[77777])
    adapter = LlamaCppAdapter(_instance_resolved("shared-architect"),
                              state_dir=state_dir)
    result = adapter.start(timeout=5)

    assert result["started"] is False
    assert "77777" in result["error"]
    assert "unmanaged llama-server" in result["error"]
    assert "refusing to adopt" in result["error"]
    assert _real_kills(fakes) == [], \
        f"foreign PID must NEVER be signalled, got: {fakes['kill_calls']}"
    assert len(fakes["spawned"]) == 0
    # No state recorded.
    assert not any(name.startswith("model-allocator-instance-")
                   for name in os.listdir(state_dir))


def test_unknown_process_on_instance_port_is_refused_not_adopted(
        monkeypatch, state_dir):
    """Port is open but no llama-server is listening on it.

    The allocator must refuse, NOT signal whatever is there.
    """
    fakes = _patch_process_layer(monkeypatch, port_open=True,
                                 server_pids=[], healthy=True)
    adapter = LlamaCppAdapter(_instance_resolved("shared-architect"),
                              state_dir=state_dir)
    result = adapter.start(timeout=5)

    assert result["started"] is False
    assert "unmanaged process" in result["error"]
    assert _real_kills(fakes) == []
    assert len(fakes["spawned"]) == 0


def test_stale_state_with_dead_pid_is_cleaned_without_killing_foreign(
        monkeypatch, state_dir):
    """Recorded PID is dead, a foreign llama-server has taken the port.

    The allocator must NOT kill the foreign process — that is adoption,
    forbidden by managed-only ownership. Stale state is removed; the
    start fails loudly.
    """
    state_path = _write_instance_state(state_dir, pid=99999,
                                       alias="shared-architect")
    fakes = _patch_process_layer(monkeypatch, alive_pids={88888},
                                 healthy=True, port_open=True,
                                 server_pids=[88888])
    adapter = LlamaCppAdapter(_instance_resolved("shared-architect"),
                              state_dir=state_dir)
    result = adapter.start(timeout=5)

    assert result["started"] is False
    assert "refusing to adopt" in result["error"]
    assert not os.path.exists(state_path), \
        "stale state must be cleaned up so no orphan record survives"
    assert _real_kills(fakes) == [], \
        f"foreign PID must NEVER be signalled, got: {fakes['kill_calls']}"


def test_stale_state_with_dead_pid_and_no_listener_recovers_cleanly(
        monkeypatch, state_dir):
    """Recorded PID is dead, port is closed — the start must succeed by
    cleaning the stale record and spawning fresh.
    """
    state_path = _write_instance_state(state_dir, pid=99999,
                                       alias="shared-architect")
    fakes = _patch_process_layer(monkeypatch, healthy=True,
                                 port_open=False, server_pids=[])
    adapter = LlamaCppAdapter(_instance_resolved("shared-architect"),
                              state_dir=state_dir)
    result = adapter.start(timeout=5)
    assert result["started"] is True
    assert result["reused"] is False
    assert len(fakes["spawned"]) == 1
    # Fresh state recorded.
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    assert state["pid"] == 10000


# ─────────────────────────────────────────────────────────────────
# Backward compat — alias WITHOUT runtime_instance keeps existing
# per-alias PID-file behaviour.
# ─────────────────────────────────────────────────────────────────

def test_alias_without_runtime_instance_uses_per_alias_pid_file(
        monkeypatch, state_dir):
    fakes = _patch_process_layer(monkeypatch, healthy=True)
    adapter = LlamaCppAdapter(_alias_resolved("legacy-llama"),
                              state_dir=state_dir)
    result = adapter.start(timeout=5)

    assert result["started"] is True
    assert len(fakes["spawned"]) == 1
    # Per-alias PID file is written, no instance JSON file appears.
    pid_file = os.path.join(
        state_dir, "model-allocator-legacy-llama-8091.pid"
    )
    assert os.path.exists(pid_file), \
        "non-instance-bound alias must keep the per-alias PID file"
    assert not any(name.startswith("model-allocator-instance-")
                   for name in os.listdir(state_dir)), \
        "non-instance-bound alias must NOT create an instance state file"


def test_alias_without_runtime_instance_status_omits_v6_fields(
        monkeypatch, state_dir):
    fakes = _patch_process_layer(monkeypatch, healthy=True)
    adapter = LlamaCppAdapter(_alias_resolved("legacy-llama"),
                              state_dir=state_dir)
    adapter.start(timeout=5)
    report = adapter.status()
    # Backward-compat: no V6 keys leak into a non-V6 alias report.
    assert "instance_name" not in report
    assert "instance_managed" not in report
    assert report["running"] is True


# ─────────────────────────────────────────────────────────────────
# Step 6 — Resolver.resolve_instance + CLI instance commands
# ─────────────────────────────────────────────────────────────────

def _seed_shared_config(tmp_path, *, aliases=("shared-architect",
                                              "shared-reviewer"),
                        instance_name="shared-llm"):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "models.yaml").write_text(
        "models:\n"
        + "".join(
            f"  {a}:\n    runtime_instance: {instance_name}\n"
            f"    clients:\n      opencode: true\n"
            for a in aliases
        )
        + f"runtime_instances:\n  {instance_name}:\n"
        + "    runtime_profile: local_llamacpp\n"
        + "    model_path: /m.gguf\n"
        + "    port: 8090\n"
        + "    context: 262144\n"
        + "    lifecycle_policy: shared_runtime\n",
        encoding="utf-8",
    )
    (cfg / "runtime_profiles.yaml").write_text(
        "runtime_profiles:\n  local_llamacpp:\n"
        "    backend: llama_cpp\n"
        "    server_bin_env: LLAMA_SERVER_BIN\n",
        encoding="utf-8",
    )
    (cfg / "roles.yaml").write_text("roles:\n", encoding="utf-8")
    return cfg


def test_resolver_resolve_instance_returns_resolved_view():
    with pytest.MonkeyPatch.context() as mp:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _seed_shared_config(Path(tmp))
            resolved = Resolver(config_dir=str(cfg)).resolve_instance(
                "shared-llm"
            )
            assert resolved["alias"] == "shared-llm"
            assert resolved["runtime_instance"] == "shared-llm"
            assert resolved["backend"] == "llama_cpp"
            assert resolved["port"] == 8090
            assert resolved["context"] == 262144


def test_resolver_resolve_instance_unknown_is_resolutionerror():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _seed_shared_config(Path(tmp))
        resolver = Resolver(config_dir=str(cfg))
        with pytest.raises(Exception) as excinfo:
            resolver.resolve_instance("ghost")
        assert "ghost" in str(excinfo.value)


class InstanceCLITests(unittest.TestCase):
    """CLI: start-instance / status-instance / stop-instance, mocked."""

    def _args(self, **over):
        ns = argparse.Namespace(name="shared-llm", timeout=30,
                                config_dir=None)
        for k, v in over.items():
            setattr(ns, k, v)
        return ns

    def _patch(self, alive_pids=None, healthy=True, port_open=False,
               server_pids=None):
        return _patch_process_layer(self, alive_pids=alive_pids,
                                    healthy=healthy, port_open=port_open,
                                    server_pids=server_pids)

    # The monkeypatch helper takes the ``monkeypatch`` fixture; unittest
    # methods don't get it, so we expose a thin wrapper using
    # ``unittest.mock.patch.object`` for the unittest tests.
    @staticmethod
    def _patch_unittest(alive_pids=None, healthy=True, port_open=False,
                        server_pids=None):
        alive = set(alive_pids or [])
        server_pids = list(server_pids or [])

        spawned = []

        def fake_popen(argv, **kwargs):
            next_pid = 10000 + len(spawned)
            proc = MagicMock()
            proc.pid = next_pid
            proc.poll.return_value = None
            spawned.append({"pid": next_pid, "argv": list(argv)})
            return proc

        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((int(pid), int(sig)))
            if sig == 0 and pid not in alive:
                raise ProcessLookupError(f"no such pid: {pid}")
            if sig != 0:
                alive.discard(int(pid))
            return

        urlopen_calls = []

        def fake_urlopen(url, **kwargs):
            urlopen_calls.append(url)
            if healthy:
                return MagicMock()
            raise Exception("unhealthy")

        patches = [
            patch.object(subprocess, "Popen", fake_popen),
            patch.object(os, "kill", fake_kill),
            patch.object(urllib.request, "urlopen", fake_urlopen),
            patch.object(LlamaCppAdapter, "_server_pids",
                         lambda self: list(server_pids)),
            patch.object(LlamaCppAdapter, "_port_open",
                         lambda self: bool(port_open)),
        ]
        for p in patches:
            p.start()
        return {"spawned": spawned, "kill_calls": kill_calls,
                "urlopen_calls": urlopen_calls, "patches": patches}

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_cfg(self):
        cfg = _seed_shared_config(Path(self.tmp))
        return str(cfg)

    def test_cmd_start_instance_creates_state(self):
        fakes = self._patch_unittest()
        try:
            captured = []
            with patch("builtins.print",
                       lambda *a, **k: captured.append(a[0] if a else "")):
                code = cli.cmd_start_instance(
                    argparse.Namespace(name="shared-llm",
                                       timeout=5,
                                       config_dir=self._make_cfg())
                )
            assert code == cli.EXIT_OK
            assert len(fakes["spawned"]) == 1
            state_path = os.path.join(
                tempfile.gettempdir(),
                "model-allocator-instance-shared-llm.json",
            )
            # The state dir default is /tmp; the file must exist there.
            # (Use the same env-default the adapter used.)
            assert any(name.endswith("model-allocator-instance-shared-llm.json")
                       for name in os.listdir(tempfile.gettempdir())), \
                f"expected instance state file in {tempfile.gettempdir()}"
        finally:
            for p in fakes["patches"]:
                p.stop()

    def test_cmd_status_instance_returns_same_identity_as_aliases(self):
        # Seed a state file in the default state dir.
        state_dir = tempfile.gettempdir()
        state_path = os.path.join(
            state_dir, "model-allocator-instance-shared-llm.json"
        )
        Path(state_path).write_text(json.dumps({
            "instance_name": "shared-llm",
            "pid": 12345,
            "port": 8090,
            "started_by_alias": "shared-architect",
            "started_at": time.time(),
        }), encoding="utf-8")
        try:
            fakes = self._patch_unittest(alive_pids={12345}, healthy=True)
            try:
                captured = []
                with patch("builtins.print",
                           lambda *a, **k: captured.append(a[0] if a else "")):
                    code = cli.cmd_status_instance(
                        argparse.Namespace(name="shared-llm",
                                           timeout=5,
                                           config_dir=self._make_cfg())
                    )
                assert code == cli.EXIT_OK
                # Last JSON object printed.
                last = json.loads(captured[-1])
                assert last["instance_name"] == "shared-llm"
                assert last["pid"] == 12345
                assert last["port"] == 8090
            finally:
                for p in fakes["patches"]:
                    p.stop()
        finally:
            try:
                os.unlink(state_path)
            except OSError:
                pass

    def test_cmd_stop_instance_signals_recorded_pid(self):
        state_dir = tempfile.gettempdir()
        state_path = os.path.join(
            state_dir, "model-allocator-instance-shared-llm.json"
        )
        Path(state_path).write_text(json.dumps({
            "instance_name": "shared-llm",
            "pid": 12345,
            "port": 8090,
            "started_by_alias": "shared-architect",
            "started_at": time.time(),
        }), encoding="utf-8")
        try:
            fakes = self._patch_unittest(alive_pids={12345}, healthy=True)
            try:
                captured = []
                with patch("builtins.print",
                           lambda *a, **k: captured.append(a[0] if a else "")):
                    code = cli.cmd_stop_instance(
                        argparse.Namespace(name="shared-llm",
                                           timeout=5,
                                           config_dir=self._make_cfg())
                    )
                assert code == cli.EXIT_OK
                real_kills = [c for c in fakes["kill_calls"] if c[1] != 0]
                assert any(p == 12345 for (p, s) in real_kills), \
                    f"stop_instance must signal the recorded PID, " \
                    f"got: {fakes['kill_calls']}"
                assert not os.path.exists(state_path)
            finally:
                for p in fakes["patches"]:
                    p.stop()
        finally:
            try:
                os.unlink(state_path)
            except OSError:
                pass

    def test_stop_all_local_servers_skips_instance_bound_aliases(self):
        """The ``stop --all-servers`` sweep must never stop a shared
        runtime — it must report it as skipped and leave the state file
        alone.
        """
        state_dir = tempfile.gettempdir()
        state_path = os.path.join(
            state_dir, "model-allocator-instance-shared-llm.json"
        )
        Path(state_path).write_text(json.dumps({
            "instance_name": "shared-llm",
            "pid": 12345,
            "port": 8090,
            "started_by_alias": "shared-architect",
            "started_at": time.time(),
        }), encoding="utf-8")
        try:
            captured = []
            with patch.object(cli, "Resolver",
                              lambda config_dir=None: _FakeResolver(
                                  config_dir)), \
                 patch.object(cli, "_get_backend_adapter",
                              lambda resolved: _FakeAdapter(
                                  resolved, port=resolved.get("port"))), \
                 patch("builtins.print",
                       lambda *a, **k: captured.append(a[0] if a else "")):
                code = cli._stop_all_local_servers(
                    argparse.Namespace(timeout=30, config_dir=self._make_cfg())
                )
            assert code == cli.EXIT_OK
            results = json.loads(captured[-1])
            # The shared alias appears once and is marked skipped.
            shared_entries = [r for r in results
                              if r.get("instance_name") == "shared-llm"]
            assert len(shared_entries) >= 1
            assert all(r.get("skipped") for r in shared_entries), \
                f"shared aliases must be skipped, got: {shared_entries}"
            # State preserved.
            assert os.path.exists(state_path)
        finally:
            try:
                os.unlink(state_path)
            except OSError:
                pass


class _FakeAdapter:
    def __init__(self, resolved, port=None):
        self.resolved = resolved
        self.port = port or resolved.get("port")
        self.kill_calls = []

    def stop(self, timeout=30):
        self.kill_calls.append(("stop", timeout))
        return {"stopped": True, "error": None}

    def status(self):
        return {"running": False, "alive": False, "healthy": False,
                "pid": None, "port": self.port}


class _FakeResolver:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        # Load the real config and project the shapes the sweep uses.
        from model_allocator.resolver import Resolver as _R
        real = _R(config_dir=config_dir)
        self._aliases = {}
        for name in real.list_aliases():
            try:
                self._aliases[name] = real.resolve_alias(name)
            except Exception:
                pass

    def list_aliases(self):
        return list(self._aliases)

    def resolve_alias(self, name):
        return dict(self._aliases[name])
