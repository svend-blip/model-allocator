# Remote llama.cpp Lifecycle Implementation Plan

> **For agentic workers:** Execute tasks in order. Steps use checkbox (`- [ ]`) syntax for tracking. Do not skip verification steps.

**Goal:** Extend the llama.cpp adapter with SSH-based remote lifecycle so llama-server instances on svend3060 (Tailscale `100.73.166.28`) can be started, health-checked, and stopped exactly like local ones — plus real start-diagnostics (log files instead of DEVNULL) for BOTH modes.

**Architecture:** `LlamaCppAdapter` (src/model_allocator/adapters/llama_cpp.py, 211 lines) currently manages only local processes: `subprocess.Popen` with stdout/stderr DEVNULL (lines 119-124), a local PID file, `os.kill` signaling, and a `http://<host>:<port>/health` poll. This plan adds an optional `ssh_host` + `remote_workdir` on the runtime profile; when present, start/stop/status run through a single mockable `run_remote(host, command, timeout)` seam (`ssh -o BatchMode=yes ...`), the remote PID is captured from `nohup ... & echo $!` into a JSON-annotated local PID file, and health polls target the Tailscale IP parsed from `ssh_host`. Local mode gains a per-alias log file under the state dir with tail-on-failure. A real svend3060 alias is added to `models.yaml` using the live-verified server parameters.

**Tech Stack:** Python 3.10+ stdlib (`subprocess`, `shlex`, `json`, `urllib.request`, `signal`), OpenSSH client (`/usr/bin/ssh`, key auth to `svend@100.73.166.28` already authorized), pyyaml (only runtime dep), unittest with `unittest.mock.patch` (tests/test_v2.py conventions).

## Cold-Start Context

- model-allocator is a standalone Python CLI at `/home/svend/model-allocator` resolving model aliases → runtime commands for ollama / llama.cpp / opencode / claude-code / onyx. Config = `models.yaml` + `runtime_profiles.yaml` + `roles.yaml` at repo root; source under `src/model_allocator/`, adapters under `src/model_allocator/adapters/`.
- The Father project `/home/svend/DPMtF-WebUI` consumes it via subprocess to `scripts/model-allocator`: `routers/bridge.py` `/allocator/status|start|stop` endpoints parse the JSON these commands print; `scripts/bridgeV002/dispatch.py:261-286` runs `stop --alias <alias>` with an outer 45s timeout.
- Run tests: `cd /home/svend/model-allocator && python3 -m pytest` → **95 passed** ~5s (baseline).
- CLI routing (src/model_allocator/cli.py): `start` → `adapter.start(timeout=args.timeout)` (line 194, default 120), `stop` → `adapter.stop(timeout=args.timeout)` (line 225, default 30), `status` → `adapter.status()` (line 130), `unload` → `adapter.unload(timeout=args.timeout)` (line 256 — **latent TypeError**: the adapter's `unload(self)` takes no timeout; fixed in Task 2).
- **Live evidence from svend3060 gathered 2026-07-12** (`ssh -o BatchMode=yes svend@100.73.166.28` works non-interactively):
  - llama.cpp build: `/home/svend/llama-cpp-turboquant/build/bin/llama-server` (exists, executable).
  - Model GGUF: `/home/svend/models/qwen36-35b-mxfp4/Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (a root-owned symlink `/Qwen3.6-35B-A3B-MXFP4_MOE.gguf` also points at it — use the real path, not the fragile symlink).
  - A manually started llama-server (pid 42035) is LISTENING on **127.0.0.1:8080** with argv: `-m /Qwen3.6-35B-A3B-MXFP4_MOE.gguf -c 262144 --parallel 1 --n-cpu-moe 26 -t 12 -b 160 --ubatch-size 128 --cache-type-k turbo4 --cache-type-v turbo3 --flash-attn on --reasoning off --no-mmap` (MoE build uses `--n-cpu-moe`, NOT `-ngl`).
  - `/usr/bin/ss` exists on the remote host; `/home/svend/llama-cpp-turboquant/logs` does NOT exist yet (start must `mkdir -p` it).

## Global Constraints

- `python3 -m py_compile <file>` MUST pass on every touched Python file.
- Single runtime dependency stays `pyyaml` (`mcp` optional extra). NO new dependencies — ssh is invoked as a subprocess, no paramiko (adding any dep requires explicit Human approval; not needed).
- All 95 existing tests stay green — the local-mode code paths (PID-file format for local, `_build_argv`, `_kill_pid`, existing test assertions in tests/test_v2.py:253-300 and 464-513) must keep working unchanged.
- TDD: failing test → implement → green. Every ssh interaction goes through the `run_remote` seam so tests never open a socket; the fixture at tests/test_v2.py:465-509 (`svend3060-llama-test`) is the starting point for remote-alias configs.
- Backwards compatibility: `start`/`stop`/`status` JSON printed by the CLI keeps all existing keys (`started`, `stopped`, `running`, `alive`, `healthy`, `pid`, `port`, `error`); new keys (`mode`, `status`, `log`, `log_tail`) are ADDITIVE. Father's `/allocator/status` reads `running`/`pid`/`port` and `/allocator/stop` reads `stopped`/`error` — verified additive-safe, so no Father code change is required (Task 7 is verification only).
- Git policy: the Human approves commits. Tasks end with `git add <files>` and STOP.

## Edge Cases a Weaker Model Would Miss

1. **Nested shell quoting — exactly ONE shell parses the remote command.** Locally we call `subprocess.run(["ssh", ..., host, command_string])` with `shell=False`, so no local shell touches `command_string`; ssh hands it to the remote login shell verbatim. Therefore quote ONLY for the remote shell: build the server invocation with `shlex.join(argv)` and quote paths with `shlex.quote(...)`. Do NOT wrap the whole command in another layer of single quotes — that is the classic double-quoting bug (it would make the remote shell see one giant literal word).
2. **`nohup <cmd> ... & echo $!` PID semantics.** The remote shell backgrounds the `nohup` process and `$!` is that process's PID. `nohup` does not fork — it exec()s the command — so `$!` IS llama-server's PID. This only holds because the command after `nohup` is a simple argv (no pipes, no `sh -c` wrapper). Keep the exact form `cd <workdir> && nohup <argv> > <log> 2>&1 & echo $!` and never add a pipeline after `nohup`.
3. **`BatchMode=yes` + `ConnectTimeout=10` + an outer `subprocess` timeout on EVERY ssh call.** Without BatchMode a missing key silently hangs on a password prompt (deadly under cron/CI); ConnectTimeout bounds TCP setup; the outer timeout bounds everything else (e.g. remote shell wedged). ssh unreachable must yield a structured envelope (`status: "unreachable"`), never a traceback.
4. **Bind host ≠ probe host.** The server must bind `0.0.0.0` (config `host: 0.0.0.0`) but health checks must probe the Tailscale IP parsed from `ssh_host` (`ssh_host.split("@")[-1]`). Probing `http://0.0.0.0:port/` "works" from the local machine by accident on some stacks and lies about the remote server. Conversely the CURRENT manually-started server on svend3060 binds 127.0.0.1 — unreachable via Tailscale — which is exactly why the port-busy pre-check must use `ss` over ssh, with health-probe only as fallback.
5. **No `_find_free_port()` for remote.** `_find_free_port` (llama_cpp.py:43-47) binds a LOCAL socket — a free local port says nothing about the remote host. Remote profiles must declare an explicit `port` (or `default_port`); missing → hard error at adapter construction.
6. **Port-busy adopt/refuse.** The user manually runs a server on remote 8080. Pre-start check `ss -tlnp | grep ':<port> '`; if occupied, refuse with `port busy — adopt the running server (use status/opencode against it) or choose another port`. Never kill a process the allocator did not start. (The new alias uses port 8090 to coexist with the manual 8080.)
7. **Stale remote PID file.** If the PID file exists but `ssh <host> kill -0 <pid>` says the process is gone, `status` must report not-running and `start` must clean the stale file and proceed — not refuse forever.
8. **PID file format compatibility.** Local PID files are bare ints (`status`/`stop` parse with `int(...)`, lines 147/199). Remote PID files are JSON (`{"pid": N, "remote": true, "ssh_host": "..."}`) so a PID file found later is self-describing. The reader must accept BOTH: try `int()` first, then JSON. Never write JSON for local aliases (old allocator versions/tests parse ints).
9. **`cli.py` `unload --alias <llama-alias>` crashes today**: `adapter.unload(timeout=args.timeout)` (cli.py:256) vs `def unload(self)` (llama_cpp.py:210). Existing tests miss it (they only test a missing alias). Fix the adapter signature to `unload(self, timeout: int = 30)` — no cli.py change needed.
10. **WAL-style Popen log file handles:** open the local log file with `open(log_path, "ab")` and pass the SAME file object as both stdout and stderr; close it in the parent after Popen (the child keeps its own descriptor). Leaving it open in the parent leaks an fd per start.
11. **Validator does local `os.path.isfile` checks** on `server_bin()`/`model_path()` (validator.py:169-186) — meaningless for remote paths. Remote aliases skip local file checks (Task 5) with an informational note instead of a bogus warning.
12. **`opencode.json` baseURL for remote aliases** must use the Tailscale IP, not the bind host: `build_opencode_config` (adapters/opencode.py:77-96) currently does `http://{host}:{port}/v1` which would emit `http://0.0.0.0:8090/v1`. Task 6 fixes the connect-host derivation.
13. **YAML bool trap:** `flash_attn: on` unquoted becomes Python `True` → `--flash-attn True` argv. The new alias MUST quote `"on"`/`"off"` (see PLAN-config-schema-doctor edge case 2 — same trap, independently required here).

---

### Task 1: TDD — `run_remote` seam + remote-aware construction

**Files:**
- Test: Create `/home/svend/model-allocator/tests/test_remote_llamacpp.py`
- Modify: `/home/svend/model-allocator/src/model_allocator/adapters/llama_cpp.py` — module imports (lines 1-14), `__init__` (lines 22-29), `_resolve_port` (lines 35-41), new module-level `run_remote`.

**Interfaces:**
- `run_remote(host: str, command: str, timeout: int = 15) -> dict` — module-level (patchable): returns `{"ok": bool, "returncode": int | None, "stdout": str, "stderr": str, "error": str | None}`. Never raises.
- `LlamaCppAdapter.__init__` gains derived attrs: `self.ssh_host: str`, `self.remote: bool`, `self.remote_workdir: str`, `self.connect_host: str`, `self.log_file: str` (local mode), `self.remote_log: str` (remote mode).
- Remote without explicit port → `LlamaCppAdapterError` at construction.

- [ ] Step 1: Create `tests/test_remote_llamacpp.py` (fixture derived from tests/test_v2.py:465-509, plus the live svend3060 params):

```python
"""Tests for SSH-based remote llama.cpp lifecycle (PLAN-remote-llamacpp-lifecycle)."""

from __future__ import annotations

import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_allocator.adapters import llama_cpp as llama_cpp_adapter


def _remote_resolved(**overrides):
    """svend3060-shaped resolved dict (mirrors tests/test_v2.py:465-509 fixture)."""
    resolved = {
        "alias": "svend3060-qwen36-35b",
        "backend": "llama_cpp",
        "ssh_host": "svend@100.73.166.28",
        "remote_workdir": "/home/svend/llama-cpp-turboquant",
        "server_bin_path": "/home/svend/llama-cpp-turboquant/build/bin/llama-server",
        "model_path": "/home/svend/models/qwen36-35b-mxfp4/Qwen3.6-35B-A3B-MXFP4_MOE.gguf",
        "context": 262144,
        "port": 8090,
        "host": "0.0.0.0",
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
    }
    resolved.update(overrides)
    return resolved


def _ssh_result(stdout="", rc=0, error=None):
    return {"ok": rc == 0 and error is None, "returncode": None if error else rc,
            "stdout": stdout, "stderr": "", "error": error}


class TestRemoteConstruction(unittest.TestCase):
    def test_remote_flag_and_connect_host_parsed_from_ssh_host(self):
        adapter = llama_cpp_adapter.LlamaCppAdapter(_remote_resolved())
        self.assertTrue(adapter.remote)
        self.assertEqual(adapter.connect_host, "100.73.166.28")
        self.assertEqual(adapter.port, 8090)

    def test_remote_requires_explicit_port_no_local_free_port_scan(self):
        resolved = _remote_resolved()
        del resolved["port"]
        with self.assertRaises(llama_cpp_adapter.LlamaCppAdapterError) as ctx:
            llama_cpp_adapter.LlamaCppAdapter(resolved)
        self.assertIn("port", str(ctx.exception))

    def test_local_alias_unaffected(self):
        adapter = llama_cpp_adapter.LlamaCppAdapter(
            {"alias": "x", "context": 4096, "default_port": 8080})
        self.assertFalse(adapter.remote)
        self.assertEqual(adapter.connect_host, "127.0.0.1")


class TestRunRemote(unittest.TestCase):
    def test_run_remote_argv_shape(self):
        captured = {}

        class FakeProc:
            returncode = 0
            stdout = "12345\n"
            stderr = ""

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return FakeProc()

        with patch.object(llama_cpp_adapter.subprocess, "run", side_effect=fake_run):
            result = llama_cpp_adapter.run_remote("svend@100.73.166.28", "echo hi", timeout=7)
        self.assertTrue(result["ok"])
        self.assertEqual(
            captured["argv"],
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "svend@100.73.166.28", "echo hi"])
        self.assertEqual(captured["kwargs"]["timeout"], 7)

    def test_run_remote_timeout_is_structured_not_raised(self):
        import subprocess as sp

        def raise_timeout(argv, **kwargs):
            raise sp.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

        with patch.object(llama_cpp_adapter.subprocess, "run", side_effect=raise_timeout):
            result = llama_cpp_adapter.run_remote("svend@100.73.166.28", "sleep 99", timeout=1)
        self.assertFalse(result["ok"])
        self.assertIn("timed out", result["error"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] Step 2: Run and confirm failures (no `remote` attr, no `run_remote`):
```bash
cd /home/svend/model-allocator && python3 -m pytest tests/test_remote_llamacpp.py -v
```

- [ ] Step 3: In `llama_cpp.py`, add `import shlex` and `import json` to the imports (lines 3-13 currently import os/shutil/signal/socket/subprocess/tempfile/time/urllib.request/pathlib/typing), then add the seam right after the exception class (line 19):

```python
SSH_BASE = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def run_remote(host: str, command: str, timeout: int = 15) -> dict:
    """Run one command on *host* over ssh. Structured result, never raises.

    Quoting contract: *command* is passed as a single argv element with
    shell=False, so no LOCAL shell parses it — exactly one shell (the remote
    login shell) does. Callers therefore quote only for the remote shell
    (shlex.quote / shlex.join).
    """
    argv = SSH_BASE + [host, command]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "",
                "error": f"ssh to {host} timed out after {timeout}s"}
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "",
                "error": f"ssh to {host} failed: {exc}"}
```

- [ ] Step 4: Replace `__init__` (lines 22-29) and `_resolve_port` (lines 35-41):

```python
    def __init__(self, resolved: dict, state_dir: str | None = None):
        self.resolved = resolved
        self.alias = resolved.get("alias", "llama")
        self.context = resolved.get("context")
        self.ssh_host = resolved.get("ssh_host") or ""
        self.remote = bool(self.ssh_host)
        self.remote_workdir = resolved.get("remote_workdir") or ""
        self.port = self._resolve_port()
        self.host = resolved.get("host", "127.0.0.1")
        # Bind host != probe host: remote servers bind 0.0.0.0 but are
        # probed via the ssh target IP; local 0.0.0.0 is probed via loopback.
        if self.remote:
            self.connect_host = self.ssh_host.split("@")[-1]
        else:
            self.connect_host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        self.state_dir = state_dir or self._default_state_dir()
        self.pid_file = os.path.join(self.state_dir, f"model-allocator-{self.alias}-{self.port}.pid")
        self.log_file = os.path.join(self.state_dir, f"model-allocator-{self.alias}-{self.port}.log")
        self.remote_log = f"{self.remote_workdir}/logs/{self.alias}.log" if self.remote else ""

    def _resolve_port(self) -> int:
        configured = self.resolved.get("port")
        if configured is None:
            configured = self.resolved.get("default_port")
        if configured is None:
            if self.remote:
                # _find_free_port binds a LOCAL socket — meaningless for a
                # remote host. Remote profiles must declare the port.
                raise LlamaCppAdapterError(
                    f"remote llama.cpp alias '{self.alias}' requires an explicit "
                    "port (or default_port) — local free-port scanning cannot "
                    "see the remote host")
            return self._find_free_port()
        return int(configured)
```

- [ ] Step 5: Verify:
```bash
python3 -m py_compile src/model_allocator/adapters/llama_cpp.py && python3 -m pytest tests/test_remote_llamacpp.py tests/test_v2.py -v 2>&1 | tail -5 && python3 -m pytest 2>&1 | tail -1
```
Expected: 6/6 new pass; existing `TestLlamaCppAdapter` (test_v2.py:253-300) and `TestResolverPreservesAliasFields` untouched and green; full suite `101 passed`.

---

### Task 2: Remote-aware `server_bin`/`model_path`, PID-record reader, `unload` fix

**Files:**
- Test: append to `/home/svend/model-allocator/tests/test_remote_llamacpp.py`.
- Modify: `/home/svend/model-allocator/src/model_allocator/adapters/llama_cpp.py` — `server_bin` (lines 49-62), new `_read_pid_record`/`_write_pid_record`, `unload` (lines 210-211).

**Interfaces:**
- `server_bin()` precedence: `server_bin_path` config field (used verbatim; local existence checks only when NOT remote) > `server_bin_env`-named env var > `LLAMA_SERVER_BIN` env > `"llama-server"` on PATH.
- `_write_pid_record(pid: int) -> None`: local → bare int text (unchanged format); remote → JSON `{"pid": pid, "remote": true, "ssh_host": self.ssh_host}`.
- `_read_pid_record() -> int | None`: tries `int()` first, then JSON `["pid"]`; `None` on missing/garbage.
- `unload(self, timeout: int = 30) -> dict` — fixes the latent cli.py:256 TypeError.

- [ ] Step 1: Append failing tests:

```python
class TestPidRecordAndServerBin(unittest.TestCase):
    def test_pid_record_roundtrip_local_is_bare_int(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = llama_cpp_adapter.LlamaCppAdapter(
                {"alias": "x", "context": 4096, "default_port": 8080}, state_dir=tmp)
            adapter._write_pid_record(4242)
            self.assertEqual(Path(adapter.pid_file).read_text(encoding="utf-8").strip(), "4242")
            self.assertEqual(adapter._read_pid_record(), 4242)

    def test_pid_record_roundtrip_remote_is_annotated_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = llama_cpp_adapter.LlamaCppAdapter(_remote_resolved(), state_dir=tmp)
            adapter._write_pid_record(31337)
            data = json.loads(Path(adapter.pid_file).read_text(encoding="utf-8"))
            self.assertEqual(data, {"pid": 31337, "remote": True,
                                    "ssh_host": "svend@100.73.166.28"})
            self.assertEqual(adapter._read_pid_record(), 31337)

    def test_read_pid_record_missing_file_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = llama_cpp_adapter.LlamaCppAdapter(_remote_resolved(), state_dir=tmp)
            self.assertIsNone(adapter._read_pid_record())

    def test_server_bin_path_field_used_verbatim_for_remote(self):
        adapter = llama_cpp_adapter.LlamaCppAdapter(_remote_resolved())
        # remote: no local isfile/which check may run
        self.assertEqual(adapter.server_bin(),
                         "/home/svend/llama-cpp-turboquant/build/bin/llama-server")

    def test_unload_accepts_timeout_kwarg(self):
        # regression: cli.py cmd_unload calls adapter.unload(timeout=args.timeout)
        with tempfile.TemporaryDirectory() as tmp:
            adapter = llama_cpp_adapter.LlamaCppAdapter(
                {"alias": "x", "context": 4096, "default_port": 8080}, state_dir=tmp)
            result = adapter.unload(timeout=1)   # no PID file -> already stopped
            self.assertTrue(result["stopped"])
```

- [ ] Step 2: Run — failures: no `_write_pid_record`, `server_bin` raises (local isfile check on a remote path), `unload() got an unexpected keyword argument`.

- [ ] Step 3: Replace `server_bin` (lines 49-62):

```python
    def server_bin(self) -> str:
        """Resolve llama-server binary.

        Precedence: server_bin_path config field > env var named by
        server_bin_env > LLAMA_SERVER_BIN env > "llama-server" on PATH.
        Remote mode uses the value verbatim — local isfile/which checks are
        meaningless for a path on another host (existence is verified over
        ssh at start time via the launch itself + log tail on failure).
        """
        binary = self.resolved.get("server_bin_path", "")
        if not binary:
            bin_env = self.resolved.get("server_bin_env")
            binary = os.environ.get(bin_env, "") if bin_env else ""
        if not binary:
            binary = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
        if self.remote:
            return binary
        if os.path.isabs(binary):
            if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
                raise LlamaCppAdapterError(f"llama-server binary not found: {binary}")
            return binary
        resolved = shutil.which(binary)
        if not resolved:
            raise LlamaCppAdapterError(f"llama-server binary not found on PATH: {binary}")
        return resolved
```

(`model_path()` at lines 64-74 needs NO change — it only assembles a string; existence checks live in the validator, made remote-aware in Task 5.)

- [ ] Step 4: Add the PID-record helpers after `_find_free_port` and replace `unload` (lines 210-211):

```python
    def _write_pid_record(self, pid: int) -> None:
        if self.remote:
            payload = {"pid": pid, "remote": True, "ssh_host": self.ssh_host}
            Path(self.pid_file).write_text(json.dumps(payload), encoding="utf-8")
        else:
            # Local format stays a bare int for backward compatibility.
            Path(self.pid_file).write_text(str(pid), encoding="utf-8")

    def _read_pid_record(self) -> int | None:
        try:
            text = Path(self.pid_file).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return int(json.loads(text)["pid"])
        except (ValueError, KeyError, TypeError):
            return None
```

```python
    def unload(self, timeout: int = 30) -> dict:
        return self.stop(timeout=timeout)
```

- [ ] Step 5: Verify:
```bash
python3 -m py_compile src/model_allocator/adapters/llama_cpp.py && python3 -m pytest tests/test_remote_llamacpp.py -v && python3 -m pytest 2>&1 | tail -1
```
Expected: full suite `106 passed`.

---

### Task 3: Remote start (quoting, nohup PID capture, port-busy refuse, health poll) + local log files

**Files:**
- Test: append to `/home/svend/model-allocator/tests/test_remote_llamacpp.py`.
- Modify: `/home/svend/model-allocator/src/model_allocator/adapters/llama_cpp.py` — `start` (lines 111-141) split into local/remote paths; new `_remote_port_busy`, `_remote_start_command`, `_health_ok`, `_tail_local_log`, `_tail_remote_log`.

**Interfaces:**
- `_remote_start_command(argv: list[str]) -> str` — pure function of the adapter; returns exactly:
  `mkdir -p <q(logs_dir)> && cd <q(workdir)> && nohup <shlex.join(argv)> > <q(remote_log)> 2>&1 & echo $!`
- `start(timeout: int = 120) -> dict` — local: `{"started", "error", "pid", "port", "log"}` (+ `"log_tail"` on failure); remote: same plus `"mode": "remote"`; port busy → `{"started": False, "error": "port busy — ..."}`.

- [ ] Step 1: Append failing tests:

```python
class TestRemoteStart(unittest.TestCase):
    def _adapter(self, tmp):
        return llama_cpp_adapter.LlamaCppAdapter(_remote_resolved(), state_dir=tmp)

    def test_remote_start_command_quoting_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp)
            argv = ["/bin/llama-server", "--model", "/models/my model.gguf",
                    "--port", "8090"]
            cmd = adapter._remote_start_command(argv)
        self.assertEqual(
            cmd,
            "mkdir -p /home/svend/llama-cpp-turboquant/logs && "
            "cd /home/svend/llama-cpp-turboquant && "
            "nohup /bin/llama-server --model '/models/my model.gguf' --port 8090 "
            "> /home/svend/llama-cpp-turboquant/logs/svend3060-qwen36-35b.log 2>&1 "
            "& echo $!")

    def test_remote_start_captures_pid_and_polls_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp)
            calls = []

            def fake_remote(host, command, timeout=15):
                calls.append(command)
                if command.startswith("ss -tlnp"):
                    return _ssh_result(stdout="", rc=1)      # grep no match: free
                if "echo $!" in command:
                    return _ssh_result(stdout="54321\n")
                return _ssh_result()

            with patch.object(llama_cpp_adapter, "run_remote", side_effect=fake_remote), \
                 patch.object(adapter, "_health_ok", return_value=True):
                result = adapter.start(timeout=5)
        self.assertTrue(result["started"])
        self.assertEqual(result["pid"], 54321)
        self.assertEqual(result["mode"], "remote")
        self.assertEqual(adapter._read_pid_record(), 54321)
        self.assertTrue(any(c.startswith("ss -tlnp") for c in calls))

    def test_remote_start_refuses_busy_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp)
            busy_line = ('LISTEN 0 512 127.0.0.1:8090 0.0.0.0:* '
                         'users:(("llama-server",pid=42035,fd=16))')

            def fake_remote(host, command, timeout=15):
                if command.startswith("ss -tlnp"):
                    return _ssh_result(stdout=busy_line, rc=0)
                raise AssertionError(f"must not launch when port busy: {command}")

            with patch.object(llama_cpp_adapter, "run_remote", side_effect=fake_remote):
                result = adapter.start(timeout=5)
        self.assertFalse(result["started"])
        self.assertIn("port busy", result["error"])

    def test_remote_start_health_timeout_reports_log_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp)

            def fake_remote(host, command, timeout=15):
                if command.startswith("ss -tlnp"):
                    return _ssh_result(stdout="", rc=1)
                if "echo $!" in command:
                    return _ssh_result(stdout="777\n")
                if command.startswith("tail "):
                    return _ssh_result(stdout="gguf load error: file not found\n")
                if command.startswith("kill "):
                    return _ssh_result()
                return _ssh_result()

            with patch.object(llama_cpp_adapter, "run_remote", side_effect=fake_remote), \
                 patch.object(adapter, "_health_ok", return_value=False):
                result = adapter.start(timeout=0)
        self.assertFalse(result["started"])
        self.assertIn("gguf load error", result.get("log_tail", ""))

    def test_remote_start_ssh_unreachable_is_structured(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter(tmp)
            with patch.object(llama_cpp_adapter, "run_remote",
                              return_value=_ssh_result(error="ssh to h timed out after 15s")):
                result = adapter.start(timeout=5)
        self.assertFalse(result["started"])
        self.assertIn("timed out", result["error"])


class TestLocalStartLogging(unittest.TestCase):
    def test_local_start_failure_includes_log_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = {"alias": "x", "context": 4096, "default_port": 8081,
                        "model_path": "/models/test.gguf"}
            adapter = llama_cpp_adapter.LlamaCppAdapter(resolved, state_dir=tmp)

            class FakeProc:
                pid = 999

                def poll(self):
                    return 1  # exited immediately

            def fake_popen(argv, stdout=None, stderr=None, start_new_session=True):
                stdout.write(b"error: model file /models/test.gguf not found\n")
                return FakeProc()

            with patch.object(adapter, "server_bin", return_value="/bin/llama-server"), \
                 patch.object(llama_cpp_adapter.subprocess, "Popen", side_effect=fake_popen):
                result = adapter.start(timeout=5)
        self.assertFalse(result["started"])
        self.assertIn("model file /models/test.gguf not found", result.get("log_tail", ""))
        self.assertTrue(Path(adapter.log_file).exists())
```

- [ ] Step 2: Run — failures (no `_remote_start_command`, DEVNULL still used, etc.).

- [ ] Step 3: Add the helpers to the adapter (place after `_build_argv`):

```python
    # ── Remote helpers (PLAN-remote-llamacpp-lifecycle) ──────────

    def _remote_logs_dir(self) -> str:
        return f"{self.remote_workdir}/logs"

    def _remote_start_command(self, argv: list[str]) -> str:
        """One remote-shell command: mkdir logs, cd, nohup-launch, echo PID.

        $! is the PID of the backgrounded nohup process; nohup exec()s the
        server (no wrapper shell survives), so $! IS llama-server's PID.
        Quoted ONLY for the remote shell — run_remote passes this string as
        a single ssh argv element with shell=False locally.
        """
        return (
            f"mkdir -p {shlex.quote(self._remote_logs_dir())} && "
            f"cd {shlex.quote(self.remote_workdir)} && "
            f"nohup {shlex.join(argv)} > {shlex.quote(self.remote_log)} 2>&1 "
            f"& echo $!"
        )

    def _remote_port_busy(self) -> dict:
        """Check the remote port before starting. ss primary, /health fallback."""
        check = run_remote(self.ssh_host, f"ss -tlnp | grep ':{self.port} '")
        if check["error"]:
            return {"busy": None, "error": check["error"]}
        if check["ok"] and check["stdout"].strip():
            return {"busy": True, "detail": check["stdout"].strip().splitlines()[0]}
        if check["returncode"] == 1:          # grep: no match
            return {"busy": False}
        # ss missing/unusable -> fall back to a health probe
        return {"busy": self._health_ok(), "detail": "health-probe fallback"}

    def _health_ok(self, timeout: int = 3) -> bool:
        try:
            urllib.request.urlopen(
                f"http://{self.connect_host}:{self.port}/health", timeout=timeout)
            return True
        except Exception:
            return False

    def _tail_local_log(self, lines: int = 40) -> str:
        try:
            content = Path(self.log_file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(content.splitlines()[-lines:])

    def _tail_remote_log(self, lines: int = 40) -> str:
        result = run_remote(self.ssh_host,
                            f"tail -n {lines} {shlex.quote(self.remote_log)}")
        return result["stdout"] if result["ok"] else (result["error"] or result["stderr"])
```

- [ ] Step 4: Replace `start` (lines 111-141) with the dual-mode version:

```python
    def start(self, timeout: int = 120) -> dict:
        try:
            argv = self._build_argv()
        except LlamaCppAdapterError as exc:
            return {"started": False, "error": str(exc)}
        os.makedirs(self.state_dir, exist_ok=True)
        if self.remote:
            return self._start_remote(argv, timeout)
        return self._start_local(argv, timeout)

    def _start_local(self, argv: list[str], timeout: int) -> dict:
        try:
            log_handle = open(self.log_file, "ab")
        except OSError as exc:
            return {"started": False, "error": f"Cannot open log file {self.log_file}: {exc}"}
        try:
            try:
                process = subprocess.Popen(
                    argv,
                    stdout=log_handle,
                    stderr=log_handle,
                    start_new_session=True,
                )
            except Exception as exc:
                return {"started": False, "error": f"Failed to start llama-server: {exc}"}
        finally:
            log_handle.close()   # child keeps its own descriptor

        self._write_pid_record(process.pid)
        start_ts = time.time()
        while time.time() - start_ts < timeout:
            if process.poll() is not None:
                return {"started": False,
                        "error": "llama-server exited early",
                        "log": self.log_file,
                        "log_tail": self._tail_local_log()}
            status = self.status(use_pid=process.pid)
            if status["running"]:
                return {"started": True, "error": None, "pid": process.pid,
                        "port": self.port, "log": self.log_file}
            time.sleep(0.5)

        self._kill_pid(process.pid, timeout=10)
        return {"started": False,
                "error": f"llama-server health endpoint did not become ready within {timeout}s",
                "log": self.log_file,
                "log_tail": self._tail_local_log()}

    def _start_remote(self, argv: list[str], timeout: int) -> dict:
        # Stale PID file: if the recorded process is dead, clean up and go on.
        stale_pid = self._read_pid_record()
        if stale_pid is not None:
            alive = run_remote(self.ssh_host, f"kill -0 {int(stale_pid)}")
            if not alive["ok"]:
                try:
                    os.unlink(self.pid_file)
                except OSError:
                    pass

        busy = self._remote_port_busy()
        if busy.get("error"):
            return {"started": False, "mode": "remote",
                    "error": f"remote pre-check failed: {busy['error']}"}
        if busy.get("busy"):
            return {"started": False, "mode": "remote",
                    "error": (f"port busy on {self.connect_host}:{self.port} — "
                              f"adopt the running server or choose another port "
                              f"({busy.get('detail', '')})")}

        launch = run_remote(self.ssh_host, self._remote_start_command(argv), timeout=30)
        if launch["error"] or not launch["ok"]:
            return {"started": False, "mode": "remote",
                    "error": launch["error"] or f"remote launch failed: {launch['stderr'].strip()}"}
        try:
            pid = int(launch["stdout"].strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"started": False, "mode": "remote",
                    "error": f"could not parse remote PID from: {launch['stdout']!r}"}
        self._write_pid_record(pid)

        start_ts = time.time()
        while True:
            if self._health_ok():
                return {"started": True, "error": None, "pid": pid,
                        "port": self.port, "mode": "remote", "log": self.remote_log}
            if time.time() - start_ts >= timeout:
                break
            time.sleep(1.0)

        tail = self._tail_remote_log()
        run_remote(self.ssh_host, f"kill {pid}")   # best-effort cleanup
        return {"started": False, "mode": "remote", "pid": pid,
                "error": f"remote llama-server health did not become ready within {timeout}s",
                "log": self.remote_log, "log_tail": tail}
```

- [ ] Step 5: Update the two existing local-start assumptions: `status`/`stop` still read the PID file — replace their raw `int(Path(self.pid_file).read_text(...))` parsing (lines 147 and 199) with `self._read_pid_record()` (returning the same not-found behavior: `status` → `{"running": False, "error": "No PID file", "pid": None}` when `None`; `stop` → `{"stopped": True, "error": None}` when `None`).

- [ ] Step 6: Verify:
```bash
python3 -m py_compile src/model_allocator/adapters/llama_cpp.py && python3 -m pytest tests/test_remote_llamacpp.py -v && python3 -m pytest 2>&1 | tail -1
```
Expected: full suite `112 passed` (106 + 6 new). Note `test_stop_removes_pid_file` (test_v2.py:290-300) must still pass — the bare-int format is preserved for local.

---

### Task 4: Remote `status` and `stop` (TERM→KILL escalation, unreachable envelope)

**Files:**
- Test: append to `/home/svend/model-allocator/tests/test_remote_llamacpp.py`.
- Modify: `/home/svend/model-allocator/src/model_allocator/adapters/llama_cpp.py` — `status` (lines 143-174), `stop` (lines 197-208).

**Interfaces:**
- `status(use_pid=None)` remote: `{"running", "alive", "healthy", "pid", "port", "mode": "remote", "error"}`; ssh unreachable → additionally `"status": "unreachable"`, `running: False`, no exception.
- `stop(timeout=30)` remote: `ssh kill <pid>` → poll `kill -0` (1s interval, bounded by timeout) → `kill -9` fallback → remove PID file. ssh unreachable → `{"stopped": False, "status": "unreachable", "error": ...}` (PID file KEPT — the server may still be running).

- [ ] Step 1: Append failing tests:

```python
class TestRemoteStatusStop(unittest.TestCase):
    def _adapter_with_pid(self, tmp, pid=777):
        adapter = llama_cpp_adapter.LlamaCppAdapter(_remote_resolved(), state_dir=tmp)
        adapter._write_pid_record(pid)
        return adapter

    def test_remote_status_alive_and_healthy(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter_with_pid(tmp)
            with patch.object(llama_cpp_adapter, "run_remote",
                              return_value=_ssh_result()), \
                 patch.object(adapter, "_health_ok", return_value=True):
                status = adapter.status()
        self.assertTrue(status["running"])
        self.assertEqual(status["mode"], "remote")
        self.assertEqual(status["pid"], 777)

    def test_remote_status_ssh_unreachable_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter_with_pid(tmp)
            with patch.object(llama_cpp_adapter, "run_remote",
                              return_value=_ssh_result(error="ssh to h timed out after 15s")), \
                 patch.object(adapter, "_health_ok", return_value=False):
                status = adapter.status()
        self.assertFalse(status["running"])
        self.assertEqual(status["status"], "unreachable")
        self.assertIn("timed out", status["error"])

    def test_remote_stop_term_then_kill_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter_with_pid(tmp)
            commands = []

            def fake_remote(host, command, timeout=15):
                commands.append(command)
                if command == "kill 777":
                    return _ssh_result()
                if command == "kill -0 777":
                    return _ssh_result()          # still alive every poll
                if command == "kill -9 777":
                    return _ssh_result()
                return _ssh_result()

            with patch.object(llama_cpp_adapter, "run_remote", side_effect=fake_remote), \
                 patch.object(llama_cpp_adapter.time, "sleep"):
                result = adapter.stop(timeout=2)
        self.assertTrue(result["stopped"])
        self.assertIn("kill 777", commands)
        self.assertIn("kill -9 777", commands)
        self.assertFalse(Path(adapter.pid_file).exists())

    def test_remote_stop_clean_exit_no_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter_with_pid(tmp)
            commands = []

            def fake_remote(host, command, timeout=15):
                commands.append(command)
                if command == "kill -0 777":
                    return _ssh_result(rc=1)      # gone after TERM
                return _ssh_result()

            with patch.object(llama_cpp_adapter, "run_remote", side_effect=fake_remote):
                result = adapter.stop(timeout=5)
        self.assertTrue(result["stopped"])
        self.assertNotIn("kill -9 777", commands)

    def test_remote_stop_ssh_unreachable_keeps_pid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = self._adapter_with_pid(tmp)
            with patch.object(llama_cpp_adapter, "run_remote",
                              return_value=_ssh_result(error="ssh to h failed: no route")):
                result = adapter.stop(timeout=2)
        self.assertFalse(result["stopped"])
        self.assertEqual(result["status"], "unreachable")
        self.assertTrue(Path(adapter.pid_file).exists())
```

- [ ] Step 2: Run — failures (local status/stop paths don't know remote).

- [ ] Step 3: Replace `status` (lines 143-174):

```python
    def status(self, use_pid: int | None = None) -> dict:
        pid = use_pid if use_pid is not None else self._read_pid_record()
        if pid is None:
            return {"running": False, "error": "No PID file", "pid": None}

        if self.remote:
            probe = run_remote(self.ssh_host, f"kill -0 {int(pid)}")
            if probe["error"]:
                return {"running": False, "alive": None, "healthy": self._health_ok(),
                        "pid": pid, "port": self.port, "mode": "remote",
                        "status": "unreachable", "error": probe["error"]}
            alive = probe["ok"]
            healthy = self._health_ok()
            return {"running": alive and healthy, "alive": alive, "healthy": healthy,
                    "pid": pid, "port": self.port, "mode": "remote",
                    "error": None if (alive and healthy) else
                    ("process dead" if not alive else "health probe failed")}

        try:
            os.kill(pid, 0)
            alive = True
        except (OSError, ProcessLookupError):
            alive = False

        health_url = f"http://{self.connect_host}:{self.port}/health"
        try:
            urllib.request.urlopen(health_url, timeout=2)
            healthy = True
            health_error = None
        except Exception as exc:
            healthy = False
            health_error = str(exc)

        running = alive and healthy
        return {
            "running": running,
            "alive": alive,
            "healthy": healthy,
            "pid": pid,
            "port": self.port,
            "error": health_error,
        }
```

- [ ] Step 4: Replace `stop` (lines 197-208):

```python
    def stop(self, timeout: int = 30) -> dict:
        pid = self._read_pid_record()
        if pid is None:
            return {"stopped": True, "error": None}

        if self.remote:
            result = self._stop_remote(pid, timeout)
            if result["stopped"]:
                try:
                    os.unlink(self.pid_file)
                except OSError:
                    pass
            return result

        result = self._kill_pid(pid, timeout=timeout)
        try:
            os.unlink(self.pid_file)
        except OSError:
            pass
        return result

    def _stop_remote(self, pid: int, timeout: int) -> dict:
        term = run_remote(self.ssh_host, f"kill {int(pid)}")
        if term["error"]:
            # Unreachable: the server may still run — keep the PID file.
            return {"stopped": False, "status": "unreachable", "error": term["error"]}
        # rc != 0 from kill usually means the process is already gone.
        deadline = time.time() + timeout
        while time.time() < deadline:
            probe = run_remote(self.ssh_host, f"kill -0 {int(pid)}")
            if probe["error"]:
                return {"stopped": False, "status": "unreachable", "error": probe["error"]}
            if not probe["ok"]:
                return {"stopped": True, "error": None}
            time.sleep(1.0)
        forced = run_remote(self.ssh_host, f"kill -9 {int(pid)}")
        if forced["error"]:
            return {"stopped": False, "status": "unreachable", "error": forced["error"]}
        return {"stopped": True, "error": None}
```

- [ ] Step 5: Verify:
```bash
python3 -m py_compile src/model_allocator/adapters/llama_cpp.py && python3 -m pytest tests/test_remote_llamacpp.py -v && python3 -m pytest 2>&1 | tail -1
```
Expected: full suite `117 passed`. Confirm `test_stop_removes_pid_file` (test_v2.py) still green (local path behavior identical).

---

### Task 5: Validator + opencode.json remote awareness

**Files:**
- Test: append to `/home/svend/model-allocator/tests/test_remote_llamacpp.py`.
- Modify: `/home/svend/model-allocator/src/model_allocator/validator.py` — `_validate_llama_cpp` (lines 169-186); `/home/svend/model-allocator/src/model_allocator/adapters/opencode.py` — `build_opencode_config` llama_cpp branch (lines 77-96).

**Interfaces:**
- `_validate_llama_cpp`: when `resolved.get("ssh_host")` is set, skip the local `os.path.isfile` checks on `server_bin()`/`model_path()` and append the informational warning `"remote alias — local file checks skipped (verified at start via ssh)"`; the running/NOT_RUNNING status probe (adapter.status()) still runs and is now remote-aware from Task 4.
- `build_opencode_config` llama branch: baseURL host = Tailscale IP for remote, loopback for a local `0.0.0.0` bind.

- [ ] Step 1: Append failing tests:

```python
class TestRemoteValidatorAndOpencodeConfig(unittest.TestCase):
    def test_validator_skips_local_file_checks_for_remote(self):
        from model_allocator.resolver import Resolver
        from model_allocator.validator import Validator
        config = {
            "models": {"svend3060-qwen36-35b": {
                **{k: v for k, v in _remote_resolved().items()
                   if k not in ("alias", "backend", "ssh_host", "remote_workdir",
                                "server_bin_path")},
                "runtime_profile": "remote_llamacpp_svend3060",
                "clients": {"opencode": True},
            }},
            "runtime_profiles": {"remote_llamacpp_svend3060": {
                "backend": "llama_cpp",
                "ssh_host": "svend@100.73.166.28",
                "remote_workdir": "/home/svend/llama-cpp-turboquant",
                "server_bin_path": "/home/svend/llama-cpp-turboquant/build/bin/llama-server",
            }},
            "roles": {},
        }
        v = Validator(resolver=Resolver(config=config))
        with patch.object(llama_cpp_adapter.LlamaCppAdapter, "status",
                          return_value={"running": True, "error": None}):
            result = v.validate("svend3060-qwen36-35b", "opencode")
        joined = " ".join(result["warnings"])
        self.assertNotIn("binary not found", joined)
        self.assertNotIn("Model file not found", joined)
        self.assertIn("local file checks skipped", joined)

    def test_opencode_config_baseurl_uses_tailscale_ip_for_remote(self):
        from model_allocator.adapters import opencode
        cfg = opencode.build_opencode_config(
            {**_remote_resolved(), "opencode_provider_name": "svend3060",
             "opencode_model_id": "qwen36-35b", "real_model": "qwen36-35b"})
        self.assertEqual(cfg["provider"]["svend3060"]["options"]["baseURL"],
                         "http://100.73.166.28:8090/v1")

    def test_opencode_config_local_zero_bind_probes_loopback(self):
        from model_allocator.adapters import opencode
        cfg = opencode.build_opencode_config(
            {"backend": "llama_cpp", "provider": "llama-local",
             "real_model": "m", "opencode_model_id": "m",
             "host": "0.0.0.0", "port": 8082})
        self.assertEqual(cfg["provider"]["llama-local"]["options"]["baseURL"],
                         "http://127.0.0.1:8082/v1")
```

- [ ] Step 2: Run — failures (validator warns `llama-server binary not found: ...` for the remote path; baseURL contains `0.0.0.0`).

- [ ] Step 3: Replace `_validate_llama_cpp` (validator.py:169-186):

```python
    def _validate_llama_cpp(self, resolved: dict, client: str, result: dict) -> None:
        try:
            adapter = llama_cpp_adapter.LlamaCppAdapter(resolved)
            if resolved.get("ssh_host"):
                result["warnings"].append(
                    "remote alias — local file checks skipped (verified at start via ssh)")
            else:
                server_bin = adapter.server_bin()
                if not os.path.isfile(server_bin):
                    result["warnings"].append(f"llama-server binary not found: {server_bin}")
                model_path = adapter.model_path()
                if not os.path.isfile(model_path):
                    result["warnings"].append(f"Model file not found: {model_path}")
            status = adapter.status()
            if status["running"]:
                result["client_support"][client] = "OK"
            else:
                result["warnings"].append(
                    f"llama.cpp server not running on port {adapter.port}: {status['error']}")
                result["client_support"][client] = "NOT_RUNNING"
        except llama_cpp_adapter.LlamaCppAdapterError as exc:
            result["warnings"].append(str(exc))
            result["client_support"][client] = "UNREACHABLE"
```

- [ ] Step 4: In `opencode.py` `build_opencode_config` llama_cpp branch, replace the two lines (80-81)

```python
        host = resolved.get("host", "127.0.0.1")
        port = resolved.get("port", resolved.get("default_port", 8080))
```

with:

```python
        port = resolved.get("port", resolved.get("default_port", 8080))
        ssh_host = resolved.get("ssh_host") or ""
        if ssh_host:
            # Remote server: clients connect via the ssh target IP, never
            # the bind address (which is 0.0.0.0 on the remote host).
            host = ssh_host.split("@")[-1]
        else:
            host = resolved.get("host", "127.0.0.1")
            if host == "0.0.0.0":
                host = "127.0.0.1"
```

- [ ] Step 5: Verify:
```bash
python3 -m py_compile src/model_allocator/validator.py src/model_allocator/adapters/opencode.py && python3 -m pytest 2>&1 | tail -1
```
Expected: full suite `120 passed` (existing `test_render_config_llama_cpp` in test_v2.py uses `port: 8080` with no host → still `http://127.0.0.1:8080/v1`, unchanged).

---

### Task 6: The real svend3060 alias + profile in the live config

**Files:**
- Modify: `/home/svend/model-allocator/runtime_profiles.yaml` (append profile), `/home/svend/model-allocator/models.yaml` (append alias), and mirror both into the `.example.yaml` files with placeholder host/paths.

All values below are live-verified on svend3060 (2026-07-12): binary and model paths exist; the manual server occupies port 8080, so this alias uses **8090**; `flash_attn`/`reasoning` are QUOTED (YAML bool trap); env-refs with defaults keep literal remote paths overridable.

- [ ] Step 1: Append to `runtime_profiles.yaml`:

```yaml
  remote_llamacpp_svend3060:
    backend: llama_cpp
    # svend3060 via Tailscale; ssh key for svend@ is authorized (BatchMode-safe).
    ssh_host: ${SVEND3060_SSH_HOST:-svend@100.73.166.28}
    remote_workdir: ${SVEND3060_LLAMA_DIR:-/home/svend/llama-cpp-turboquant}
    server_bin_path: ${SVEND3060_LLAMA_BIN:-/home/svend/llama-cpp-turboquant/build/bin/llama-server}
    gpu: svend3060
```

- [ ] Step 2: Append to `models.yaml`:

```yaml
  svend3060-qwen36-35b:
    runtime_profile: remote_llamacpp_svend3060
    # Real path on svend3060 (the / symlink is root-owned and fragile).
    model_path: ${SVEND3060_MODEL_GGUF:-/home/svend/models/qwen36-35b-mxfp4/Qwen3.6-35B-A3B-MXFP4_MOE.gguf}
    real_model: qwen36-35b-mxfp4
    opencode_provider_name: svend3060
    opencode_model_id: qwen36-35b-mxfp4
    context: 262144
    # Live-verified server params (MoE build: --n-cpu-moe, NOT -ngl):
    parallel: 1
    n_cpu_moe: 26
    threads: 12
    batch: 160
    ubatch_size: 128
    cache_type_k: turbo4
    cache_type_v: turbo3
    flash_attn: "on"    # quoted — bare on/off is a YAML boolean
    reasoning: "off"
    no_mmap: true
    # Bind all interfaces on the remote host; probed via the Tailscale IP.
    host: 0.0.0.0
    # 8080 is the user's manually-started server — do not collide.
    port: 8090
    lifecycle_policy: stop_after_step
    clients:
      opencode: true
      claude-code: false
```

- [ ] Step 3: Mirror both blocks into `runtime_profiles.example.yaml` / `models.example.yaml`, replacing the concrete defaults with `user@100.64.0.99`-style placeholders and a comment `# see PLAN-remote-llamacpp-lifecycle.md`.

- [ ] Step 4: Static verification (no server started):
```bash
cd /home/svend/model-allocator && python3 - <<'EOF'
from model_allocator.resolver import Resolver
from model_allocator.adapters.llama_cpp import LlamaCppAdapter
r = Resolver(config_dir=".").resolve_alias("svend3060-qwen36-35b")
a = LlamaCppAdapter(r)
assert a.remote and a.connect_host == "100.73.166.28" and a.port == 8090, (a.remote, a.connect_host, a.port)
print("ARGV:", " ".join(a._build_argv()))
EOF
```
Expected: `ARGV:` line containing `--ctx-size 262144 --host 0.0.0.0 --port 8090 --parallel 1 --n-cpu-moe 26 -t 12 -b 160 --ubatch-size 128 --cache-type-k turbo4 --cache-type-v turbo3 --flash-attn on --reasoning off --no-mmap` and the absolute remote binary/model paths. No `True`/`False` tokens anywhere in the argv.

- [ ] Step 5: Live lifecycle check (network required; svend3060 reachable):
```bash
./scripts/model-allocator start --alias svend3060-qwen36-35b --timeout 300
./scripts/model-allocator status --alias svend3060-qwen36-35b
curl -s http://100.73.166.28:8090/health && echo
./scripts/model-allocator stop --alias svend3060-qwen36-35b
./scripts/model-allocator status --alias svend3060-qwen36-35b
```
Expected: start → `"started": true`, `"mode": "remote"`, numeric `"pid"`; status → `"running": true`; curl → `{"status":"ok"}`; stop → `"stopped": true`; final status → `"running": false, "error": "No PID file"`. Also verify the port-busy refusal against the user's manual server: temporarily set `port: 8080` via env-free test (`python3` snippet constructing the adapter with `port=8080`) and confirm `start` returns `port busy — adopt the running server or choose another port` without killing pid 42035.

- [ ] Step 6: Stage and STOP — await Human commit approval:
```bash
git add src/model_allocator/adapters/llama_cpp.py src/model_allocator/validator.py src/model_allocator/adapters/opencode.py tests/test_remote_llamacpp.py models.yaml runtime_profiles.yaml models.example.yaml runtime_profiles.example.yaml
git status --short
```
Suggested commit message: `[V6] Remote llama.cpp lifecycle — ssh start/status/stop for svend3060, port-busy adopt/refuse, log files + tail-on-failure for local and remote`

---

### Task 7: CROSS-REPO — Father consumer verification (no code change required)

> **Cross-repo verification in `/home/svend/DPMtF-WebUI` — read-only. No output format changed (all new JSON keys are additive), so no Father modification is needed; this task PROVES it. Any change that did become necessary would require Human approval for `routers/` and a Human commit.**

- [ ] Step 1: Confirm the stop path: `scripts/bridgeV002/dispatch.py:261-286` runs `stop --alias <alias>` with an outer `timeout=45`. Remote stop worst case = ssh connect (≤10s) + TERM/poll loop (≤ CLI default 30s) + one escalation call — within an ssh-reachable host this fits 45s; when ssh is UNREACHABLE, `run_remote`'s ConnectTimeout bounds each call to ~10-15s and stop returns a structured failure (exit 1), which dispatch already handles as a WARNING without hanging (dispatch.py:275-278). No change.
- [ ] Step 2: Confirm the status endpoint: `routers/bridge.py` `/allocator/status` (lines 1125-1168) returns the adapter JSON to `static/js/dpmtf-app.js` `updateRuntimeSection` (lines 538-559), which reads only `running`/`pid`/`port` — additive keys (`mode`, `status`) are ignored. No change.
- [ ] Step 3: Confirm the start endpoint timeout: `/allocator/start` (routers/bridge.py:1171-1208) uses a 200s subprocess timeout while the CLI start default is 120s — a remote cold load of a 35B MXFP4 model can exceed 120s. If the LIVE check in Task 6 Step 5 needed `--timeout 300`, record a follow-up for the Human: consider passing `--timeout 180` from the endpoint (one-line change in routers/bridge.py, requires Human approval). Do not change it in this plan.
- [ ] Step 4: Run Father's test suite subset as regression proof:
```bash
cd /home/svend/DPMtF-WebUI && python3 -m pytest tests/test_allocator_config_endpoints.py tests/test_bridge_endpoints.py -q
```
Expected: all pass, zero Father modifications (`git -C /home/svend/DPMtF-WebUI status --short` → clean).

## Acceptance Criteria

1. `cd /home/svend/model-allocator && python3 -m pytest` → `120 passed` (95 baseline + 25 new across Tasks 1-5), 0 failures — with NO network: every ssh interaction in tests goes through the mocked `run_remote` seam.
2. `python3 -m py_compile src/model_allocator/adapters/llama_cpp.py src/model_allocator/validator.py src/model_allocator/adapters/opencode.py` → exit 0.
3. Quoting proof (unit): `python3 -m pytest tests/test_remote_llamacpp.py::TestRemoteStart::test_remote_start_command_quoting_exact -v` → passes, pinning the exact single-shell command string including `nohup ... > ... 2>&1 & echo $!` and `shlex`-quoted spaced path.
4. Task 6 Step 4 static check prints an argv with quoted-string `on`/`off` values and no `True`/`False` tokens.
5. Live (svend3060 reachable): `start`/`status`/`stop` sequence from Task 6 Step 5 behaves as specified, including `curl http://100.73.166.28:8090/health` succeeding from the local machine while the server binds 0.0.0.0 remotely.
6. Port-busy refusal proven live against the manual server on 8080: `"port busy"` error, and `ssh svend@100.73.166.28 'kill -0 42035'` still succeeds afterwards (allocator never killed a server it didn't start; use the current manual-server PID if it differs).
7. Unreachable envelope: with Tailscale down (or `ssh_host` pointed at a black-hole IP via `SVEND3060_SSH_HOST=svend@100.73.166.99` env override), `status --alias svend3060-qwen36-35b` exits 2 (WARNING) printing JSON containing `"status": "unreachable"` — no traceback.
8. `unload --alias <any-llama-alias>` no longer raises TypeError (regression test `test_unload_accepts_timeout_kwarg` green; `./scripts/model-allocator unload --alias svend3060-qwen36-35b` returns JSON, exit 0 when nothing to unload).
9. Local diagnostics: after a failed local start, the JSON contains `"log"` and non-empty `"log_tail"`, and the file `model-allocator-<alias>-<port>.log` exists under `$MODEL_ALLOCATOR_STATE_DIR` (or the system tmp dir).
10. Father untouched: `git -C /home/svend/DPMtF-WebUI status --short` → clean; Father test subset (Task 7 Step 4) passes.
11. Allocator staged files match Task 6 Step 6 exactly; nothing committed without the Human.
