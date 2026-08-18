"""llama.cpp server backend adapter for model-allocator."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


def served_model_id(resolved: dict) -> str:
    """The ONE name a llama.cpp model is served under and requested by.

    Used by this adapter for `--alias` and by the opencode config builder
    for the model id, because the two were previously assembled in
    different places from different key orders: the adapter preferred
    `served_model_name`, the config builder did not know the key existed.
    An alias setting only `served_model_name` got a server serving one name
    and a config requesting another -- a mismatch that shows on the first
    request, after preflight has passed.

    Empty string when nothing names the model; callers emit no flag and
    fall back to their own defaults.
    """
    return str(
        resolved.get("served_model_name")
        or resolved.get("opencode_model_id")
        or resolved.get("real_model")
        or ""
    )


class LlamaCppAdapterError(Exception):
    pass


class LlamaCppAdapter:
    def __init__(self, resolved: dict, state_dir: str | None = None):
        self.resolved = resolved
        self.alias = resolved.get("alias", "llama")
        self.context = resolved.get("context")
        self.port = self._resolve_port()
        self.host = resolved.get("host", "127.0.0.1")
        self.state_dir = state_dir or self._default_state_dir()
        # V6: an instance-bound alias uses an instance-keyed JSON state
        # file (one canonical record per instance). Aliases without
        # ``runtime_instance`` keep the existing per-alias PID file
        # behaviour untouched.
        self.instance_name = resolved.get("runtime_instance") or None
        if self.instance_name:
            self.state_file = os.path.join(
                self.state_dir, f"model-allocator-instance-{self.instance_name}.json"
            )
            self.pid_file = None
        else:
            self.state_file = None
            self.pid_file = os.path.join(
                self.state_dir, f"model-allocator-{self.alias}-{self.port}.pid"
            )

    @staticmethod
    def _default_state_dir() -> str:
        return os.environ.get("MODEL_ALLOCATOR_STATE_DIR", tempfile.gettempdir())

    def _resolve_port(self) -> int:
        configured = self.resolved.get("port")
        if configured is None:
            configured = self.resolved.get("default_port")
        if configured is None:
            return self._find_free_port()
        return int(configured)

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def server_bin(self) -> str:
        """Resolve llama-server binary from config, env-var names, or PATH."""
        # Direct path from config takes precedence
        binary = self.resolved.get("server_bin_path", "")
        if binary:
            if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
                raise LlamaCppAdapterError(f"llama-server binary not found: {binary}")
            return binary
        bin_env = self.resolved.get("server_bin_env")
        binary = os.environ.get(bin_env, "") if bin_env else ""
        if not binary:
            binary = os.environ.get("LLAMA_SERVER_BIN", "llama-server")
        if os.path.isabs(binary):
            if not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
                raise LlamaCppAdapterError(f"llama-server binary not found: {binary}")
            return binary
        resolved = shutil.which(binary)
        if not resolved:
            raise LlamaCppAdapterError(f"llama-server binary not found on PATH: {binary}")
        return resolved

    def model_path(self) -> str:
        path = self.resolved.get("model_path", "")
        if not path:
            model_name = self.resolved.get("model_name", "")
            root_env = self.resolved.get("model_root_env")
            root = os.environ.get(root_env, "") if root_env else ""
            if root and model_name:
                path = f"{root}/{model_name}"
        if not path:
            raise LlamaCppAdapterError("No model_path configured for llama.cpp alias")
        return path

    def _build_argv(self) -> list[str]:
        argv = [
            self.server_bin(),
            "--model", self.model_path(),
            "--ctx-size", str(self.context or 131072),
            "--host", self.host,
            "--port", str(self.port),
        ]
        # Serve under the name the client will ask for.
        #
        # Without --alias, llama.cpp names the model after the GGUF file, so
        # /v1/models returns "qwen2.5-coder-14b-instruct-q4_K_M.gguf" while
        # the opencode config -- built from `real_model` -- tells OpenCode to
        # request "qwen2.5-coder-14b-instruct-q4_K_M". The two are assembled
        # in different places from different sources and had no reason to
        # agree.
        served = served_model_id(self.resolved)
        if served:
            argv += ["--alias", str(served)]
        flags = self.resolved
        if "parallel" in flags:
            argv += ["--parallel", str(flags["parallel"])]
        if "n_cpu_moe" in flags:
            argv += ["--n-cpu-moe", str(flags["n_cpu_moe"])]
        if "threads" in flags:
            argv += ["-t", str(flags["threads"])]
        if "batch" in flags:
            argv += ["-b", str(flags["batch"])]
        if "ubatch_size" in flags:
            argv += ["--ubatch-size", str(flags["ubatch_size"])]
        if "cache_type_k" in flags:
            argv += ["--cache-type-k", str(flags["cache_type_k"])]
        if "cache_type_v" in flags:
            argv += ["--cache-type-v", str(flags["cache_type_v"])]
        if "flash_attn" in flags:
            argv += ["--flash-attn", str(flags["flash_attn"])]
        if "temp" in flags:
            argv += ["--temp", str(flags["temp"])]
        if "reasoning" in flags:
            argv += ["--reasoning", str(flags["reasoning"])]
        if "reasoning_budget" in flags:
            argv += ["--reasoning-budget", str(flags["reasoning_budget"])]
        if flags.get("jinja"):
            argv.append("--jinja")
        if "load_mode" in flags:
            argv += ["--load-mode", str(flags["load_mode"])]
        if flags.get("no_mmap"):
            argv.append("--no-mmap")
        if "gpu_layers" in flags:
            argv += ["--n-gpu-layers", str(flags["gpu_layers"])]
        if "tensor_split" in flags:
            argv += ["--tensor-split", str(flags["tensor_split"])]
        if "spec_type" in flags:
            argv += ["--spec-type", str(flags["spec_type"])]
        if "spec_draft_n_max" in flags:
            argv += ["--spec-draft-n-max", str(flags["spec_draft_n_max"])]
        return argv

    # ─── V6: instance-keyed state file I/O ────────────────────────

    def _read_instance_state(self) -> dict | None:
        """Load the recorded instance state (one canonical record per instance).

        Returns None when the file is absent or unreadable — the caller
        treats that as "no managed runtime exists", the same way a missing
        per-alias pid file is treated.
        """
        if not self.state_file:
            return None
        try:
            with open(self.state_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        if not data.get("instance_name") or data.get("instance_name") != self.instance_name:
            # The state file's instance name must agree with this adapter —
            # a stale file from a renamed instance must not be honoured.
            return None
        return data

    def _write_instance_state(self, pid: int, port: int) -> dict:
        """Write a fresh instance state record (atomic temp + rename)."""
        if not self.state_file:
            return {}
        state = {
            "instance_name": self.instance_name,
            "pid": int(pid),
            "port": int(port),
            "started_by_alias": self.alias,
            "started_at": time.time(),
        }
        os.makedirs(self.state_dir, exist_ok=True)
        tmp_path = self.state_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self.state_file)
        return state

    def _forget_instance_state(self) -> None:
        """Remove the instance state file (best effort)."""
        if not self.state_file:
            return
        try:
            os.unlink(self.state_file)
        except OSError:
            pass

    # ─── start / stop / status / unload ───────────────────────────

    def start(self, timeout: int = 120) -> dict:
        if self.instance_name:
            return self._start_instance(timeout=timeout)
        return self._start_alias(timeout=timeout)

    def _start_alias(self, timeout: int) -> dict:
        """Per-alias start (unchanged pre-V6 behaviour)."""
        try:
            argv = self._build_argv()
        except LlamaCppAdapterError as exc:
            return {"started": False, "error": str(exc)}

        os.makedirs(self.state_dir, exist_ok=True)
        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            return {"started": False, "error": f"Failed to start llama-server: {exc}"}

        Path(self.pid_file).write_text(str(process.pid), encoding="utf-8")

        start_ts = time.time()
        while time.time() - start_ts < timeout:
            if process.poll() is not None:
                return {"started": False, "error": "llama-server exited early"}
            status = self.status(use_pid=process.pid)
            if status["running"]:
                return {"started": True, "error": None, "pid": process.pid, "port": self.port}
            time.sleep(0.5)

        # Timeout: try to stop the half-started process.
        self._kill_pid(process.pid, timeout=10)
        return {"started": False, "error": f"llama-server health endpoint did not become ready within {timeout}s"}

    def _start_instance(self, timeout: int) -> dict:
        """Instance-bound start: record-keyed, start-once / reuse-by-identity.

        Reuse decisions are made ONLY from the recorded instance state —
        never from mere port or model-path equality. A foreign process on
        the instance's configured port is never signalled and never
        adopted.
        """
        try:
            argv = self._build_argv()
        except LlamaCppAdapterError as exc:
            return {"started": False, "error": str(exc)}

        os.makedirs(self.state_dir, exist_ok=True)
        state = self._read_instance_state()

        # Step A — recorded identity: if we previously started this
        # instance and the recorded PID is still alive, wait briefly for
        # the health endpoint and reuse without spawning.
        if state is not None and self._pid_alive(state.get("pid")):
            if self._wait_healthy(timeout=min(timeout, 30)):
                return {
                    "started": True,
                    "reused": True,
                    "pid": state["pid"],
                    "port": state["port"],
                    "instance_name": self.instance_name,
                    "error": None,
                }
            # Recorded PID is alive but the server isn't responding. The
            # recorded PID is the authority — fall through to the
            # ownership check below, which will refuse to spawn while the
            # recorded PID holds the port.

        # Step B-pre — stale state cleanup. If a state record exists but
        # its PID is dead, the record is stale and must NOT influence the
        # ownership decision below — a foreign llama-server on the port
        # now is not ours, and we are not the cause of the dead PID.
        if state is not None and not self._pid_alive(state.get("pid")):
            self._forget_instance_state()
            state = None

        # Step B — ownership check: before we spawn anything we must
        # confirm the port is ours. If the port is open, refuse unless
        # the listener is our recorded PID.
        if self._port_open():
            owners = self._server_pids()
            if state is not None and state.get("pid") in owners:
                # Our recorded PID is the listener; the earlier health
                # wait didn't see it ready in time. Try once more.
                if self._wait_healthy(timeout=min(timeout, 30)):
                    return {
                        "started": True,
                        "reused": True,
                        "pid": state["pid"],
                        "port": state["port"],
                        "instance_name": self.instance_name,
                        "error": None,
                    }
                # Recorded PID is alive but never came up — clean up
                # the stale record and refuse (no kill by port-scan).
                self._kill_pid(state["pid"], timeout=10)
                self._forget_instance_state()
                return {
                    "started": False,
                    "error": (
                        f"recorded PID {state['pid']} for instance "
                        f"'{self.instance_name}' did not become healthy; "
                        f"removed stale state — start again"
                    ),
                }
            if not owners:
                # Something other than llama-server is listening on the
                # configured port. The allocator never adopts foreign
                # processes.
                return {
                    "started": False,
                    "error": (
                        f"port {self.port} for instance "
                        f"'{self.instance_name}' is occupied by an "
                        f"unmanaged process; refusing to start"
                    ),
                }
            # A llama-server we did not record is holding the port.
            return {
                "started": False,
                "error": (
                    f"port {self.port} for instance "
                    f"'{self.instance_name}' is occupied by an "
                    f"unmanaged llama-server (pid={owners[0]}); "
                    f"refusing to adopt — stop it manually if you "
                    f"want the allocator to manage it"
                ),
            }

        # Step C — spawn and record.
        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            return {
                "started": False,
                "error": f"failed to start llama-server for instance "
                         f"'{self.instance_name}': {exc}",
            }

        self._write_instance_state(process.pid, self.port)

        start_ts = time.time()
        while time.time() - start_ts < timeout:
            if process.poll() is not None:
                self._forget_instance_state()
                return {
                    "started": False,
                    "error": "llama-server exited early",
                }
            if self._wait_healthy(timeout=2):
                return {
                    "started": True,
                    "reused": False,
                    "pid": process.pid,
                    "port": self.port,
                    "instance_name": self.instance_name,
                    "error": None,
                }
            time.sleep(0.5)

        # Timeout — clean up.
        self._kill_pid(process.pid, timeout=10)
        self._forget_instance_state()
        return {
            "started": False,
            "error": (
                f"llama-server for instance '{self.instance_name}' did "
                f"not become healthy within {timeout}s"
            ),
        }

    def status(self, use_pid: int | None = None) -> dict:
        if self.instance_name:
            return self._status_instance()
        return self._status_alias(use_pid=use_pid)

    def _status_alias(self, use_pid: int | None = None) -> dict:
        pid = use_pid
        if pid is None:
            try:
                pid = int(Path(self.pid_file).read_text(encoding="utf-8").strip())
            except (FileNotFoundError, ValueError):
                # A lost pid file must not hide a live server — that is how
                # an orphan comes to hold the GPU while status says "stopped".
                owners = self._server_pids()
                if not owners and not self._port_open():
                    return {"running": False, "error": "No PID file", "pid": None}
                pid = owners[0] if owners else None

        alive = False
        if pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False

        healthy, health_error = self._check_health()
        # The health endpoint answering IS the server being up. llama-server
        # returns 503 while loading weights, so this still means "ready",
        # which is what start() waits on.
        running = healthy
        return {
            "running": running,
            "alive": alive,
            "healthy": healthy,
            "pid": pid,
            "port": self.port,
            "error": health_error,
        }

    def _status_instance(self) -> dict:
        """Status keyed by recorded instance identity.

        The reported ``instance_name``, ``pid`` and ``port`` are the same
        for every sharing alias and for ``status-instance --name`` —
        that is the contract.
        """
        state = self._read_instance_state()
        pid = state.get("pid") if state else None
        alive = bool(pid and self._pid_alive(pid))
        healthy, health_error = self._check_health()
        running = bool(state) and alive and healthy
        return {
            "running": running,
            "alive": alive,
            "healthy": healthy,
            "pid": pid,
            "port": self.port,
            "instance_name": self.instance_name,
            "instance_managed": state is not None,
            "error": health_error,
        }

    def _check_health(self) -> tuple[bool, str | None]:
        health_url = f"http://{self.host}:{self.port}/health"
        try:
            urllib.request.urlopen(health_url, timeout=2)
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _wait_healthy(self, timeout: int) -> bool:
        deadline = time.time() + max(0, int(timeout))
        while time.time() < deadline:
            ok, _ = self._check_health()
            if ok:
                return True
            time.sleep(0.2)
        return False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ProcessLookupError, TypeError, ValueError):
            return False

    @staticmethod
    def _kill_pid(pid: int, timeout: int = 30) -> dict:
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            return {"stopped": True, "error": None}

        start_ts = time.time()
        while time.time() - start_ts < timeout:
            try:
                os.kill(pid, 0)
            except (OSError, ProcessLookupError):
                return {"stopped": True, "error": None}
            time.sleep(0.2)

        # SIGTERM didn't work — escalate to SIGKILL
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            return {"stopped": True, "error": None}

        # Verify SIGKILL actually terminated the process
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            return {"stopped": False, "error": f"PID {pid} survived SIGKILL"}
        except (OSError, ProcessLookupError):
            return {"stopped": True, "error": None}

    def _port_open(self) -> bool:
        """Whether anything is still listening on the server port.

        llama-server does not hand off to another process the way SGLang
        does, so its recorded pid is normally accurate — but a pid is still
        only bookkeeping. The port is the fact.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                return sock.connect_ex((self.host, self.port)) == 0
        except OSError:
            return False

    def _server_pids(self) -> list[int]:
        """PIDs running llama-server on this port, found by scanning /proc."""
        pids: list[int] = []
        port_token = str(self.port)
        try:
            entries = os.listdir("/proc")
        except OSError:
            return []
        for entry in entries:
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as handle:
                    parts = handle.read().decode("utf-8", "replace").split("\0")
            except OSError:
                continue
            if "llama-server" in " ".join(parts) and port_token in parts:
                pids.append(int(entry))
        return pids

    def _forget_pid_file(self) -> None:
        try:
            os.unlink(self.pid_file)
        except OSError:
            pass

    def stop(self, timeout: int = 30) -> dict:
        if self.instance_name:
            return self._stop_instance_alias_level()
        return self._stop_alias(timeout=timeout)

    def _stop_alias(self, timeout: int) -> dict:
        """Stop the server and prove it against the port.

        Deleting the pid file after an unconfirmed kill is what turns a
        surviving server into one the allocator can never stop again: every
        later attempt finds no pid file and reports success while the model
        keeps the whole GPU. The file now outlives an unconfirmed stop.
        """
        deadline = time.time() + timeout
        step_timeout = max(5, timeout // 3)

        try:
            pid = int(Path(self.pid_file).read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            pid = None

        if pid is not None:
            self._kill_pid(pid, timeout=step_timeout)

        if not self._port_open():
            self._forget_pid_file()
            return {"stopped": True, "error": None}

        for target in self._server_pids():
            self._kill_pid(target, timeout=step_timeout)

        while time.time() < deadline:
            if not self._port_open():
                self._forget_pid_file()
                return {"stopped": True, "error": None}
            time.sleep(0.5)

        return {
            "stopped": False,
            "error": (f"port {self.port} is still accepting connections — "
                      f"server not confirmed down"),
        }

    def _stop_instance_alias_level(self) -> dict:
        """Alias-level stop on an instance-bound adapter is a NO-OP.

        The instance may be shared by several aliases; only the explicit
        ``stop-instance --name`` operation may terminate the recorded
        process. ``cmd_stop --alias`` and ``cmd_unload --alias`` (and the
        step-completion paths that call them) land here and must leave
        the process running.
        """
        state = self._read_instance_state()
        return {
            "stopped": True,
            "skipped": True,
            "instance_name": self.instance_name,
            "pid": state.get("pid") if state else None,
            "port": state.get("port") if state else self.port,
            "reason": (
                "instance-bound runtime is shared; stop via "
                "'model-allocator stop-instance --name <instance>'"
            ),
            "error": None,
        }

    def stop_instance(self, timeout: int = 30) -> dict:
        """Instance-level stop: kill the recorded PID and remove the state.

        Deterministic: only the recorded PID is signalled. If no state is
        recorded the call returns OK-without-action — the allocator does
        not adopt a foreign listener by killing what it finds on the port.
        """
        if not self.instance_name:
            return {
                "stopped": False,
                "error": "stop_instance called on a non-instance-bound adapter",
            }
        state = self._read_instance_state()
        if state is None:
            return {
                "stopped": True,
                "skipped": True,
                "instance_name": self.instance_name,
                "pid": None,
                "port": self.port,
                "reason": "no recorded state for this instance",
                "error": None,
            }
        pid = state.get("pid")
        result = self._kill_pid(pid, timeout=timeout)
        # Always forget the state on instance-level stop, even if the kill
        # failed — the caller has accepted responsibility for this instance.
        self._forget_instance_state()
        return {
            "stopped": result.get("stopped", False),
            "instance_name": self.instance_name,
            "pid": pid,
            "port": state.get("port", self.port),
            "error": result.get("error"),
        }

    def unload(self, timeout: int = 30) -> dict:
        if self.instance_name:
            # Same rule as alias-level stop: instance-bound unloads are
            # no-ops. The instance is shared; only instance-level stop
            # may terminate it.
            return self._stop_instance_alias_level()
        return self._stop_alias(timeout=timeout)
