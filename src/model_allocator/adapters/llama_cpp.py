"""llama.cpp server backend adapter for model-allocator."""

from __future__ import annotations

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
        self.pid_file = os.path.join(self.state_dir, f"model-allocator-{self.alias}-{self.port}.pid")

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
        return argv

    def start(self, timeout: int = 120) -> dict:
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

    def status(self, use_pid: int | None = None) -> dict:
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

        health_url = f"http://{self.host}:{self.port}/health"
        try:
            urllib.request.urlopen(health_url, timeout=2)
            healthy = True
            health_error = None
        except Exception as exc:
            healthy = False
            health_error = str(exc)

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

    def unload(self) -> dict:
        return self.stop()
