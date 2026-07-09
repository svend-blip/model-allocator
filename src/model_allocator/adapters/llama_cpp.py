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
        """Resolve llama-server binary from env-var names only."""
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
        if "reasoning" in flags:
            argv += ["--reasoning", str(flags["reasoning"])]
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
                return {"running": False, "error": "No PID file", "pid": None}

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

        running = alive and healthy
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

        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        return {"stopped": True, "error": None}

    def stop(self, timeout: int = 30) -> dict:
        try:
            pid = int(Path(self.pid_file).read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            return {"stopped": True, "error": None}

        result = self._kill_pid(pid, timeout=timeout)
        try:
            os.unlink(self.pid_file)
        except OSError:
            pass
        return result

    def unload(self) -> dict:
        return self.stop()
