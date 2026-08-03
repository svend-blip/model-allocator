"""SGLang server backend adapter for model-allocator."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


class SGLangAdapterError(Exception):
    pass


class SGLangAdapter:
    def __init__(self, resolved: dict, state_dir: str | None = None):
        self.resolved = resolved
        self.alias = resolved.get("alias", "sglang")
        self.port = self._resolve_port()
        self.host = resolved.get("host", resolved.get("default_host", "127.0.0.1"))
        self.state_dir = state_dir or self._default_state_dir()
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

    def _python_bin(self) -> str:
        venv = self.resolved.get("venv", "")
        if venv:
            python = os.path.join(venv, "bin", "python")
            if not (os.path.isfile(python) and os.access(python, os.X_OK)):
                raise SGLangAdapterError(f"Python not found in venv: {python}")
            return python
        return "python3"

    def _build_argv(self) -> list[str]:
        model_path = self.resolved.get("model_path", "")
        if not model_path:
            raise SGLangAdapterError("No model_path configured for SGLang alias")

        served_name = self.resolved.get("served_model_name", "qwen-shared")
        context = self.resolved.get("context", 32768)
        mem_frac = self.resolved.get("mem_fraction_static", 0.82)
        max_requests = self.resolved.get("max_running_requests", 2)
        tool_parser = self.resolved.get("tool_call_parser", "qwen")

        argv = [
            self._python_bin(), "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--served-model-name", served_name,
            "--host", self.host,
            "--port", str(self.port),
            "--context-length", str(context),
            "--mem-fraction-static", str(mem_frac),
            "--max-running-requests", str(max_requests),
            "--tool-call-parser", tool_parser,
        ]

        if self.resolved.get("enable_cache_report"):
            argv.append("--enable-cache-report")

        return argv

    def start(self, timeout: int = 120) -> dict:
        try:
            argv = self._build_argv()
        except SGLangAdapterError as exc:
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
            return {"started": False, "error": f"Failed to start sglang server: {exc}"}

        Path(self.pid_file).write_text(str(process.pid), encoding="utf-8")

        start_ts = time.time()
        while time.time() - start_ts < timeout:
            if process.poll() is not None:
                return {"started": False, "error": "sglang server exited early"}
            status = self.status(use_pid=process.pid)
            if status["running"]:
                return {
                    "started": True, "error": None,
                    "pid": process.pid, "port": self.port,
                }
            time.sleep(0.5)

        self._kill_pid(process.pid, timeout=10)
        return {
            "started": False,
            "error": f"sglang server health endpoint did not become ready within {timeout}s",
        }

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
