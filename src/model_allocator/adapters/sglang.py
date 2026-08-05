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
        # Ensure the venv bin directory is on PATH so that tools like ninja
        # (required by FlashInfer's JIT compilation) are found.
        env = os.environ.copy()
        venv = self.resolved.get("venv", "")
        if venv:
            venv_bin = os.path.join(venv, "bin")
            env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
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
                # Record the pid that actually owns the port, not the one we
                # spawned: the launcher hands off to the real server and
                # exits, so persisting its pid leaves the file pointing at a
                # corpse and every later stop silently no-ops.
                owners = self._server_pids()
                server_pid = owners[0] if owners else process.pid
                Path(self.pid_file).write_text(str(server_pid), encoding="utf-8")
                return {
                    "started": True, "error": None,
                    "pid": server_pid, "port": self.port,
                }
            time.sleep(0.5)

        self._kill_pid(process.pid, timeout=10)
        return {
            "started": False,
            "error": f"sglang server health endpoint did not become ready within {timeout}s",
        }

    def status(self, use_pid: int | None = None) -> dict:
        """Report status, trusting the port over the pid file.

        A missing pid file used to mean "not running", which hid a live
        orphaned server from every caller. The health endpoint answering is
        proof the server is up regardless of what the bookkeeping says.
        """
        pid = use_pid
        if pid is None:
            try:
                pid = int(Path(self.pid_file).read_text(encoding="utf-8").strip())
            except (FileNotFoundError, ValueError):
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

        # A answering health endpoint IS the server running. Requiring the
        # recorded pid to be alive as well made a healthy orphan report as
        # stopped, which is how one came to hold the GPU unnoticed.
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
        # SGLang spawns child processes (scheduler, detokenizer) that must
        # also be killed. Try the process group first, then fall back to the
        # single PID.
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

        start_ts = time.time()
        while time.time() - start_ts < timeout:
            try:
                os.kill(pid, 0)
            except (OSError, ProcessLookupError):
                return {"stopped": True, "error": None}
            time.sleep(0.2)

        # SIGTERM didn't work — escalate to SIGKILL for the process group
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

        # Verify the process actually died
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            return {"stopped": False, "error": f"PID {pid} survived SIGKILL"}
        except (OSError, ProcessLookupError):
            return {"stopped": True, "error": None}

    def _port_open(self) -> bool:
        """Whether anything is still listening on the server port.

        The port is the only trustworthy identity for this backend. A pid
        can be gone while the server keeps running, and it was exactly that
        gap that let a stop report success over a live server.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                return sock.connect_ex((self.host, self.port)) == 0
        except OSError:
            return False

    def _server_pids(self) -> list[int]:
        """PIDs belonging to this SGLang server, found by scanning /proc.

        Launchers (matched on both the module name and this port) come
        first, then the renamed worker processes SGLang spawns
        (`sglang::scheduler`, `sglang::detokenizer`). The workers carry no
        port in their command line, so they can only be matched by name —
        which is why they are used as a last resort, after the port has
        proven the server is still alive.
        """
        launchers: list[int] = []
        workers: list[int] = []
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
            joined = " ".join(parts)
            if "sglang.launch_server" in joined and port_token in parts:
                launchers.append(int(entry))
            elif joined.startswith("sglang::"):
                workers.append(int(entry))
        return launchers + workers

    def _forget_pid_file(self) -> None:
        try:
            os.unlink(self.pid_file)
        except OSError:
            pass

    def stop(self, timeout: int = 30) -> dict:
        """Stop the server and prove it, or say plainly that it is still up.

        The previous version killed whatever pid the file named, deleted
        the file, and returned that kill's result. SGLang's launcher exits
        once the real server processes are running, so the recorded pid was
        usually already dead: every stop reported instant success, the pid
        file was removed, and the live server became unreachable to the
        allocator forever — holding the whole GPU. Confirmation now comes
        from the port, and the pid file survives an unconfirmed stop so a
        later attempt can still find its way back.
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

        # Still serving — the recorded pid was not the server. Go after the
        # processes that actually own the port.
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
