"""FreeToken server backend adapter for model-allocator.

FreeToken is a GPU-resident inference runtime with an OpenAI-compatible API,
so it belongs to the same family as the llama.cpp and SGLang adapters: the
allocator owns the process, the port is the trustworthy identity, and a stop
is not reported as done until the port confirms it.

Two things make it different from SGLang and shape most of this file.

First, the runtime answers for its own readiness. `/health` reports
`loading` -> `ok` -> `error` with real load progress, so this adapter never
has to infer "ready" from "the process exists" — a guess that, on a runtime
which holds ~30 GB of a 32 GB card, would hand a client an endpoint that
cannot answer and cost a whole model load to discover.

Second, a FreeToken model reference is not necessarily a path. Hugging Face
repo IDs (`vrfai/Qwen3.8-27B-NVFP4`) are first-class, so validating the model
by asking the filesystem whether it exists would reject exactly the
configuration that was qualified on this machine.

The harness boundary is deliberate: FreeToken ships `ft launch codex` and
`ft launch claude`, which would start a coding harness from inside the model
allocator. This adapter only ever runs `ft serve`. Choosing and starting an
interface is the harness allocator's authority, and the seam between them is
`endpoint()` below.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# The version this adapter was qualified against on the RTX 5090 workstation
# (FreeToken source 2757bb5). A newer runtime is allowed but reported, because
# FreeToken's backend selection moves fast and an unqualified version is a
# thing to notice before it becomes a production default, not after.
QUALIFIED_RUNTIME_VERSION = "0.1.2"

# Arguments this adapter owns. extra_args exists so a fast-moving runtime can
# be reached without a schema change, but it must not be able to move the
# server off the port, host, GPU or model the allocator is tracking it by —
# that would leave the ownership records describing a process that is not
# there.
PROTECTED_ARGS = frozenset({
    "--model-path", "--model", "--host", "--port", "--gpu",
})

# Alias/profile keys that map onto a `ft serve` flag taking one value. Only
# keys with a configured value are emitted: an unset option must produce no
# flag at all, so the runtime's own defaults and auto-selection stay in
# charge of everything the profile does not deliberately pin.
_VALUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("memory_ratio", "--memory-ratio"),
    ("nvfp4_backend", "--nvfp4-backend"),
    ("moe_backend", "--moe-backend"),
    ("moe_cache_size", "--moe-cache-size"),
    ("moe_cache_rate", "--moe-cache-rate"),
    ("kv_reserve_tokens", "--kv-reserve-tokens"),
    # The KV budget, in tokens, shared by prompt and generation. FreeToken
    # sizes this from whatever VRAM is left after weights and MoE cache, which
    # on a full card lands far below the model's context — see FT-6 in the
    # README. Set it deliberately when a client needs a working context.
    ("num_tokens", "--num-tokens"),
    ("num_pages", "--num-pages"),
    ("sampling_defaults", "--sampling-defaults"),
    ("reasoning_parser", "--reasoning-parser"),
    ("tool_call_parser", "--tool-call-parser"),
    ("attention_backend", "--attention-backend"),
    ("cache_type", "--cache-type"),
    ("model_source", "--model-source"),
    ("dtype", "--dtype"),
    ("served_model_name", "--served-model-name"),
    ("max_output_tokens", "--max-output-tokens"),
    ("max_running_requests", "--max-running-requests"),
    ("max_seq_len_override", "--max-seq-len-override"),
    ("cuda_graph_max_bs", "--cuda-graph-max-bs"),
    ("max_prefill_length", "--max-prefill-length"),
)

# Boolean keys that become a bare flag when true.
_BOOL_FLAGS: tuple[tuple[str, str], ...] = (
    ("moe_cache_auto", "--moe-cache-auto"),
    ("enable_cache_report", "--enable-cache-report"),
    ("disable_moe_prefill_overlap", "--disable-moe-prefill-overlap"),
)


class FreeTokenAdapterError(Exception):
    pass


class FreeTokenAdapter:
    def __init__(self, resolved: dict, state_dir: str | None = None):
        self.resolved = resolved
        self.alias = resolved.get("alias", "freetoken")
        self.port = self._resolve_port()
        self.host = resolved.get("host", resolved.get("default_host", "127.0.0.1"))
        self.state_dir = state_dir or self._default_state_dir()
        self.pid_file = os.path.join(
            self.state_dir, f"model-allocator-{self.alias}-{self.port}.pid"
        )

    # ---------------------------------------------------------------- config

    @staticmethod
    def _default_state_dir() -> str:
        return os.environ.get("MODEL_ALLOCATOR_STATE_DIR", tempfile.gettempdir())

    def _resolve_port(self) -> int:
        configured = self.resolved.get("port")
        if configured is None:
            configured = self.resolved.get("default_port")
        if configured is None:
            raise FreeTokenAdapterError(
                f"No port configured for FreeToken alias "
                f"'{self.resolved.get('alias', '?')}' — set port on the alias "
                f"or default_port on the runtime profile"
            )
        return int(configured)

    def resolve_executable(self) -> str:
        """Locate the `ft` binary without shell activation.

        The qualified install lives in a project-local virtualenv, so `ft` is
        not on the PATH of a systemd service or any other non-interactive
        parent. Executing the resolved binary directly is both deterministic
        and the only thing that works from a service environment; requiring
        `source .venv/bin/activate` in a production startup path is not an
        option.
        """
        for key in ("executable", "server_bin_path"):
            candidate = self.resolved.get(key, "")
            if candidate:
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return candidate
                raise FreeTokenAdapterError(
                    f"FreeToken executable not found or not executable: {candidate}"
                )
        found = shutil.which("ft")
        if found:
            return found
        raise FreeTokenAdapterError(
            "FreeToken executable not found: set `executable` on the runtime "
            "profile (the qualified install is a project-local venv, so `ft` "
            "is normally absent from PATH)"
        )

    def version(self, timeout: int = 15) -> dict:
        """Preflight `ft --version`, and say whether it is the qualified one.

        An unqualified version is not refused here. FreeToken's backend
        selection has already been measured to change throughput by more than
        an order of magnitude, so a version change is something to qualify
        deliberately — but that judgement belongs to whoever is promoting the
        runtime, not to a startup path that would otherwise fail closed on a
        patch release.
        """
        try:
            executable = self.resolve_executable()
        except FreeTokenAdapterError as exc:
            return {"ok": False, "error": str(exc), "version": None,
                    "executable": None, "qualified": None}
        try:
            completed = subprocess.run(
                [executable, "--version"],
                capture_output=True, text=True, timeout=timeout,
            )
        except Exception as exc:
            return {"ok": False, "error": f"Failed to run `ft --version`: {exc}",
                    "version": None, "executable": executable, "qualified": None}

        output = f"{completed.stdout} {completed.stderr}".strip()
        match = re.search(r"(\d+\.\d+\.\d+)", output)
        if completed.returncode != 0 or not match:
            return {
                "ok": False,
                "error": f"Could not parse a version from `ft --version`: {output!r}",
                "version": None, "executable": executable, "qualified": None,
            }

        detected = match.group(1)
        expected = self.resolved.get("qualified_runtime_version",
                                     QUALIFIED_RUNTIME_VERSION)
        return {
            "ok": True,
            "error": None,
            "version": detected,
            "executable": executable,
            "qualified": detected == expected,
            "qualified_version": expected,
            "warning": (None if detected == expected else
                        f"FreeToken {detected} has not been qualified on this "
                        f"machine (qualified: {expected})"),
        }

    # --------------------------------------------------------------- command

    def _model_reference(self) -> str:
        """The `--model-path` value: a local directory OR a Hugging Face repo ID.

        Deliberately not checked against the filesystem. Both qualified
        profiles name Hugging Face repos that resolve out of the local HF
        cache, and `Path(model).exists()` would reject them while claiming
        the configuration was wrong.
        """
        model = self.resolved.get("model_path", "")
        if not model:
            raise FreeTokenAdapterError(
                "No model_path configured for FreeToken alias — expected a "
                "local checkpoint directory or a Hugging Face repo ID"
            )
        return str(model)

    def child_env(self, base: dict | None = None) -> dict:
        """Environment for `ft serve`, with the runtime's own bin on PATH.

        Executing the resolved binary directly is deterministic and avoids
        sourcing an activate script — but it is not sufficient. FreeToken
        JIT-compiles FlashInfer kernels during model load and shells out to
        `ninja`, which lives in the same project-local venv and is nowhere on
        a system PATH. Without this the server starts, answers /health, loads
        weights, and only then dies with FileNotFoundError: 'ninja'.

        Prepending rather than replacing: the venv's tools win, everything
        else the parent had stays reachable. The SGLang adapter carries the
        same lesson for the same reason.
        """
        env = dict(os.environ if base is None else base)
        bin_dir = os.path.dirname(self.resolve_executable())
        if bin_dir:
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        return env

    def gpu_index(self) -> int | None:
        """FreeToken's `--gpu` index, from the allocator's own GPU naming.

        The runtime profiles name devices `cuda0`, which is the allocator's
        device identity and not something FreeToken accepts. Translating here
        keeps one GPU vocabulary across backends instead of adding a
        FreeToken-shaped duplicate of a field that already exists.
        """
        raw = self.resolved.get("gpu")
        if raw is None or raw == "":
            return None
        match = re.search(r"(\d+)\s*$", str(raw))
        if not match:
            raise FreeTokenAdapterError(
                f"Cannot read a GPU index from gpu={raw!r} — expected a form "
                f"like 'cuda0', 'cuda:1' or '0'"
            )
        return int(match.group(1))

    def build_argv(self) -> list[str]:
        if self.resolved.get("num_tokens") and self.resolved.get("num_pages"):
            raise FreeTokenAdapterError(
                "num_tokens and num_pages both set — FreeToken accepts only "
                "one of them (they size the same KV allocation)"
            )
        argv = [
            self.resolve_executable(), "serve",
            "--model-path", self._model_reference(),
            "--host", str(self.host),
            "--port", str(self.port),
        ]

        gpu = self.gpu_index()
        if gpu is not None:
            argv += ["--gpu", str(gpu)]

        for key, flag in _VALUE_FLAGS:
            value = self.resolved.get(key)
            if value is None or value == "":
                continue
            argv += [flag, str(value)]

        for key, flag in _BOOL_FLAGS:
            if self.resolved.get(key):
                argv.append(flag)

        extra = self.resolved.get("extra_args") or []
        if extra:
            if not isinstance(extra, list):
                raise FreeTokenAdapterError("extra_args must be a list")
            for item in extra:
                token = str(item)
                head = token.split("=", 1)[0]
                if head in PROTECTED_ARGS:
                    raise FreeTokenAdapterError(
                        f"extra_args may not set {head}: it is owned by the "
                        f"allocator, which tracks this runtime by host, port, "
                        f"GPU and model"
                    )
                argv.append(token)

        return argv

    # Kept for symmetry with the other adapters, which expose the private name.
    _build_argv = build_argv

    # ------------------------------------------------------------- lifecycle

    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _get_json(self, path: str, timeout: float = 2.0) -> tuple[dict | None, str | None]:
        url = f"{self._base_url()}{path}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                body = response.read().decode("utf-8", "replace")
            return json.loads(body), None
        except urllib.error.HTTPError as exc:
            return None, f"HTTP {exc.code} from {path}"
        except Exception as exc:
            return None, str(exc)

    def health(self, timeout: float = 2.0) -> dict:
        """Runtime state as the runtime itself reports it.

        `/health` distinguishes loading from serving from fatal, which is why
        this adapter never equates a live process with a usable endpoint.
        """
        doc, error = self._get_json("/health", timeout=timeout)
        if doc is None:
            return {"state": None, "reachable": False, "error": error, "doc": None}
        status = str(doc.get("status", "")).lower()
        state = {"ok": "ready", "loading": "loading", "error": "failed"}.get(
            status, "unhealthy"
        )
        return {"state": state, "reachable": True, "error": doc.get("message"),
                "doc": doc}

    def models(self, timeout: float = 5.0) -> dict:
        """Models the endpoint actually exposes, per the OpenAI-compatible route."""
        doc, error = self._get_json("/v1/models", timeout=timeout)
        if doc is None:
            return {"ok": False, "error": error, "models": []}
        entries = doc.get("data") or []
        names = [str(entry.get("id")) for entry in entries if entry.get("id")]
        return {"ok": True, "error": None, "models": names, "doc": doc}

    def stats(self, timeout: float = 5.0) -> dict:
        """Runtime telemetry. Optional by contract.

        A stats failure says nothing about whether inference works, so callers
        must not treat it as a runtime failure — health and the completion
        route are the authorities on that.
        """
        doc, error = self._get_json("/v1/stats", timeout=timeout)
        if doc is None:
            return {"ok": False, "error": error}
        model = doc.get("model") or {}
        kv = doc.get("kv") or {}
        throughput = doc.get("throughput") or {}
        requests = doc.get("requests") or {}
        gpus = doc.get("gpus") or []
        return {
            "ok": True,
            "error": None,
            "model": model.get("id"),
            "context_max": model.get("ctx"),
            "moe": model.get("moe"),
            "attention": model.get("attn"),
            "vram_bytes": doc.get("vram_bytes"),
            "kv_used": kv.get("used_pages"),
            "kv_capacity": kv.get("total_pages"),
            "decode_tps": throughput.get("decode_tps"),
            "requests_completed": requests.get("completed"),
            # GPU identity rather than index: FreeToken keys its bandwidth
            # calibration per device, and an index is not a stable name for
            # one across a reordered bus.
            "gpus": [{"index": g.get("index"), "name": g.get("name"),
                      "uuid": g.get("uuid"), "total_bytes": g.get("total_bytes")}
                     for g in gpus],
            "doc": doc,
        }

    def expected_model_name(self) -> str:
        """The model name the endpoint is expected to expose.

        FreeToken serves a Hugging Face repo under its bare name, so
        `vrfai/Qwen3.8-27B-NVFP4` is exposed as `Qwen3.8-27B-NVFP4` unless a
        served_model_name overrides it.
        """
        served = self.resolved.get("served_model_name")
        if served:
            return str(served)
        return self._model_reference().rstrip("/").split("/")[-1]

    # ----------------------------------------------------------- GPU policy

    def _nvidia_smi(self, args: list[str], timeout: float = 10.0) -> str | None:
        """Run nvidia-smi, or return None when it cannot be reached.

        Absence is not treated as a failure. A machine without nvidia-smi is
        a machine this check cannot speak about, and refusing to start on a
        missing diagnostic tool would break working setups to protect against
        a hazard we have no evidence of.
        """
        binary = shutil.which("nvidia-smi")
        if not binary:
            return None
        try:
            completed = subprocess.run([binary] + args, capture_output=True,
                                       text=True, timeout=timeout)
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout

    def gpu_occupancy(self) -> list[dict]:
        """Compute processes holding this profile's GPU, largest first.

        Reported per process rather than as a single free-memory number so a
        refusal can name what is in the way. `nvidia-smi` scopes the query to
        one device with `-i`, which matters on a multi-GPU host where another
        card being busy is none of this profile's business.
        """
        index = self.gpu_index()
        args = ["--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits"]
        if index is not None:
            args = ["-i", str(index)] + args
        output = self._nvidia_smi(args)
        if output is None:
            return []
        apps = []
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            try:
                used = int(parts[1])
            except ValueError:
                continue
            apps.append({"pid": int(parts[0]), "used_mib": used,
                         "name": parts[2] if len(parts) > 2 else ""})
        return sorted(apps, key=lambda a: a["used_mib"], reverse=True)

    def gpu_free_mib(self) -> int | None:
        index = self.gpu_index()
        args = ["--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        if index is not None:
            args = ["-i", str(index)] + args
        output = self._nvidia_smi(args)
        if output is None:
            return None
        first = output.strip().splitlines()[0] if output.strip() else ""
        try:
            return int(first.strip())
        except ValueError:
            return None

    def check_gpu_available(self) -> dict:
        """Decide whether this profile can have the card, before spending
        minutes discovering it cannot.

        A qualified FreeToken profile claims very nearly the whole device, so
        it is an exclusive workload in practice. Starting it beside a resident
        llama.cpp or SGLang server does not degrade gracefully: weights load
        for minutes and then the worker dies on CUDA OOM, which on an
        autonomous chain is a failed step with a misleading cause.

        This refuses rather than reclaims. Releasing another runtime is a
        decision about somebody else's work, and the dispatch layer already
        stops the outgoing role's model before starting the incoming one — so
        an occupied card here means something unexpected is running, which is
        exactly when acting automatically is wrong. The report names the
        occupant so the caller can decide.
        """
        required = self.resolved.get("min_free_vram_mib")
        free = self.gpu_free_mib()
        occupants = self.gpu_occupancy()
        ours = self._server_pids()
        foreign = [a for a in occupants if a["pid"] not in ours]

        if free is None:
            return {"ok": True, "checked": False, "free_mib": None,
                    "occupants": foreign, "error": None,
                    "note": "nvidia-smi unavailable — GPU policy not enforced"}
        if required and free < int(required):
            names = ", ".join(
                f"pid {a['pid']} ({a['name'] or 'unknown'}) {a['used_mib']} MiB"
                for a in foreign[:3]) or "no compute process reported"
            return {
                "ok": False, "checked": True, "free_mib": free,
                "occupants": foreign,
                "error": (f"GPU {self.gpu_index()} has {free} MiB free but this "
                          f"profile needs {required} MiB; held by: {names}"),
            }
        return {"ok": True, "checked": True, "free_mib": free,
                "occupants": foreign, "error": None}

    def _port_open(self) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                return sock.connect_ex((self.host, self.port)) == 0
        except OSError:
            return False

    def _server_pids(self) -> list[int]:
        """PIDs of FreeToken servers on THIS port.

        Both conditions are required. Matching `ft serve` alone would put
        every FreeToken instance on the machine within reach of a stop, and
        this adapter must only ever act on the runtime it is tracking.
        """
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
            joined = " ".join(parts)
            if "serve" in parts and ("freetoken" in joined or "/ft" in joined
                                     or parts[:1] == ["ft"]):
                if port_token in parts:
                    pids.append(int(entry))
        return pids

    def inspect_port(self) -> dict:
        """Classify whatever already holds the port, before starting anything.

        Starting a second GPU runtime on an occupied port is not a recoverable
        mistake on a card this full, so an occupant is identified rather than
        assumed away: an allocator-owned instance serving this exact model can
        be reused, another FreeToken cannot be taken over, and something that
        is not FreeToken at all must never be disturbed.
        """
        if not self._port_open():
            return {"occupied": False, "kind": "free", "owned": False,
                    "model": None, "reusable": False}

        health = self.health()
        if not health["reachable"]:
            return {"occupied": True, "kind": "unknown_process", "owned": False,
                    "model": None, "reusable": False,
                    "error": "port is open but /health did not answer"}
        if health["state"] is None or health["state"] == "unhealthy":
            return {"occupied": True, "kind": "incompatible_service",
                    "owned": False, "model": None, "reusable": False,
                    "error": "port answers, but not with a FreeToken health document"}

        owned = self._recorded_pid() is not None and bool(self._server_pids())
        served = (health["doc"] or {}).get("model")
        expected = self.expected_model_name()
        matches = served is not None and str(served) == expected
        return {
            "occupied": True,
            "kind": "allocator_owned_freetoken" if owned else "external_freetoken",
            "owned": owned,
            "state": health["state"],
            "model": served,
            "expected_model": expected,
            "reusable": bool(matches and health["state"] in ("ready", "loading")),
        }

    def _recorded_pid(self) -> int | None:
        try:
            return int(Path(self.pid_file).read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            return None

    def _forget_pid_file(self) -> None:
        try:
            os.unlink(self.pid_file)
        except OSError:
            pass

    def start(self, timeout: int = 900) -> dict:
        """Start `ft serve` and wait for the runtime to declare itself ready.

        The default timeout is generous because a 27B checkpoint cold from
        disk legitimately takes minutes to load; the wait ends early on a
        fatal health status or a process that exits, so a real failure is
        still reported quickly.
        """
        occupant = self.inspect_port()
        if occupant["occupied"]:
            if occupant["reusable"]:
                pids = self._server_pids()
                return {
                    "started": True, "reused": True, "error": None,
                    "pid": pids[0] if pids else self._recorded_pid(),
                    "port": self.port, "state": occupant.get("state"),
                }
            return {
                "started": False, "reused": False,
                "error": (f"port {self.port} is already held by "
                          f"{occupant['kind']}"
                          + (f" serving '{occupant['model']}' (expected "
                             f"'{occupant['expected_model']}')"
                             if occupant.get("model") else "")),
                "port": self.port, "occupant": occupant,
            }

        preflight = self.version()
        if not preflight["ok"]:
            return {"started": False, "error": preflight["error"], "port": self.port}

        gpu = self.check_gpu_available()
        if not gpu["ok"]:
            return {"started": False, "state": "gpu_unavailable",
                    "error": gpu["error"], "port": self.port, "gpu": gpu}

        try:
            argv = self.build_argv()
        except FreeTokenAdapterError as exc:
            return {"started": False, "error": str(exc), "port": self.port}

        os.makedirs(self.state_dir, exist_ok=True)
        log_path = os.path.join(
            self.state_dir, f"model-allocator-{self.alias}-{self.port}.log"
        )
        try:
            log_handle = open(log_path, "wb")
        except OSError:
            log_handle = subprocess.DEVNULL
        try:
            process = subprocess.Popen(
                argv,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=self.child_env(),
            )
        except Exception as exc:
            return {"started": False,
                    "error": f"Failed to start FreeToken server: {exc}",
                    "port": self.port}
        finally:
            if log_handle is not subprocess.DEVNULL:
                log_handle.close()

        Path(self.pid_file).write_text(str(process.pid), encoding="utf-8")

        start_ts = time.time()
        last_state = "starting"
        while time.time() - start_ts < timeout:
            if process.poll() is not None:
                self._forget_pid_file()
                return {
                    "started": False, "state": "failed",
                    "error": (f"FreeToken exited before becoming ready "
                              f"(rc={process.returncode}); see {log_path}"),
                    "port": self.port, "log": log_path,
                }
            health = self.health()
            if health["reachable"]:
                last_state = health["state"]
                if last_state == "ready":
                    verification = self.verify_model()
                    return {
                        "started": verification["ok"], "reused": False,
                        "state": "ready" if verification["ok"] else "model_mismatch",
                        "error": verification["error"],
                        "pid": process.pid, "port": self.port,
                        "model": verification.get("served"),
                        "log": log_path,
                    }
                if last_state == "failed":
                    return {
                        "started": False, "state": "failed",
                        "error": health["error"] or "FreeToken reported a fatal error",
                        "pid": process.pid, "port": self.port, "log": log_path,
                    }
            time.sleep(1.0)

        return {
            "started": False, "state": last_state,
            "error": (f"FreeToken did not become ready within {timeout}s "
                      f"(last state: {last_state}); see {log_path}"),
            "pid": process.pid, "port": self.port, "log": log_path,
        }

    def verify_model(self) -> dict:
        """Confirm the endpoint exposes the model that was asked for."""
        expected = self.expected_model_name()
        discovered = self.models()
        if not discovered["ok"]:
            return {"ok": False, "expected": expected, "served": None,
                    "error": f"could not read /v1/models: {discovered['error']}"}
        if expected in discovered["models"]:
            return {"ok": True, "expected": expected, "served": expected,
                    "error": None}
        return {
            "ok": False, "expected": expected,
            "served": discovered["models"][0] if discovered["models"] else None,
            "error": (f"endpoint exposes {discovered['models']!r}, "
                      f"expected '{expected}'"),
        }

    def status(self, use_pid: int | None = None) -> dict:
        """Report state, trusting the runtime's health over the bookkeeping.

        `running` covers loading as well as ready on purpose: a caller that
        auto-starts on "not running" would otherwise launch a second server
        onto a GPU that the first one is in the middle of filling.
        """
        pid = use_pid if use_pid is not None else self._recorded_pid()
        if pid is None:
            pids = self._server_pids()
            pid = pids[0] if pids else None

        alive = False
        if pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False

        health = self.health()
        if health["reachable"]:
            state = health["state"]
        elif self._port_open():
            state = "unhealthy"
        elif alive:
            state = "starting"
        else:
            state = "stopped"

        doc = health["doc"] or {}
        return {
            "running": state in ("ready", "loading"),
            "ready": state == "ready",
            "state": state,
            "alive": alive,
            "healthy": state == "ready",
            "pid": pid,
            "port": self.port,
            "model": doc.get("model"),
            "error": health["error"] if state in ("failed", "unhealthy") else None,
        }

    def endpoint(self) -> dict:
        """The normalized descriptor handed to the harness allocator.

        This is the whole seam. Everything above it — which venv holds `ft`,
        how the NVFP4 backend was chosen, what the GPU calibration says — is
        the model allocator's business, and nothing below it needs to know.
        """
        base_url = self._base_url()
        served = self.expected_model_name()
        context = None
        kv_capacity = None
        telemetry = self.stats()
        if telemetry["ok"]:
            served = telemetry.get("model") or served
            context = telemetry.get("context_max")
            kv_capacity = telemetry.get("kv_capacity")
        if context is None:
            context = self.resolved.get("context")
        if kv_capacity is None:
            # /v1/stats reports kv as null until the runtime has served its
            # first request, and the budget is most needed BEFORE that — a
            # caller sizing its opening prompt has nothing else to go on. Fall
            # back to what the profile asked for, which is what the runtime
            # logs allocating at startup.
            kv_capacity = self.resolved.get("num_tokens")

        compatibility = ["openai"]
        doc, _ = self._get_json("/openapi.json", timeout=3.0)
        if doc and "/v1/messages" in (doc.get("paths") or {}):
            compatibility.append("anthropic")

        return {
            "provider": "freetoken",
            "base_url": base_url,
            "api_base": f"{base_url}/v1",
            "model": served,
            # The architecture's maximum. NOT what fits right now.
            "context_length": context,
            # What actually fits: the allocated KV budget, in tokens, for
            # prompt AND generation together. On a card this full the two
            # numbers are not close — the qualified Qwen3.8 profile reports
            # 262144 context against a 14303-token KV budget, and a caller
            # that sized its prompt by context_length had its very first
            # request refused (FT-6). Consumers must budget by this field.
            "usable_context_tokens": kv_capacity,
            "api_compatibility": compatibility,
        }

    def capabilities(self) -> dict:
        return {
            "provider": "freetoken",
            "capabilities": {
                "openai_api": True,
                "runtime_stats": True,
                "gpu_selection": True,
                "huggingface_models": True,
                "offline_cached_models": True,
            },
        }

    # ------------------------------------------------------------------ stop

    @staticmethod
    def _signal_pid(pid: int, sig: int, timeout: float) -> bool:
        try:
            os.kill(pid, sig)
        except (OSError, ProcessLookupError):
            return True
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except (OSError, ProcessLookupError):
                return True
            time.sleep(0.2)
        return False

    def _terminate(self, pid: int, timeout: float) -> bool:
        """Escalate politely: SIGINT, then SIGTERM, then the whole group.

        SIGINT first because that is what a clean interactive shutdown sends,
        and it is the path FreeToken's own qualification exercised.
        """
        step = max(2.0, timeout / 3.0)
        if self._signal_pid(pid, signal.SIGINT, step):
            return True
        if self._signal_pid(pid, signal.SIGTERM, step):
            return True
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            return False
        except (OSError, ProcessLookupError):
            return True

    def stop(self, timeout: int = 60, drain: bool = True) -> dict:
        """Stop the runtime this adapter owns, and prove it by the port.

        The pid file survives an unconfirmed stop deliberately: a stop that
        deletes its own bookkeeping and then reports success is how a live
        server becomes unreachable to the allocator while still holding the
        card.
        """
        if drain and self._port_open():
            # Close admission first so in-flight generations are not cut off
            # mid-token. Best effort by design — a runtime too wedged to
            # accept this is exactly the one that needs the signals below.
            try:
                request = urllib.request.Request(
                    f"{self._base_url()}/v1/admin/prepare-stop",
                    data=b"{}", method="POST",
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(request, timeout=5)
            except Exception:
                pass

        deadline = time.time() + timeout
        pid = self._recorded_pid()
        if pid is not None:
            self._terminate(pid, timeout=max(5.0, timeout / 3.0))

        if not self._port_open():
            self._forget_pid_file()
            return {"stopped": True, "error": None, "port": self.port}

        for target in self._server_pids():
            self._terminate(target, timeout=max(5.0, timeout / 3.0))

        while time.time() < deadline:
            if not self._port_open():
                self._forget_pid_file()
                return {"stopped": True, "error": None, "port": self.port}
            time.sleep(0.5)

        return {
            "stopped": False,
            "error": (f"port {self.port} is still accepting connections — "
                      f"FreeToken not confirmed down"),
            "port": self.port,
        }

    def unload(self) -> dict:
        return self.stop()
