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

import hashlib
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

# Failure classes. A start() or readiness() result that is not a success
# carries one of these under `code`, so a caller can tell an incomplete
# download from a crashed process from a port held by someone else without
# parsing prose. They are strings rather than an enum because every result
# in this package is a plain dict that ends up in JSON.
MODEL_CACHE_INCOMPLETE = "MODEL_CACHE_INCOMPLETE"
RUNTIME_START_FAILED = "RUNTIME_START_FAILED"
RUNTIME_NOT_READY = "RUNTIME_NOT_READY"
MODEL_IDENTITY_MISMATCH = "MODEL_IDENTITY_MISMATCH"
# Not in the original five: the readiness chain checks the live context
# against the alias, and an endpoint advertising less than its alias
# promises is neither a wrong model nor a runtime that failed to come up.
CONTEXT_CAPABILITY_MISMATCH = "CONTEXT_CAPABILITY_MISMATCH"
RESOURCE_CONFLICT = "RESOURCE_CONFLICT"

# Startup log lines that look alarming and are not. Qwen3.8-Flash-Next
# builds its MoE expert banks at load, and on this machine FreeToken reports
# taking the serial path when free RAM is short. Both lines appeared on the
# qualified, working run, so they are surfaced as slow initialisation and
# never as a failure — the process exiting or readiness failing is what
# decides that.
SLOW_INIT_MARKERS: tuple[str, ...] = (
    "expert banks: low free RAM -> serial build",
    "expert banks: slow path (serial build)",
)

# The safetensors index that an indexed (sharded) checkpoint is keyed by.
SAFETENSORS_INDEX = "model.safetensors.index.json"

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
    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


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
        # Written beside the pid file on every start, compared on reuse.
        self.fingerprint_file = os.path.join(
            self.state_dir,
            f"model-allocator-{self.alias}-{self.port}.fingerprint.json",
        )
        # The last `ft --version` answer, so diagnostics and the fingerprint
        # can name the runtime version without running the binary again.
        self._version_seen: str | None = None

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
        # A profile that names the variable its executable expands from has
        # pinned a specific install. The loader turns an unset ${VAR} into
        # an empty string, which is how we get here with `executable` blank;
        # the variable name is what makes the error actionable, and the
        # absence of a PATH fallback is deliberate — the Flash-Next
        # qualification lives in its own venv, and whatever other `ft` is on
        # PATH is a different, unqualified runtime.
        env_name = self.resolved.get("executable_env", "")
        if env_name:
            candidate = os.environ.get(str(env_name), "")
            if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
            if candidate:
                raise FreeTokenAdapterError(
                    f"FreeToken executable not found or not executable: "
                    f"{candidate} (from ${env_name})"
                )
            raise FreeTokenAdapterError(
                f"FreeToken executable unresolved: ${env_name} is not set. "
                f"The runtime profile expands `executable` from it — export "
                f"it to the qualified venv's `ft`. No PATH fallback: a "
                f"different ft is an unqualified runtime"
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
        self._version_seen = detected
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
        cuda_bin = self.cuda_toolkit_bin(env)
        if cuda_bin:
            # After the venv bin, before everything else: FreeToken's NVFP4
            # path JIT-builds kernels with the first `nvcc` on PATH and refuses
            # a toolkit that does not match torch's CUDA. A systemd service
            # (the bridge broker) has no toolkit on PATH, so the apt wrapper
            # /usr/bin/nvcc (CUDA 12) wins and the server dies after loading
            # 29 GB of weights. Measured 2026-09-02, 9000-02-ELOOP handoff 46.
            env["PATH"] = env["PATH"].replace(
                bin_dir + os.pathsep, bin_dir + os.pathsep + cuda_bin + os.pathsep, 1
            ) if bin_dir else cuda_bin + os.pathsep + env.get("PATH", "")
        return env

    @staticmethod
    def cuda_toolkit_bin(env: dict) -> str:
        """bin dir of the CUDA toolkit to compile against, or "" when unknown.

        `CUDA_HOME` (the convention every CUDA build tool honours) wins; the
        distro's `/usr/local/cuda` symlink is the fallback. No version is
        guessed here — the runtime checks the toolkit against its torch build
        and reports a mismatch itself.
        """
        home = (env.get("CUDA_HOME") or "").strip()
        candidates = [home] if home else []
        candidates.append(os.path.join(os.sep, "usr", "local", "cuda"))
        for candidate in candidates:
            bin_dir = os.path.join(candidate, "bin")
            if os.path.isfile(os.path.join(bin_dir, "nvcc")):
                return bin_dir
        return ""

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
        """Models the endpoint actually exposes, per the OpenAI-compatible route.

        `context` maps each model id to the context length the endpoint
        advertises for it (FreeToken reports `max_model_len` and
        `context_length`, both 262144 on the qualified Flash-Next profile),
        or None when the entry carries neither. This is the route to verify
        capabilities on: it is generic, it is what clients read, and unlike
        /v1/stats it describes configuration rather than telemetry.
        """
        doc, error = self._get_json("/v1/models", timeout=timeout)
        if doc is None:
            return {"ok": False, "error": error, "models": [], "context": {}}
        entries = doc.get("data") or []
        names = [str(entry.get("id")) for entry in entries if entry.get("id")]
        context: dict[str, int | None] = {}
        for entry in entries:
            if not entry.get("id"):
                continue
            context[str(entry["id"])] = self._advertised_context(entry)
        return {"ok": True, "error": None, "models": names, "context": context,
                "doc": doc}

    @staticmethod
    def _advertised_context(entry: dict) -> int | None:
        """Read a model entry's context length under the names servers use."""
        for source in (entry, entry.get("meta") or {}):
            for key in ("max_model_len", "context_length", "n_ctx"):
                value = source.get(key) if isinstance(source, dict) else None
                if value is None:
                    continue
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return None

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

        doc = health["doc"] or {}
        served = doc.get("model")
        if health["state"] == "ready" and "model" not in doc:
            # `{"status": "ok"}` is what llama-server answers too, and the
            # Flash-Next profile shares port 8090 with a llama.cpp instance.
            # A serving FreeToken always names its model; a health document
            # that does not is some other server, and not ours to touch.
            return {"occupied": True, "kind": "incompatible_service",
                    "owned": False, "model": None, "reusable": False,
                    "error": "port answers /health without naming a model — "
                             "not a FreeToken runtime"}

        owned = self._recorded_pid() is not None and bool(self._server_pids())
        expected = self.expected_model_name()
        matches = served is not None and str(served) == expected
        reusable = bool(matches and health["state"] in ("ready", "loading"))

        # An owned runtime is reused only when it is the runtime we would
        # have started: same model, profile, GPU and launch arguments. A pid
        # file alone proves we started *something* on this port, not that it
        # was started from this alias's configuration. Older records have no
        # fingerprint and fall back to the model-identity check above.
        fingerprint_mismatch = False
        recorded = self._recorded_fingerprint() if owned else None
        if recorded is not None:
            current = self.runtime_fingerprint().get("digest")
            fingerprint_mismatch = recorded.get("digest") != current
            if fingerprint_mismatch:
                reusable = False
        return {
            "occupied": True,
            "kind": "allocator_owned_freetoken" if owned else "external_freetoken",
            "owned": owned,
            "state": health["state"],
            "model": served,
            "expected_model": expected,
            "reusable": reusable,
            "fingerprint_mismatch": fingerprint_mismatch,
        }

    def _recorded_pid(self) -> int | None:
        try:
            return int(Path(self.pid_file).read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            return None

    def _recorded_fingerprint(self) -> dict | None:
        try:
            doc = json.loads(Path(self.fingerprint_file).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return None
        return doc if isinstance(doc, dict) else None

    def _record_fingerprint(self, fingerprint: dict) -> None:
        try:
            Path(self.fingerprint_file).write_text(
                json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            # Bookkeeping, not correctness: a start must not fail because the
            # sidecar could not be written. Reuse then falls back to the
            # model-identity check, as it does for records that predate it.
            pass

    def _forget_pid_file(self) -> None:
        for path in (self.pid_file, self.fingerprint_file):
            try:
                os.unlink(path)
            except OSError:
                pass

    @staticmethod
    def _cgroup_unit_of(pid: int) -> str | None:
        """Name the systemd unit whose cgroup holds `pid`, if any."""
        try:
            with open(f"/proc/{pid}/cgroup", encoding="utf-8") as handle:
                line = handle.readline().strip()
        except OSError:
            return None
        unit = line.rpartition("/")[2]
        if unit.endswith((".scope", ".service")):
            return unit
        return None

    def _shield_from_oomd(self, pid: int | None) -> dict:
        """Ask systemd-oomd to spare the unit this server landed in.

        `ft serve` inherits the cgroup of whoever invoked the allocator —
        in practice a tmux pane's scope — and on a machine where the engine
        holds ~60 GB of shared memory it is also the most expensive victim
        oomd can pick: a reload costs minutes and stalls every role wired
        to the endpoint (it was killed twice on 2026-08-29, once mid-load,
        which is why this runs before the readiness wait). The preference
        is runtime-only and dies with the scope, so it must be re-applied
        on every start; that is exactly why it lives here. Best effort by
        design: a start never fails because the shield could not be set,
        the outcome is reported in the start result instead.
        """
        if pid is None:
            return {"applied": False, "reason": "no pid to inspect"}
        unit = self._cgroup_unit_of(pid)
        if unit is None:
            return {"applied": False,
                    "reason": f"pid {pid} is not in a systemd unit cgroup"}
        try:
            probe = subprocess.run(
                ["systemctl", "--user", "set-property", unit,
                 "ManagedOOMPreference=avoid"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception as exc:
            return {"applied": False, "reason": str(exc)}
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout or "").strip()
            return {"applied": False,
                    "reason": detail or f"systemctl rc={probe.returncode}"}
        return {"applied": True, "unit": unit}

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
                pid = pids[0] if pids else self._recorded_pid()
                return {
                    "started": True, "reused": True, "error": None,
                    "code": None, "pid": pid,
                    "port": self.port, "state": occupant.get("state"),
                    "oomd_shield": self._shield_from_oomd(pid),
                }
            detail = ""
            if occupant.get("fingerprint_mismatch"):
                detail = (" started by this allocator from a different "
                          "configuration (runtime fingerprint differs)")
            elif occupant.get("model"):
                detail = (f" serving '{occupant['model']}' (expected "
                          f"'{occupant['expected_model']}')")
            return {
                "started": False, "reused": False,
                "code": RESOURCE_CONFLICT,
                "error": (f"port {self.port} is already held by "
                          f"{occupant['kind']}{detail}"),
                "port": self.port, "occupant": occupant,
                "diagnostics": self.diagnostics(
                    runtime_state=occupant.get("state") or occupant.get("kind")),
            }

        preflight = self.version()
        if not preflight["ok"]:
            return {"started": False, "code": RUNTIME_START_FAILED,
                    "error": preflight["error"], "port": self.port,
                    "diagnostics": self.diagnostics(runtime_state="not_started")}

        gpu = self.check_gpu_available()
        if not gpu["ok"]:
            return {"started": False, "state": "gpu_unavailable",
                    "code": RESOURCE_CONFLICT,
                    "error": gpu["error"], "port": self.port, "gpu": gpu,
                    "diagnostics": self.diagnostics(runtime_state="not_started")}

        # Before spending minutes on a load: are the weights actually all
        # there? FreeToken opens shards lazily, so a missing one surfaces as
        # a process crash deep into initialisation — or, worse, as a
        # download the allocator never asked for. This reads; it never
        # repairs, deletes or fetches.
        checkpoint = self.checkpoint_preflight()
        if not checkpoint["ok"]:
            return {"started": False, "state": "cache_incomplete",
                    "code": MODEL_CACHE_INCOMPLETE,
                    "error": checkpoint["error"], "port": self.port,
                    "checkpoint": checkpoint,
                    "diagnostics": self.diagnostics(
                        runtime_state="not_started", checkpoint=checkpoint)}

        try:
            argv = self.build_argv()
        except FreeTokenAdapterError as exc:
            return {"started": False, "code": RUNTIME_START_FAILED,
                    "error": str(exc), "port": self.port,
                    "diagnostics": self.diagnostics(runtime_state="not_started")}

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
            return {"started": False, "code": RUNTIME_START_FAILED,
                    "error": f"Failed to start FreeToken server: {exc}",
                    "port": self.port,
                    "diagnostics": self.diagnostics(runtime_state="not_started")}
        finally:
            if log_handle is not subprocess.DEVNULL:
                log_handle.close()

        Path(self.pid_file).write_text(str(process.pid), encoding="utf-8")
        # Record what was started, so a later start() from a different
        # configuration cannot mistake this process for its own.
        fingerprint = self.runtime_fingerprint(checkpoint=checkpoint)
        self._record_fingerprint(fingerprint)

        # Shield before the readiness wait, not after: the load itself is
        # the window where the engine's shmem footprint grows fastest.
        oomd_shield = self._shield_from_oomd(process.pid)

        start_ts = time.time()
        last_state = "starting"
        initialisation: dict = {"slow_path": False, "notes": []}
        while time.time() - start_ts < timeout:
            # Expected-but-slow log lines are reported as they appear and
            # change nothing about the wait: the run they were observed on
            # reached READY. Only an exit or a failed readiness check ends it.
            self._note_slow_initialisation(log_path, initialisation)
            if process.poll() is not None:
                self._forget_pid_file()
                tail = self._log_tail(log_path)
                code = self._classify_exit(tail)
                return {
                    "started": False, "state": "failed", "code": code,
                    "error": (f"FreeToken exited before becoming ready "
                              f"(rc={process.returncode}); see {log_path}"),
                    "port": self.port, "log": log_path, "log_tail": tail,
                    "initialisation": initialisation,
                    "diagnostics": self.diagnostics(
                        runtime_state="exited", checkpoint=checkpoint),
                }
            health = self.health()
            if health["reachable"]:
                last_state = health["state"]
                if last_state == "ready":
                    readiness = self.readiness(alive=process.poll() is None)
                    return {
                        "started": readiness["ready"], "reused": False,
                        "state": readiness["state"],
                        "code": readiness["code"],
                        "error": readiness["error"],
                        "pid": process.pid, "port": self.port,
                        "model": readiness.get("served"),
                        "context": readiness.get("context"),
                        "log": log_path,
                        "initialisation": initialisation,
                        "oomd_shield": oomd_shield,
                        "fingerprint": fingerprint.get("digest"),
                        "diagnostics": (None if readiness["ready"] else
                                        self.diagnostics(
                                            runtime_state=readiness["state"],
                                            checkpoint=checkpoint)),
                    }
                if last_state == "failed":
                    return {
                        "started": False, "state": "failed",
                        "code": RUNTIME_START_FAILED,
                        "error": health["error"] or "FreeToken reported a fatal error",
                        "pid": process.pid, "port": self.port, "log": log_path,
                        "initialisation": initialisation,
                        "oomd_shield": oomd_shield,
                        "diagnostics": self.diagnostics(
                            runtime_state="failed", checkpoint=checkpoint),
                    }
            time.sleep(1.0)

        return {
            "started": False, "state": last_state, "code": RUNTIME_NOT_READY,
            "error": (f"FreeToken did not become ready within {timeout}s "
                      f"(last state: {last_state}); see {log_path}"),
            "pid": process.pid, "port": self.port, "log": log_path,
            "initialisation": initialisation,
            "oomd_shield": oomd_shield,
            "diagnostics": self.diagnostics(
                runtime_state=last_state, checkpoint=checkpoint),
        }

    # ------------------------------------------------------ startup log

    @staticmethod
    def _log_tail(log_path: str, limit: int = 65536) -> str:
        """The end of the server log, for diagnostics. Never raises."""
        try:
            with open(log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - limit))
                return handle.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def _note_slow_initialisation(self, log_path: str, report: dict) -> None:
        """Record known slow-path markers from the log, once each."""
        tail = self._log_tail(log_path)
        if not tail:
            return
        for marker in SLOW_INIT_MARKERS:
            if marker in tail and marker not in report["notes"]:
                report["notes"].append(marker)
                report["slow_path"] = True

    @staticmethod
    def _classify_exit(log_tail: str) -> str:
        """Name the failure class of a process that exited before ready.

        The checkpoint preflight should have caught a missing shard before
        launch; this is the second line, for an artifact the runtime asked
        for that the index did not name. Anything else is a start failure
        with the log tail attached — never an unexplained crash.
        """
        text = log_tail.lower()
        if "safetensors" in text and any(
                marker in text for marker in
                ("no such file", "not found", "does not exist", "missing",
                 "incomplete")):
            return MODEL_CACHE_INCOMPLETE
        return RUNTIME_START_FAILED

    # -------------------------------------------------------- readiness

    def verify_model(self) -> dict:
        """Confirm the endpoint exposes the model that was asked for."""
        expected = self.expected_model_name()
        discovered = self.models()
        if not discovered["ok"]:
            return {"ok": False, "expected": expected, "served": None,
                    "code": RUNTIME_NOT_READY,
                    "error": f"could not read /v1/models: {discovered['error']}"}
        if expected in discovered["models"]:
            return {"ok": True, "expected": expected, "served": expected,
                    "code": None, "error": None,
                    "context": discovered["context"].get(expected)}
        return {
            "ok": False, "expected": expected,
            "code": MODEL_IDENTITY_MISMATCH,
            "served": discovered["models"][0] if discovered["models"] else None,
            "error": (f"endpoint exposes {discovered['models']!r}, "
                      f"expected '{expected}'"),
        }

    def verify_context(self, discovered: dict | None = None) -> dict:
        """Check the live context against what the alias promises.

        The alias's `context` is what endpoint() hands to consumers, who
        budget by it. An endpoint advertising less than that is not READY:
        the promise would be broken on the first long prompt. More is fine —
        the alias states a floor it was qualified at, not a ceiling. An
        endpoint that does not advertise a figure at all cannot be checked
        and is reported as unverified rather than refused, because the
        27B and 35B aliases were qualified before /v1/models carried one.
        """
        expected = self.resolved.get("context")
        served = self.expected_model_name()
        if discovered is None:
            discovered = self.models()
        if not discovered.get("ok"):
            return {"ok": False, "checked": False, "code": RUNTIME_NOT_READY,
                    "expected": expected, "live": None,
                    "error": f"could not read /v1/models: {discovered.get('error')}"}
        live = (discovered.get("context") or {}).get(served)
        if expected is None or live is None:
            return {"ok": True, "checked": False, "code": None,
                    "expected": expected, "live": live, "error": None}
        if int(live) < int(expected):
            return {"ok": False, "checked": True,
                    "code": CONTEXT_CAPABILITY_MISMATCH,
                    "expected": expected, "live": live,
                    "error": (f"endpoint advertises context {live} for "
                              f"'{served}', alias promises {expected}")}
        return {"ok": True, "checked": True, "code": None,
                "expected": expected, "live": live, "error": None}

    def readiness(self, alive: bool | None = None) -> dict:
        """The whole chain, in order, stopping at the first link that fails.

        process alive -> /health ok -> /v1/models answers -> a model is
        served -> it is the expected one -> its context is acceptable ->
        READY. A process that exists is the *first* of six conditions, not
        the conclusion; on a runtime this size, a client handed an endpoint
        that fails any later link pays for a full load to find out.

        `alive` is a fact the caller already holds (start() from poll(),
        status() from its pid check); None means "no process is tracked",
        which is the case for an external runtime being inspected.
        """
        base = {"ready": False, "served": None, "context": None}
        if alive is False:
            return {**base, "stage": "process", "state": "stopped",
                    "code": RUNTIME_NOT_READY,
                    "error": "the tracked process is not alive"}

        health = self.health()
        if not health["reachable"]:
            return {**base, "stage": "health", "state": "starting",
                    "code": RUNTIME_NOT_READY,
                    "error": f"/health unreachable: {health['error']}"}
        if health["state"] != "ready":
            return {**base, "stage": "health", "state": health["state"],
                    "code": (RUNTIME_START_FAILED if health["state"] == "failed"
                             else RUNTIME_NOT_READY),
                    "error": health["error"] or f"/health reports {health['state']}"}

        discovered = self.models()
        if not discovered["ok"]:
            return {**base, "stage": "models", "state": "unhealthy",
                    "code": RUNTIME_NOT_READY,
                    "error": f"could not read /v1/models: {discovered['error']}"}
        if not discovered["models"]:
            return {**base, "stage": "served_model", "state": "unhealthy",
                    "code": RUNTIME_NOT_READY,
                    "error": "/v1/models lists no model"}

        identity = self.verify_model()
        if not identity["ok"]:
            return {**base, "stage": "identity", "state": "model_mismatch",
                    "served": identity.get("served"),
                    "code": identity["code"], "error": identity["error"]}

        context = self.verify_context(discovered)
        if not context["ok"]:
            return {**base, "stage": "context", "state": "context_mismatch",
                    "served": identity["served"], "context": context["live"],
                    "code": context["code"], "error": context["error"]}

        return {"ready": True, "stage": "ready", "state": "ready", "code": None,
                "error": None, "served": identity["served"],
                "context": context["live"] if context["checked"]
                else self.resolved.get("context")}

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
        code = None
        error = None
        if health["reachable"]:
            state = health["state"]
            if state == "ready":
                # /health saying ok is a necessary condition, not READY:
                # the served model and its context still have to be the
                # ones this alias promises. A mismatch is still `running`
                # — a caller that auto-starts on "not running" would put a
                # second runtime on the card — but it is not `ready`.
                # Liveness is deliberately not passed: an answering /health
                # outranks a pid record, which can be stale or belong to a
                # runtime we did not start. start() holds the real fact.
                readiness = self.readiness()
                state = readiness["state"]
                code = readiness["code"]
                error = readiness["error"]
        elif self._port_open():
            state = "unhealthy"
        elif alive:
            state = "starting"
        else:
            state = "stopped"

        doc = health["doc"] or {}
        if error is None and state in ("failed", "unhealthy"):
            error = health["error"]
        return {
            "running": state in ("ready", "loading", "model_mismatch",
                                 "context_mismatch"),
            "ready": state == "ready",
            "state": state,
            "alive": alive,
            "healthy": state == "ready",
            "pid": pid,
            "port": self.port,
            "model": doc.get("model"),
            "code": code,
            "error": error,
        }

    def diagnostics(self, runtime_state: str | None = None,
                    checkpoint: dict | None = None) -> dict:
        """What a failure report has to carry to be acted on.

        Every field is read from configuration, bookkeeping or the
        filesystem; nothing here runs the binary or touches the network, so
        it can be attached to any result without changing the outcome.
        """
        try:
            executable = self.resolve_executable()
        except FreeTokenAdapterError as exc:
            executable = None
            executable_error = str(exc)
        else:
            executable_error = None
        if checkpoint is None:
            checkpoint = self.checkpoint_preflight()
        try:
            gpu = self.gpu_index()
        except FreeTokenAdapterError:
            gpu = self.resolved.get("gpu")
        return {
            "alias": self.alias,
            "provider": "freetoken",
            "model": self.resolved.get("model_path"),
            "served_model_expected": self._safe_expected_model_name(),
            "revision": checkpoint.get("revision"),
            "snapshot": checkpoint.get("snapshot"),
            "missing_artifact": (checkpoint.get("missing") or
                                 checkpoint.get("incomplete") or
                                 checkpoint.get("broken_symlinks") or None),
            "executable": executable,
            "executable_error": executable_error,
            "version": self._version_seen,
            "host": self.host,
            "port": self.port,
            "gpu": gpu,
            "runtime_profile": self.resolved.get("runtime_profile"),
            "runtime_state": runtime_state,
        }

    def _safe_expected_model_name(self) -> str | None:
        try:
            return self.expected_model_name()
        except FreeTokenAdapterError:
            return None

    # ------------------------------------------------------ fingerprint

    def runtime_fingerprint(self, version: str | None = None,
                            checkpoint: dict | None = None) -> dict:
        """Deterministic identity of the runtime this configuration starts.

        Covers what makes two runtimes interchangeable: provider, model
        reference and revision, served name, GPU, profile, the launch
        arguments that shape the process, and the executable (with its
        version when one has been observed). Request-level settings —
        reasoning effort above all — are deliberately absent: they vary per
        call and change nothing about which process is running.

        `version` is taken from the argument, else from the last version()
        preflight; it is never probed here, so the fingerprint is pure.
        """
        if checkpoint is None:
            checkpoint = self.checkpoint_preflight()
        try:
            executable = self.resolve_executable()
        except FreeTokenAdapterError:
            executable = None
        try:
            launch_args = self.build_argv()[2:]  # after <executable> serve
        except FreeTokenAdapterError:
            launch_args = None
        try:
            gpu = self.gpu_index()
        except FreeTokenAdapterError:
            gpu = None
        material = {
            "provider": "freetoken",
            "model": self.resolved.get("model_path"),
            "revision": checkpoint.get("revision"),
            "served_model": self._safe_expected_model_name(),
            "gpu": gpu,
            "runtime_profile": self.resolved.get("runtime_profile"),
            "host": self.host,
            "port": self.port,
            "launch_args": launch_args,
            "executable": executable,
            "version": version or self._version_seen,
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return {**material, "digest": digest}

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
        # Context comes from /v1/models first: it is configuration as the
        # endpoint states it, and the value the readiness chain verified.
        # /v1/stats is telemetry — its page_size has been seen to disagree
        # with the resolved runtime config — and is only the last resort.
        discovered = self.models()
        if discovered["ok"]:
            context = discovered["context"].get(served)
        telemetry = self.stats()
        if telemetry["ok"]:
            served = telemetry.get("model") or served
            kv_capacity = telemetry.get("kv_capacity")
        if context is None:
            context = self.resolved.get("context")
        if context is None and telemetry["ok"]:
            context = telemetry.get("context_max")
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

    def hub_cache_dir(self) -> str:
        """The Hugging Face hub cache this profile's weights would come from.

        Honours HF_HUB_CACHE, then HF_HOME, then the documented default —
        the same order the hub library itself uses.
        """
        explicit = os.environ.get("HF_HUB_CACHE")
        if explicit:
            return explicit
        home = os.environ.get("HF_HOME")
        if home:
            return os.path.join(home, "hub")
        return os.path.expanduser("~/.cache/huggingface/hub")

    def _hub_cache_candidates(self) -> list[str]:
        """Cache roots to look in, in the order the runtime resolves them.

        Two layouts have been seen on the qualified machine for the same
        repo: `~/.cache/huggingface/hub/models--…` and, beside it,
        `~/.cache/huggingface/models--…`. FreeToken resolved the `hub/` one,
        so it comes first; the sibling is checked only when hub/ has no
        entry, and never picked over an existing hub/ entry — that would be
        guessing a snapshot path the runtime will not use.
        """
        hub = self.hub_cache_dir()
        candidates = [hub]
        head, tail = os.path.split(hub.rstrip(os.sep))
        if tail == "hub" and head:
            candidates.append(head)
        return candidates

    def resolve_snapshot(self) -> dict:
        """Locate the snapshot the runtime would load, without guessing.

        For a repo id: the cache entry's `refs/main` names the revision and
        the snapshot is `snapshots/<revision>`. With no refs, a single
        snapshot directory is unambiguous; several without a ref are not,
        and are reported as such rather than picked from.
        """
        report = {"model": None, "kind": None, "cache_dir": None,
                  "entry": None, "snapshot": None, "revision": None,
                  "error": None}
        try:
            model = self._model_reference()
        except FreeTokenAdapterError as exc:
            report["error"] = str(exc)
            return report
        report["model"] = model
        if os.path.isdir(model):
            report.update(kind="local_directory", snapshot=model)
            return report
        if "/" not in model:
            report.update(kind="unknown", error="not a repo id or a directory")
            return report
        report["kind"] = "hub_repo"
        entry_name = "models--" + model.replace("/", "--")
        for cache in self._hub_cache_candidates():
            entry = os.path.join(cache, entry_name)
            if not os.path.isdir(entry):
                continue
            report.update(cache_dir=cache, entry=entry)
            ref = os.path.join(entry, "refs", "main")
            revision = None
            try:
                revision = Path(ref).read_text(encoding="utf-8").strip() or None
            except OSError:
                revision = None
            snapshots_dir = os.path.join(entry, "snapshots")
            if revision:
                snapshot = os.path.join(snapshots_dir, revision)
                if os.path.isdir(snapshot):
                    report.update(snapshot=snapshot, revision=revision)
                else:
                    report["error"] = (f"refs/main names {revision} but "
                                       f"snapshots/{revision} is missing")
                return report
            try:
                names = sorted(n for n in os.listdir(snapshots_dir)
                               if os.path.isdir(os.path.join(snapshots_dir, n)))
            except OSError:
                names = []
            if len(names) == 1:
                report.update(snapshot=os.path.join(snapshots_dir, names[0]),
                              revision=names[0])
            elif not names:
                report["error"] = "cache entry exists but holds no snapshot"
            else:
                report["error"] = (f"no refs/main and {len(names)} snapshots — "
                                   f"cannot tell which one the runtime loads")
            return report
        report["error"] = "not in the hub cache"
        return report

    def checkpoint_preflight(self) -> dict:
        """Prove the weights are complete, or say exactly what is not.

        For an indexed safetensors checkpoint "the directory exists" and "no
        .incomplete files" prove nothing: the Flash-Next repo was found on
        this machine in a second cache layout holding the index and none of
        its 206 shards. So the check is the index's own: it exists, it
        parses, every unique file in weight_map is present, every symlink
        among them resolves, and none of them is a partial download.

        Read-only by design. An incomplete cache is reported with the
        missing artifacts named; it is never repaired, pruned or
        re-downloaded from here — that is a decision about 100 GB of
        somebody's disk and network, not a startup step.
        """
        location = self.resolve_snapshot()
        report = {
            "ok": True, "checked": False, "code": None, "error": None,
            "model": location["model"], "kind": location["kind"],
            "snapshot": location["snapshot"], "revision": location["revision"],
            "cache_dir": location["cache_dir"],
            "index": None, "shards": 0,
            "missing": [], "incomplete": [], "broken_symlinks": [],
        }
        if location["kind"] == "unknown" or location["model"] is None:
            report["error"] = location["error"]
            return report
        if location["kind"] == "hub_repo" and location["entry"] is None:
            # Not cached at all. There is nothing to verify, and whether
            # the runtime fetches is the runtime's decision, not ours.
            report["error"] = location["error"]
            return report
        if location["snapshot"] is None:
            report.update(ok=False, checked=True, code=MODEL_CACHE_INCOMPLETE,
                          error=f"checkpoint cache unusable: {location['error']}")
            return report

        snapshot = location["snapshot"]
        index_path = os.path.join(snapshot, SAFETENSORS_INDEX)
        report["index"] = index_path
        report["checked"] = True
        if not os.path.lexists(index_path):
            single = os.path.join(snapshot, "model.safetensors")
            if os.path.lexists(single):
                # Not an indexed checkpoint; verify the one file the same way.
                report["index"] = None
                self._check_artifact(snapshot, "model.safetensors", report)
                report["shards"] = 1
                return self._finish_preflight(report)
            report.update(ok=False, code=MODEL_CACHE_INCOMPLETE,
                          missing=[SAFETENSORS_INDEX],
                          error=(f"{SAFETENSORS_INDEX} missing from {snapshot} "
                                 f"— indexed checkpoint cannot be verified"))
            return report
        if os.path.islink(index_path) and not os.path.exists(index_path):
            report.update(ok=False, code=MODEL_CACHE_INCOMPLETE,
                          broken_symlinks=[SAFETENSORS_INDEX],
                          error=f"{SAFETENSORS_INDEX} is a symlink that does not resolve")
            return report
        try:
            with open(index_path, encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index["weight_map"]
            if not isinstance(weight_map, dict):
                raise ValueError("weight_map is not a mapping")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report.update(ok=False, code=MODEL_CACHE_INCOMPLETE,
                          error=f"{SAFETENSORS_INDEX} is malformed: {exc}")
            return report

        files = sorted({str(name) for name in weight_map.values()})
        report["shards"] = len(files)
        for name in files:
            self._check_artifact(snapshot, name, report)
        return self._finish_preflight(report)

    @staticmethod
    def _check_artifact(snapshot: str, name: str, report: dict) -> None:
        """Classify one referenced file: present, missing, broken or partial."""
        path = os.path.join(snapshot, name)
        if name.endswith(".incomplete"):
            report["incomplete"].append(name)
            return
        if not os.path.lexists(path):
            report["missing"].append(name)
            return
        if os.path.islink(path) and not os.path.exists(path):
            # A hub symlink whose blob is still downloading points at a
            # name that only exists with an .incomplete suffix.
            try:
                target = os.path.join(os.path.dirname(path), os.readlink(path))
            except OSError:
                target = ""
            if target and os.path.exists(target + ".incomplete"):
                report["incomplete"].append(name)
            else:
                report["broken_symlinks"].append(name)
            return
        if os.path.realpath(path).endswith(".incomplete"):
            report["incomplete"].append(name)

    @staticmethod
    def _finish_preflight(report: dict) -> dict:
        problems = []
        if report["missing"]:
            problems.append(f"{len(report['missing'])} missing "
                            f"(first: {report['missing'][0]})")
        if report["broken_symlinks"]:
            problems.append(f"{len(report['broken_symlinks'])} unresolvable "
                            f"symlink(s) (first: {report['broken_symlinks'][0]})")
        if report["incomplete"]:
            problems.append(f"{len(report['incomplete'])} partial download(s) "
                            f"(first: {report['incomplete'][0]})")
        if problems:
            report.update(ok=False, code=MODEL_CACHE_INCOMPLETE,
                          error=(f"checkpoint at {report['snapshot']} is "
                                 f"incomplete: " + "; ".join(problems)))
        return report

    def model_is_cached(self) -> bool | None:
        """Whether this profile's weights are already on disk.

        Returns None when the question does not apply or cannot be answered
        — a local checkpoint path that exists needs no hub cache, and an
        unreadable cache directory is not evidence of absence. This is a
        presence question; checkpoint_preflight() is the integrity one.
        """
        try:
            model = self._model_reference()
        except FreeTokenAdapterError:
            return None
        if os.path.isdir(model):
            return True
        if "/" not in model:
            return None
        entry_name = "models--" + model.replace("/", "--")
        try:
            return any(os.path.isdir(os.path.join(cache, entry_name))
                       for cache in self._hub_cache_candidates())
        except OSError:
            return None

    def capabilities(self) -> dict:
        """Report capabilities, detected where they can be.

        `offline_cached_models` used to be hardcoded True. That was an
        assumption wearing a measurement's clothes: nothing had checked
        whether this profile's weights were actually on disk, and a consumer
        planning an offline run would have been told yes regardless. It is
        now derived from the hub cache, and None when the question cannot be
        answered rather than a guess in either direction.
        """
        checkpoint = self.checkpoint_preflight()
        return {
            "provider": "freetoken",
            "capabilities": {
                "openai_api": True,
                "runtime_stats": True,
                "gpu_selection": self.gpu_index() is not None,
                "huggingface_models": True,
                "offline_cached_models": self.model_is_cached(),
                # Presence is not completeness: the cache can hold an index
                # and none of its shards. None when there is nothing to
                # verify (not cached, or not a checkpoint we can locate).
                "checkpoint_complete": (checkpoint["ok"] if checkpoint["checked"]
                                        else None),
                # The alias's promise; the readiness chain verifies the live
                # figure from /v1/models against it before READY.
                "context_length": self.resolved.get("context"),
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
