"""Local Ollama backend adapter (V1A/V1B)."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from typing import Any


class OllamaAdapterError(Exception):
    pass


class OllamaAdapter:
    def __init__(self, api_base: str = "", real_model: str = "", context: int | None = None):
        self.api_base = api_base.rstrip("/")
        self.real_model = real_model
        self.context = context

    def _request(
        self,
        path: str,
        method: str = "GET",
        data: bytes | None = None,
        timeout: int = 5,
    ) -> Any:
        if not self.api_base:
            raise OllamaAdapterError("OLLAMA_BASE_URL not configured")
        url = f"{self.api_base}{path}"
        req = urllib.request.Request(url, method=method, data=data)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise OllamaAdapterError(f"Ollama HTTP {exc.code}: {exc.reason}")
        except Exception as exc:
            raise OllamaAdapterError(f"Ollama unreachable: {exc}")

    def is_api_reachable(self) -> dict:
        """Return whether the Ollama API base is reachable."""
        try:
            self._request("/api/tags")
            return {"reachable": True, "error": None}
        except OllamaAdapterError as exc:
            return {"reachable": False, "error": str(exc)}

    def is_model_available(self) -> dict:
        """Return whether real_model is known to the Ollama instance."""
        try:
            data = self._request("/api/tags")
        except OllamaAdapterError as exc:
            return {"available": False, "error": str(exc)}
        models = data.get("models", [])
        names = {m.get("name", "") for m in models}
        # Ollama API always returns tags (:latest); real_model may omit them.
        variants = {self.real_model, self.real_model.rsplit(":", 1)[0]}
        # If real_model has no tag, also try with :latest appended.
        if ":" not in self.real_model:
            variants.add(self.real_model + ":latest")
        available = bool(names & variants)
        return {"available": available, "error": None if available else "Model not found in Ollama"}

    def runtime_status(self) -> dict:
        """Return runtime status from Ollama ps endpoint."""
        try:
            data = self._request("/api/ps")
        except OllamaAdapterError as exc:
            return {"running": False, "error": str(exc), "models": []}
        running_models = [m.get("name", "") for m in data.get("models", [])]
        return {
            "running": bool(running_models),
            "models": running_models,
            "error": None,
        }

    def _name_variants(self) -> set:
        """Every spelling Ollama might use for this model."""
        variants = {self.real_model, self.real_model.rsplit(":", 1)[0]}
        if ":" not in self.real_model:
            variants.add(self.real_model + ":latest")
        return variants

    def placement(self) -> dict:
        """Where Ollama actually put the model — GPU, CPU, or split.

        Ollama does not fail when VRAM is short. It loads whatever fits and
        runs the remainder on CPU, reporting success either way. The role
        still works; it is simply five to ten times slower, with nothing in
        any log to say why. Observed 2026-08-05: qwen3.6-27b ran at 99% CPU
        because another server still held the GPU, and the only trace was a
        human noticing the chain had gone quiet.

        gpu_fraction is size_vram/size — 1.0 when fully resident on the GPU,
        near 0 when the model is effectively running on the CPU.
        """
        try:
            data = self._request("/api/ps")
        except OllamaAdapterError as exc:
            return {"known": False, "error": str(exc)}
        variants = self._name_variants()
        for model in data.get("models", []):
            name = model.get("name", "")
            if name in variants or name.rsplit(":", 1)[0] in variants:
                total = model.get("size") or 0
                vram = model.get("size_vram") or 0
                return {
                    "known": True,
                    "size": total,
                    "size_vram": vram,
                    "gpu_fraction": (vram / total) if total else 0.0,
                    "error": None,
                }
        return {"known": False, "error": "model is not resident in Ollama"}

    def start_model(self) -> dict:
        """Warm up the Ollama model via the generate API, then check placement.

        Uses a minimal prompt and keep_alive so the runtime loads the model
        into memory without performing useful generation. A load that landed
        on the CPU is reported as a failure: handing a role a model that
        works but crawls is far harder to diagnose later than an error here.
        """
        if not self.api_base:
            return {"started": False, "error": "OLLAMA_BASE_URL not configured"}
        payload = {
            "model": self.real_model,
            "prompt": " ",
            "stream": False,
            "keep_alive": "5m",
        }
        if self.context:
            payload["options"] = {"num_ctx": self.context}
        data = json.dumps(payload).encode("utf-8")
        try:
            self._request("/api/generate", method="POST", data=data, timeout=60)
        except Exception as exc:
            return {"started": False, "error": f"Ollama start failed: {exc}"}

        min_gpu = float(os.environ.get("DPMTF_OLLAMA_MIN_GPU_FRACTION", "0.9"))
        place = self.placement()
        if not place.get("known"):
            # Cannot tell. Do not invent a failure, but say the check was not
            # made rather than implying the placement was verified.
            return {"started": True, "error": None, "placement": "unknown",
                    "placement_error": place.get("error")}

        fraction = place["gpu_fraction"]
        gib = 1024 ** 3
        result = {
            "started": True,
            "error": None,
            "gpu_fraction": round(fraction, 3),
            "size_vram": place["size_vram"],
            "size": place["size"],
        }
        if fraction < min_gpu:
            result["started"] = False
            result["error"] = (
                f"model loaded at {fraction:.0%} GPU "
                f"({place['size_vram'] / gib:.1f} of {place['size'] / gib:.1f} GiB "
                f"in VRAM) — below the {min_gpu:.0%} floor. Something else is "
                f"holding the GPU, or the model does not fit. Running it here "
                f"would be far slower than it looks."
            )
        return result

    def stop_model(self, timeout: int = 30) -> dict:
        """Stop the Ollama model via the `ollama` CLI with a timeout.

        Returns success if the model was stopped or was already unloaded.
        """
        if not self.real_model:
            return {"stopped": True, "error": None}
        try:
            result = subprocess.run(
                ["ollama", "stop", self.real_model],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return {"stopped": True, "error": None}
            stderr = (result.stderr or "").lower()
            if "not loaded" in stderr or "not found" in stderr:
                return {"stopped": True, "error": None}
            return {"stopped": False, "error": result.stderr.strip() or result.stdout.strip()}
        except subprocess.TimeoutExpired:
            return {"stopped": False, "error": f"ollama stop timed out after {timeout}s"}
        except Exception as exc:
            return {"stopped": False, "error": str(exc)}

    @staticmethod
    def api_base_from_profile(profile: dict) -> str:
        """Resolve API base from env var name or default."""
        env_name = profile.get("api_base_env", "OLLAMA_BASE_URL")
        default_base = profile.get("default_api_base", "http://127.0.0.1:11434")
        value = os.environ.get(env_name, "")
        return value or default_base
