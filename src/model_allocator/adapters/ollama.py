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

    def start_model(self) -> dict:
        """Warm up the Ollama model via the generate API.

        Uses a minimal prompt and keep_alive so the runtime loads the model
        into memory without performing useful generation.
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
            return {"started": True, "error": None}
        except Exception as exc:
            return {"started": False, "error": f"Ollama start failed: {exc}"}

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
