"""Local Ollama backend adapter (V1A — read-only)."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


class OllamaAdapterError(Exception):
    pass


class OllamaAdapter:
    def __init__(self, api_base: str = "", real_model: str = "", context: int | None = None):
        self.api_base = api_base.rstrip("/")
        self.real_model = real_model
        self.context = context

    def _request(self, path: str, method: str = "GET", data: bytes | None = None) -> Any:
        if not self.api_base:
            raise OllamaAdapterError("OLLAMA_BASE_URL not configured")
        url = f"{self.api_base}{path}"
        req = urllib.request.Request(url, method=method, data=data)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
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
        # Ollama tag names sometimes omit :latest.
        variants = {self.real_model, self.real_model.rsplit(":", 1)[0]}
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

    @staticmethod
    def api_base_from_profile(profile: dict) -> str:
        """Resolve API base from env var name or default."""
        env_name = profile.get("api_base_env", "OLLAMA_BASE_URL")
        default_base = profile.get("default_api_base", "http://127.0.0.1:11434")
        value = os.environ.get(env_name, "")
        return value or default_base
