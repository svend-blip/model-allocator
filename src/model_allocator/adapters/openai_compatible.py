"""OpenAI-compatible cloud backend adapter (OpenRouter, Minimax, etc.)."""

from __future__ import annotations

import os
import urllib.request
from typing import Any


class OpenAICompatibleAdapterError(Exception):
    pass


class OpenAICompatibleAdapter:
    def __init__(self, api_base: str = "", api_key_env: str = ""):
        self.api_base = (api_base or "").rstrip("/")
        self.api_key_env = api_key_env

    @staticmethod
    def api_base_from_profile(profile: dict) -> str:
        """Resolve API base from env var name or default."""
        env_name = profile.get("api_base_env")
        default_base = profile.get("default_api_base") or ""
        if not env_name:
            return default_base
        return os.environ.get(env_name, "") or default_base

    def _request(self, path: str = "", method: str = "GET", timeout: int = 5) -> Any:
        if not self.api_base:
            raise OpenAICompatibleAdapterError("API base not configured")
        url = f"{self.api_base}{path}"
        req = urllib.request.Request(url, method=method)
        req.add_header("User-Agent", "model-allocator/0.2.0")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return {"status_code": resp.getcode(), "body": resp.read().decode("utf-8", errors="ignore")}
        except urllib.error.HTTPError as exc:
            # A 4xx/5xx from the base URL still means the endpoint is reachable.
            return {"status_code": exc.code, "body": "", "error": exc.reason}
        except Exception as exc:
            raise OpenAICompatibleAdapterError(f"API base unreachable: {exc}")

    def is_api_reachable(self) -> dict:
        """Probe the configured API base with a GET request.

        Returns a dict with keys ``reachable`` (bool), ``status_code``
        (int or None) and ``error`` (str or None). A 4xx/5xx response
        still counts as reachable -- only a transport failure is
        reported as unreachable.
        """
        try:
            result = self._request("/")
            return {"reachable": True, "status_code": result.get("status_code"), "error": None}
        except OpenAICompatibleAdapterError as exc:
            return {"reachable": False, "status_code": None, "error": str(exc)}

    def are_credentials_present(self) -> dict:
        if not self.api_key_env:
            return {"present": False, "error": "No API key environment variable configured"}
        value = os.environ.get(self.api_key_env, "")
        if not value:
            return {"present": False, "error": f"Environment variable '{self.api_key_env}' is not set"}
        return {"present": True, "error": None}

    def status(self) -> dict:
        reachable = self.is_api_reachable()
        credentials = self.are_credentials_present()
        return {
            "reachable": reachable["reachable"],
            "credentials_present": credentials["present"],
            "api_base": self.api_base,
            "error": reachable.get("error") or credentials.get("error"),
        }

    def start(self) -> dict:
        """Cloud warm-up is validation-only."""
        status = self.status()
        if not status["reachable"]:
            return {"started": False, "error": status["error"]}
        if not status["credentials_present"]:
            return {"started": False, "error": "API key not configured"}
        return {"started": True, "error": None}

    def stop(self) -> dict:
        """Cloud stop is a no-op; backend lifecycle is remote."""
        return {"stopped": True, "error": None}

    def unload(self) -> dict:
        """Cloud unload is a no-op."""
        return {"unloaded": True, "error": None}
