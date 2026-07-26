"""Anthropic native API adapter for model-allocator.

Handles runtime lifecycle for the anthropic backend. This is a cloud
backend — start/stop/unload are credential-check-only (cloud_noop).
"""

from __future__ import annotations

import os
from typing import Any


class AnthropicAdapter:
    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY"):
        self.api_key_env = api_key_env

    def are_credentials_present(self) -> dict:
        value = os.environ.get(self.api_key_env, "")
        if not value:
            return {
                "present": False,
                "error": f"Environment variable '{self.api_key_env}' is not set",
            }
        return {"present": True, "error": None}

    def status(self) -> dict:
        credentials = self.are_credentials_present()
        return {
            "reachable": True,
            "credentials_present": credentials["present"],
            "error": credentials.get("error"),
        }

    def start(self) -> dict[str, Any]:
        status = self.status()
        if not status["credentials_present"]:
            return {"started": False, "error": status["error"]}
        return {"started": True, "error": None}

    def stop(self) -> dict[str, Any]:
        return {"stopped": True, "error": None}

    def unload(self) -> dict[str, Any]:
        return {"unloaded": True, "error": None}
