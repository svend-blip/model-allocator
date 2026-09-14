"""OpenAI-compatible cloud backend adapter (OpenRouter, Minimax, etc.)."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Any

from model_allocator.invoke_result import make_invoke_result


class OpenAICompatibleAdapterError(Exception):
    pass


# One lock per endpoint, process-wide. A remote node declared as a single
# worker (``max_running_requests: 1`` on its profile) serializes with its
# runtime's own request limit; this lock keeps THIS process from ever
# holding two in-flight invocations against such a node, so the second
# caller queues here instead of at the node. It is not a scheduler and
# never routes anywhere else.
_ENDPOINT_LOCKS: dict[str, threading.Lock] = {}
_ENDPOINT_LOCKS_GUARD = threading.Lock()


def _endpoint_lock(api_base: str) -> threading.Lock:
    with _ENDPOINT_LOCKS_GUARD:
        lock = _ENDPOINT_LOCKS.get(api_base)
        if lock is None:
            lock = threading.Lock()
            _ENDPOINT_LOCKS[api_base] = lock
        return lock


class OpenAICompatibleAdapter:
    def __init__(
        self,
        api_base: str = "",
        api_key_env: str = "",
        real_model: str = "",
        enable_thinking: bool | None = None,
        max_running_requests: int | None = None,
        provider: str = "",
        invoke_timeout: int | None = None,
    ):
        self.api_base = (api_base or "").rstrip("/")
        self.api_key_env = api_key_env
        # Invocation identity and per-alias request options (all optional;
        # the routing-only uses of this adapter never set them).
        self.real_model = real_model or ""
        self.enable_thinking = enable_thinking
        self.max_running_requests = max_running_requests
        self.provider = provider or "openai_compatible"
        self.invoke_timeout = invoke_timeout or 600

    @property
    def single_worker(self) -> bool:
        """True when the node is declared to run one request at a time."""
        return self.max_running_requests == 1

    @staticmethod
    def api_base_from_profile(profile: dict) -> str:
        """Resolve API base from env var name or default."""
        env_name = profile.get("api_base_env")
        default_base = profile.get("default_api_base") or ""
        if not env_name:
            return default_base
        return os.environ.get(env_name, "") or default_base

    def _request(
        self,
        path: str = "",
        method: str = "GET",
        timeout: int = 5,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> Any:
        if not self.api_base:
            raise OpenAICompatibleAdapterError("API base not configured")
        url = f"{self.api_base}{path}"
        req = urllib.request.Request(url, method=method, data=body)
        req.add_header("User-Agent", "model-allocator/0.2.0")
        for name, value in (headers or {}).items():
            req.add_header(name, value)
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

    def is_model_available(self, model: str = "") -> dict:
        """Check the configured model is listed at the endpoint's model list.

        Probes ``/models`` (when the base already includes ``/v1``) and
        ``/v1/models`` (when the base stops at the host), so it works for
        both base-URL conventions used across the runtime profiles. Model
        availability is a *where-practical* check per the Remote AI-PC
        Model Endpoints addendum: an endpoint that does not expose a model
        list, or a transport failure, is reported as an inability to
        confirm (``available`` False with an ``error``) rather than a hard
        failure, which the validator surfaces as a warning.
        """
        if not model:
            return {"available": False, "error": "no model configured"}
        last_error = None
        for path in ("/models", "/v1/models"):
            try:
                result = self._request(path)
            except OpenAICompatibleAdapterError as exc:
                last_error = str(exc)
                continue
            if result.get("status_code") not in (200, None):
                last_error = f"{path} returned status {result.get('status_code')}"
                continue
            try:
                doc = json.loads(result.get("body") or "{}")
            except ValueError as exc:
                last_error = f"could not parse {path}: {exc}"
                continue
            entries = doc.get("data") if isinstance(doc, dict) else None
            ids = [e.get("id") for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
            if model in ids:
                return {"available": True, "error": None}
            last_error = f"model '{model}' not listed at {path}"
        return {"available": False, "error": last_error or "no model list endpoint responded"}

    def build_chat_payload(self, prompt: str) -> dict:
        """The OpenAI-compatible chat body for one prompt.

        ``enable_thinking`` travels as ``chat_template_kwargs`` — the field
        FreeToken (and vLLM/SGLang) read for hybrid-thinking models — only
        when the alias sets it; an alias without the field sends nothing and
        leaves the model's own default in force.
        """
        payload: dict[str, Any] = {
            "model": self.real_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        return payload

    def invoke(self, prompt: str, timeout: int | None = None, **_ignored) -> dict:
        """One-shot chat completion; returns the generic InvokeResult envelope.

        Never raises for upstream failures: an unreachable endpoint, a
        non-200 status or an unparsable body all land in ``error``. On a
        single-worker node (``max_running_requests: 1``) invocations from
        this process are serialized per endpoint, so a second caller waits
        for the first instead of running concurrently against the node.
        """
        if not self.real_model:
            return make_invoke_result(self.provider, "", error="no model configured")
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = json.dumps(self.build_chat_payload(prompt)).encode("utf-8")
        lock = _endpoint_lock(self.api_base) if self.single_worker else None
        started = time.monotonic()
        if lock is not None:
            lock.acquire()
        try:
            result = self._request(
                "/chat/completions",
                method="POST",
                timeout=timeout or self.invoke_timeout,
                body=body,
                headers=headers,
            )
        except OpenAICompatibleAdapterError as exc:
            return make_invoke_result(
                self.provider, "", error=str(exc),
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                metadata={"model": self.real_model, "api_base": self.api_base},
            )
        finally:
            if lock is not None:
                lock.release()
        elapsed_ms = (time.monotonic() - started) * 1000.0
        metadata = {"model": self.real_model, "api_base": self.api_base}
        status_code = result.get("status_code")
        if status_code != 200:
            return make_invoke_result(
                self.provider, "",
                error=f"/chat/completions returned status {status_code}: {result.get('error') or result.get('body', '')[:200]}",
                elapsed_ms=elapsed_ms, metadata=metadata,
            )
        try:
            doc = json.loads(result.get("body") or "{}")
            choice = (doc.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            metadata["finish_reason"] = choice.get("finish_reason")
            metadata["usage"] = doc.get("usage")
        except (ValueError, AttributeError, IndexError, TypeError) as exc:
            return make_invoke_result(
                self.provider, "", error=f"could not parse /chat/completions body: {exc}",
                elapsed_ms=elapsed_ms, metadata=metadata,
            )
        return make_invoke_result(self.provider, text, elapsed_ms=elapsed_ms, metadata=metadata)

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
