"""ONYX enterprise-knowledge backend adapter (invoke-only runtime).

ONYX (github.com/onyx-dot-app/onyx) is an OPTIONAL runtime: it exposes
assistants (personas) over a chat API. This adapter owns every
ONYX-specific concern — authentication, session handling, invocation, and
response normalization — so the allocator core stays provider-agnostic.

Execution model: stateless one-shot invoke. Each invoke() creates a fresh
chat session via `chat_session_info` in the send-chat-message request
(verified live 2026-07-11: POST /chat/send-chat-message with stream=false
returns a complete ChatFullResponse JSON).

Authentication (in priority order):
1. API key (Business tier): env var named by `api_key_env` -> Bearer header
2. Basic login: env vars named by `email_env`/`password_env` ->
   POST /auth/login (form-encoded) -> session cookie forwarded to the chat
   call. Community/Lite instances gate API keys, so this is the default.

Lifecycle: start()/stop() are no-ops (`cloud_noop` policy) — the ONYX
stack's lifecycle belongs to docker compose, not the allocator.

The HTTP client is injectable for testability:
    http_client(url, method, headers, body, timeout)
        -> (status_code: int, body: str, response_headers: dict)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from model_allocator.invoke_result import make_invoke_result

DEFAULT_TIMEOUT_SECONDS = 120
LOGIN_TIMEOUT_SECONDS = 15
HEALTH_TIMEOUT_SECONDS = 5
USER_AGENT = "model-allocator/0.3 (onyx-adapter)"


class OnyxAdapterError(Exception):
    pass


def _default_http_client(
    url: str, method: str, headers: dict, body: Optional[bytes], timeout: int
) -> tuple[int, str, dict]:
    req = urllib.request.Request(url, data=body, method=method)
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (
                resp.getcode(),
                resp.read().decode("utf-8", errors="replace"),
                dict(resp.headers),
            )
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            exc.read().decode("utf-8", errors="replace") if exc.fp else "",
            dict(exc.headers or {}),
        )
    except Exception as exc:
        raise OnyxAdapterError(f"ONYX unreachable: {exc}") from exc


class OnyxAdapter:
    """Invoke-only adapter for ONYX assistants."""

    def __init__(
        self,
        api_base: str = "",
        api_key_env: str = "",
        email_env: str = "",
        password_env: str = "",
        persona_id: int = 0,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        http_client: Optional[Callable] = None,
    ):
        self.api_base = (api_base or "").rstrip("/")
        self.api_key_env = api_key_env
        self.email_env = email_env
        self.password_env = password_env
        self.persona_id = persona_id
        self.timeout = timeout
        self._http_client = http_client or _default_http_client

    @staticmethod
    def from_resolved(resolved: dict, http_client: Optional[Callable] = None) -> "OnyxAdapter":
        """Build the adapter from a resolver output (alias + profile merge)."""
        env_name = resolved.get("api_base_env")
        api_base = ""
        if env_name:
            api_base = os.environ.get(env_name, "")
        if not api_base:
            api_base = resolved.get("default_api_base") or ""
        return OnyxAdapter(
            api_base=api_base,
            api_key_env=resolved.get("api_key_env", ""),
            email_env=resolved.get("email_env", ""),
            password_env=resolved.get("password_env", ""),
            persona_id=int(resolved.get("persona_id", 0) or 0),
            timeout=int(resolved.get("invoke_timeout", DEFAULT_TIMEOUT_SECONDS)),
            http_client=http_client,
        )

    # ── Authentication ───────────────────────────────────────────

    def _auth_headers(self) -> dict:
        """Resolve auth: API key if present, else login for a session cookie."""
        api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if api_key:
            return {"Authorization": f"Bearer {api_key}"}

        email = os.environ.get(self.email_env, "") if self.email_env else ""
        password = os.environ.get(self.password_env, "") if self.password_env else ""
        if not email or not password:
            raise OnyxAdapterError(
                "No ONYX credentials: neither API key nor email/password "
                f"env vars ({self.api_key_env or 'unset'} / "
                f"{self.email_env or 'unset'}) are configured"
            )

        body = urllib.parse.urlencode(
            {"username": email, "password": password}
        ).encode("utf-8")
        status, text, resp_headers = self._http_client(
            f"{self.api_base}/auth/login",
            "POST",
            {
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": USER_AGENT,
            },
            body,
            LOGIN_TIMEOUT_SECONDS,
        )
        if status not in (200, 204):
            raise OnyxAdapterError(
                f"ONYX login failed: HTTP {status} {text[:120]}"
            )
        cookie = self._extract_cookies(resp_headers)
        if not cookie:
            raise OnyxAdapterError("ONYX login returned no session cookie")
        return {"Cookie": cookie}

    @staticmethod
    def _extract_cookies(resp_headers: dict) -> str:
        """Collect Set-Cookie values into a single Cookie header value."""
        cookies = []
        for key, value in resp_headers.items():
            if key.lower() == "set-cookie":
                values = value if isinstance(value, list) else [value]
                for item in values:
                    cookies.append(item.split(";", 1)[0])
        return "; ".join(c for c in cookies if c)

    # ── Invoke (the adapter's core capability) ───────────────────

    def invoke(
        self,
        prompt: str,
        persona_id: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """One-shot stateless invocation of an ONYX assistant.

        Returns the generic InvokeResult envelope; never raises for
        upstream answer errors (they land in the envelope's error field).
        """
        started = time.monotonic()
        effective_persona = self.persona_id if persona_id is None else persona_id
        try:
            headers = self._auth_headers()
            headers.update(
                {"Content-Type": "application/json", "User-Agent": USER_AGENT}
            )
            payload = {
                "message": prompt,
                "stream": False,
                "chat_session_info": {"persona_id": effective_persona},
            }
            status, text, _ = self._http_client(
                f"{self.api_base}/chat/send-chat-message",
                "POST",
                headers,
                json.dumps(payload).encode("utf-8"),
                timeout or self.timeout,
            )
            elapsed_ms = (time.monotonic() - started) * 1000.0
            if status != 200:
                return make_invoke_result(
                    provider="onyx",
                    text="",
                    error=f"HTTP {status}: {text[:200]}",
                    elapsed_ms=elapsed_ms,
                    metadata={"persona_id": effective_persona},
                )
            data = json.loads(text)
            answer = data.get("answer") or ""
            error_msg = data.get("error_msg")
            if not answer and error_msg:
                return make_invoke_result(
                    provider="onyx",
                    text="",
                    error=str(error_msg),
                    elapsed_ms=elapsed_ms,
                    metadata={"persona_id": effective_persona},
                )
            return make_invoke_result(
                provider="onyx",
                text=answer,
                citations=self._normalize_citations(data),
                elapsed_ms=elapsed_ms,
                metadata={
                    "persona_id": effective_persona,
                    "chat_session_id": data.get("chat_session_id"),
                    "message_id": data.get("message_id"),
                    "reasoning_present": bool(data.get("pre_answer_reasoning")),
                },
            )
        except OnyxAdapterError as exc:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            return make_invoke_result(
                provider="onyx", text="", error=str(exc), elapsed_ms=elapsed_ms
            )

    @staticmethod
    def _normalize_citations(data: dict) -> list[dict]:
        """Map ONYX citation_info/top_documents to the generic shape."""
        citations = []
        docs_by_id = {}
        for doc in data.get("top_documents") or []:
            if isinstance(doc, dict) and doc.get("document_id") is not None:
                docs_by_id[doc["document_id"]] = doc
        for item in data.get("citation_info") or []:
            if not isinstance(item, dict):
                continue
            doc_id = item.get("document_id")
            doc = docs_by_id.get(doc_id, {})
            citations.append(
                {
                    "citation_number": item.get("citation_num"),
                    "document_id": doc_id,
                    "title": doc.get("semantic_identifier"),
                    "link": doc.get("link"),
                    "source_type": doc.get("source_type"),
                }
            )
        return citations

    # ── Status / lifecycle ───────────────────────────────────────

    def status(self) -> dict:
        reachable = False
        error = None
        try:
            status, _, _ = self._http_client(
                f"{self.api_base}/health",
                "GET",
                {"User-Agent": USER_AGENT},
                None,
                HEALTH_TIMEOUT_SECONDS,
            )
            reachable = status == 200
            if not reachable:
                error = f"/health returned HTTP {status}"
        except OnyxAdapterError as exc:
            error = str(exc)

        credentials_present = bool(
            (self.api_key_env and os.environ.get(self.api_key_env))
            or (
                self.email_env
                and os.environ.get(self.email_env)
                and self.password_env
                and os.environ.get(self.password_env)
            )
        )
        return {
            "reachable": reachable,
            "credentials_present": credentials_present,
            "api_base": self.api_base,
            "error": error
            if error
            else (None if credentials_present else "No ONYX credentials configured"),
        }

    def start(self) -> dict:
        """No-op: the ONYX stack lifecycle belongs to docker compose."""
        status = self.status()
        if not status["reachable"]:
            return {"started": False, "error": status["error"]}
        return {"started": True, "error": None}

    def stop(self) -> dict:
        """No-op (cloud_noop lifecycle)."""
        return {"stopped": True, "error": None}
