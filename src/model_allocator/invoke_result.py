"""Generic InvokeResult envelope (ONYX Adapter Architecture spec §5/§7).

Every invoke-capable provider returns this exact shape so consumers (the
headless runner, the MCP server, bridgeV002 roles) never need
provider-specific parsing. Citations are empty for plain LLM providers and
populated by knowledge providers such as ONYX.
"""

from __future__ import annotations

from typing import Optional

INVOKE_RESULT_VERSION = "1.0"


def make_invoke_result(
    provider: str,
    text: str,
    citations: Optional[list] = None,
    error: Optional[str] = None,
    elapsed_ms: Optional[float] = None,
    metadata: Optional[dict] = None,
) -> dict:
    return {
        "invoke_result_version": INVOKE_RESULT_VERSION,
        "status": "error" if error else "ok",
        "provider": provider,
        "text": text,
        "citations": citations or [],
        "error": error,
        "metadata": {
            **(metadata or {}),
            "elapsed_ms": round(elapsed_ms, 1) if elapsed_ms is not None else None,
        },
    }
