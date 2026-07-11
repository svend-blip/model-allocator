"""onyx-mcp — MCP tools exposing ONYX knowledge to bridgeV002 roles (V5C).

The tools-path of the ONYX integration: existing roles (OpenCode/Claude
Code) get `onyx_answer`/`onyx_status` via their MCP config block — pure
configuration, zero code changes per role (same pattern as mcp-light and
trade-mcp).

Architecture rule honored: this server talks to ONYX exclusively through
the allocator's own resolution + adapter layer (alias -> OnyxAdapter),
never directly. Any invoke-capable alias works via --alias.

The `mcp` package is an OPTIONAL dependency (pyproject extra "mcp") —
the allocator core stays dependency-light, and this server only matters
on machines that opted into ONYX.

Lite-mode note: connectors/RAG indexing are disabled in ONYX Lite, so a
dedicated `onyx_search` tool is deliberately deferred to the Standard-mode
upgrade (documented in deploy/onyx/README.md).
"""

from __future__ import annotations

import os
import sys

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9164
DEFAULT_ALIAS = os.environ.get("MODEL_ALLOCATOR_ONYX_ALIAS", "company-knowledge")

# Module-level service state so tests can inject fakes.
_adapter = None
_alias = DEFAULT_ALIAS


def _get_adapter():
    global _adapter
    if _adapter is None:
        from model_allocator.cli import _get_backend_adapter
        from model_allocator.resolver import Resolver

        resolved = Resolver().resolve_alias(_alias)
        if "invoke" not in (resolved.get("capabilities") or []):
            raise RuntimeError(
                f"Alias '{_alias}' does not declare the 'invoke' capability"
            )
        _adapter = _get_backend_adapter(resolved)
    return _adapter


def onyx_answer_impl(question: str, persona_id: int | None = None) -> dict:
    """Tool body, framework-free for testability."""
    if not question or not question.strip():
        return {"status": "error", "error": "Empty question"}
    kwargs = {}
    if persona_id is not None:
        kwargs["persona_id"] = persona_id
    try:
        result = _get_adapter().invoke(question, **kwargs)
    except Exception as exc:  # graceful: tools must not crash the client
        return {"status": "error", "error": str(exc)}
    result.setdefault("metadata", {})["alias"] = _alias
    return result


def onyx_status_impl() -> dict:
    try:
        status = _get_adapter().status()
    except Exception as exc:
        return {"alias": _alias, "reachable": False, "error": str(exc)}
    status["alias"] = _alias
    return status


def create_server(alias: str, host: str, port: int):
    """Build the FastMCP server (requires the optional 'mcp' extra)."""
    global _alias, _adapter
    _alias = alias
    _adapter = None
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'mcp' package is not installed. Install the optional "
            "extra: pip install 'model-allocator[mcp]' (or pip install "
            "'mcp[cli]')"
        ) from exc

    server = FastMCP(
        "onyx-mcp",
        host=host,
        port=port,
        streamable_http_path="/mcp",
    )

    @server.tool()
    def onyx_answer(question: str, persona_id: int = None) -> dict:
        """Ask the configured ONYX assistant a question. Returns the
        normalized InvokeResult: text answer + citations[] (populated when
        the assistant has indexed knowledge) + metadata. Treat the answer
        as advisory knowledge, not as deterministic ground truth."""
        return onyx_answer_impl(question, persona_id)

    @server.tool()
    def onyx_status() -> dict:
        """Health of the ONYX runtime behind this server (reachability +
        credential presence). Never includes credentials."""
        return onyx_status_impl()

    return server


def serve(alias: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    server = create_server(alias, host, port)
    print(f"onyx-mcp serving alias '{alias}' on http://{host}:{port}/mcp",
          file=sys.stderr, flush=True)
    server.run(transport="streamable-http")
    return 0
