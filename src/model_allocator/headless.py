"""Headless runner — the generic client for invoke-capable runtimes (V5B).

BridgeV002's execution model is: a client runs in the role's tmux session,
dispatch pastes a prompt into the pane and presses Enter. TUI clients
(OpenCode/Claude Code) consume that natively; API-only runtimes such as
ONYX have no TUI. This runner closes the gap WITHOUT any bridge changes:

    tmux pane -> headless runner (stdin) -> allocator invoke() -> answer
    printed to the pane + full InvokeResult JSON written to output-dir

Prompt framing: dispatch pastes multi-line text in one burst, so the
runner batches stdin lines and treats an idle gap (default 2.0s) after
received content as end-of-prompt. EOF ends the runner.

The runner is a TRANSPORT, not an agent: it never interprets prompt
content, never writes bridge deliverables itself, and never executes
instructions. Roles that need convention files get them via bridge step
configuration, not via prompt parsing here.
"""

from __future__ import annotations

import json
import os
import select
import sys
import time
from typing import Callable, Optional, TextIO

DEFAULT_IDLE_SECONDS = 2.0
POLL_SECONDS = 0.1
READY_BANNER = "HEADLESS RUNNER READY"


def read_prompt(
    stream: TextIO,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    poll_seconds: float = POLL_SECONDS,
) -> Optional[str]:
    """Collect one prompt from the stream.

    Data is accumulated with unbuffered os.read (buffered readline would
    swallow bytes into userspace where select can no longer see them);
    once content exists and no new data arrives for `idle_seconds`, the
    batch is returned as one prompt. Returns None on EOF with no pending
    content (caller should exit).
    """
    chunks: list[bytes] = []
    idle_deadline: Optional[float] = None
    fd = stream.fileno()
    while True:
        readable, _, _ = select.select([fd], [], [], poll_seconds)
        if readable:
            data = os.read(fd, 65536)
            if data == b"":  # EOF
                if chunks:
                    return b"".join(chunks).decode("utf-8", errors="replace")
                return None
            chunks.append(data)
            idle_deadline = time.monotonic() + idle_seconds
            continue
        if chunks and idle_deadline is not None and time.monotonic() >= idle_deadline:
            return b"".join(chunks).decode("utf-8", errors="replace")


def _write_result(output_dir: str, sequence: int, result: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    epoch_ms = int(time.time() * 1000)
    path = os.path.join(output_dir, f"invoke-{sequence:04d}-{epoch_ms}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    latest = os.path.join(output_dir, "latest.json")
    tmp = latest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, latest)
    return path


def run_loop(
    stream: TextIO,
    invoke_fn: Callable[[str], dict],
    out: TextIO,
    output_dir: Optional[str] = None,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    banner_extra: str = "",
) -> int:
    """The runner loop, decoupled from CLI/adapter wiring for testability."""
    print(f"{READY_BANNER} {banner_extra}".rstrip(), file=out, flush=True)
    sequence = 0
    while True:
        prompt = read_prompt(stream, idle_seconds=idle_seconds)
        if prompt is None:
            print("[headless] EOF — exiting", file=out, flush=True)
            return 0
        if not prompt.strip():
            continue
        sequence += 1
        print(f"[headless] invoke #{sequence} ({len(prompt)} chars)…",
              file=out, flush=True)
        result = invoke_fn(prompt)
        text = result.get("text") or ""
        status = result.get("status")
        print(text, file=out, flush=True)
        citations = result.get("citations") or []
        if citations:
            print("[headless] citations:", file=out)
            for c in citations:
                print(f"  [{c.get('citation_number')}] "
                      f"{c.get('title')} {c.get('link') or ''}",
                      file=out, flush=True)
        if status != "ok":
            print(f"[headless] ERROR: {result.get('error')}",
                  file=out, flush=True)
        if output_dir:
            path = _write_result(output_dir, sequence, result)
            print(f"[headless] result written: {path}", file=out, flush=True)
        elapsed = (result.get("metadata") or {}).get("elapsed_ms")
        print(f"[headless] done #{sequence} status={status} "
              f"elapsed_ms={elapsed}", file=out, flush=True)


def main_from_args(args, resolver_factory) -> int:
    """CLI wiring: resolve the alias once, then loop until EOF."""
    from model_allocator.cli import _get_backend_adapter  # late: avoid cycle

    resolver = resolver_factory()
    resolved = resolver.resolve_alias(args.alias)
    capabilities = resolved.get("capabilities") or []
    if "invoke" not in capabilities:
        print(
            f"ERROR: alias '{args.alias}' does not declare the 'invoke' "
            "capability — the headless runner requires it",
            file=sys.stderr,
        )
        return 1
    adapter = _get_backend_adapter(resolved)

    def invoke_fn(prompt: str) -> dict:
        kwargs = {}
        if args.persona is not None:
            kwargs["persona_id"] = args.persona
        if args.timeout:
            kwargs["timeout"] = args.timeout
        result = adapter.invoke(prompt, **kwargs)
        result.setdefault("metadata", {})["alias"] = args.alias
        return result

    return run_loop(
        sys.stdin,
        invoke_fn,
        sys.stdout,
        output_dir=args.output_dir,
        idle_seconds=args.idle_seconds,
        banner_extra=f"alias={args.alias} backend={resolved.get('backend')}",
    )
