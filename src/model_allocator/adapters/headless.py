"""Headless client adapter — builds the tmux command for the runner (V5B).

Parallel to the opencode/claude_code client adapters: `run --client
headless` resolves a role's alias and emits a tmux-safe shell string that
starts the headless runner in the role's session. Any invoke-capable
backend works; bridgeV002 needs zero changes (same start_coding + dispatch
injection contract as TUI clients).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any


def _resolve_allocator_invocation() -> list[str]:
    """Resolve how to start the allocator CLI, without hardcoded paths.

    Priority: MODEL_ALLOCATOR_BIN env -> `model-allocator` on PATH ->
    repo wrapper script (deterministic relative to this file) ->
    `python3 -m model_allocator`.
    """
    override = os.environ.get("MODEL_ALLOCATOR_BIN")
    if override:
        return [override]
    on_path = shutil.which("model-allocator")
    if on_path:
        return [on_path]
    wrapper = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "model-allocator"
    if wrapper.exists():
        return [str(wrapper)]
    return [sys.executable or "python3", "-m", "model_allocator"]


def build_headless_command(resolved: dict, role_key: str) -> dict[str, Any]:
    """Build the command object for the headless runner."""
    alias = resolved.get("alias")
    if not alias:
        raise ValueError("Resolved alias is missing its alias name")
    if "invoke" not in (resolved.get("capabilities") or []):
        raise ValueError(
            f"Alias '{alias}' does not declare the 'invoke' capability "
            "required by the headless client"
        )
    argv = _resolve_allocator_invocation() + ["headless", "--alias", alias]
    output_dir = resolved.get("headless_output_dir")
    if output_dir:
        argv += ["--output-dir", str(output_dir)]
    idle = resolved.get("headless_idle_seconds")
    if idle:
        argv += ["--idle-seconds", str(idle)]
    return {"env": {}, "argv": argv}
