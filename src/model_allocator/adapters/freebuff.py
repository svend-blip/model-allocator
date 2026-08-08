"""freebuff client adapter.

freebuff is a self-contained coding TUI started by its own name -- it
carries its own model configuration and needs nothing resolved for it. The
adapter exists so roles using freebuff take the SAME start path as every
other client (one allocator, one `run` command, one place start_coding
calls) instead of growing a parallel launch mechanism in the bridge.
Deliberately thin: if freebuff ever takes an allocator-selected model or
environment, this is already the place it goes.
"""

from __future__ import annotations

import os
import shutil
from typing import Any


def build_freebuff_command(resolved: dict) -> dict[str, Any]:
    binary = os.environ.get("FREEBUFF_BIN", "freebuff")
    if not os.path.isabs(binary) and shutil.which(binary) is None:
        raise ValueError(
            "freebuff binary not found on PATH (set FREEBUFF_BIN to override)")
    return {"env": {}, "argv": [binary]}
