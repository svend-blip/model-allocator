"""Render command objects to tmux-safe shell strings."""

from __future__ import annotations

import shlex
from typing import Any


def render_tmux_shell_string(command_object: dict[str, Any]) -> str:
    """Render a command object to a tmux-safe shell string.

    Mirrors scripts/bridgeV002/command_builder.render_tmux_shell_string:
    env vars precede the binary, values containing '$' use double quotes to
    allow shell expansion, all other values use shlex.quote().
    """
    env = command_object.get("env", {})
    argv = command_object.get("argv", [])

    if not argv:
        raise ValueError("Command object missing argv")

    env_parts = []
    for key, value in env.items():
        val_str = str(value)
        if "$" in val_str:
            env_parts.append(f'{key}="{val_str}"')
        else:
            env_parts.append(f"{key}={shlex.quote(val_str)}")

    argv_part = " ".join(shlex.quote(str(arg)) for arg in argv)

    if env_parts:
        return " ".join(env_parts + [argv_part])
    return argv_part
