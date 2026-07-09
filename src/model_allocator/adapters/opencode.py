"""OpenCode client adapter for model-allocator."""

from __future__ import annotations

import os
import shutil
from typing import Any


def build_opencode_command(resolved: dict, config_dir: str) -> dict[str, Any]:
    """Build an OpenCode + Ollama command object equivalent to command_builder.

    Equivalent to command_builder.build_opencode_ollama_command for the
    local_ollama provider: argv `[opencode, "--model", "ollama/<real_model>"]`
    and env `OPENCODE_CONFIG_DIR` + `OPENCODE_CONFIG` under the role's
    config directory.

    Paths are never hardcoded. `OPENCODE_ROLES_CONFIG_BASE` env var overrides
    the default config base; `OPENCODE_BIN` overrides the opencode binary.
    """
    real_model = resolved.get("real_model", "")
    if not real_model:
        raise ValueError("Resolved alias is missing real_model")

    opencode_bin = os.environ.get("OPENCODE_BIN", "opencode")
    if os.path.isabs(opencode_bin):
        if not (os.path.isfile(opencode_bin) and os.access(opencode_bin, os.X_OK)):
            raise ValueError(f"OpenCode binary not found: {opencode_bin}")
    elif shutil.which(opencode_bin) is None:
        raise ValueError(f"OpenCode binary not found on PATH: {opencode_bin}")

    config_base = os.environ.get("OPENCODE_ROLES_CONFIG_BASE", "$HOME/.config/opencode-roles")
    full_config_dir = f"{config_base}/{config_dir}"

    env: dict[str, str] = {
        "OPENCODE_CONFIG_DIR": full_config_dir,
        "OPENCODE_CONFIG": f"{full_config_dir}/opencode.json",
    }

    return {
        "env": env,
        "argv": [opencode_bin, "--model", f"ollama/{real_model}"],
    }
