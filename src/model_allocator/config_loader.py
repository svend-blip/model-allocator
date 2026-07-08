"""Configuration loader for model-allocator.

Resolution rules (V1A):
- Load allocator-local YAML/JSON config files.
- Environment variable references use the syntax ${ENV_VAR_NAME} or $ENV_VAR_NAME.
- Secrets are referenced by variable NAME only, never inlined.
- No opaque shell strings are parsed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


ENV_RE = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*)\}?)?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)


def _load_yaml(path: Path) -> Any:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_file(path: Path | str) -> Any:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    if suffix == ".json":
        return _load_json(path)
    # Fallback by content sniff: try JSON first, then YAML.
    try:
        return _load_json(path)
    except json.JSONDecodeError:
        return _load_yaml(path)


def resolve_env(value: Any) -> Any:
    """Recursively resolve environment variable references in structured data."""
    if isinstance(value, str):
        return _resolve_env_string(value)
    if isinstance(value, list):
        return [resolve_env(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve_env(val) for key, val in value.items()}
    return value


def _resolve_env_string(value: str) -> str:
    if not value:
        return value

    def repl(match: re.Match) -> str:
        name = match.group(1) or match.group(3)
        default = match.group(2)
        if name is None:
            return match.group(0)
        return os.environ.get(name, default if default is not None else "")

    return ENV_RE.sub(repl, value)


def load_config(config_dir: Path | str | None = None) -> dict:
    """Load allocator-local configuration.

    Loads models, roles, and runtime_profiles files from *config_dir*.
    If *config_dir* is omitted, the current working directory is used.
    Files are searched in this order (first match wins):
      models.yaml / models.json
      roles.yaml / roles.json
      runtime_profiles.yaml / runtime_profiles.json
    """
    config_dir = Path(config_dir) if config_dir else Path.cwd()

    def load(name: str) -> dict:
        for ext in (".yaml", ".yml", ".json"):
            path = config_dir / f"{name}{ext}"
            if path.exists():
                return resolve_env(load_file(path)) or {}
        return {}

    models = load("models").get("models", {})
    runtime_profiles = load("runtime_profiles").get("runtime_profiles", {})
    roles = load("roles").get("roles", {})

    return {
        "models": models,
        "runtime_profiles": runtime_profiles,
        "roles": roles,
    }
