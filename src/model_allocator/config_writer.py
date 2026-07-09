"""Read/write allocator config (models.yaml, roles.yaml) for the config editor.

Unlike config_loader.load_config, this loads RAW values (no ${ENV} resolution)
so edits round-trip without baking resolved env values back into the files.
runtime_profiles.yaml is read-only here and is never written.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml


class ConfigWriteError(Exception):
    """Raised when a config write is rejected (validation or IO)."""


def _find(config_dir: Path, name: str) -> Path:
    for ext in (".yaml", ".yml"):
        candidate = config_dir / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return config_dir / f"{name}.yaml"


def _raw_load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_raw(config_dir: str | Path) -> dict:
    """Load raw aliases/roles/profiles without env-var resolution."""
    d = Path(config_dir)
    return {
        "aliases": _raw_load(_find(d, "models")).get("models", {}) or {},
        "roles": _raw_load(_find(d, "roles")).get("roles", {}) or {},
        "profiles": _raw_load(_find(d, "runtime_profiles")).get("runtime_profiles", {}) or {},
    }


def _safe_write(path: Path, top_key: str, body: dict) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump({top_key: body}, fh, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)


def _role_alias_refs(role: dict) -> list:
    refs = []
    if role.get("default_alias"):
        refs.append(role["default_alias"])
    refs.extend((role.get("client_aliases") or {}).values())
    return refs


def set_alias(config_dir: str | Path, name: str, definition: dict) -> None:
    if not name:
        raise ConfigWriteError("alias name is required")
    d = Path(config_dir)
    raw = load_raw(d)
    profile = definition.get("runtime_profile")
    if profile and profile not in raw["profiles"]:
        raise ConfigWriteError(f"unknown runtime_profile: {profile}")
    aliases = raw["aliases"]
    aliases[name] = definition
    _safe_write(_find(d, "models"), "models", aliases)


def delete_alias(config_dir: str | Path, name: str) -> None:
    d = Path(config_dir)
    raw = load_raw(d)
    for role_name, role in raw["roles"].items():
        if name in _role_alias_refs(role):
            raise ConfigWriteError(f"alias '{name}' is referenced by role '{role_name}'")
    aliases = raw["aliases"]
    if name not in aliases:
        raise ConfigWriteError(f"unknown alias: {name}")
    del aliases[name]
    _safe_write(_find(d, "models"), "models", aliases)


def set_role(config_dir: str | Path, name: str, definition: dict) -> None:
    if not name:
        raise ConfigWriteError("role name is required")
    d = Path(config_dir)
    raw = load_raw(d)
    known = set(raw["aliases"].keys())
    for ref in _role_alias_refs(definition):
        if ref not in known:
            raise ConfigWriteError(f"role references unknown alias: {ref}")
    roles = raw["roles"]
    roles[name] = definition
    _safe_write(_find(d, "roles"), "roles", roles)


def delete_role(config_dir: str | Path, name: str) -> None:
    d = Path(config_dir)
    raw = load_raw(d)
    roles = raw["roles"]
    if name not in roles:
        raise ConfigWriteError(f"unknown role: {name}")
    del roles[name]
    _safe_write(_find(d, "roles"), "roles", roles)
