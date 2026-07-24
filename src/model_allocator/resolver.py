"""Logical alias resolver.

Given an alias name or a (role_key, client) pair, resolve the effective
backend, real_model, context, lifecycle_policy, clients, and profile fields.
"""

from __future__ import annotations

from typing import Any

from model_allocator.config_loader import load_config


class ResolutionError(Exception):
    pass


class Resolver:
    def __init__(self, config: dict | None = None, config_dir: str | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_dir)

    def resolve_alias(self, alias_name: str) -> dict:
        models = self.config.get("models", {})
        if alias_name not in models:
            raise ResolutionError(f"Alias '{alias_name}' not found")
        alias = dict(models[alias_name])
        profile_name = alias.get("runtime_profile")
        if not profile_name:
            raise ResolutionError(f"Alias '{alias_name}' has no runtime_profile")
        profiles = self.config.get("runtime_profiles", {})
        if profile_name not in profiles:
            raise ResolutionError(f"Runtime profile '{profile_name}' not found for alias '{alias_name}'")
        profile = dict(profiles[profile_name])
        merged = {
            "alias": alias_name,
            "runtime_profile": profile_name,
            "backend": profile.get("backend"),
            "api_base_env": profile.get("api_base_env"),
            "default_api_base": profile.get("default_api_base"),
            "gpu": profile.get("gpu"),
        }
        # Preserve every field declared on the alias. Alias fields override
        # matching profile fields, which is why profile backfill happens last.
        reserved = {"alias", "runtime_profile"}
        for key, value in alias.items():
            if key not in reserved:
                merged[key] = value
        # Merge any remaining backend-specific profile fields that are not already set.
        for key, value in profile.items():
            if key not in merged:
                merged[key] = value
        return merged

    def resolve_role_client(self, role_key: str, client: str) -> dict:
        roles = self.config.get("roles", {})
        role = roles.get(role_key)
        if not role:
            raise ResolutionError(f"Role '{role_key}' not found")
        alias_name = role.get("client_aliases", {}).get(client)
        if not alias_name:
            alias_name = role.get("default_alias")
        if not alias_name:
            raise ResolutionError(f"No alias configured for role '{role_key}' and client '{client}'")
        resolved = self.resolve_alias(alias_name)
        resolved["role_key"] = role_key
        if "config_dir" in role:
            resolved["config_dir"] = role["config_dir"]
        return resolved

    def list_aliases(self) -> list[str]:
        return list(self.config.get("models", {}).keys())

    def get_clients(self, alias_name: str) -> list[str]:
        """Return the list of declared client keys for an alias."""
        models = self.config.get("models", {})
        alias = models.get(alias_name, {})
        clients = alias.get("clients", {})
        return list(clients.keys()) if isinstance(clients, dict) else []
