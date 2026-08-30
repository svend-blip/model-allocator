"""Logical alias resolver.

Given an alias name or a (role_key, client) pair, resolve the effective
backend, real_model, context, lifecycle_policy, clients, and profile fields.

V6 extension: an alias may declare ``runtime_instance`` and/or
``inference_profile`` references. ``runtime_instance`` owns physical
identity (model path, port, context, placement, KV cache, server bin,
profile binding) and the alias must NOT also set those fields — silent
precedence between alias and instance is forbidden. The alias's
``runtime_profile`` (if any) wins for backward compatibility; otherwise
the instance's ``runtime_profile`` is used.
"""

from __future__ import annotations

from typing import Any

from model_allocator.config_loader import load_config
from model_allocator.schema import (
    INFERENCE_PROFILE_DEFERRED_FIELDS,
    INSTANCE_OWNED_FIELDS,
)


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

        # V6 reference resolution: validate references and field ownership
        # BEFORE merging anything, so a misconfigured alias surfaces here
        # as a clean error rather than as a silently-wrong resolved view.
        runtime_instances = self.config.get("runtime_instances", {}) or {}
        inference_profiles = self.config.get("inference_profiles", {}) or {}

        instance_name = alias.get("runtime_instance")
        instance: dict | None = None
        if instance_name is not None:
            if instance_name not in runtime_instances:
                raise ResolutionError(
                    f"Alias '{alias_name}' references unknown "
                    f"runtime_instance '{instance_name}'"
                )
            instance = dict(runtime_instances[instance_name])
            for owned in INSTANCE_OWNED_FIELDS:
                if owned in alias:
                    raise ResolutionError(
                        f"Alias '{alias_name}' is bound to runtime_instance "
                        f"'{instance_name}' and must not also set "
                        f"'{owned}' (owned by the instance)"
                    )

        inference_profile_name = alias.get("inference_profile")
        inference_profile: dict | None = None
        if inference_profile_name is not None:
            if inference_profile_name not in inference_profiles:
                raise ResolutionError(
                    f"Alias '{alias_name}' references unknown "
                    f"inference_profile '{inference_profile_name}'"
                )
            inference_profile = dict(inference_profiles[inference_profile_name])
            # Deferred-field rejection: temperature / top_p have no
            # transport path in V6, so accepting them in the resolved view
            # would be the silent-do-nothing failure the Mission Contract
            # forbids. The doctor already rejects these; the resolver
            # catches them again so a config that bypasses the doctor
            # (e.g. a programmatic resolve_alias call) still fails loudly.
            for deferred_field in INFERENCE_PROFILE_DEFERRED_FIELDS:
                if deferred_field in inference_profile:
                    raise ResolutionError(
                        f"inference_profile '{inference_profile_name}' "
                        f"field '{deferred_field}' is not implemented in "
                        f"V6 (deferred) — remove it"
                    )

        # Backward compat: aliases without runtime_instance must keep
        # working exactly as before — runtime_profile on the alias is the
        # only source.
        profile_name = alias.get("runtime_profile")
        if not profile_name and instance is not None:
            profile_name = instance.get("runtime_profile")
        if not profile_name:
            raise ResolutionError(f"Alias '{alias_name}' has no runtime_profile")
        profiles = self.config.get("runtime_profiles", {})
        if profile_name not in profiles:
            raise ResolutionError(
                f"Runtime profile '{profile_name}' not found for alias "
                f"'{alias_name}'"
            )
        profile = dict(profiles[profile_name])
        merged = {
            "alias": alias_name,
            "runtime_profile": profile_name,
            "backend": profile.get("backend"),
            "api_base_env": profile.get("api_base_env"),
            "default_api_base": profile.get("default_api_base"),
            "gpu": profile.get("gpu"),
        }

        # V6: instance fields are merged next, before the alias fields,
        # so an alias's own references (runtime_instance, inference_profile)
        # remain visible to downstream consumers and instance-owned
        # identity takes precedence over the alias — but a valid
        # instance-bound alias never sets instance-owned fields anyway.
        if instance is not None:
            for key, value in instance.items():
                merged[key] = value
        if inference_profile is not None:
            for key, value in inference_profile.items():
                merged[key] = value

        # Preserve every field declared on the alias. Alias fields override
        # matching profile fields, which is why profile backfill happens
        # last. ``alias`` itself is excluded because ``merged`` already has
        # the canonical alias name. The two V6 reference keys are passed
        # through so downstream consumers can still see which instance /
        # profile the alias was bound to.
        reserved = {"alias"}
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

    def resolve_instance(self, instance_name: str) -> dict:
        """Resolve a runtime instance by name (V6 — instance-level CLI).

        Returns a dict shaped like ``resolve_alias`` so a single
        ``LlamaCppAdapter`` can drive both alias-bound and
        instance-direct lifecycle operations. ``alias`` is the instance
        name (so log lines and JSON output name the runtime
        unambiguously); the canonical ``runtime_instance`` reference is
        also preserved so consumers can tell which instance they got.
        """
        runtime_instances = self.config.get("runtime_instances", {}) or {}
        if instance_name not in runtime_instances:
            raise ResolutionError(
                f"Runtime instance '{instance_name}' not found"
            )
        instance = dict(runtime_instances[instance_name])
        profile_name = instance.get("runtime_profile")
        if not profile_name:
            raise ResolutionError(
                f"Runtime instance '{instance_name}' has no runtime_profile"
            )
        profiles = self.config.get("runtime_profiles", {})
        if profile_name not in profiles:
            raise ResolutionError(
                f"Runtime profile '{profile_name}' not found for runtime "
                f"instance '{instance_name}'"
            )
        profile = dict(profiles[profile_name])
        merged: dict = {
            "alias": instance_name,
            "runtime_profile": profile_name,
            "backend": profile.get("backend"),
            "runtime_instance": instance_name,
        }
        for key, value in profile.items():
            merged.setdefault(key, value)
        # Instance fields override profile defaults — the instance is the
        # authoritative source of physical identity.
        for key, value in instance.items():
            merged[key] = value
        return merged

    def list_aliases(self) -> list[str]:
        return list(self.config.get("models", {}).keys())

    def list_runtime_instances(self) -> list[str]:
        return list(self.config.get("runtime_instances", {}) or {})

    def get_clients(self, alias_name: str) -> list[str]:
        """Return the ENABLED client keys for an alias.

        A declared-but-false client is a recorded decision not to allow that
        harness; returning it here made `list` validate aliases against
        clients they explicitly disable, while the web UI filtered them out
        — two callers disagreeing about what the same YAML means.
        """
        models = self.config.get("models", {})
        alias = models.get(alias_name, {})
        clients = alias.get("clients", {})
        if not isinstance(clients, dict):
            return []
        return [name for name, enabled in clients.items() if enabled]
