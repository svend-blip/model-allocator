"""Config schema: per-backend field allow-lists + validation (doctor).

The resolver (V2.1) does a blind generic merge, so typo'd fields silently
survive resolution and are then silently ignored by the adapters
(e.g. ``n_cpu_mo`` instead of ``n_cpu_moe`` never reaches llama-server's
argv). This module encodes the fields the code ACTUALLY reads.

Rules:
- unknown field           -> warning (forward compatibility)
- wrong type              -> error
- str field holding bool  -> error with a quote hint (YAML 1.1 turns bare
                             on/off/yes/no into booleans)
- missing required field  -> error
- string containing an env reference (${VAR} / $VAR) -> exempt from type
  checks: the doctor lints RAW (unresolved) config, matching the
  config_writer path.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_allocator.config_loader import ENV_RE


@dataclass
class Issue:
    level: str  # "error" | "warning"
    field: str
    message: str


BACKENDS = ("ollama", "llama_cpp", "openai_compatible", "onyx", "anthropic")

COMMON_ALIAS_FIELDS: dict[str, object] = {
    "runtime_profile": str,
    "real_model": str,
    "context": int,
    "lifecycle_policy": str,
    "clients": dict,
    "display_name": str,
    "opencode_provider_name": str,
    "opencode_model_id": str,
    "persona_id": int,
    "invoke_timeout": int,
    "headless_output_dir": str,
    "headless_idle_seconds": (int, float),
    "max_output_tokens": int,
    "disable_adaptive_thinking": bool,
    "claude_binary": str,
    "claude_extra_args": list,
    # Pre-seeded for in-flight plans
    "ssh_host": str,
    "remote_workdir": str,
    "server_bin_path": str,
}

LLAMACPP_ALIAS_FIELDS: dict[str, object] = {
    "model_path": str,
    "model_name": str,
    "port": int,
    "host": str,
    "parallel": int,
    "n_cpu_moe": int,
    "threads": int,
    "batch": int,
    "ubatch_size": int,
    "cache_type_k": str,
    "cache_type_v": str,
    "flash_attn": str,
    "reasoning": str,
    "no_mmap": bool,
    "gpu_layers": int,
    "tensor_split": str,
}

PROFILE_FIELDS: dict[str, object] = {
    "backend": str,
    "api_base_env": str,
    "default_api_base": str,
    "api_key_env": str,
    "email_env": str,
    "password_env": str,
    "provider": str,
    # "subscription" (use the client's own login) | "api_key" (bill $api_key_env).
    # Anthropic profiles default to "api_key" when absent, for compatibility.
    "credentials": str,
    "gpu": str,
    "server_bin_env": str,
    "model_root_env": str,
    "default_port": int,
    "default_ctx": int,
    "default_gpu_layers": int,
    "host": str,
    "capabilities": list,
    # Pre-seeded for in-flight plans
    "ssh_host": str,
    "remote_workdir": str,
}

ROLE_FIELDS: dict[str, object] = {
    "default_alias": str,
    "config_dir": str,
    "client_aliases": dict,
}

# Fields that are strings but are commonly broken by YAML bool trap
_YAML_BOOL_TRAP_FIELDS = {"flash_attn", "reasoning"}


def _is_env_ref(value) -> bool:
    """Check if a value is a string containing an env var reference."""
    if not isinstance(value, str):
        return False
    return bool(ENV_RE.search(value))


def _check_type(field_name, value, expected_type, issues: list[Issue]):
    """Type-check a field value, handling bool-is-not-int and env-ref wildcards."""
    if _is_env_ref(value):
        return  # env-ref strings are exempt from type checks

    if field_name in _YAML_BOOL_TRAP_FIELDS and isinstance(value, bool):
        issues.append(Issue(
            "error", field_name,
            f"YAML loaded '{field_name}' as bool ({value}) — "
            f"quote the value in models.yaml (e.g. flash_attn: \"on\")"
        ))
        return

    if isinstance(expected_type, tuple):
        valid_types = expected_type
    else:
        valid_types = (expected_type,)

    # bool is a subclass of int in Python — reject it for int fields
    if int in valid_types and isinstance(value, bool):
        issues.append(Issue(
            "error", field_name,
            f"field '{field_name}' must be an integer, got bool ({value})"
        ))
        return

    if not isinstance(value, valid_types):
        type_name = " or ".join(t.__name__ for t in valid_types)
        issues.append(Issue(
            "error", field_name,
            f"field '{field_name}' must be {type_name}, got {type(value).__name__}"
        ))


def validate_alias(alias_name: str, definition: dict, profiles: dict) -> list[Issue]:
    """Validate a single alias definition against the schema."""
    issues: list[Issue] = []

    # runtime_profile is required
    profile_name = definition.get("runtime_profile")
    if not profile_name:
        issues.append(Issue("error", "runtime_profile", "runtime_profile is required"))
        return issues

    # Profile must exist
    if profile_name not in profiles:
        issues.append(Issue(
            "error", "runtime_profile",
            f"unknown runtime_profile: {profile_name}"
        ))
        return issues

    profile = profiles[profile_name]
    backend = profile.get("backend", "")

    # Build the allow-list for this alias: common + backend-specific + profile fields
    allow_list = dict(COMMON_ALIAS_FIELDS)
    if backend == "llama_cpp":
        allow_list.update(LLAMACPP_ALIAS_FIELDS)
    # Profile fields are overridable on alias — add them
    for key, val_type in PROFILE_FIELDS.items():
        if key not in allow_list:
            allow_list[key] = val_type

    # Check each field in the definition
    for field_name, value in definition.items():
        if field_name in ("runtime_profile",):
            continue  # already checked
        if field_name in allow_list:
            expected_type = allow_list[field_name]
            _check_type(field_name, value, expected_type, issues)
        else:
            issues.append(Issue(
                "warning", field_name,
                f"unknown field '{field_name}' on alias '{alias_name}'"
            ))

    # Backend-specific required fields
    if backend == "ollama":
        if "real_model" not in definition or not definition.get("real_model"):
            issues.append(Issue("error", "real_model",
                                "ollama backend requires real_model"))
    elif backend == "llama_cpp":
        if not definition.get("model_path") and not definition.get("model_name"):
            issues.append(Issue("error", "model_path",
                                "llama_cpp backend requires model_path or model_name"))

    return issues


def validate_profile(profile_name: str, definition: dict) -> list[Issue]:
    """Validate a runtime profile definition."""
    issues: list[Issue] = []

    backend = definition.get("backend")
    if not backend:
        issues.append(Issue("error", "backend", "backend is required"))
    elif backend not in BACKENDS:
        issues.append(Issue(
            "error", "backend",
            f"unknown backend: {backend} (expected one of {', '.join(BACKENDS)})"
        ))

    # Check types of known fields
    for field_name, value in definition.items():
        if field_name in PROFILE_FIELDS:
            expected_type = PROFILE_FIELDS[field_name]
            _check_type(field_name, value, expected_type, issues)
        else:
            issues.append(Issue(
                "warning", field_name,
                f"unknown field '{field_name}' on profile '{profile_name}'"
            ))

    return issues


def validate_role(role_name: str, definition: dict, aliases: dict) -> list[Issue]:
    """Validate a role definition."""
    issues: list[Issue] = []

    # Check types of known fields
    for field_name, value in definition.items():
        if field_name in ROLE_FIELDS:
            expected_type = ROLE_FIELDS[field_name]
            _check_type(field_name, value, expected_type, issues)
        else:
            issues.append(Issue(
                "warning", field_name,
                f"unknown field '{field_name}' on role '{role_name}'"
            ))

    # default_alias must reference an existing alias
    default_alias = definition.get("default_alias")
    if default_alias and default_alias not in aliases:
        issues.append(Issue(
            "error", "default_alias",
            f"role references unknown alias: {default_alias}"
        ))

    # client_aliases values must reference existing aliases
    client_aliases = definition.get("client_aliases") or {}
    if isinstance(client_aliases, dict):
        for client, alias_ref in client_aliases.items():
            if alias_ref and alias_ref not in aliases:
                issues.append(Issue(
                    "error", "client_aliases",
                    f"role references unknown alias: {alias_ref} (client {client})"
                ))

    return issues


def lint_config(raw: dict) -> dict:
    """Lint a full raw config (from config_writer.load_raw).

    Returns a dict with keys "aliases", "profiles", "roles", each mapping
    name -> list[Issue]. Only entries with issues are included.
    """
    aliases = raw.get("aliases", {})
    profiles = raw.get("profiles", {})
    roles = raw.get("roles", {})

    report: dict[str, dict[str, list[Issue]]] = {"aliases": {}, "profiles": {}, "roles": {}}

    for name, definition in aliases.items():
        issues = validate_alias(name, definition, profiles)
        if issues:
            report["aliases"][name] = issues

    for name, definition in profiles.items():
        issues = validate_profile(name, definition)
        if issues:
            report["profiles"][name] = issues

    for name, definition in roles.items():
        issues = validate_role(name, definition, aliases)
        if issues:
            report["roles"][name] = issues

    return report
