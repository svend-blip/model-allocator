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


BACKENDS = ("ollama", "llama_cpp", "openai_compatible", "onyx", "anthropic",
            "sglang", "freetoken")

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
    # Pi frontend. `pi_tools` is the one of these that carries governance
    # rather than configuration: it is an allowlist the client enforces, so a
    # reviewer that must not write can be prevented from writing instead of
    # asked not to.
    "pi_provider_name": str,
    "pi_model_id": str,
    "pi_agent_dir": str,
    "pi_api_key": str,
    "pi_no_session": bool,
    "pi_thinking": str,
    "pi_tools": list,
    "pi_skills": list,
    "pi_extra_args": list,
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
    "reasoning_budget": int,
    "jinja": bool,
    "load_mode": str,
    "no_mmap": bool,
    "gpu_layers": int,
    "tensor_split": str,
    # Speculative decoding (llama.cpp --spec-type / --spec-draft-n-max).
    # spec_type values are llama-server's own, e.g. "draft-mtp" for models
    # with a baked-in MTP head (Qwen3.8).
    "spec_type": str,
    "spec_draft_n_max": int,
}

SGLANG_ALIAS_FIELDS: dict[str, object] = {
    "model_path": str,
    "served_model_name": str,
    "port": int,
    "host": str,
    "venv": str,
    "context": int,
    "mem_fraction_static": (int, float),
    "max_running_requests": int,
    "tool_call_parser": str,
    "enable_cache_report": bool,
    "max_output_tokens": int,
}

FREETOKEN_ALIAS_FIELDS: dict[str, object] = {
    # model_path holds either a local checkpoint directory or a Hugging Face
    # repo ID — FreeToken accepts both, and the qualified profiles use repos.
    "model_path": str,
    "served_model_name": str,
    "port": int,
    "host": str,
    "executable": str,
    "context": int,
    "memory_ratio": (int, float),
    # Backend selection. These are not cosmetic tuning: on the qualified
    # RTX 5090 profile, nvfp4_backend decided between ~4 and ~63 tokens/sec.
    "nvfp4_backend": str,
    "moe_backend": str,
    "moe_cache_auto": bool,
    "moe_cache_size": str,
    "moe_cache_rate": (int, float),
    "kv_reserve_tokens": int,
    "attention_backend": str,
    "cache_type": str,
    "sampling_defaults": str,
    "reasoning_parser": str,
    "tool_call_parser": str,
    "model_source": str,
    "dtype": str,
    "max_running_requests": int,
    "max_output_tokens": int,
    # The runtime reports its own maximum context; this caps it deliberately
    # and is left unset by the qualified profiles.
    "max_seq_len_override": int,
    "cuda_graph_max_bs": int,
    "max_prefill_length": int,
    "enable_cache_report": bool,
    "disable_moe_prefill_overlap": bool,
    "extra_args": list,
    "qualified_runtime_version": str,
    "qualification": dict,
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
    # Absolute path to a runtime binary that is not on PATH. FreeToken's
    # qualified install is a project-local venv, so a service environment
    # never finds `ft` by name.
    "executable": str,
    "qualified_runtime_version": str,
    "server_bin_env": str,
    "model_root_env": str,
    "default_port": int,
    "default_ctx": int,
    "default_gpu_layers": int,
    "host": str,
    "capabilities": list,
    "venv": str,
    "default_host": str,
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


# V6 — runtime instance owns physical identity. An alias bound to a
# runtime_instance MUST NOT also set any of these fields; the resolver
# would have to pick one, and silent precedence is forbidden by contract.
INSTANCE_OWNED_FIELDS: tuple[str, ...] = (
    "model_path",
    "port",
    "context",
    "n_cpu_moe",
    "gpu_layers",
    "cache_type_k",
    "cache_type_v",
    "runtime_profile",
    "server_bin_path",
)


# V6 — runtime instance allow-list (mirrors the per-backend alias
# shape but with the physical-identity fields an instance always owns).
# Unknown fields stay a warning (forward compatibility), matching aliases.
RUNTIME_INSTANCE_FIELDS: dict[str, object] = {
    "runtime_profile": str,
    "model_path": str,
    "model_name": str,
    "server_bin_path": str,
    "port": int,
    "host": str,
    "context": int,
    "n_cpu_moe": int,
    "gpu_layers": int,
    "cache_type_k": str,
    "cache_type_v": str,
    "lifecycle_policy": str,
    "model_root_env": str,
    "default_port": int,
    "default_ctx": int,
}


# V6 — inference_profiles: split into implemented and deferred.
# Implemented fields have a clean existing transport path and ride on it
# (reasoning_budget -> llama-server argv, max_output_tokens -> opencode
# limit block + claude_code env + pi models.json + CLI override). The
# deferred fields (temperature, top_p) have NO existing transport path
# and were accepted silently before; declaring them must now be a loud
# validation error with a "deferred in V6" message — the contract
# forbids silent acceptance of fields that would do nothing.
INFERENCE_PROFILE_IMPLEMENTED_FIELDS: dict[str, object] = {
    "reasoning_budget": int,
    "max_output_tokens": int,
}

INFERENCE_PROFILE_DEFERRED_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
)

# Union used in error messages listing the implemented + deferred candidates.
INFERENCE_PROFILE_KNOWN_FIELDS: tuple[str, ...] = (
    *INFERENCE_PROFILE_IMPLEMENTED_FIELDS,
    *INFERENCE_PROFILE_DEFERRED_FIELDS,
)

# Backward-compat alias for any caller that imported the old name. New
# callers should prefer INFERENCE_PROFILE_IMPLEMENTED_FIELDS.
INFERENCE_PROFILE_FIELDS: dict[str, object] = INFERENCE_PROFILE_IMPLEMENTED_FIELDS


# Alias-owned keys that reference runtime/inference profiles. They are
# exempt from the instance-ownership conflict rule: an alias is allowed
# to declare its references, just not physical-identity fields.
ALIAS_REFERENCE_KEYS: tuple[str, ...] = (
    "runtime_instance",
    "inference_profile",
)


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


def validate_alias(alias_name: str, definition: dict, profiles: dict,
                   runtime_instances: dict | None = None,
                   inference_profiles: dict | None = None) -> list[Issue]:
    """Validate a single alias definition against the schema.

    V6 extension: aliases may declare ``runtime_instance`` and/or
    ``inference_profile`` references. The reference must resolve, and an
    alias bound to a runtime_instance must NOT also set any field the
    instance owns (model_path, port, context, n_cpu_moe, gpu_layers,
    cache_type_k, cache_type_v, runtime_profile, server_bin_path) —
    the contract forbids silent precedence between the two.
    """
    issues: list[Issue] = []
    runtime_instances = runtime_instances or {}
    inference_profiles = inference_profiles or {}

    # V6 reference checks first: both refs are independent of the
    # runtime_profile requirement below, so unknown refs surface as their
    # own errors instead of being hidden by an early return.
    instance_name = definition.get("runtime_instance")
    if instance_name is not None:
        if instance_name not in runtime_instances:
            issues.append(Issue(
                "error", "runtime_instance",
                f"alias '{alias_name}' references unknown runtime_instance "
                f"'{instance_name}'"
            ))

    inference_profile_name = definition.get("inference_profile")
    if inference_profile_name is not None:
        if inference_profile_name not in inference_profiles:
            issues.append(Issue(
                "error", "inference_profile",
                f"alias '{alias_name}' references unknown inference_profile "
                f"'{inference_profile_name}'"
            ))

    # Field-ownership check (only meaningful when the reference resolves;
    # a dangling reference already surfaced above and will still trip the
    # resolver later, but the schema here reports it on its own terms).
    if instance_name is not None and instance_name in runtime_instances:
        instance = runtime_instances[instance_name]
        for owned in INSTANCE_OWNED_FIELDS:
            if owned in definition:
                issues.append(Issue(
                    "error", owned,
                    f"alias '{alias_name}' is bound to runtime_instance "
                    f"'{instance_name}' and must not also set '{owned}' "
                    f"(owned by the instance)"
                ))

    # runtime_profile is required UNLESS the alias is bound to a runtime
    # instance that provides one — in which case the alias must NOT have
    # one (enforced by the ownership check above).
    profile_name = definition.get("runtime_profile")
    if not profile_name:
        if instance_name is not None and instance_name in runtime_instances:
            profile_name = runtime_instances[instance_name].get("runtime_profile")
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
    elif backend == "sglang":
        allow_list.update(SGLANG_ALIAS_FIELDS)
    elif backend == "freetoken":
        allow_list.update(FREETOKEN_ALIAS_FIELDS)
    # Profile fields are overridable on alias — add them
    for key, val_type in PROFILE_FIELDS.items():
        if key not in allow_list:
            allow_list[key] = val_type
    # V6 reference keys are accepted on the alias.
    for key in ALIAS_REFERENCE_KEYS:
        if key not in allow_list:
            allow_list[key] = str

    # Check each field in the definition
    for field_name, value in definition.items():
        if field_name in ("runtime_profile",):
            continue  # already checked
        if field_name in ALIAS_REFERENCE_KEYS:
            # The reference itself was checked above; just type-check.
            if not isinstance(value, str):
                issues.append(Issue(
                    "error", field_name,
                    f"field '{field_name}' must be a string, got "
                    f"{type(value).__name__}"
                ))
            continue
        if field_name in allow_list:
            expected_type = allow_list[field_name]
            _check_type(field_name, value, expected_type, issues)
        else:
            issues.append(Issue(
                "warning", field_name,
                f"unknown field '{field_name}' on alias '{alias_name}'"
            ))

    # Backend-specific required fields (only meaningful when the alias
    # itself declares the backend-binding fields — an instance-bound alias
    # is exempt because the instance owns them).
    has_instance = (instance_name is not None
                    and instance_name in runtime_instances)
    if backend == "ollama":
        if "real_model" not in definition or not definition.get("real_model"):
            issues.append(Issue("error", "real_model",
                                "ollama backend requires real_model"))
    elif backend == "llama_cpp":
        if not has_instance and not definition.get("model_path") \
                and not definition.get("model_name"):
            issues.append(Issue("error", "model_path",
                                "llama_cpp backend requires model_path or model_name"))
    elif backend == "sglang":
        if not has_instance and not definition.get("model_path"):
            issues.append(Issue("error", "model_path",
                                "sglang backend requires model_path"))
    elif backend == "freetoken":
        if not has_instance and not definition.get("model_path"):
            issues.append(Issue(
                "error", "model_path",
                "freetoken backend requires model_path (a local checkpoint "
                "directory or a Hugging Face repo ID)"))

    return issues


def validate_runtime_instance(instance_name: str, definition: dict) -> list[Issue]:
    """Validate a single runtime_instance definition (V6).

    Field ownership is exclusive: the instance owns physical identity,
    but the linter does not enforce required fields here — an instance
    referencing ``runtime_profile`` lets that profile supply the
    backend-specific required fields. Unknown fields stay a warning
    (forward compatibility), matching alias validation.
    """
    issues: list[Issue] = []
    if not isinstance(definition, dict):
        issues.append(Issue(
            "error", instance_name,
            f"runtime_instance '{instance_name}' must be a mapping"
        ))
        return issues

    for field_name, value in definition.items():
        if field_name in RUNTIME_INSTANCE_FIELDS:
            expected_type = RUNTIME_INSTANCE_FIELDS[field_name]
            _check_type(field_name, value, expected_type, issues)
        else:
            issues.append(Issue(
                "warning", field_name,
                f"unknown field '{field_name}' on runtime_instance "
                f"'{instance_name}'"
            ))
    return issues


def validate_inference_profile(profile_name: str, definition: dict) -> list[Issue]:
    """Validate a single inference_profile definition (V6).

    Three tiers of verdict:

    - ``INFERENCE_PROFILE_IMPLEMENTED_FIELDS`` -> type-checked and accepted.
    - ``INFERENCE_PROFILE_DEFERRED_FIELDS``   -> ERROR with a "deferred in V6"
      message naming the field. These have no transport path; declaring them
      silently would do nothing, which the Mission Contract forbids.
    - anything else                              -> ERROR as an unknown field.
    """
    issues: list[Issue] = []
    if not isinstance(definition, dict):
        issues.append(Issue(
            "error", profile_name,
            f"inference_profile '{profile_name}' must be a mapping"
        ))
        return issues

    for field_name, value in definition.items():
        if field_name in INFERENCE_PROFILE_IMPLEMENTED_FIELDS:
            expected_type = INFERENCE_PROFILE_IMPLEMENTED_FIELDS[field_name]
            _check_type(field_name, value, expected_type, issues)
        elif field_name in INFERENCE_PROFILE_DEFERRED_FIELDS:
            issues.append(Issue(
                "error", field_name,
                f"inference_profile '{profile_name}' field '{field_name}' "
                f"is not implemented in V6 (deferred) — remove it"
            ))
        else:
            issues.append(Issue(
                "error", field_name,
                f"inference_profile '{profile_name}' has unknown field "
                f"'{field_name}' (allowed: "
                f"{', '.join(sorted(INFERENCE_PROFILE_KNOWN_FIELDS))})"
            ))
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

    Returns a dict with keys ``aliases``, ``runtime_instances``,
    ``inference_profiles``, ``profiles`` and ``roles`` (each mapping
    name -> list[Issue]). Only entries with issues are included.
    """
    aliases = raw.get("aliases", {})
    profiles = raw.get("profiles", {})
    roles = raw.get("roles", {})
    runtime_instances = raw.get("runtime_instances", {}) or {}
    inference_profiles = raw.get("inference_profiles", {}) or {}

    report: dict[str, dict[str, list[Issue]]] = {
        "aliases": {},
        "runtime_instances": {},
        "inference_profiles": {},
        "profiles": {},
        "roles": {},
    }

    for name, definition in aliases.items():
        issues = validate_alias(
            name, definition, profiles,
            runtime_instances=runtime_instances,
            inference_profiles=inference_profiles,
        )
        if issues:
            report["aliases"][name] = issues

    for name, definition in runtime_instances.items():
        issues = validate_runtime_instance(name, definition)
        if issues:
            report["runtime_instances"][name] = issues

    for name, definition in inference_profiles.items():
        issues = validate_inference_profile(name, definition)
        if issues:
            report["inference_profiles"][name] = issues

    for name, definition in profiles.items():
        issues = validate_profile(name, definition)
        if issues:
            report["profiles"][name] = issues

    for name, definition in roles.items():
        issues = validate_role(name, definition, aliases)
        if issues:
            report["roles"][name] = issues

    return report
