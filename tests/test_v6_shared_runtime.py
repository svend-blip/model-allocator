"""Tests for V6 shared foundation runtime (SCHEMA + VALIDATION layer only).

The full feature splits across several handoffs. This file covers B1 only:
schema loading, field ownership validation, reference resolution, and the
backward-compatibility guarantee that an alias WITHOUT the new V6 references
behaves identically to the pre-V6 resolver.

Process-lifecycle changes (B2/B3) live elsewhere; nothing in this file
should start or stop a real server.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from model_allocator import schema
from model_allocator.resolver import ResolutionError, Resolver


PROFILES = {
    "local_llamacpp": {"backend": "llama_cpp", "server_bin_env": "LLAMA_SERVER_BIN"},
    "local_ollama": {"backend": "ollama", "default_api_base": "http://127.0.0.1:11434"},
}


# ─────────────────────────────────────────────────────────────
# Schema layer (doctor / lint_config)
# ─────────────────────────────────────────────────────────────

def _errors(issues, field=None):
    out = [i for i in issues if i.level == "error"]
    if field is not None:
        out = [i for i in out if i.field == field]
    return out


def test_validate_runtime_instance_accepts_canonical_shape():
    definition = {
        "runtime_profile": "local_llamacpp",
        "model_path": "/m.gguf",
        "server_bin_path": "/bin/llama-server",
        "port": 8090,
        "context": 262144,
        "n_cpu_moe": 31,
        "gpu_layers": 99,
        "cache_type_k": "q4_0",
        "cache_type_v": "q4_0",
        "lifecycle_policy": "shared_runtime",
    }
    issues = schema.validate_runtime_instance("shared-llm", definition)
    assert issues == [], f"unexpected issues: {issues}"


def test_validate_runtime_instance_unknown_field_is_warning():
    """Unknown fields on a runtime_instance are a warning, matching the
    alias behavior — forward compatibility, not a hard fail."""
    issues = schema.validate_runtime_instance("ri",
                                             {"runtime_profile": "p",
                                              "mystery_field": 1})
    warnings = [i for i in issues if i.level == "warning"]
    assert any(i.field == "mystery_field" for i in warnings)
    assert not _errors(issues)


def test_validate_runtime_instance_type_error_on_bad_port():
    issues = schema.validate_runtime_instance("ri",
                                             {"runtime_profile": "p",
                                              "port": "8080"})
    assert any(i.field == "port" and i.level == "error" for i in issues)


def test_validate_inference_profile_accepts_the_implemented_fields():
    """The V6 transport rides on ``reasoning_budget`` and ``max_output_tokens``.
    Both pass the schema cleanly — anything else must surface as an error.
    """
    definition = {
        "reasoning_budget": 4096,
        "max_output_tokens": 16384,
    }
    issues = schema.validate_inference_profile("careful", definition)
    assert issues == [], f"unexpected issues: {issues}"


def test_validate_inference_profile_rejects_temperature_as_deferred():
    """temperature has no transport path in V6. The schema rejects it
    with a message naming the field and stating it is deferred."""
    issues = schema.validate_inference_profile("bad",
                                             {"temperature": 0.3})
    errs = _errors(issues, "temperature")
    assert errs, "temperature must surface as an error, not a warning"
    assert any("deferred" in i.message and "temperature" in i.message
               for i in errs)


def test_validate_inference_profile_rejects_top_p_as_deferred():
    issues = schema.validate_inference_profile("bad",
                                             {"top_p": 0.92})
    errs = _errors(issues, "top_p")
    assert errs, "top_p must surface as an error, not a warning"
    assert any("deferred" in i.message and "top_p" in i.message
               for i in errs)


def test_validate_inference_profile_rejects_unknown_field_with_error():
    """The schema knows exactly the implemented + deferred candidate
    fields. Anything else is an ERROR — silent acceptance is the failure
    mode the contract forbids."""
    issues = schema.validate_inference_profile("bad",
                                             {"system_prompt": "be quiet",
                                              "reasoning_budget": 100})
    errs = _errors(issues, "system_prompt")
    assert errs, "system_prompt must surface as an error, not a warning"
    assert any("unknown field" in i.message and "system_prompt" in i.message
               for i in errs)


def test_validate_inference_profile_type_error_on_bad_reasoning_budget():
    """Type-checking still applies to the implemented fields."""
    issues = schema.validate_inference_profile("bad",
                                             {"reasoning_budget": "lots"})
    assert any(i.field == "reasoning_budget" and i.level == "error"
               for i in issues)


def test_validate_alias_unknown_runtime_instance_reference_is_error():
    definition = {"runtime_instance": "ghost",
                  "runtime_profile": "local_llamacpp",
                  "model_path": "/m.gguf"}
    issues = schema.validate_alias("a", definition, PROFILES,
                                  runtime_instances={},
                                  inference_profiles={})
    errs = _errors(issues, "runtime_instance")
    assert any("unknown runtime_instance 'ghost'" in i.message for i in errs)


def test_validate_alias_unknown_inference_profile_reference_is_error():
    definition = {"runtime_profile": "local_llamacpp",
                  "model_path": "/m.gguf",
                  "inference_profile": "no-such-profile"}
    issues = schema.validate_alias("a", definition, PROFILES,
                                  runtime_instances={},
                                  inference_profiles={"careful": {}})
    errs = _errors(issues, "inference_profile")
    assert any("unknown inference_profile 'no-such-profile'" in i.message
               for i in errs)


@pytest.mark.parametrize("owned_field,owned_value", [
    ("model_path", "/m.gguf"),
    ("port", 8081),
    ("context", 131072),
    ("n_cpu_moe", 26),
    ("gpu_layers", 99),
    ("cache_type_k", "q4_0"),
    ("cache_type_v", "q4_0"),
    ("runtime_profile", "local_llamacpp"),
    ("server_bin_path", "/bin/llama-server"),
])
def test_validate_alias_instance_bound_alias_forbidden_from_owning_fields(
        owned_field, owned_value):
    """Field ownership is exclusive: an alias bound to a runtime_instance
    MUST NOT also set any field the instance owns. No silent precedence.
    """
    instances = {"shared-llm": {"runtime_profile": "local_llamacpp",
                                "model_path": "/shared.gguf",
                                "port": 8090,
                                "context": 262144}}
    alias = {"runtime_instance": "shared-llm", owned_field: owned_value}
    issues = schema.validate_alias("a", alias, PROFILES,
                                  runtime_instances=instances,
                                  inference_profiles={})
    errs = [i for i in issues if i.level == "error" and i.field == owned_field]
    assert errs, (
        f"alias bound to runtime_instance must not also set "
        f"'{owned_field}' but no error was raised. issues={issues}"
    )
    assert any("owned by the instance" in i.message for i in errs)


def test_validate_alias_instance_bound_alias_no_runtime_profile_is_ok():
    """An alias bound to an instance that supplies runtime_profile does
    NOT need to also declare it on the alias."""
    instances = {"shared-llm": {"runtime_profile": "local_llamacpp",
                                "model_path": "/shared.gguf",
                                "port": 8090,
                                "context": 262144}}
    alias = {"runtime_instance": "shared-llm"}
    issues = schema.validate_alias("a", alias, PROFILES,
                                  runtime_instances=instances,
                                  inference_profiles={})
    assert issues == [], f"unexpected issues: {issues}"


def test_lint_config_reports_unknown_runtime_instance():
    raw = {
        "aliases": {"a": {"runtime_instance": "ghost",
                          "runtime_profile": "local_llamacpp",
                          "model_path": "/m.gguf"}},
        "profiles": PROFILES,
        "roles": {},
        "runtime_instances": {},
        "inference_profiles": {},
    }
    report = schema.lint_config(raw)
    assert "a" in report["aliases"]
    assert any(i.field == "runtime_instance" and i.level == "error"
               for i in report["aliases"]["a"])


def test_lint_config_reports_instance_field_ownership_violation():
    raw = {
        "aliases": {"a": {"runtime_instance": "shared-llm",
                          "runtime_profile": "local_llamacpp",
                          "port": 8081}},
        "profiles": PROFILES,
        "roles": {},
        "runtime_instances": {"shared-llm": {
            "runtime_profile": "local_llamacpp",
            "model_path": "/m.gguf",
            "port": 8090,
            "context": 262144,
        }},
        "inference_profiles": {},
    }
    report = schema.lint_config(raw)
    assert "a" in report["aliases"]
    assert any(i.field == "port" and i.level == "error"
               for i in report["aliases"]["a"])


def test_lint_config_reports_inference_profile_unknown_field():
    raw = {
        "aliases": {},
        "profiles": PROFILES,
        "roles": {},
        "runtime_instances": {},
        "inference_profiles": {"bad": {"system_prompt": "no"}},
    }
    report = schema.lint_config(raw)
    assert "bad" in report["inference_profiles"]
    assert any(i.level == "error" for i in report["inference_profiles"]["bad"])


# ─────────────────────────────────────────────────────────────
# Resolver layer (resolve_alias)
# ─────────────────────────────────────────────────────────────

def _seed_v6(tmp: Path, *, alias_name="shared-architect",
             instance_name="shared-llm",
             inference_profile_name="profile-careful",
             alias_runtime_profile=None,
             alias_extras=None) -> Path:
    (tmp / "models.yaml").write_text(
        "models:\n"
        f"  {alias_name}:\n"
        f"    runtime_instance: {instance_name}\n"
        f"    inference_profile: {inference_profile_name}\n"
        + (f"    runtime_profile: {alias_runtime_profile}\n"
           if alias_runtime_profile else "")
        + (alias_extras or "")
        + "    clients:\n"
        + "      opencode: true\n"
        + "runtime_instances:\n"
        f"  {instance_name}:\n"
        + "    runtime_profile: local_llamacpp\n"
        + "    model_path: /shared-118b.gguf\n"
        + "    port: 8090\n"
        + "    context: 262144\n"
        + "    n_cpu_moe: 31\n"
        + "    gpu_layers: 99\n"
        + "    lifecycle_policy: shared_runtime\n"
        + "inference_profiles:\n"
        f"  {inference_profile_name}:\n"
        + "    reasoning_budget: 4096\n"
        + "    max_output_tokens: 16384\n",
        encoding="utf-8",
    )
    (tmp / "runtime_profiles.yaml").write_text(
        "runtime_profiles:\n"
        "  local_llamacpp:\n"
        "    backend: llama_cpp\n"
        "    server_bin_env: LLAMA_SERVER_BIN\n",
        encoding="utf-8",
    )
    (tmp / "roles.yaml").write_text("roles:\n", encoding="utf-8")
    return tmp


def test_resolve_alias_instance_bound_takes_physical_fields_from_instance():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _seed_v6(Path(tmp))
        resolved = Resolver(config_dir=str(cfg)).resolve_alias("shared-architect")
        assert resolved["runtime_profile"] == "local_llamacpp"
        assert resolved["model_path"] == "/shared-118b.gguf"
        assert resolved["port"] == 8090
        assert resolved["context"] == 262144
        assert resolved["n_cpu_moe"] == 31
        assert resolved["gpu_layers"] == 99
        assert resolved["lifecycle_policy"] == "shared_runtime"
        assert resolved["runtime_instance"] == "shared-llm"
        assert resolved["inference_profile"] == "profile-careful"
        assert resolved["reasoning_budget"] == 4096
        assert resolved["max_output_tokens"] == 16384
        # Deferred fields are NOT in the resolved view — the resolver
        # rejects them before merging.
        assert "temperature" not in resolved
        assert "top_p" not in resolved


def test_resolve_alias_unknown_runtime_instance_is_resolutionerror():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp)
        _seed_v6(cfg)
        # Break the alias-side reference only — the runtime_instance block
        # below must stay defined, otherwise the alias resolves to itself.
        text = cfg.joinpath("models.yaml").read_text(encoding="utf-8")
        text = text.replace(
            "runtime_instance: shared-llm",
            "runtime_instance: ghost-instance",
        )
        cfg.joinpath("models.yaml").write_text(text, encoding="utf-8")
        resolver = Resolver(config_dir=str(cfg))
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve_alias("shared-architect")
        assert "unknown runtime_instance 'ghost-instance'" in str(excinfo.value)


def test_resolve_alias_unknown_inference_profile_is_resolutionerror():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp)
        _seed_v6(cfg)
        text = cfg.joinpath("models.yaml").read_text(encoding="utf-8")
        text = text.replace(
            "inference_profile: profile-careful",
            "inference_profile: no-such-profile",
        )
        cfg.joinpath("models.yaml").write_text(text, encoding="utf-8")
        resolver = Resolver(config_dir=str(cfg))
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve_alias("shared-architect")
        assert "unknown inference_profile 'no-such-profile'" in str(excinfo.value)


def test_resolve_alias_instance_bound_alias_with_port_is_resolutionerror():
    """Field ownership error surfaces at resolve time, not silently."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp)
        _seed_v6(cfg, alias_extras="    port: 8081\n")
        resolver = Resolver(config_dir=str(cfg))
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve_alias("shared-architect")
        msg = str(excinfo.value)
        assert "shared-llm" in msg
        assert "port" in msg
        assert "owned by the instance" in msg


def test_resolve_alias_instance_bound_alias_with_runtime_profile_is_resolutionerror():
    """A V6 alias bound to an instance must NOT also declare runtime_profile."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp)
        _seed_v6(cfg, alias_runtime_profile="local_llamacpp")
        resolver = Resolver(config_dir=str(cfg))
        with pytest.raises(ResolutionError) as excinfo:
            resolver.resolve_alias("shared-architect")
        msg = str(excinfo.value)
        assert "shared-llm" in msg
        assert "runtime_profile" in msg


# ─────────────────────────────────────────────────────────────
# Backward compat — an alias without the V6 references behaves
# exactly as the pre-V6 resolver did.
# ─────────────────────────────────────────────────────────────

def test_resolve_alias_without_runtime_instance_is_byte_equivalent_to_pre_v6():
    """The pre-V6 resolver path must be byte-equivalent to the V6 path
    when no runtime_instance / inference_profile references are used."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp)
        (cfg / "models.yaml").write_text(
            "models:\n"
            "  legacy-llama:\n"
            "    runtime_profile: local_llamacpp\n"
            "    model_path: /m.gguf\n"
            "    context: 131072\n"
            "    port: 8081\n"
            "    n_cpu_moe: 26\n"
            "    clients:\n"
            "      opencode: true\n",
            encoding="utf-8",
        )
        (cfg / "runtime_profiles.yaml").write_text(
            "runtime_profiles:\n"
            "  local_llamacpp:\n"
            "    backend: llama_cpp\n"
            "    server_bin_env: LLAMA_SERVER_BIN\n",
            encoding="utf-8",
        )
        (cfg / "roles.yaml").write_text("roles:\n", encoding="utf-8")

        resolved = Resolver(config_dir=str(cfg)).resolve_alias("legacy-llama")

        # All previously-visible fields stay visible, unchanged.
        assert resolved["alias"] == "legacy-llama"
        assert resolved["runtime_profile"] == "local_llamacpp"
        assert resolved["backend"] == "llama_cpp"
        assert resolved["model_path"] == "/m.gguf"
        assert resolved["context"] == 131072
        assert resolved["port"] == 8081
        assert resolved["n_cpu_moe"] == 26
        # No new V6 keys leaked into a non-V6 alias.
        assert "runtime_instance" not in resolved
        assert "inference_profile" not in resolved


def test_load_config_without_v6_sections_returns_empty_subkeys():
    """Configs without the V6 sections still load — empty subkeys, not
    KeyError. The pre-V6 contract guarantees ``models``, ``runtime_profiles``
    and ``roles``; the V6 additions surface as empty dicts."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp)
        (cfg / "models.yaml").write_text(
            "models:\n"
            "  a:\n"
            "    runtime_profile: local_ollama\n"
            "    real_model: m\n",
            encoding="utf-8",
        )
        (cfg / "runtime_profiles.yaml").write_text(
            "runtime_profiles:\n"
            "  local_ollama:\n"
            "    backend: ollama\n"
            "    default_api_base: http://127.0.0.1:11434\n",
            encoding="utf-8",
        )
        (cfg / "roles.yaml").write_text("roles:\n", encoding="utf-8")

        from model_allocator.config_loader import load_config
        cfg_loaded = load_config(str(cfg))
        assert "a" in cfg_loaded["models"]
        assert "local_ollama" in cfg_loaded["runtime_profiles"]
        assert cfg_loaded["runtime_instances"] == {}
        assert cfg_loaded["inference_profiles"] == {}

        # And the doctor is silent on the missing sections.
        from model_allocator.config_writer import load_raw
        raw = load_raw(cfg)
        report = schema.lint_config(raw)
        assert report == {"aliases": {}, "runtime_instances": {},
                          "inference_profiles": {}, "profiles": {}, "roles": {}}


def test_resolve_alias_with_only_inference_profile_still_resolves_profile():
    """An alias can reference an inference_profile WITHOUT a
    runtime_instance. The alias's own runtime_profile is the only source
    for physical identity."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp)
        (cfg / "models.yaml").write_text(
            "models:\n"
            "  thinking-alias:\n"
            "    runtime_profile: local_ollama\n"
            "    real_model: qwen:latest\n"
            "    inference_profile: profile-careful\n"
            "    clients:\n"
            "      opencode: true\n"
            "inference_profiles:\n"
            "  profile-careful:\n"
            "    reasoning_budget: 4096\n"
            "    max_output_tokens: 16384\n",
            encoding="utf-8",
        )
        (cfg / "runtime_profiles.yaml").write_text(
            "runtime_profiles:\n"
            "  local_ollama:\n"
            "    backend: ollama\n"
            "    default_api_base: http://127.0.0.1:11434\n",
            encoding="utf-8",
        )
        (cfg / "roles.yaml").write_text("roles:\n", encoding="utf-8")

        resolved = Resolver(config_dir=str(cfg)).resolve_alias("thinking-alias")
        assert resolved["runtime_profile"] == "local_ollama"
        assert resolved["inference_profile"] == "profile-careful"
        assert resolved["reasoning_budget"] == 4096
        assert resolved["max_output_tokens"] == 16384
        # Deferred fields are NOT in the resolved view.
        assert "temperature" not in resolved
        assert "top_p" not in resolved
        # No runtime_instance reference was set.
        assert "runtime_instance" not in resolved
