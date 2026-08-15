"""Tests for the config schema module (PLAN-config-schema-doctor)."""

from model_allocator import schema


PROFILES = {
    "local_llamacpp": {"backend": "llama_cpp", "server_bin_env": "LLAMA_SERVER_BIN"},
    "local_ollama": {"backend": "ollama", "default_api_base": "http://127.0.0.1:11434"},
}


def _levels(issues, level):
    return [i for i in issues if i.level == level]


def test_typo_field_is_warning_not_error():
    definition = {"runtime_profile": "local_llamacpp", "model_path": "/m.gguf",
                  "n_cpu_mo": 26}
    issues = schema.validate_alias("a", definition, PROFILES)
    warnings = _levels(issues, "warning")
    assert any(i.field == "n_cpu_mo" for i in warnings)
    assert not any(i.field == "n_cpu_mo" for i in _levels(issues, "error"))


def test_yaml_bool_trap_on_flash_attn_is_error_with_quote_hint():
    definition = {"runtime_profile": "local_llamacpp", "model_path": "/m.gguf",
                  "flash_attn": True, "reasoning": False}
    issues = schema.validate_alias("a", definition, PROFILES)
    errors = _levels(issues, "error")
    assert any(i.field == "flash_attn" and "quote" in i.message.lower() for i in errors)
    assert any(i.field == "reasoning" for i in errors)


def test_bool_does_not_satisfy_int():
    definition = {"runtime_profile": "local_llamacpp", "model_path": "/m.gguf",
                  "port": True}
    issues = schema.validate_alias("a", definition, PROFILES)
    assert any(i.field == "port" for i in _levels(issues, "error"))


def test_env_ref_string_is_wildcard():
    definition = {"runtime_profile": "local_llamacpp",
                  "model_path": "${MODEL_ROOT_GGUF}/x.gguf",
                  "port": "${LLAMA_PORT}"}
    issues = schema.validate_alias("a", definition, PROFILES)
    assert not _levels(issues, "error")


def test_missing_runtime_profile_and_unknown_profile_are_errors():
    assert any(i.field == "runtime_profile"
               for i in _levels(schema.validate_alias("a", {}, PROFILES), "error"))
    issues = schema.validate_alias(
        "a", {"runtime_profile": "ghost", "real_model": "m"}, PROFILES)
    assert any("unknown runtime_profile: ghost" in i.message
               for i in _levels(issues, "error"))


def test_ollama_requires_real_model_llama_requires_model_source():
    issues = schema.validate_alias("a", {"runtime_profile": "local_ollama"}, PROFILES)
    assert any(i.field == "real_model" for i in _levels(issues, "error"))
    issues = schema.validate_alias("a", {"runtime_profile": "local_llamacpp"}, PROFILES)
    assert any(i.field == "model_path" for i in _levels(issues, "error"))


def test_profile_field_overridable_on_alias_no_warning():
    definition = {"runtime_profile": "local_llamacpp", "model_path": "/m.gguf",
                  "port": 8081, "default_api_base": "http://x:1"}
    issues = schema.validate_alias("a", definition, PROFILES)
    assert not issues or all(i.field not in ("port", "default_api_base") for i in issues)


def test_validate_profile_backend_enum():
    assert any(i.field == "backend"
               for i in _levels(schema.validate_profile("p", {}), "error"))
    issues = schema.validate_profile("p", {"backend": "vllm"})
    assert any("vllm" in i.message for i in _levels(issues, "error"))
    assert schema.validate_profile("p", {"backend": "ollama", "gpu": "cuda0"}) == []


def test_validate_role_refs():
    aliases = {"good": {}}
    issues = schema.validate_role(
        "r", {"default_alias": "ghost", "client_aliases": {"opencode": "good"}}, aliases)
    assert any("role references unknown alias: ghost" in i.message
               for i in _levels(issues, "error"))
    assert schema.validate_role(
        "r", {"default_alias": "good", "config_dir": "r",
              "client_aliases": {"opencode": "good"}}, aliases) == []


def test_lint_config_shapes():
    raw = {
        "aliases": {"a": {"runtime_profile": "local_ollama", "real_model": "m"}},
        "profiles": PROFILES,
        "roles": {"r": {"default_alias": "a", "config_dir": "r"}},
        "runtime_instances": {},
        "inference_profiles": {},
    }
    report = schema.lint_config(raw)
    assert report == {"aliases": {}, "runtime_instances": {},
                      "inference_profiles": {}, "profiles": {}, "roles": {}}
    raw["aliases"]["bad"] = {"runtime_profile": "ghost"}
    report = schema.lint_config(raw)
    assert "bad" in report["aliases"]
