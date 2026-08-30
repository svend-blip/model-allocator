# Config Schema Doctor Implementation Plan

> **For agentic workers:** Execute tasks in order. Steps use checkbox (`- [ ]`) syntax for tracking. Do not skip verification steps.

**Goal:** Add a schema module with per-backend field allow-lists and a `doctor` CLI command, so typo'd or mistyped config fields are caught instead of silently dropped, and config writes are validated.

**Architecture:** `Resolver.resolve_alias` (V2.1, src/model_allocator/resolver.py:47-54) does a blind generic merge — any alias field survives resolution, and `LlamaCppAdapter._build_argv` (adapters/llama_cpp.py:76-109) only reads known keys, so `n_cpu_mo` (typo for `n_cpu_moe`) silently vanishes from the server argv. `config_writer.set_alias` (config_writer.py:63-73) validates only `runtime_profile` existence. This plan adds `src/model_allocator/schema.py` (Issue dataclass + validate_alias/validate_profile/validate_role built from the REAL fields the code reads — tables below), wires error-blocking into `config_writer`, and adds `model-allocator doctor` (human report + `--json`, non-zero exit on errors) — after first running doctor against the live config and fixing what it flags.

**Tech Stack:** Python 3.10+ stdlib (`dataclasses`, `json`), pyyaml (only runtime dep — NO jsonschema), pytest (tests/test_config_writer.py conventions).

## Cold-Start Context

- model-allocator is a standalone Python CLI at `/home/svend/model-allocator` resolving model aliases → runtime commands for ollama / llama.cpp / opencode / claude-code / onyx backends.
- Config = `models.yaml` (aliases, top key `models`), `runtime_profiles.yaml` (top key `runtime_profiles`), `roles.yaml` (top key `roles`) at the repo root. `config_loader.load_config` resolves `${ENV}` refs; `config_writer.load_raw` loads UNRESOLVED values (for round-tripping edits).
- Source under `src/model_allocator/`; adapters under `src/model_allocator/adapters/`.
- Father (`/home/svend/DPMtF-WebUI`) consumes the CLI via subprocess to `scripts/model-allocator`; the config dashboard calls `config set-alias/set-role/...` (routers/bridge.py `_run_allocator`, lines 1249-1282) and surfaces `{"error": ...}` stderr JSON as HTTP 400 (`test_post_alias_validation_error_is_400` in Father's tests/test_allocator_config_endpoints.py depends on the error text containing the offending name).
- Run tests: `cd /home/svend/model-allocator && python3 -m pytest` → **95 passed** ~5s (baseline).

## Global Constraints

- `python3 -m py_compile <file>` MUST pass on every touched Python file.
- Single runtime dependency stays `pyyaml` (`mcp` optional extra). **No jsonschema** — the allow-lists are plain dicts.
- All 95 existing tests stay green, including `tests/test_config_writer.py` and `tests/test_config_cli.py` (their fixtures are schema-clean; keep existing `ConfigWriteError` message substrings `unknown runtime_profile:` and `role references unknown alias:` intact — Father asserts on them).
- TDD: failing test → implement → green (pytest style, tmp-dir configs, like tests/test_config_writer.py).
- Backwards compatibility: `doctor` is a NEW command (no consumer yet). `config set-alias/set-role` keep their stdout/stderr JSON contract (`{"ok": true, "name": ...}` / `{"error": "..."}`); new validation only ADDS error cases and stderr warnings.
- Write-blocking (Task 4) is enabled only AFTER the live config passes doctor (Task 3) — otherwise the Father dashboard could no longer re-save existing entries.
- Git policy: the Human approves commits. Tasks end with `git add <files>` and STOP.

## Edge Cases a Weaker Model Would Miss

1. **`isinstance(True, int)` is `True` in Python.** A YAML `port: true` would pass a naive int check. Every int/float check must exclude bool explicitly.
2. **YAML 1.1 booleanizes bare `on`/`off`/`yes`/`no`.** The live `models.yaml` alias `llama-test` has `flash_attn: on` and `reasoning: off` (lines 56-57) which PyYAML loads as `True`/`False` — `_build_argv` would emit `--flash-attn True` / `--reasoning False`, which llama-server rejects. This is a REAL latent bug in the repo today (verified 2026-07-12: `yaml.safe_load` returns `True`/`False` for these). Doctor must turn "str-typed field holding a bool" into an ERROR with the hint *quote the value* — and Task 3 fixes the live file.
3. **Env-ref values are strings until resolution.** Doctor lints the RAW config (`config_writer.load_raw` — deliberately unresolved so it matches the write path). A field typed int holding `"${LLAMA_PORT}"` must NOT be a type error: any string matching `config_loader.ENV_RE` is exempt from type checks (wildcard).
4. **The alias allow-list is the UNION of alias fields and profile fields.** The resolver copies every alias key over the profile merge (resolver.py:47-54), so `port`, `host`, `default_api_base`, etc. are legitimately overridable on an alias. Flagging them as unknown-on-alias would produce false warnings. Each backend's table below notes the source (alias/profile/both).
5. **Unknown field = WARNING, not error (forward compatibility).** Other in-flight plans (PLAN-claude-env-equivalence, PLAN-remote-llamacpp-lifecycle) introduce fields; their names are pre-seeded in the allow-lists below, which is harmless even if those plans have not landed (allow-lists only suppress warnings). A field NOT in any list may still be a future feature — warn, don't block.
6. **`default_gpu_layers` and `default_ctx` in the live `runtime_profiles.yaml` are dead** — verified: no source file reads them (only test fixture dicts mention them). Doctor will warn; the fix is to DELETE them from the live file (project principle: config must be visible-and-used or deleted), not to whitelist dead names.
7. **`lifecycle_policy` is declarative metadata** — no allocator code reads it, but Father's dispatch semantics and humans do. It stays in the allow-list; do not "clean it up".
8. **Blocking writes must not break Father's dashboard round-trip.** Father re-saves EXISTING aliases through `config set-alias`. If the live config has schema errors when blocking lands, every save fails. Hence the strict task order: doctor first, fix live config, THEN enable blocking.
9. **Warnings from `set_alias` must go to stderr as plain lines, never stdout.** `cmd_config_set_alias` prints `{"ok": true, ...}` JSON on stdout that Father parses (routers/bridge.py `_run_allocator` + `_config_payload`); polluting stdout breaks the dashboard.

---

### Field tables (the ground truth — extracted from the code 2026-07-12)

**Alias-level, all backends** (readers in parentheses):

| Field | Type | Read by |
|-------|------|---------|
| `runtime_profile` | str, **required** | resolver.py:30 |
| `real_model` | str (required for ollama/openai_compatible) | all client adapters |
| `context` | int | validator.py:57-61, llama_cpp.py:80, opencode.py:129-133, ollama.py:90 |
| `lifecycle_policy` | str | declarative only (Father semantics) |
| `clients` | dict[str, bool] | validator.py:48-55, cli.py:149-151 |
| `display_name` | str | opencode.py:91,115 |
| `opencode_provider_name` | str | opencode.py:44,78,99; cli.py:377 |
| `opencode_model_id` | str | opencode.py:45,79,100 |
| `persona_id` | int | onyx.py:107 |
| `invoke_timeout` | int | onyx.py:108 |
| `headless_output_dir` | str | adapters/headless.py:49 |
| `headless_idle_seconds` | int or float | adapters/headless.py:52 |
| `max_output_tokens` | int | claude_code adapter (PLAN-claude-env-equivalence) |
| `disable_adaptive_thinking` | bool | claude_code adapter (PLAN-claude-env-equivalence) |
| `claude_binary` | str | claude_code adapter (PLAN-claude-env-equivalence) |
| `claude_extra_args` | list | claude_code adapter (PLAN-claude-env-equivalence) |

**Alias-level, llama_cpp backend** (adapters/llama_cpp.py):

| Field | Type | Read at |
|-------|------|---------|
| `model_path` | str | model_path() line 65 |
| `model_name` | str | line 67 |
| `port` | int | _resolve_port() line 36 |
| `host` | str | line 27 |
| `parallel` | int | _build_argv line 86 |
| `n_cpu_moe` | int | line 88 |
| `threads` | int | line 90 |
| `batch` | int | line 92 |
| `ubatch_size` | int | line 94 |
| `cache_type_k` | str | line 96 |
| `cache_type_v` | str | line 98 |
| `flash_attn` | str (`"on"`/`"off"` — MUST be quoted in YAML) | line 100 |
| `reasoning` | str (quoted) | line 102 |
| `no_mmap` | bool | line 103 |
| `gpu_layers` | int | line 106 |
| `tensor_split` | str | line 108 |

**Profile-level** (all overridable on an alias via the resolver union merge):

| Field | Type | Read by |
|-------|------|---------|
| `backend` | str, **required**, one of `ollama`/`llama_cpp`/`openai_compatible`/`onyx` | resolver.py:40, cli.py:42-60 |
| `api_base_env` | str | ollama/openai/onyx/claude_code adapters |
| `default_api_base` | str | same |
| `api_key_env` | str | openai_compatible, onyx, claude_code |
| `provider` | str | opencode.py, validator.py:89 |
| `gpu` | str | resolver.py:43, validator.py:46 |
| `email_env` / `password_env` | str | onyx.py:105-106 |
| `capabilities` | list[str] | cli.py:460, validator.py:164, headless.py:43 |
| `server_bin_env` | str | llama_cpp.py:51 |
| `model_root_env` | str | llama_cpp.py:68 |
| `default_port` | int | llama_cpp.py:38, opencode.py:81 |
| `host` | str | llama_cpp.py:27, opencode.py:80 |
| `ssh_host` / `remote_workdir` / `server_bin_path` | str | PLAN-remote-llamacpp-lifecycle (pre-seeded) |

**Role-level** (resolver.py:57-71, config_writer.py:55-60):

| Field | Type |
|-------|------|
| `default_alias` | str, must reference an existing alias |
| `config_dir` | str |
| `client_aliases` | dict[str, str], every value must reference an existing alias |

---

### Task 1: TDD — `src/model_allocator/schema.py`

**Files:**
- Create: `/home/svend/model-allocator/src/model_allocator/schema.py`
- Test: Create `/home/svend/model-allocator/tests/test_schema.py`

**Interfaces:**
- `@dataclass Issue: level: str ("error"|"warning"), field: str, message: str`
- `validate_alias(alias_name: str, definition: dict, profiles: dict) -> list[Issue]`
- `validate_profile(profile_name: str, definition: dict) -> list[Issue]`
- `validate_role(role_name: str, definition: dict, aliases: dict) -> list[Issue]`
- `lint_config(raw: dict) -> dict` — `raw` is `config_writer.load_raw()` output (`{"aliases", "roles", "profiles"}`); returns `{"aliases": {name: [Issue,...]}, "profiles": {...}, "roles": {...}}` with only non-empty entries.

- [ ] Step 1: Create `tests/test_schema.py`:

```python
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
                  "n_cpu_mo": 26}  # typo for n_cpu_moe
    issues = schema.validate_alias("a", definition, PROFILES)
    warnings = _levels(issues, "warning")
    assert any(i.field == "n_cpu_mo" for i in warnings)
    assert not any(i.field == "n_cpu_mo" for i in _levels(issues, "error"))


def test_yaml_bool_trap_on_flash_attn_is_error_with_quote_hint():
    # yaml.safe_load("flash_attn: on") == {"flash_attn": True}
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
                  "port": "${LLAMA_PORT}"}  # int field, but env-ref -> exempt
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
    }
    report = schema.lint_config(raw)
    assert report == {"aliases": {}, "profiles": {}, "roles": {}}
    raw["aliases"]["bad"] = {"runtime_profile": "ghost"}
    report = schema.lint_config(raw)
    assert "bad" in report["aliases"]
```

- [ ] Step 2: Run — all fail with `ModuleNotFoundError` / `ImportError`:
```bash
cd /home/svend/model-allocator && python3 -m pytest tests/test_schema.py -v
```

- [ ] Step 3: Create `src/model_allocator/schema.py`:

```python
"""Config schema: per-backend field allow-lists + validation (doctor).

The resolver (V2.1) does a blind generic merge, so typo'd fields silently
survive resolution and are then silently ignored by the adapters
(e.g. ``n_cpu_mo`` instead of ``n_cpu_moe`` never reaches llama-server's
argv). This module encodes the fields the code ACTUALLY reads — sources are
noted in PLAN-config-schema-doctor.md's field tables.

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


BACKENDS = ("ollama", "llama_cpp", "openai_compatible", "onyx")

# Fields legal on every alias (see plan field table for readers).
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
    # PLAN-claude-env-equivalence (pre-seeded; harmless if not yet landed)
    "max_output_tokens": int,
    "disable_adaptive_thinking": bool,
    "claude_binary": str,
    "claude_extra_args": list,
}

BACKEND_ALIAS_FIELDS: dict[str, dict[str, object]] = {
    "ollama": {},
    "openai_compatible": {},
    "onyx": {},
    "llama_cpp": {
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
        "flash_attn": str,   # "on"/"off" — must be quoted in YAML
        "reasoning": str,    # "on"/"off" — must be quoted in YAML
        "no_mmap": bool,
        "gpu_layers": int,
        "tensor_split": str,
    },
}

# Profile fields. The resolver merges profile fields under alias fields,
# so all of these are also legal as alias-level overrides (UNION rule).
PROFILE_FIELDS: dict[str, object] = {
    "backend": str,
    "api_base_env": str,
    "default_api_base": str,
    "api_key_env": str,
    "provider": str,
    "gpu": str,
    "email_env": str,
    "password_env": str,
    "capabilities": list,
    "server_bin_env": str,
    "model_root_env": str,
    "default_port": int,
    "host": str,
    # PLAN-remote-llamacpp-lifecycle (pre-seeded)
    "ssh_host": str,
    "remote_workdir": str,
    "server_bin_path": str,
    # PLAN-claude-env-equivalence overrides on the profile
    "claude_binary": str,
    "claude_extra_args": list,
    "max_output_tokens": int,
    "disable_adaptive_thinking": bool,
}

ROLE_FIELDS: dict[str, object] = {
    "default_alias": str,
    "config_dir": str,
    "client_aliases": dict,
}


def _is_env_ref(value: object) -> bool:
    return isinstance(value, str) and bool(ENV_RE.search(value))


def _check_type(field: str, value: object, expected: object) -> Issue | None:
    if _is_env_ref(value):
        return None  # strings until resolution — wildcard
    if expected is bool:
        if not isinstance(value, bool):
            return Issue("error", field,
                         f"expected bool, got {type(value).__name__}")
        return None
    if expected is str and isinstance(value, bool):
        return Issue("error", field,
                     "expected string but YAML parsed a boolean — quote the "
                     "value (YAML 1.1 turns bare on/off/yes/no into booleans)")
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            return Issue("error", field,
                         f"expected int, got {type(value).__name__}")
        return None
    if isinstance(expected, tuple):
        if isinstance(value, bool) or not isinstance(value, expected):
            names = "/".join(t.__name__ for t in expected)
            return Issue("error", field,
                         f"expected {names}, got {type(value).__name__}")
        return None
    if not isinstance(value, expected):
        return Issue("error", field,
                     f"expected {expected.__name__}, got {type(value).__name__}")
    return None


def _lint_fields(definition: dict, allowed: dict, skip: set) -> list[Issue]:
    issues: list[Issue] = []
    for field, value in definition.items():
        if field in skip:
            continue
        expected = allowed.get(field)
        if expected is None:
            issues.append(Issue("warning", field,
                                "unknown field (typo?) — the resolver keeps "
                                "it but no adapter reads it"))
            continue
        issue = _check_type(field, value, expected)
        if issue:
            issues.append(issue)
    return issues


def validate_alias(alias_name: str, definition: dict, profiles: dict) -> list[Issue]:
    issues: list[Issue] = []
    profile: dict = {}
    profile_name = definition.get("runtime_profile")
    if not profile_name:
        issues.append(Issue("error", "runtime_profile", "required field is missing"))
    elif not isinstance(profile_name, str):
        issues.append(Issue("error", "runtime_profile",
                            f"expected str, got {type(profile_name).__name__}"))
    elif profile_name not in profiles:
        issues.append(Issue("error", "runtime_profile",
                            f"unknown runtime_profile: {profile_name}"))
    else:
        profile = profiles[profile_name] or {}

    backend = profile.get("backend")
    allowed = dict(PROFILE_FIELDS)          # UNION: profile fields overridable on alias
    allowed.update(COMMON_ALIAS_FIELDS)
    allowed.update(BACKEND_ALIAS_FIELDS.get(backend, {}))
    issues += _lint_fields(definition, allowed, skip={"runtime_profile"})

    clients = definition.get("clients")
    if isinstance(clients, dict):
        for client_key, flag in clients.items():
            if not isinstance(flag, bool):
                issues.append(Issue("error", f"clients.{client_key}",
                                    f"expected bool, got {type(flag).__name__}"))

    if backend in ("ollama", "openai_compatible") and not definition.get("real_model"):
        issues.append(Issue("error", "real_model",
                            f"{backend} alias requires real_model"))
    if backend == "llama_cpp":
        has_path = bool(definition.get("model_path"))
        has_name = bool(definition.get("model_name")) and bool(
            definition.get("model_root_env") or profile.get("model_root_env"))
        if not (has_path or has_name):
            issues.append(Issue("error", "model_path",
                                "llama_cpp alias needs model_path, or "
                                "model_name + model_root_env"))
    return issues


def validate_profile(profile_name: str, definition: dict) -> list[Issue]:
    issues: list[Issue] = []
    backend = definition.get("backend")
    if not backend:
        issues.append(Issue("error", "backend", "required field is missing"))
    elif backend not in BACKENDS:
        issues.append(Issue("error", "backend",
                            f"unknown backend '{backend}' "
                            f"(expected one of: {', '.join(BACKENDS)})"))
    issues += _lint_fields(definition, PROFILE_FIELDS, skip={"backend"})
    return issues


def validate_role(role_name: str, definition: dict, aliases: dict) -> list[Issue]:
    issues: list[Issue] = []
    issues += _lint_fields(definition, ROLE_FIELDS, skip=set())
    refs = []
    if definition.get("default_alias"):
        refs.append(definition["default_alias"])
    client_aliases = definition.get("client_aliases")
    if isinstance(client_aliases, dict):
        refs.extend(client_aliases.values())
    for ref in refs:
        if ref not in aliases:
            issues.append(Issue("error", "default_alias",
                                f"role references unknown alias: {ref}"))
    return issues


def lint_config(raw: dict) -> dict:
    """Lint a config_writer.load_raw() dict. Returns per-section issue maps."""
    profiles = raw.get("profiles", {}) or {}
    aliases = raw.get("aliases", {}) or {}
    roles = raw.get("roles", {}) or {}
    return {
        "aliases": {name: issues for name, definition in aliases.items()
                    if (issues := validate_alias(name, definition or {}, profiles))},
        "profiles": {name: issues for name, definition in profiles.items()
                     if (issues := validate_profile(name, definition or {}))},
        "roles": {name: issues for name, definition in roles.items()
                  if (issues := validate_role(name, definition or {}, aliases))},
    }
```

- [ ] Step 4: Verify:
```bash
python3 -m py_compile src/model_allocator/schema.py && python3 -m pytest tests/test_schema.py -v && python3 -m pytest 2>&1 | tail -1
```
Expected: 10/10 new; full suite `105 passed`.

---

### Task 2: TDD — `doctor` CLI command (human report + `--json`)

**Files:**
- Test: Create `/home/svend/model-allocator/tests/test_doctor_cli.py`
- Modify: `/home/svend/model-allocator/src/model_allocator/cli.py` — new `cmd_doctor` (place after `cmd_config_delete_role`, line 444) + parser registration in `build_parser` (after the `p_config` block, line 630).

**Interfaces:**
- `model-allocator doctor [--json]` → lints ALL aliases + profiles + roles from `config_writer.load_raw(config_dir)`.
- Exit codes: 1 if any error, 2 if warnings only, 0 if clean (mirrors validate's OK/WARNING/ERROR mapping).
- `--json` stdout: `{"status": "OK"|"WARNING"|"ERROR", "errors": N, "warnings": N, "aliases": {name: [{level, field, message}]}, "profiles": {...}, "roles": {...}}`.

- [ ] Step 1: Create `tests/test_doctor_cli.py`:

```python
"""Tests for the doctor CLI command (PLAN-config-schema-doctor)."""

import json
from pathlib import Path

import yaml

from model_allocator.cli import main


def _seed(tmp_path: Path, models: dict) -> Path:
    (tmp_path / "models.yaml").write_text(yaml.safe_dump({"models": models}),
                                          encoding="utf-8")
    (tmp_path / "runtime_profiles.yaml").write_text(yaml.safe_dump({
        "runtime_profiles": {
            "local_llamacpp": {"backend": "llama_cpp",
                               "server_bin_env": "LLAMA_SERVER_BIN"},
        }}), encoding="utf-8")
    (tmp_path / "roles.yaml").write_text(yaml.safe_dump({"roles": {}}),
                                         encoding="utf-8")
    return tmp_path


def test_doctor_clean_config_exit_0(tmp_path, capsys):
    d = _seed(tmp_path, {"ok": {"runtime_profile": "local_llamacpp",
                                "model_path": "/m.gguf",
                                "flash_attn": "on"}})
    rc = main(["--config-dir", str(d), "doctor"])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_doctor_typo_warning_exit_2(tmp_path, capsys):
    d = _seed(tmp_path, {"a": {"runtime_profile": "local_llamacpp",
                               "model_path": "/m.gguf",
                               "n_cpu_mo": 26}})
    rc = main(["--config-dir", str(d), "doctor"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "n_cpu_mo" in out and "WARNING" in out


def test_doctor_bool_trap_error_exit_1_and_json(tmp_path, capsys):
    d = _seed(tmp_path, {"a": {"runtime_profile": "local_llamacpp",
                               "model_path": "/m.gguf",
                               "flash_attn": True}})
    rc = main(["--config-dir", str(d), "doctor", "--json"])
    assert rc == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ERROR"
    assert report["errors"] >= 1
    alias_issues = report["aliases"]["a"]
    assert any(i["field"] == "flash_attn" and i["level"] == "error"
               for i in alias_issues)
```

- [ ] Step 2: Run — fails with argparse `invalid choice: 'doctor'`.

- [ ] Step 3: Add `cmd_doctor` to `cli.py` (after `cmd_config_delete_role`):

```python
def cmd_doctor(args: argparse.Namespace) -> int:
    """Lint ALL aliases, profiles, and roles against the field schema."""
    from dataclasses import asdict

    from model_allocator import schema

    raw = config_writer.load_raw(_config_dir(args))
    report = schema.lint_config(raw)
    n_errors = sum(1 for section in report.values() for issues in section.values()
                   for issue in issues if issue.level == "error")
    n_warnings = sum(1 for section in report.values() for issues in section.values()
                     for issue in issues if issue.level == "warning")
    status = "ERROR" if n_errors else ("WARNING" if n_warnings else "OK")

    if getattr(args, "json", False):
        payload = {
            "status": status,
            "errors": n_errors,
            "warnings": n_warnings,
        }
        for section, entries in report.items():
            payload[section] = {
                name: [asdict(issue) for issue in issues]
                for name, issues in entries.items()
            }
        print(json.dumps(payload, indent=2))
    else:
        print(f"{status}: {n_errors} error(s), {n_warnings} warning(s)")
        section_labels = {"aliases": "alias", "profiles": "profile", "roles": "role"}
        for section, entries in report.items():
            for name, issues in entries.items():
                for issue in issues:
                    print(f"  {issue.level.upper():7s} {section_labels[section]} "
                          f"{name}.{issue.field}: {issue.message}")

    if n_errors:
        return EXIT_ERROR
    if n_warnings:
        return EXIT_WARNING
    return EXIT_OK
```

- [ ] Step 4: Register the subcommand in `build_parser` (after the `config` subparser block, before `return parser`):

```python
    p_doctor = sub.add_parser(
        "doctor",
        help="Lint all aliases/profiles/roles against the field schema (typos, YAML bool traps, bad refs)",
    )
    p_doctor.add_argument("--json", action="store_true",
                          help="Machine-readable report")
    p_doctor.set_defaults(func=cmd_doctor)
```

- [ ] Step 5: Verify:
```bash
python3 -m py_compile src/model_allocator/cli.py && python3 -m pytest tests/test_doctor_cli.py -v && python3 -m pytest 2>&1 | tail -1
```
Expected: 3/3 new; full suite `108 passed`.

---

### Task 3: Run doctor against the LIVE config and fix what it flags

**Files:**
- Modify: `/home/svend/model-allocator/models.yaml` (lines 56-57), `/home/svend/model-allocator/runtime_profiles.yaml` (lines 14-15), and mirror in `models.example.yaml` / `runtime_profiles.example.yaml`.

Expected findings (pre-computed 2026-07-12 by applying the rules to the live files — verify by actually running doctor):

1. **ERROR** alias `llama-test.flash_attn`: bool (`on` unquoted → `True`).
2. **ERROR** alias `llama-test.reasoning`: bool (`off` unquoted → `False`).
3. **WARNING** profile `local_llamacpp_cuda0.default_gpu_layers`: unknown field (dead — no reader in src/).
4. **WARNING** profile `local_llamacpp_cuda0.default_ctx`: unknown field (dead — no reader in src/).

- [ ] Step 1: Run doctor against the live config:
```bash
cd /home/svend/model-allocator && ./scripts/model-allocator doctor; echo "rc=$?"
```
Expected: `rc=1` with exactly the four findings above (plus none others; if others appear, evaluate each: real bug → fix; legitimate field the tables missed → add to the allow-list in schema.py WITH a reader reference in the plan-table style, and note it in the commit message).

- [ ] Step 2: Fix `models.yaml` lines 56-57 — quote the values:
```yaml
    flash_attn: "on"
    reasoning: "off"
```
(This fixes a real latent bug: unquoted they resolve to `--flash-attn True` / `--reasoning False` in the llama-server argv.)

- [ ] Step 3: Fix `runtime_profiles.yaml` — delete the two dead lines from `local_llamacpp_cuda0`:
```yaml
    default_gpu_layers: 99
    default_ctx: 131072
```
(Verified unread by any source file; project principle: config is visible-and-used or deleted.)

- [ ] Step 4: Apply the same two fixes to `models.example.yaml` (`llama-*` example alias, if it has unquoted `on`/`off`) and `runtime_profiles.example.yaml` (same dead fields).

- [ ] Step 5: Re-run and require a clean bill:
```bash
./scripts/model-allocator doctor; echo "rc=$?"
```
Expected: `OK: 0 error(s), 0 warning(s)` and `rc=0`. Do not proceed to Task 4 until this passes.

---

### Task 4: Wire write-blocking into `config_writer.set_alias` / `set_role`

**Files:**
- Test: append to `/home/svend/model-allocator/tests/test_config_writer.py`.
- Modify: `/home/svend/model-allocator/src/model_allocator/config_writer.py` — `set_alias` (lines 63-73), `set_role` (lines 89-100); add `import sys` and `from model_allocator import schema` to the imports (after `import yaml`, line 14).

**Interfaces:**
- Errors from `schema.validate_alias`/`validate_role` raise `ConfigWriteError` with `"; ".join(f"{field}: {message}")` — the existing substrings `unknown runtime_profile: <name>` and `role references unknown alias: <ref>` are preserved verbatim inside the messages (Father's HTTP-400 tests grep for them).
- Warnings print one line each to **stderr** (`WARNING: alias <name>.<field>: <message>`); the write proceeds.

- [ ] Step 1: Add failing tests to `tests/test_config_writer.py`:

```python
def test_set_alias_blocks_on_type_error(tmp_path):
    d = _seed(tmp_path)
    with pytest.raises(cw.ConfigWriteError) as exc:
        cw.set_alias(d, "bad", {"runtime_profile": "local_ollama_cuda0",
                                "real_model": "m", "context": "not-an-int"})
    assert "context" in str(exc.value)


def test_set_alias_warns_but_writes_on_unknown_field(tmp_path, capsys):
    d = _seed(tmp_path)
    cw.set_alias(d, "warned", {"runtime_profile": "local_ollama_cuda0",
                               "real_model": "m", "n_cpu_mo": 26})
    assert "warned" in cw.load_raw(d)["aliases"]
    assert "n_cpu_mo" in capsys.readouterr().err


def test_set_alias_env_ref_not_blocked(tmp_path):
    d = _seed(tmp_path)
    cw.set_alias(d, "env-ok", {"runtime_profile": "local_ollama_cuda0",
                               "real_model": "${REAL_MODEL_ENV}"})
    assert "env-ok" in cw.load_raw(d)["aliases"]


def test_set_role_blocks_on_bad_type(tmp_path):
    d = _seed(tmp_path)
    with pytest.raises(cw.ConfigWriteError):
        cw.set_role(d, "r2", {"default_alias": "imple-fast",
                              "config_dir": 123})
```

- [ ] Step 2: Run — `test_set_alias_blocks_on_type_error`, `test_set_alias_warns_but_writes_on_unknown_field` (stderr empty), and `test_set_role_blocks_on_bad_type` fail.

- [ ] Step 3: Replace `set_alias` (config_writer.py:63-73) with:

```python
def set_alias(config_dir: str | Path, name: str, definition: dict) -> None:
    if not name:
        raise ConfigWriteError("alias name is required")
    d = Path(config_dir)
    raw = load_raw(d)
    issues = schema.validate_alias(name, definition, raw["profiles"])
    _enforce(issues, f"alias {name}")
    aliases = raw["aliases"]
    aliases[name] = definition
    _safe_write(_find(d, "models"), "models", aliases)
```

and `set_role` (lines 89-100) with:

```python
def set_role(config_dir: str | Path, name: str, definition: dict) -> None:
    if not name:
        raise ConfigWriteError("role name is required")
    d = Path(config_dir)
    raw = load_raw(d)
    issues = schema.validate_role(name, definition, raw["aliases"])
    _enforce(issues, f"role {name}")
    roles = raw["roles"]
    roles[name] = definition
    _safe_write(_find(d, "roles"), "roles", roles)
```

and add the shared helper above `set_alias` (plus remove the now-redundant inline profile check — `schema.validate_alias` covers it with the same message text; keep `_role_alias_refs` because `delete_alias` still uses it):

```python
def _enforce(issues, subject: str) -> None:
    """Errors block the write; warnings go to stderr and the write proceeds."""
    errors = [i for i in issues if i.level == "error"]
    if errors:
        raise ConfigWriteError(
            "; ".join(f"{i.field}: {i.message}" for i in errors))
    for issue in issues:
        print(f"WARNING: {subject}.{issue.field}: {issue.message}",
              file=sys.stderr)
```

- [ ] Step 4: Verify existing behavior contracts still hold — `test_set_alias_rejects_unknown_profile` (message contains `unknown runtime_profile: nope`) and `test_set_role_rejects_dangling_alias` (`role references unknown alias: ghost`) must pass unchanged:
```bash
python3 -m py_compile src/model_allocator/config_writer.py && python3 -m pytest tests/test_config_writer.py tests/test_config_cli.py -v && python3 -m pytest 2>&1 | tail -1
```
Expected: full suite `112 passed`.

- [ ] Step 5: Cross-repo check (read-only, no Father change): Father's dashboard endpoints pass alias/role definitions through `config set-alias` / `set-role` (routers/bridge.py `_run_allocator`, lines 1249-1282) and map stderr `{"error": ...}` to HTTP 400 — but note `_enforce` warnings are plain lines, not the error JSON; they only appear when the write SUCCEEDS (rc 0), and Father ignores stderr on success (verified: `_run_allocator` raises only on `returncode != 0`). Run Father's tests to prove no regression:
```bash
cd /home/svend/DPMtF-WebUI && python3 -m pytest tests/test_allocator_config_endpoints.py -q
```
Expected: all pass with no Father modification.

- [ ] Step 6: Stage and STOP — await Human commit approval:
```bash
cd /home/svend/model-allocator && git add src/model_allocator/schema.py src/model_allocator/cli.py src/model_allocator/config_writer.py tests/test_schema.py tests/test_doctor_cli.py tests/test_config_writer.py models.yaml runtime_profiles.yaml models.example.yaml runtime_profiles.example.yaml
git status --short
```
Suggested commit message: `[V5.3] Config schema doctor — per-backend allow-lists, doctor CLI (--json), write-blocking in config_writer; fix YAML bool trap + drop dead profile fields`

## Acceptance Criteria

1. `cd /home/svend/model-allocator && python3 -m pytest` → `112 passed` (95 baseline + 10 schema + 3 doctor + 4 writer), 0 failures.
2. `python3 -m py_compile src/model_allocator/schema.py src/model_allocator/cli.py src/model_allocator/config_writer.py` → exit 0.
3. `./scripts/model-allocator doctor; echo rc=$?` on the fixed live config → `OK: 0 error(s), 0 warning(s)`, `rc=0`.
4. `./scripts/model-allocator doctor --json | python3 -m json.tool > /dev/null && echo JSON_OK` → `JSON_OK`.
5. Bool-trap regression proof: `python3 -c "import yaml; print(yaml.safe_load(open('models.yaml'))['models']['llama-test']['flash_attn'])"` → `on` (string), no longer `True`.
6. `./scripts/model-allocator config set-alias --name typo-test --json '{"runtime_profile": "local_ollama_cuda0", "real_model": "m", "context": "big"}'; echo rc=$?` → stderr JSON `{"error": "context: expected int, got str"}`, `rc=1`, and `typo-test` NOT in models.yaml. Clean up any successful test aliases afterwards with `config delete-alias`.
7. `grep -rn "jsonschema" src/ pyproject.toml` → no matches (pyyaml-only honored).
8. `cd /home/svend/DPMtF-WebUI && python3 -m pytest tests/test_allocator_config_endpoints.py -q` → all pass with zero Father changes.
9. `git status --short` in the allocator repo shows exactly the Task 4 Step 6 file list staged; nothing committed.
