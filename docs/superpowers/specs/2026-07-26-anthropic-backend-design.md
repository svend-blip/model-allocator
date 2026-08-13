# Anthropic Native API Backend — Design Spec

**Date:** 2026-07-26
**Status:** Draft
**Project:** model-allocator

## 1. Motivation

model-allocator understøtter i dag fire backends: `ollama`, `llama_cpp`,
`openai_compatible`, og `onyx`. Claude Code-adapteren kan route til Ollama
(lokal Anthropic-kompatibel endpoint) eller OpenRouter (cloud
Anthropic-kompatibel proxy). Der findes **ingen** mulighed for at lade
Claude Code tale direkte med den native Anthropic API (`api.anthropic.com`).

Dette blokerer brug af Anthropic's egne modeller — specifikt Fable 5
(`claude-fable-5`) — via model-allocator.

## 2. Design

### 2.1 Ny backend: `anthropic`

Tilføj `"anthropic"` til `BACKENDS` i `schema.py`.

### 2.2 Ny runtime profile: `cloud_anthropic`

```yaml
runtime_profiles:
  cloud_anthropic:
    backend: anthropic
    api_key_env: ANTHROPIC_API_KEY
    provider: anthropic
```

### 2.3 Claude Code adapter (`adapters/claude_code.py`)

Ny gren i `build_claude_code_command()` for `backend == "anthropic"`:

- **Sæt** `ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY` (dollar-reference, shell-ekspanderes)
- **Ikke** sæt `ANTHROPIC_BASE_URL` — Claude Code bruger sin default (`api.anthropic.com`)
- **Ikke** sæt `ANTHROPIC_AUTH_TOKEN`
- `max_output_tokens` og `disable_adaptive_thinking` håndteres som for andre backends
- `MODEL_ALLOCATOR_ACTIVE_MODEL` sættes som sædvanligt

### 2.4 Ny adapter: `adapters/anthropic.py`

En tynd `AnthropicAdapter` klasse til runtime-livscyklus (start/stop/status):

```python
class AnthropicAdapter:
    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY"): ...
    def are_credentials_present(self) -> dict: ...
    def status(self) -> dict: ...
    def start(self) -> dict: ...    # cloud_noop — tjek kun credentials
    def stop(self) -> dict: ...     # noop
    def unload(self) -> dict: ...   # noop
```

`start()` returnerer `{started: True}` hvis `ANTHROPIC_API_KEY` er sat,
ellers `{started: False, error: ...}`.

### 2.5 Validator (`validator.py`)

- `_validate_anthropic()`: tjek at `ANTHROPIC_API_KEY` er sat i miljøet
- `_check_client_backend_compatibility()`: `anthropic` + `claude-code` = OK;
  `anthropic` + `opencode` = ERROR (OpenCode understøtter ikke native Anthropic)

### 2.6 CLI (`cli.py`)

- `_get_backend_adapter()`: `anthropic` → `AnthropicAdapter`
- `cmd_start()`: `anthropic` → `adapter.start()`
- `cmd_stop()`: `anthropic` → `adapter.stop()`
- `cmd_unload()`: `anthropic` → `adapter.unload()`
- `cmd_status()`: `anthropic` → `adapter.status()`
- `cmd_preflight()`: `anthropic` → `adapter.start()`

### 2.7 Model-alias: `fable5`

```yaml
models:
  fable5:
    runtime_profile: cloud_anthropic
    real_model: claude-fable-5
    context: 200000
    lifecycle_policy: cloud_noop
    clients:
      claude-code: true
```

Ingen `max_output_tokens` — ingen kunstig begrænsning.

### 2.8 Frontend (web UI)

**Ingen ændringer nødvendige.** UI'et er datadrevet:
- `/api/profiles` returnerer `cloud_anthropic` med `backend: anthropic`
- Model-formularens runtime_profile-dropdown inkluderer automatisk `cloud_anthropic`
- Profiles-tabellen viser backend-kolonnen som "anthropic"

## 3. Fil-oversigt

| Fil | Ændring | Est. linjer |
|-----|---------|-------------|
| `src/model_allocator/schema.py` | Tilføj `"anthropic"` til BACKENDS | +1 |
| `src/model_allocator/adapters/anthropic.py` | **NY** AnthropicAdapter | ~40 |
| `src/model_allocator/adapters/claude_code.py` | Håndter `anthropic` backend | +15 |
| `src/model_allocator/validator.py` | `_validate_anthropic()` + kompatibilitet | +20 |
| `src/model_allocator/cli.py` | Håndter `anthropic` i adapter/router | +15 |
| `runtime_profiles.yaml` | Ny `cloud_anthropic` profil | +5 |
| `models.yaml` | Ny `fable5` alias | +9 |
| `tests/test_v2.py` | Tests for anthropic backend | +30 |

## 4. Test-plan

1. **Resolver:** `resolve_alias("fable5")` returnerer `backend: anthropic`,
   `api_key_env: ANTHROPIC_API_KEY`
2. **Claude Code adapter:** `build_claude_code_command()` for anthropic:
   - `ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY` i env
   - Ingen `ANTHROPIC_BASE_URL` i env
   - Ingen `ANTHROPIC_AUTH_TOKEN` i env
   - `argv` indeholder `--model claude-fable-5`
3. **Validator:** `validate("fable5", "claude-code")` → OK når API key er sat
4. **Validator:** `validate("fable5", "opencode")` → ERROR
5. **AnthropicAdapter:** `start()` → `{started: True}` når key sat
6. **AnthropicAdapter:** `start()` → `{started: False}` når key mangler
7. **Doctor:** `lint_config()` accepterer `anthropic` backend uden warnings

## 5. Afhængigheder

- Bruger skal have `ANTHROPIC_API_KEY` sat i sit miljø (Anthropic API-nøgle fra
  console.anthropic.com)
- Claude Code binary skal være installeret og på PATH
- Ingen nye Python-pakker kræves

## 6. Risici

- **Lav:** `api.anthropic.com` connectivity kan ikke valideres uden at foretage
  et ægte API-kald — `start()` tjekker kun at API-nøglen er sat, ikke at den
  er gyldig. Dette er samme mønster som `openai_compatible` backenden.
