# Model Allocator

> A validated runtime/model-alias layer for the DPMtF ecosystem.
> Decouples *which model a role uses* from *how that model is started, stopped,
> validated, and resolved* across local Ollama, llama.cpp (TurboQuant), cloud
> OpenAI-compatible APIs, and Minimax.

Status: **V1A → V5, fully built and live-validated.** 95 tests. All
adapters validated against real backends. Wired into the Father WebUI,
including a full config dashboard (alias/role CRUD). V5 adds ONYX as an
OPTIONAL invoke-only knowledge runtime + a generic headless client that
turns any invoke-capable API runtime into a bridgeV002 role with zero
bridge changes.

---

## What it is

Model Allocator is a small standalone Python CLI (plus a Father-side proxy + UI)
that lets bridgeV002 (the DPMtF role/step orchestrator) refer to models by a
**stable logical alias** (`imple-fast`, `review-cloud`, `llama-test`,
`svend3060-llama-test`) instead of backend-specific start commands. The
allocator resolves the alias to a real backend/model/runtime, validates that the
combination works, and manages the runtime lifecycle (start/stop/status).

### Three-layer separation

```
bridgeV002        → role/step orchestration (which role uses which alias)
Model Allocator   → alias resolution + validation + lifecycle (this repo)
Backend adapters  → concrete runtime commands (Ollama, llama.cpp, cloud, …)
```

> bridgeV002 chooses role / step / model-alias.
> Model Allocator owns runtime lifecycle.
> Backend adapters own concrete commands.

---

## Adapters

### Backend adapters

| Adapter | Backend | start | stop / unload | status | context / offload |
|---------|---------|-------|---------------|--------|-------------------|
| `ollama` | Local Ollama | warm via `/api/generate` + keep_alive | `ollama stop <model>` | `ollama ps` / API | num_ctx per-request (informational) |
| `llama_cpp` | llama-server (TurboQuant build) | spawn `llama-server` with full flag set + PID file + free/configured port + `/health` polling | kill PID (timeout) | `/health` + PID alive | `--ctx-size`, `--n-cpu-moe` (MoE) / `--n-gpu-layers` (dense), `--cache-type-k/v`, `--flash-attn`, `--tensor-split`, … |
| `openai_compatible` | Cloud OpenAI-compatible | validate only | no-op | API reachability | model max context |
| `onyx` | ONYX assistants (optional) | no-op (docker compose owns lifecycle) | no-op | `/health` + credentials | invoke-only: one-shot chat with citations |

### Client adapters

| Adapter | Launches | Model prefix |
|----------|----------|--------------|
| `opencode` | `opencode` TUI | `ollama/<model>`, `openrouter/<model>`, bare Minimax id, `<provider>/<model_id>` for llama.cpp |
| `claude_code` | `claude` TUI | `claude --model <model>` (valid flag on Claude Code, unlike OpenCode) |
| `headless` | allocator runner loop | any invoke-capable alias — pasted tmux prompt -> invoke() -> answer in pane + InvokeResult JSON file |

### Client/backend matrix

- **Claude Code** supports Ollama + OpenRouter (Anthropic-compatible endpoints). Minimax via Claude Code is rejected (no Anthropic-compatible endpoint) — `validate` returns a clear ERROR.
- **OpenCode** supports Ollama + OpenRouter + Minimax (built-in) + openai_compatible.

---

## CLI

```
model-allocator resolve  --role <role> --client <client>
model-allocator validate --alias <alias> --client <client>
model-allocator list     [--only-ok] [--client <client>]
model-allocator status   --alias <alias>
model-allocator start    --alias <alias>
model-allocator stop     --alias <alias>
model-allocator unload   --alias <alias>
model-allocator run      --role <role> --client <client>
model-allocator env      --role <role> --client claude-code
model-allocator render-config --role <role> --client opencode [--output <path>]
model-allocator preflight --role <role> --client <client>
model-allocator config show
model-allocator config set-alias    --name <alias> --json '<definition>'
model-allocator config delete-alias --name <alias>
model-allocator config set-role     --name <role>  --json '<definition>'
model-allocator config delete-role  --name <role>
model-allocator invoke   --alias <alias> [--prompt <p>|-] [--persona N] [--timeout S]
model-allocator headless --alias <alias> [--output-dir D] [--idle-seconds F]
model-allocator mcp-serve [--alias <alias>] [--host H] [--port 9164]
```

Run via `python3 -m model_allocator <command>` (dev) or the `scripts/model-allocator`
wrapper (installed/PATH, used by bridgeV002 subprocess calls).

### Key command semantics

- **`run`** prints a tmux-safe shell string (env + argv) to stdout. bridgeV002's
  `start_coding.py` captures it and sends it to the role's tmux session. The
  allocator does **not** own tmux orchestration.
- **`render-config --output <path>`** writes a model-specific `opencode.json`
  atomically (temp + rename), **merging** with an existing file (preserves
  `$schema`, `permission`, `mcp`, other providers; sets the top-level `model`
  field + the role's provider block). Required because the OpenCode TUI
  silently ignores the `--model` CLI flag — the `model` field in `opencode.json`
  is the only reliable way to set the model.
- **`stop` / `start`** have internal timeouts and never hang (the
  `dispatch.py` post-dispatch hang is not inherited).
- **`invoke`** (V5A) performs a one-shot stateless invocation of an
  invoke-capable alias and prints the generic **InvokeResult envelope**
  `{status, provider, text, citations[], error, metadata}` — citations are
  empty for plain LLM providers and populated by knowledge providers
  (ONYX). Capability is declared as DATA on the runtime profile
  (`capabilities: [invoke]`), so partial providers need no interface stubs.
- **`headless`** (V5B) runs the generic client loop inside a role's tmux
  session: dispatch pastes a prompt exactly as for TUI clients, the runner
  batches it (idle-gap framing), calls invoke(), prints the answer to the
  pane and writes the InvokeResult to `--output-dir`. This is what makes
  API-only runtimes usable as bridge roles with ZERO bridge changes.
- **`mcp-serve`** (V5C) exposes `onyx_answer`/`onyx_status` MCP tools on
  streamable-http (default 127.0.0.1:9164/mcp) so EXISTING roles gain
  knowledge lookup via a config-block only. Requires the optional `mcp`
  extra: `pip install -e ".[mcp]"`.
- **`config`** (V4) is the validated write layer for `models.yaml` /
  `roles.yaml`: `show` prints the full config (aliases, roles, profiles) as
  JSON; `set-alias`/`set-role` create or update an entry from a JSON
  definition (validated before write, atomic temp + rename); the delete
  subcommands remove entries. This is the layer the Father config dashboard
  writes through — the YAML files stay the single source of truth.

---

## Configuration

Model Allocator resolves from (no source is replaced — all are combined):

1. `machine.json` / Machine Profile
2. environment variables (secrets referenced by **name**, never inlined)
3. database values (Father `dpmtf.db`)
4. allocator config files: `models.yaml`, `roles.yaml`, `runtime_profiles.yaml`
5. role/step selections from bridgeV002

### Logical aliases (`models.yaml`)

```yaml
models:
  imple-fast:
    runtime_profile: local_ollama_cuda0
    real_model: qwen36-27b-q4km:latest
    context: 131072
    lifecycle_policy: persistent
    clients: {opencode: true, claude-code: true}
  llama-test:
    runtime_profile: local_llamacpp_cuda0
    model_path: ${MODEL_ROOT_GGUF}/model.gguf      # env-var name, not hardcoded path
    context: 65536
    n_cpu_moe: 26                                   # MoE CPU expert offload (NOT -ngl)
    cache_type_k: turbo4                            # TurboQuant KV cache type
    cache_type_v: turbo3
    flash_attn: "on"
    port: 8085
    lifecycle_policy: stop_after_step
    clients: {opencode: true}
```

### Runtime profiles (`runtime_profiles.yaml`)

```yaml
runtime_profiles:
  local_ollama_cuda0:
    backend: ollama
    api_base_env: OLLAMA_BASE_URL
    default_api_base: http://127.0.0.1:11434
    gpu: cuda0
  local_llamacpp:
    backend: llama_cpp
    server_bin_env: LLAMA_SERVER_BIN               # env-var name
  cloud_minimax:
    backend: openai_compatible
    api_base_env: MINIMAX_API_BASE
    api_key_env: MINIMAX_API_KEY
    provider: minimax
```

### Role mappings (`roles.yaml`)

```yaml
roles:
  imple01:
    default_alias: imple01-local
    config_dir: imple01
    client_aliases:
      opencode: imple01-local
      claude-code: imple01-claude
```

### Lifecycle policies

| Policy | Behavior |
|--------|----------|
| `persistent` | keep model warm/running |
| `stop_after_step` | stop/unload after the step is done |
| `shared_runtime` | leave running (shared by multiple roles/steps) |
| `cloud_noop` | cloud backend; no local start/stop |

---

## bridgeV002 integration

Model Allocator is **optional per role/step** — existing `direct_*` roles are
unchanged. A role opts in via `bridge_roles.default_model_source = 'model_allocator'`
+ `default_model_alias = '<alias>'` (or step-level `model_source`/`model_alias`
override). Resolution priority: **step > role > system default**.

### What lives where

| Layer | Owns |
|-------|------|
| bridgeV002 (Father) | roles, steps, flow order, tmux, deliverable routing, `model_source`/`model_alias` selection |
| Model Allocator | alias resolution, validation, start/stop/status, context/offload, client compatibility |
| Backend adapter | concrete Ollama / llama.cpp / cloud commands |

### Father-side integration points

- `start_coding.py` — when `model_source == 'model_allocator'`: calls
  `model-allocator run` (allocator branch); for **direct** opencode roles:
  `bridge_lib.ensure_opencode_model_field()` writes the `model` field into the
  role's `opencode.json` before start (V2.3 — OpenCode TUI ignores `--model`).
  `command_builder.py` is **unchanged** throughout.
- `dispatch.py` — at ollama-unload points, when the active role is
  allocator-managed: calls `model-allocator stop --alias` (45s timeout); else
  existing `unload_ollama_model()`.
- `routers/bridge.py` — proxy endpoints: `GET /allocator/aliases`,
  `POST /allocator/{validate,status,start,stop}` (shell out, 30/30/200/60s
  timeouts, 502 on failure); config CRUD: `GET /allocator/config`,
  `POST /allocator/config/{alias,role}`,
  `DELETE /allocator/config/{alias,role}/{name}` (V4 — shell out to the
  `config` subcommands, hardened against non-dict error JSON and non-string
  names).
- WebUI (`dpmtf-app.js`) — `model_source` dropdown + alias picker + Validate
  button in Roles/Steps editors (V3A); runtime status card + Start/Stop/
  Refresh + persistent last-validation (localStorage) on allocator-managed
  role cards (V3B); allocator config dashboard — alias/role lists, profile
  overview, alias + role detail forms with create/update/delete and runtime
  status controls (V4).

### The OpenCode `--model` bug (closed)

The OpenCode TUI (1.17.16) silently ignores the `--model <model>` CLI flag. So
neither `command_builder` (direct) nor the allocator's `run` output could set
the model via the flag — OpenCode fell back to config-default/session-resumption.
Fix: set a top-level `model` field in `opencode.json` (the only reliable way).
- V2.2: allocator roles — `render-config --output` regenerates the opencode.json.
- V2.3: direct roles — `ensure_opencode_model_field()` in `bridge_lib.py`.
- Claude Code roles are unaffected — `claude --model` IS a valid flag.

---

## Quick start

```bash
git clone https://github.com/svend-blip/model-allocator.git
cd model-allocator
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                      # pyyaml is the only dependency
python3 -m model_allocator --help
python3 -m model_allocator list --client opencode
python3 -m model_allocator validate --alias imple01-local --client opencode
python3 -m pytest tests/              # 69 tests (unittest + pytest style)
```

For a local TurboQuant llama.cpp setup:

```bash
export LLAMA_SERVER_BIN=/path/to/llama-cpp-turboquant/build/bin/llama-server
export MODEL_ROOT_GGUF=/path/to/models/gguf
python3 -m model_allocator start --alias llama-test    # spawns llama-server
python3 -m model_allocator status --alias llama-test
python3 -m model_allocator stop --alias llama-test
```

---

## Config-file rules

- Committed config may contain: alias names, backend names, model names,
  **environment-variable names**, relative/configured paths, lifecycle policies,
  numeric context/offload values.
- Committed config must **not** contain: long tmux launch commands, inline API
  keys, machine-specific absolute paths, ad-hoc shell pipelines.
- `*.example.*` files may show illustrative machine-specific paths in
  **comments** only. Reusable code never hardcodes `/home/...` paths.

---

## Project structure

```
model-allocator/
  README.md                          (this file)
  docs/governance-templates-v2/11_SCOPE.md   (authoritative scope)
  machine.example.json
  models.example.yaml
  roles.example.yaml
  runtime_profiles.example.yaml
  pyproject.toml                     (pyyaml dep; entry point model-allocator)
  scripts/model-allocator            (PATH wrapper)
  src/model_allocator/
    cli.py                           (12 commands incl. the config subcommand group)
    config_loader.py                 (YAML config loading + merge)
    config_writer.py                 (validated safe write for aliases/roles; atomic temp + rename)
    resolver.py                      (alias → backend/model/flags; generic field merge)
    validator.py                     (§10.1 checks + §10.2 output)
    renderer.py                      (tmux-safe shell string)
    adapters/
      ollama.py                      (local Ollama: status/availability/start/stop)
      llama_cpp.py                   (llama-server: start/stop/status, PID/port/health, full flag set)
      openai_compatible.py           (cloud: validate/reachability)
      opencode.py                    (OpenCode client: run + render-config + provider block)
      claude_code.py                 (Claude Code client: run + env)
  tests/
    test_v1a.py                      (V1A core: resolve/validate/list/status + Ollama)
    test_v2.py                       (V2/V2.1/V2.2: adapters + render-config merge + resolver field-preservation)
    test_config_writer.py            (V4: validated write layer)
    test_config_cli.py               (V4: config subcommands)
```

---

## Version history

| Version | Scope | Commit |
|---------|-------|--------|
| V1A | Minimal proof: CLI, config, resolve/validate/list/status, Ollama adapter | `48ed685` |
| V1B | bridgeV002 pilot: run/start/stop, OpenCode adapter, Father migration 003 + start_coding/dispatch integration | `05a3bf6` + Father `7a18b06` |
| V2 | Claude Code + openai_compatible + llama.cpp adapters; unload/preflight/env/render-config commands | `4dac69e` |
| V2.1 | Resolver bug fix (was dropping backend-specific alias fields: port/n_cpu_moe/cache_type_k/…) | `6298345` |
| V2.2 | Model-specific opencode.json (allocator roles): `render-config --output` + start_coding integration | `8b0451e` + Father `0b687a6` |
| V2.3 | Model-specific opencode.json (direct roles): `ensure_opencode_model_field` in bridge_lib | Father `74a1179` |
| V3A | WebUI core: model_source dropdown + alias picker + Validate button + `/allocator/{aliases,validate}` endpoints | Father `aac539a` |
| V3B | WebUI status + lifecycle: status cards + Start/Stop/Refresh + localStorage + `/allocator/{status,start,stop}` | Father `1e879c7` |
| V4 | Config dashboard: `config_writer` write layer + `config` CLI subcommands + Father config CRUD endpoints + WebUI alias/role dashboard | `dbc2c9a` + `2309d66` + Father `9f52840`…`b166dc5` |

### Live validation

- **imple01** (allocator, Ollama) → uses `qwen3-coder:30b-256k` (V2.2, `/api/ps`).
- **review01/review02** (direct, Ollama) → use their declared models (V2.3, `/api/ps`).
- **archi01** (Claude Code) → `--model` works (no bug for Claude Code).
- **llama.cpp** (TurboQuant) → full start/status/stop lifecycle validated locally
  with real `--n-cpu-moe 26 --cache-type-k turbo4 --cache-type-v turbo3
  --flash-attn on --reasoning off --no-mmap` argv (V2.1 + local TurboQuant build).

---

## Known limitations

- The llama.cpp adapter cannot **adopt** an already-running external
  llama-server (status/stop read only the adapter's own PID file). It manages
  servers it starts itself (validated).
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS` + absolute binary path from the Father Machine
  Profile are not read by the allocator (minor env-equivalence gap; model
  selection works via the opencode.json `model` field regardless).
- Optional FastAPI service (scope §16.4) is deferred — the Father-proxy
  approach is used instead.
