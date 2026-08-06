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

### Context handling — ollama vs llama.cpp

For `local_ollama_cuda0` aliases, the `context` field in `models.yaml` is a **warm-up
hint**: it sets `options.num_ctx` on the allocator's own warm-up request to
`/api/generate`, but it does **not** bind the model to that size. If a client
request (e.g. Claude Code, OpenCode) omits its own `num_ctx` parameter, Ollama
reloads the model at the value **baked** into the model file — which may differ
from the warm-up value.

```
models.yaml: context: 65536   → allocator warms the model with num_ctx=65536
Claude Code connects without num_ctx  → Ollama reloads at baked default (e.g. 131072)
```

To **bind** a specific context size with ollama, point `real_model` at a variant
that was baked at that exact size (e.g. `qwen3.6-27b-32k`, `-48k`, `-64k`).

For `llama_cpp`, there is no such problem: `--ctx-size` is passed when the server
starts and no client request can override it. The server abides by the flag
permanently until restarted.

| Adapter | context field | Binding? | Override risk |
|---------|--------------|----------|---------------|
| `ollama` | `num_ctx` per-request (warm-up hint) | No — model file has baked value | Client requests without explicit `num_ctx` revert to baked default |
| `llama_cpp` | `--ctx-size` at server start | Yes — enforced for lifetime of process | None |

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

## ONYX runtime (optional, V5)

> **Integration Pattern for LLMs**

The model-allocator integration pattern is designed to be clear enough for an LLM to implement without mistakes:

### Three-layer architecture
1. **bridgeV002** chooses which role uses which alias (database-driven)
2. **Model Allocator** resolves alias → backend/model/runtime (models.yaml)
3. **Backend adapters** own concrete start/stop/validate commands

### How to wire model-allocator into any project
1. **Step 1:** Define aliases in models.yaml (alias → real_model + runtime_profile + context) 
2. **Step 2:** Reference aliases in bridge_roles.default_model_alias (database)
3. **Step 3:** dispatch.py resolves via get_effective_model_source() → model_allocator CLI  
4. **Step 4:** start_coding.py generates opencode.json with resolved model/provider
5. **Step 5:** dispatch.py uses LeaseRegistry.acquire() for reference-counted lifecycle

### Common LLM mistakes to avoid
- DO NOT hardcode model names in dispatch code — always use aliases
- DO NOT call ollama stop/start directly — use model-allocator CLI or LeaseRegistry
- DO NOT assume model context size — resolve via allocator validate
- DO NOT skip the LeaseRegistry — unconditional stop kills other jobs sharing the alias
- DO NOT hardcode /home/svend paths — use config.py getters

### The dispatch integration points (exact functions in dispatch.py)
- `signal_send`: LeaseRegistry.acquire(job_id, alias) — acquires lease + starts model 
- `signal_complete`: LeaseRegistry.release(handoff_id, alias) — releases lease, stops if last
- `signal_complete`: _advance_chain(job) — fallback chain advancement  
- `start_coding.py`: get_effective_model_source() → model-allocator CLI → opencode.json

ONYX (github.com/onyx-dot-app/onyx) is an OPTIONAL knowledge runtime.
Nothing requires it: only aliases whose profile declares `backend: onyx`
touch it, and with the stack down those aliases fail cleanly while every
other alias is untouched (guaranteed by tests).

- Deployment: ONYX **Lite** via docker compose — see `deploy/onyx/README.md`
  (API http://127.0.0.1:9162, web UI http://127.0.0.1:9163; LLM backend is
  the local Ollama via its OpenAI-compatible /v1 endpoint — zero cloud cost)
- Auth: basic login via env names `ONYX_EMAIL`/`ONYX_ADMIN_PASSWORD`
  (session cookie per invoke); `ONYX_API_KEY` supported where API keys are
  available (Business tier)
- Two integration paths, both zero-bridge-change:
  1. **Runtime path**: a role's alias -> `run --client headless` command in
     the role's tmux session (see `advisor01` example role)
  2. **Tools path**: `mcp-serve` + an `onyx-mcp` block in the role's MCP
     config — existing roles keep their LLM and gain `onyx_answer`

### Dependencies

| Component | Requires |
|-----------|----------|
| allocator core | Python 3.10+, `pyyaml` (nothing else) |
| ONYX runtime (optional) | docker + compose plugin (user-local install OK), ~1 GB RAM (Lite), local Ollama |
| `mcp-serve` (optional) | `pip install -e ".[mcp]"` (mcp[cli] >= 1.28.1) |
| dev/tests | `pytest` |

> **Full installation guide:** For comprehensive installation instructions covering all local runtimes
> (llama.cpp, SGLang, Ollama, OpenCode, Claude Code) and environment configuration, see
> [`/home/svend/DPMtF-WebUI/SETUP.md`](file:///home/svend/DPMtF-WebUI/SETUP.md).

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
    invoke_result.py                 (V5: generic InvokeResult envelope)
    headless.py                      (V5B: runner loop — stdin framing, invoke, output files)
    mcp_server.py                    (V5C: onyx-mcp tools; optional 'mcp' extra)
    resolver.py                      (alias → backend/model/flags; generic field merge)
    validator.py                     (§10.1 checks + §10.2 output)
    renderer.py                      (tmux-safe shell string)
    adapters/
      ollama.py                      (local Ollama: status/availability/start/stop)
      llama_cpp.py                   (llama-server: start/stop/status, PID/port/health, full flag set)
      openai_compatible.py           (cloud: validate/reachability)
      opencode.py                    (OpenCode client: run + render-config + provider block)
      claude_code.py                 (Claude Code client: run + env)
      onyx.py                        (V5A: ONYX backend — auth, one-shot invoke, citations)
      headless.py                    (V5B: headless client command builder)
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
| V5-fase0 | ONYX Lite local deployment (docker compose, basic auth, Ollama /v1 LLM backend, clean persona) | `18dfe20` |
| V5A | `invoke()` capability + InvokeResult envelope + ONYX backend adapter + capability-as-data | `17bfc58` |
| V5B | Headless runner client adapter (API runtimes as bridge roles, zero bridge changes) | `74ba425` |
| V5C | onyx-mcp: `mcp-serve` with onyx_answer/onyx_status tools (optional `mcp` extra) | `8c971a5` |
| V5.1 | Claude Code env equivalence: max_output_tokens, adaptive thinking, active model, binary+extra-args, `--max-output-tokens` passthrough | `e08d825` |
| V5.2 | SGLang adapter + Laguna (llama.cpp) config + `llama_SG` flow support: auto-start in `run`, `--no-auto-start`, `server_bin_path`, `--jinja`/`--load-mode`/`--reasoning-budget` flags, process-group kill for SGLang, `ANTHROPIC_API_KEY` blanking in Ollama/llama_cpp backends | `6ed84b3` |

### Live validation

- **imple01** (allocator, Ollama) → uses `qwen3-coder:30b-256k` (V2.2, `/api/ps`).
- **review01/review02** (direct, Ollama) → use their declared models (V2.3, `/api/ps`).
- **archi01** (Claude Code) → `--model` works (no bug for Claude Code).
- **llama.cpp** (TurboQuant) → full start/status/stop lifecycle validated locally
  with real `--n-cpu-moe 26 --cache-type-k turbo4 --cache-type-v turbo3
  --flash-attn on --reasoning off --no-mmap` argv (V2.1 + local TurboQuant build).
- **llama.cpp** (Laguna) → `laguna-local` alias with `--jinja --load-mode none
  --reasoning-budget 2048` flags, `server_bin_path` for custom build, Claude Code
  client (V5.2).
- **SGLang** (Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit) → `qwen-shared-sglang` alias,
  persistent lifecycle, `qwen3_coder` tool-call parser, OpenCode client via
  `@ai-sdk/openai-compatible` provider. Live-validated: health OK, basic chat OK,
  tool-call OK (V5.2).

---

## Known limitations

- The llama.cpp adapter cannot **adopt** an already-running external
  llama-server (status/stop read only the adapter's own PID file). It manages
  servers it starts itself (validated).
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS` + absolute binary path from the Father Machine
  Profile are now read by the allocator (fixed in V5.1 via `max_output_tokens`,
  `claude_binary`, `claude_extra_args` config fields).
- Optional FastAPI service (scope §16.4) is deferred — the Father-proxy
  approach is used instead.
- ONYX Lite has no connectors/RAG indexing — `onyx_answer` works, a
  dedicated `onyx_search` tool awaits the Standard-mode upgrade
  (deploy/onyx/README.md documents the path). ONYX API keys are
  Business-tier gated; the adapter therefore defaults to cookie login.
- The headless runner is a transport, not an agent: roles that must write
  bridge convention files need that handled via bridge step configuration,
  not prompt parsing in the runner.
- SGLang adapter requires the model to be downloaded and the venv to be
  set up before first use (`/home/svend/venvs/sglang`). The adapter does
  not auto-install SGLang or download models.
- Only one large model (Laguna 23.5 GB or SGLang/Qwen 17 GB) fits in the
  RTX 5090's 32 GB VRAM at a time. The `llama_SG` flow handles this via
  pre/post-dispatch scripts that stop one server before starting the other.

## OpenCode external-directory permissions

OpenCode role sessions may stall on "Allow edit" dialogs when accessing
directories outside the session's cwd. Two directories need allowlisting:

| Directory | Configured by | Purpose |
|-----------|--------------|---------|
| `{DPMTF_BRIDGE_DIR}` | `.env` / `dpmtf.ini` (default: `~/flows`) | Handoffs, results, verdicts, run artifacts |
| `{project_root}/docs/` | `config.get_project_root()` (Father repo) | Governance templates read by all roles |

The fix is a `permission.external_directory` block in each role's
`opencode.json`. Replace `{BRIDGE_DIR}` and `{FATHER}/docs` with the
actual paths from your configuration:

```json
"permission": {
  "external_directory": {
    "{BRIDGE_DIR}/*": "allow",
    "{BRIDGE_DIR}/**": "allow",
    "{FATHER}/docs/*": "allow",
    "{FATHER}/docs/**": "allow"
  }
}
```

This block survives `render-config` because the merge preserves keys not
in the rendered output (`model` and `provider` only). The global
`~/.config/opencode/opencode.json` should also carry this block as a
fallback — OpenCode merges the global config under `OPENCODE_CONFIG`.

All OpenCode roles under `~/.config/opencode-roles/` should have this
block. To add it, resolve the paths from your config first:

```bash
# Resolve paths from config
BRIDGE_DIR="${DPMTF_BRIDGE_DIR:-$HOME/flows}"
FATHER_DOCS="$(python3 -c "import sys; sys.path.insert(0,'.'); import config; print(config.get_project_root())")/docs"

# Add permission block to all role configs
for dir in ~/.config/opencode-roles/*/; do
  file="${dir}opencode.json"
  [ -f "$file" ] && python3 -c "
import json, os
cfg = json.load(open('$file'))
if 'permission' not in cfg:
    bd = os.environ.get('BRIDGE_DIR', os.path.expanduser('~/flows'))
    fd = os.environ.get('FATHER_DOCS', '')
    cfg['permission'] = {'external_directory': {
        bd + '/*': 'allow', bd + '/**': 'allow',
        fd + '/*': 'allow', fd + '/**': 'allow'
    }}
    json.dump(cfg, open('$file','w'), indent=2)
    print('  added')
" && echo "  $(basename $dir): added" || echo "  $(basename $dir): already present"
done
```
