# Model Allocator

> A validated runtime/model-alias layer for the DPMtF ecosystem
> (DPMtF — Deterministic Process Management to Finalisation: a deterministic
> multi-agent process orchestration framework for taking defined work from
> intent to verified finalisation through governed flows, steps, roles,
> harnesses, models, gates, and artifacts).
> Decouples *which model a role uses* from *how that model is started, stopped,
> validated, and resolved* across local Ollama, llama.cpp (TurboQuant), cloud
> OpenAI-compatible APIs, and Minimax.

Status: **V1A → V6, fully built and live-validated.** 279 tests. All
adapters validated against real backends. Wired into the Father WebUI,
including a full config dashboard (alias/role CRUD). V5 adds ONYX as an
OPTIONAL invoke-only knowledge runtime + a generic headless client that
turns any invoke-capable API runtime into a bridgeV002 role with zero
bridge changes. V6 adds a shared foundation runtime: one physical
serving process may be shared by several aliases, with recorded
instance identity driving lifecycle and per-alias inference profiles
riding the existing backend/client transport paths.

---

## Place in the DPMtF Ecosystem

Four components, one machine boundary:

```
   model-allocator                  model-allocator
   (Father's copy)                  (worker's copy)
         │ resolves role→model            │
         ▼                                ▼
   DPMtF-WebUI ("Father") ◄──────── DPMtF-LightWorker
   flows · dispatch · evidence      polls Father over Tailscale,
   gates · SQLite · port 9130       executes one role at a time in
         │                          disposable worktrees
         └── mcp-light (port 9135)
             read-only context: loopback for Father's own
             roles, a second tailnet instance for workers
```

| Component | Depends on | Provides |
|-----------|-----------|----------|
| model-allocator | its own machine's `models.yaml`/`roles.yaml` | role→model resolution, runtime lifecycle, client configs |
| DPMtF-WebUI | model-allocator (same machine), SQLite | flows, dispatch, evidence gates, LightWorker endpoints, watchdog |
| mcp-light | read access to DPMtF-WebUI's files and database | governance/flow/verdict lookup over MCP |
| DPMtF-LightWorker | model-allocator (worker machine), Father reachable over Tailscale | remote role execution |

**Install order — each step's preflight checks the one before it:**

1. **model-allocator** — on every machine that runs models (Father and
   each worker), with that machine's own config files.
2. **DPMtF-WebUI** — on Father: `init_db` → `migrate` → uvicorn on 9130.
3. **mcp-light** — on Father (optional but standard): loopback unit, plus
   the tailnet unit if remote workers should reach it.
4. **DPMtF-LightWorker** — on each worker: venv → `worker.yaml` → auth
   token → base client config → `preflight.sh` 16/16 → daemon.

Each repository's own Installation section covers its steps in detail.

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
| `sglang` | SGLang server (venv) | spawn `python -m sglang.launch_server` + health polling + VRAM settle | kill PID (timeout) | health + PID alive | `--context-length`, venv/model_path from runtime profile |
| `freetoken` | FreeToken runtime (external venv) | spawn `ft serve` with only the configured flags + PID file + `/health` polling (`loading` -> `ok`) + model verification via `/v1/models` | `prepare-stop` drain, then SIGINT/SIGTERM/SIGKILL, confirmed by the port | `/health` state + `/v1/stats` telemetry | runtime reports its own max context; `--max-seq-len-override` caps it deliberately |
| `anthropic` | Anthropic API / Claude subscription | no-op (hosted) | no-op | credentials presence (`ANTHROPIC_API_KEY` or Claude Code login) | model max context |
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
| `pi` | `pi` TUI | built-in provider ids, or custom providers declared in Pi's `models.json` |
| `freebuff` | Freebuff frontend | alias resolved through the freebuff wrapper (cloud_llm flow) |
| `headless` | allocator runner loop | any invoke-capable alias — pasted tmux prompt -> invoke() -> answer in pane + InvokeResult JSON file |

### Client/backend matrix

- **Claude Code** supports Ollama + OpenRouter (Anthropic-compatible endpoints). Minimax via Claude Code is rejected (no Anthropic-compatible endpoint) — `validate` returns a clear ERROR.
- **OpenCode** supports Ollama + OpenRouter + Minimax (built-in) + openai_compatible.

---

## Runtime instances (V6)

A **runtime instance** is one physical serving process (one llama-server,
one SGLang process) that several logical aliases may share. The instance
owns physical identity — port, model path, context, placement, KV cache,
server binary — and the alias owns the logical DPMtF-facing name plus
its `runtime_instance` reference and its `inference_profile` reference.

### Field ownership — exclusive

| Field | Owner | Why |
|-------|-------|-----|
| `model_path`, `model_name`, `server_bin_path`, `port`, `host`, `context`, `n_cpu_moe`, `gpu_layers`, `cache_type_k`, `cache_type_v`, `runtime_profile` | **runtime_instance** | Physical identity. Lives for the lifetime of the process. |
| `runtime_instance`, `inference_profile` | **alias** | Logical references. An alias may declare both, one, or neither. |
| `clients`, `display_name`, `opencode_*`, `claude_*`, `pi_*`, `invoke_timeout`, `lifecycle_policy` | **alias** | Logical, role-facing config. |

**Conflict rule:** an alias bound to a `runtime_instance` MUST NOT also
set any instance-owned field. The schema linter and the resolver both
fail loudly with a message naming the alias, the instance, and the
field. The V6 contract forbids silent precedence between the two.

### Shared lifecycle (recorded instance identity)

A V6 instance-bound alias starts its process once, then reuses the
recorded instance for every subsequent alias that points at the same
instance:

- **First alias start** spawns the server and writes a JSON state file
  `model-allocator-instance-<name>.json` recording `{instance_name,
  pid, port, started_by_alias, started_at}`. The decision to spawn is
  *only* from the absence of a recorded state — never from "port open"
  or "model path matches".
- **Second alias start** finds the recorded state, verifies the PID is
  alive, waits for the health endpoint, and returns `reused: true`
  without spawning. No kill, no port scan, no adoption.
- **Alias-level `stop` / `unload`** on an instance-bound alias is a
  no-op (`stopped: true, skipped: true, reason: "instance-bound runtime
  is shared; stop via stop-instance"`). The state file survives.
- **`stop --all-servers`** skips instance-bound aliases and surfaces
  them in its JSON output with `skipped: true`.
- **Step-completion never auto-stops a `shared_runtime` instance.** The
  contract forbids it.

### Managed-only ownership

The allocator stops only processes it started and recorded. The
ownership decision is keyed on the recorded PID, not on whatever happens
to be listening on the port:

- A foreign (unrecorded) llama-server on the instance port —
  refused, never signalled, never adopted. The `start` returns
  `error: "port N for instance <name> is occupied by an unmanaged
  llama-server (pid=…); refusing to adopt — stop it manually if you
  want the allocator to manage it"`.
- An unknown (non-llama-server) process on the instance port — refused
  with the same shape.
- Stale state (recorded PID dead) is cleaned up before the ownership
  check runs. A foreign listener on the port is still never killed —
  adopting it would be the failure mode the contract forbids.

### Instance-level CLI (V6)

Three commands operate directly on a runtime instance by name,
bypassing alias resolution. They are the only ways to terminate a
shared runtime:

```
model-allocator start-instance   --name <ri> [--timeout S]
model-allocator status-instance  --name <ri>
model-allocator stop-instance    --name <ri> [--timeout S]
```

`status-instance` reports the same `instance_name`, `pid`, `port` as
`status` does for any sharing alias — one canonical record per
instance. The CLI handlers are wired for `llama_cpp` and `sglang`
backends (the two local-server backends); cloud/ollama/anthropic/onyx
aliases reject the instance-level verb with a clear error.

---

## Inference profiles (V6)

An **inference_profile** is a per-alias tuning block that rides on
*existing* backend/client transport paths. Aliases reference one via
`inference_profile: <name>`. The profile is a separate top-level
key in `models.yaml` alongside `models:`, `runtime_instances:`, and
`roles:`.

```yaml
inference_profiles:
  profile-careful:
    reasoning_budget: 4096
    max_output_tokens: 16384
  profile-fast:
    reasoning_budget: 1024
    max_output_tokens: 8192
```

### Implemented fields and their transport

| Field | Transport path | Where it lands |
|-------|----------------|----------------|
| `max_output_tokens` | Resolver merge → opencode `model.<id>.limit.output`, claude_code `CLAUDE_CODE_MAX_OUTPUT_TOKENS` env, pi `models.json` `maxTokens`, `cmd run --max-output-tokens` CLI override | Every client adapter that supports an output budget |
| `reasoning_budget` | Resolver merge → llama-server argv `--reasoning-budget N` | llama.cpp adapter flag rendering |

The transport rides the existing resolver merge — the only place
inference-profile fields land is the resolved view, which the adapters
already read. No new parameter-transport abstraction, no generic
passthrough.

### Precedence rule

`profile` is the default tuning. An alias-level field
(e.g. `max_output_tokens: 4096` on the alias) wins over the profile.
The `cmd run --max-output-tokens <N>` CLI override wins over both
(the override is applied to the resolved view after `resolve_alias`
returns). Concretely: the merge order in `resolve_alias` is
`instance → profile → alias` (`alias` last, so it overrides), and the
CLI override is applied after the merge.

### Deferred fields — loud rejection

`temperature` and `top_p` are part of the four candidate fields but
have no existing transport path in V6. Declaring them in a profile
would silently do nothing, which the Mission Contract forbids. Both
fields are **loudly rejected**:

- The doctor (`model-allocator doctor`) reports an error of the form
  `inference_profile '<name>' field '<field>' is not implemented in
  V6 (deferred) — remove it`.
- `resolve_alias` raises `ResolutionError` with the same message, so a
  programmatic caller that skips the doctor still fails loudly.

Unknown fields are an error too (the schema knows precisely the
implemented + deferred candidate set).

### Deferred / reserved (documentation only)

Two features are reserved for a future allocator version and are
**not** implemented in V6:

- The **`adopted` ownership mode** — a foreign llama-server could be
  adopted by the allocator so the allocator may stop it. V6 only
  manages processes it started. Documented here only; do not set
  `adopted` in any config.
- The **`adapter:` alias key** (LoRA adapter selection). Reserved
  to avoid a future config-rewrite; declaring it raises a schema
  warning today (unknown field), an error in the future. No alias
  declaration should set it.

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
model-allocator start-instance  --name <ri> [--timeout S]      # V6
model-allocator status-instance --name <ri>                    # V6
model-allocator stop-instance   --name <ri> [--timeout S]      # V6
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
| `shared_runtime` (V6) | leave running; reuse via recorded instance identity across all sharing aliases; only `stop-instance --name <ri>` may terminate it |
| `cloud_noop` | cloud backend; no local start/stop |

### Runtime instances (V6 config surface)

Two new top-level sections in `models.yaml` declare the shared
runtime:

```yaml
runtime_instances:
  shared-llm-118b:
    runtime_profile: local_llamacpp_cuda0
    model_path: ${MODEL_ROOT_GGUF}/Laguna-S-2.1-118B-A8B-IQ4_XS.gguf
    server_bin_path: ${LLAMA_SERVER_BIN}
    port: 8090                     # NOT 8080: see note below
    context: 262144
    n_cpu_moe: 31
    gpu_layers: 99
    cache_type_k: q4_0
    cache_type_v: q4_0
    lifecycle_policy: shared_runtime

inference_profiles:
  profile-careful:
    reasoning_budget: 4096
    max_output_tokens: 16384
  profile-fast:
    reasoning_budget: 1024
    max_output_tokens: 8192
```

An alias declares its references (V6 keys are optional on the alias):

```yaml
shared-architect:
  runtime_instance: shared-llm-118b
  inference_profile: profile-careful
  clients: {opencode: true}
```

The dedicated port must be **non-8080**: 8080 is the machine's
deliberate one-at-a-time port (the trade-engine reclaim sweeps it
weekdays 14:00Z), and a shared instance needs its own dedicated port
to coexist. The example above uses 8090; pick whatever port is free
on your machine, but never 8080.

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

## FreeToken runtime (optional)

FreeToken is a GPU-resident inference runtime with an OpenAI-compatible API.
It is optional: nothing changes on a machine without it, and it activates only
when an alias explicitly resolves to `backend: freetoken`.

### Installation stays outside the allocator

The allocator discovers and validates FreeToken; it never installs or upgrades
it. That keeps the runtime independently upgradeable, and it keeps an
upgrade — which can change backend selection and with it throughput — a
deliberate act rather than a side effect.

```bash
git clone https://github.com/FlashML-org/FreeToken.git freetoken
cd freetoken
uv venv --python /usr/bin/python3.12 .venv
uv pip install -e ".[accel]"
```

Because that venv is project-local, `ft` is absent from the PATH of any
service environment. The runtime profile therefore names the binary
explicitly, and the allocator executes it directly rather than sourcing an
activate script:

```bash
export FREETOKEN_BIN=/path/to/freetoken/.venv/bin/ft
```

### `ft serve` only — never `ft launch`

FreeToken ships `ft launch codex` and `ft launch claude`, which start a coding
harness against a runtime it also starts. This allocator does not use them and
must not: choosing, configuring and starting an interface is the harness
allocator's authority. The seam between the two is a normalized endpoint
descriptor:

```json
{
  "provider": "freetoken",
  "base_url": "http://127.0.0.1:8088",
  "api_base": "http://127.0.0.1:8088/v1",
  "model": "Qwen3.8-27B-NVFP4",
  "context_length": 262144,
  "api_compatibility": ["openai", "anthropic"]
}
```

`api_compatibility` is detected from the running server's own route table, not
declared: `anthropic` appears only when `/v1/messages` is actually served.

### Qualified profiles

Two configurations have been measured working on the RTX 5090 workstation
against FreeToken 0.1.2 (source `2757bb5`), and both are shipped as aliases:

| Alias | Model | Shape | Key options |
|-------|-------|-------|-------------|
| `freetoken-qwen38-27b` | `vrfai/Qwen3.8-27B-NVFP4` | dense, NVFP4 | `memory_ratio: 0.90`, `nvfp4_backend: auto` |
| `freetoken-qwen36-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | MoE, offload | `moe_backend: auto`, `moe_cache_auto`, `kv_reserve_tokens: 16384` |

`model_path` holds either a local checkpoint directory or a Hugging Face repo
ID — both qualified profiles use repos, resolved from the local HF cache, so
startup needs no network once the weights are provisioned.

**`nvfp4_backend: auto` is load-bearing.** On the qualified card the default
Triton NVFP4 path measured ~4.3 tokens/sec against ~63 with automatic backend
selection; the same 219-token coding workload went from 51 seconds to 3.5.
Both configurations start and serve correctly, so nothing but
`test_qualified_qwen38_keeps_nvfp4_backend` stands between a configuration
tidy-up and a fifteenfold slowdown.

Options are emitted only when configured. An unset option produces no flag,
which leaves FreeToken's own automatic selection in charge of everything a
profile does not deliberately pin — and keeps MoE flags off dense models,
where they would configure something that is not there.

### Context: what is advertised versus what fits

The two numbers are far apart, and the gap decides whether a client works at
all. `endpoint()` therefore carries both:

- `context_length` — the architecture's maximum, as the runtime reports it.
  Both qualified profiles say 262144.
- `usable_context_tokens` — the allocated KV budget, shared by prompt AND
  generation. On the qualified Qwen3.8 profile at `memory_ratio: 0.90` that is
  **14303 tokens**, about five percent of the advertised figure.

Sizing a prompt by `context_length` fails on contact. Measured (FT-6): a
coding harness whose baseline context is ~35k tokens was refused immediately
with `prompt is too long: 34937 tokens > 14303 maximum`. Raising the budget
with `--num-tokens 49152` (3.00 GiB of KV) let the server start, and then the
prefill exhausted the card — `torch.OutOfMemoryError`, 36 MiB free.

The KV cost per token is not a property of FreeToken but of the model. The
dense 27B profile spends ~3.00 GiB on 49152 tokens; the MoE 35B-A3B profile
spends **0.32 GiB on 16542 tokens** — roughly a third per token, because far
fewer parameters are active. On a 32 GB card that difference decides which
models can host an agent rather than answer a question.

So: the qualified throughput figures were measured on short prompts and say
nothing about whether a model can hold a repository in context. Check
`usable_context_tokens` before pointing a harness at a profile.

### Ownership and arbitration

A qualified profile at `memory_ratio: 0.90` consumes roughly 30 GB of the
card, so a FreeToken instance is an exclusive GPU workload in practice. It is
registered in `LOCAL_SERVER_BACKENDS`, so `stop-all-servers` reaches it like
any other resident runtime.

Before starting, the adapter classifies whatever already holds the port: an
allocator-owned instance serving the same model is reused rather than
restarted (a 27B load is minutes), another FreeToken is refused rather than
taken over, and a service that is not FreeToken is never disturbed. A stop is
not reported as done until the port confirms it, and the PID file survives an
unconfirmed stop so a later attempt can still find its way back.

### Request-level parameters are not configured here

`--reasoning-parser` is a server option and belongs to the runtime profile.
`reasoning_effort` is a per-request choice — measurably worth setting to
`none` for short deterministic coding tasks, where the model otherwise spends
its output budget on reasoning content — and it belongs to whoever builds the
request. That is the harness allocator or the DPMtF role profile, not this
adapter.

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
python3 -m pytest tests/              # 327 tests (unittest + pytest style)
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
    cli.py                           (CLI incl. the config subcommand group + V6 instance commands)
    config_loader.py                 (YAML config loading + merge)
    config_writer.py                 (validated safe write for aliases/roles; atomic temp + rename)
    invoke_result.py                 (V5: generic InvokeResult envelope)
    headless.py                      (V5B: runner loop — stdin framing, invoke, output files)
    mcp_server.py                    (V5C: onyx-mcp tools; optional 'mcp' extra)
    resolver.py                      (alias → backend/model/flags; V6 reference resolution + field-ownership merge)
    schema.py                        (doctor + config validation; V6: runtime_instances / inference_profiles)
    validator.py                     (§10.1 checks + §10.2 output)
    renderer.py                      (tmux-safe shell string)
    adapters/
      ollama.py                      (local Ollama: status/availability/start/stop)
      llama_cpp.py                   (llama-server: start/stop/status, PID/port/health, full flag set; V6 instance-keyed state)
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
    test_v6_shared_runtime.py        (V6: schema + resolver + parity — Father fixture)
    test_v6_instance_lifecycle.py    (V6: start-once / reuse-by-identity / managed-only ownership)
    test_v6_inference_profile_transport.py  (V6: profile fields ride existing transport paths)
    test_v6_worker_parity.py         (V6: same machinery on a worker-style config)
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
| V6 | Shared foundation runtime: `runtime_instances` (one physical process per instance), `inference_profiles` (per-alias tuning — `max_output_tokens` / `reasoning_budget` implemented, `temperature` / `top_p` deferred with loud rejection), `shared_runtime` lifecycle (start-once / reuse-by-identity, never auto-stopped), managed-only ownership (no foreign-port kill, no adoption), `start-instance` / `status-instance` / `stop-instance` CLI; 279 tests | run 016 |

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
- **V6 — `adopted` ownership mode** is reserved for a future allocator
  version. The allocator will not adopt a foreign llama-server on the
  instance port; it only manages processes it started and recorded. Stop
  the foreign server manually if you want the allocator to take over.
- **V6 — `adapter:` alias key** (LoRA adapter selection) is reserved for
  a future allocator version. The schema accepts it as a warning today
  (unknown field). Do not declare it on any alias.

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
