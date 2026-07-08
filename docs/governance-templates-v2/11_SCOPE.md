# Model Allocator — Scope & Governance

> **en-US is the standard language for this document.**
> Status: **Scope definition (V1) — no implementation yet.**
> Owner: DPMtF Architect (Father project). Cross-project component.

---

## 0. Document Purpose

This document is the **authoritative scope boundary** for the Model Allocator
component. It defines what Model Allocator owns, what it explicitly does NOT
own, and the verified impact it has on **bridgeV002** — the DPMtF role/step
orchestration layer in the Father project.

It is a **scope/governance document**, not an implementation plan. No code is
written under this scope. Implementation handoffs will be produced separately
and must stay within the boundaries defined here.

The source design spec was provided by the Human on 2026-07-08. This document
codifies it as governance and adds the **verified bridgeV002 impact
assessment in Section 5** based on inspection of the live Father codebase.

**Core scope rule:**

> V1 must prove alias resolution and validation first. bridgeV002 integration
> is opt-in, role/step scoped, and must preserve the existing `direct_*` path
> until explicitly migrated.

---

## 1. Purpose

Create a central **Model Allocator** as a small standalone runtime layer that
simplifies how bridgeV002 starts, stops, validates, and resolves models for:

- OpenCode
- Claude Code
- llama.cpp
- Ollama local runtimes
- cloud Ollama-compatible APIs
- cloud Minimax / OpenAI-compatible APIs
- other OpenAI-compatible providers

**Goal:** avoid storing long backend-specific start commands directly inside
bridgeV002 roles or steps. Instead, bridgeV002 uses a stable command surface:

```
model-allocator resolve  --role imple01 --client opencode
model-allocator validate --alias imple-fast --client opencode
model-allocator run      --role imple01 --client opencode
model-allocator stop     --alias imple-fast
```

The actual backend may be local Ollama, llama.cpp, cloud API, Minimax, or
another provider. **bridgeV002 must not need to know backend-specific
start/stop/context/offload details.**

---

## 2. Core Principle — Three-Layer Separation

```
bridgeV002          → role/step orchestration layer
Model Allocator     → validated runtime/model alias layer
Backend adapter     → concrete model runtime commands
```

| Layer | Owns | Does NOT own |
|-------|------|--------------|
| **bridgeV002** | roles, steps, flow execution, tmux orchestration, whether a role/step uses direct model config or Model Allocator | model alias resolution, runtime validation, backend lifecycle commands, context/offload policy |
| **Model Allocator** | model alias resolution, runtime validation, start/stop/unload/status, context policy, offload policy, client compatibility, backend command generation | roles, steps, flow order, tmux target selection, deliverable routing |
| **Backend adapter** | Ollama commands, llama.cpp server commands, cloud API validation, OpenCode config rendering, Claude Code env/wrapper setup | alias naming, lifecycle policy choice, which role uses which alias |

**Hard rule:**

> BridgeV002 chooses role / step / model alias.
> Model Allocator owns runtime lifecycle.
> Backend adapter owns concrete commands.

---

## 3. Separate Repository — Confirmed

Model Allocator lives in its **own repository** at:

```
/home/svend/model-allocator/
```

**Why separate:** it is consumed by multiple surfaces — bridgeV002, OpenCode
wrappers, Claude Code wrappers, DPMtF WebUI, AI PC Resource WebUI, standalone
terminal workflows, and future multi-user setups. Embedding it inside
bridgeV002 would make it bridge-specific.

**Proposed structure — target, not yet created:**

```
model-allocator/
  README.md
  docs/governance-templates-v2/11_SCOPE.md
  machine.example.json
  models.example.yaml
  roles.example.yaml
  src/model_allocator/
    cli.py
    resolver.py
    validator.py
    runtime.py
    lifecycle.py
    adapters/
      ollama.py
      llama_cpp.py
      openai_compatible.py
      minimax.py
      opencode.py
      claude_code.py
  scripts/
    model-allocator
  tests/
```

**Form factor:** start as a simple Python CLI. Add FastAPI later **only if**
another WebUI needs live status/control.

---

## 4. Configuration Sources

Model Allocator **does not replace** existing configuration sources. It
**resolves** from them:

- `machine.json` / Machine Profile
- environment variables
- database values from Father `dpmtf.db`
- allocator config files such as `models.yaml` and `roles.yaml`
- role/step selections from bridgeV002

**Resolution flow:**

```
machine.json + env + database + allocator config
        ↓
Model Allocator resolve / validate
        ↓
OK / Warning / Error
        ↓
bridgeV002 selects allocator alias
        ↓
OpenCode / Claude Code starts against a known-good backend
```

### 4.1 Configuration Precedence

Unless an implementation handoff explicitly defines a narrower rule, use this
precedence:

1. bridgeV002 step override
2. bridgeV002 role default
3. allocator role config
4. allocator model config
5. machine profile
6. environment variables for secrets/runtime endpoints
7. allocator defaults

Secrets must **not** be stored in bridgeV002 or allocator model files. API keys
are referenced by environment variable name only.

### 4.2 Structured Config Rule

Model/runtime configuration must be **structured data**, not opaque shell
strings.

**Committed config may contain:**

- alias names
- backend names
- model names
- environment variable names
- relative/configured paths
- lifecycle policies
- numeric context/offload values

**Committed config must not contain:**

- long tmux launch commands
- inline API keys
- machine-specific absolute paths
- ad-hoc shell pipelines
- backend stop/unload shell snippets

---

## 5. bridgeV002 Impact Assessment — VERIFIED

This section is the result of inspecting the live Father codebase
`/home/svend/DPMtF-WebUI` on 2026-07-08. It documents **what bridgeV002
currently owns** regarding models/runtime, and **what changes** when Model
Allocator is introduced.

### 5.1 Current State — What bridgeV002 Owns Today

| Artifact | Location | Role |
|----------|----------|------|
| Role model config | `bridge_roles` columns: `model_type`, `cloud_model`, `ollama_model`, `default_runtime`, `default_provider`, `default_model`, `config_dir` | Per-role model/runtime/provider selection |
| Step model override | `bridge_flow_steps` columns: `runtime_override`, `provider_override`, `model_override` | Step-level override of role defaults |
| Machine Profile | `profiles/machine.local.json` symlinked + `machine.*.example.json` | Provider configs, model lists, paths, runtime templates |
| Command builder | `scripts/bridgeV002/command_builder.py` — `build_start_command(runtime, provider, model, role_key, machine_profile, config_dir)` | Renders the concrete client launch command |
| Start runner | `scripts/bridgeV002/start_coding.py` | Sends built command to each role's tmux session; applies override chain step > role |
| Post-dispatch unload | `scripts/bridgeV002/dispatch.py` — `unload_ollama_model()` called at multiple sites | No-kill `ollama stop` after dispatch |

### 5.2 Overlap with Model Allocator

**Model Allocator does not delete or replace `command_builder.py` in V1.**

For allocator-enabled roles/steps, bridgeV002 bypasses backend-specific command
construction and delegates runtime/client resolution to Model Allocator.

The existing `command_builder.py` remains the `direct_*` compatibility path
until a later explicit migration removes or shrinks it.

| bridgeV002 artifact today | Model Allocator role | Notes |
|----------------------------|----------------------|-------|
| `command_builder.build_start_command()` runtime/model resolution | `model-allocator resolve / validate / run` | Used only when `model_source == model_allocator` |
| `dispatch.py::unload_ollama_model()` | `model-allocator stop / unload --alias` | Replaces direct `ollama stop` only for allocator-enabled roles/steps |
| Machine Profile `machine.json` | Stays as config source | Not deleted; allocator may read it |

### 5.3 Required bridgeV002 Schema Additions

New columns are **nullable and opt-in**. Existing columns remain for the
`direct_*` path.

**`bridge_roles`:**

```
default_model_source TEXT DEFAULT NULL
default_model_alias  TEXT DEFAULT NULL
```

Allowed values for `bridge_roles.default_model_source`:

- `direct_ollama`
- `direct_cloud`
- `direct_llama_cpp`
- `model_allocator`
- `NULL` = use system default

**Important:** `inherit_from_role` is **not valid at role level**.

**`bridge_flow_steps`:**

```
model_source TEXT DEFAULT NULL
model_alias  TEXT DEFAULT NULL
```

Allowed values for `bridge_flow_steps.model_source`:

- `inherit_from_role`
- `direct_ollama`
- `direct_cloud`
- `direct_llama_cpp`
- `model_allocator`
- `NULL` = inherit from role

**Migration mechanism:** schema additions must use a versioned SQL migration:

```
scripts/db/00X_*.sql
python3 scripts/migrate.py
```

**Do not edit `init_db.py` for this migration.**

The existing `runtime_override`, `provider_override`, and `model_override`
columns stay untouched for the `direct_*` path.

### 5.4 Required bridgeV002 Code Changes

| File | Change | Gated by |
|------|--------|----------|
| `start_coding.py` | When `model_source == model_allocator`, call `model-allocator run --role <role> --client <client>` instead of `build_start_command`. When `direct_*` or NULL, existing path is unchanged. | `model_source` |
| `dispatch.py` | When active role/step uses allocator, call `model-allocator stop --alias <alias>` or `model-allocator unload --alias <alias>` according to lifecycle policy. `direct_*` path unchanged. | `model_source` |
| `bridge_lib.py` | Add helper returning effective `(model_source, model_alias)` after applying step > role > system-default priority. | required |
| Frontend | Later V3 scope: role/step editor dropdown + alias picker + validation status. | V3 only |

### 5.5 What Stays in bridgeV002

bridgeV002 continues to own:

- `bridge_roles` and `bridge_flow_steps`
- role/step/flow order
- tmux session orchestration
- deliverable routing
- convention injection
- the selection of `model_source` and `model_alias` per role/step
- the existing `direct_*` model path

### 5.6 Impact Summary

> Model Allocator is **additive and opt-in**. bridgeV002 gains nullable
> model-source fields and conditional call-sites for allocator-enabled
> roles/steps. Nothing in the existing `direct_*` path changes. The allocator
> provides validated alias/runtime resolution and backend lifecycle handling
> **only when explicitly selected**.

---

## 6. Model Source Selection

Model Allocator is **optional**. It can be selected at role level or step
level.

### 6.1 Role-Level Selection

Role-level `default_model_source` values:

- `direct_ollama`
- `direct_cloud`
- `direct_llama_cpp`
- `model_allocator`
- `NULL` = system default

Role-level `default_model_alias` is used only when:

```
default_model_source == model_allocator
```

### 6.2 Step-Level Selection

Step-level `model_source` values:

- `inherit_from_role`
- `direct_ollama`
- `direct_cloud`
- `direct_llama_cpp`
- `model_allocator`
- `NULL` = inherit from role

Step-level `model_alias` is used only when:

```
model_source == model_allocator
```

### 6.3 Resolution Priority

1. Step override
2. Role default
3. System default

**System default must never force Model Allocator globally.**

### 6.4 Examples

```yaml
# Role using the allocator
role_key: imple01
client: opencode
default_model_source: model_allocator
default_model_alias: imple-fast

# Step inheriting from role
step_key: sim01
role_key: imple01
model_source: inherit_from_role

# Step overriding role
step_key: review01
role_key: review01
model_source: model_allocator
model_alias: review-deep
```

**Migration rule:** Model Allocator must not be forced globally. It must be
possible to test it on one role or one step first.

---

## 7. Logical Model Aliases

A logical alias is a user-friendly name resolving to a real
backend/model/runtime.

```yaml
models:
  imple-fast:
    runtime_profile: local_ollama_cuda0
    real_model: qwen36-27b-q4km:latest
    context: 131072
    lifecycle_policy: persistent
    clients:
      opencode: true
      claude-code: true
  review-cloud:
    runtime_profile: cloud_minimax
    real_model: minimax-m3
    lifecycle_policy: cloud_noop
    clients:
      opencode: true
      claude-code: false
  llama-test:
    runtime_profile: local_llamacpp_cuda0
    model_path: /home/svend/ai-data/models/gguf/model.gguf
    context: 65536
    gpu_layers: 35
    lifecycle_policy: stop_after_step
    clients:
      opencode: true
      claude-code: true
```

**Visibility rule:** the logical alias must be visible in bridgeV002, and the
resolved real backend/model must also be visible for debugging.

---

## 8. Runtime Profiles

Runtime profiles describe the actual backend/provider/machine target. They keep
machine/backend/provider details **outside** bridgeV002.

```yaml
runtime_profiles:
  local_ollama_cuda0:
    backend: ollama
    api_base_env: OLLAMA_BASE_URL
    default_api_base: http://127.0.0.1:11434
    gpu: cuda0
  local_llamacpp_cuda0:
    backend: llama_cpp
    server_bin: /home/svend/bin/llama-server
    model_root: /home/svend/ai-data/models/gguf
    default_gpu_layers: 99
    default_ctx: 131072
    gpu: cuda0
  cloud_minimax:
    backend: openai_compatible
    api_base_env: MINIMAX_API_BASE
    api_key_env: MINIMAX_API_KEY
    provider: minimax
```

Committed **examples** may use `/home/svend/...` for local documentation, but
committed **reusable code** must not hardcode machine-specific absolute paths.

---

## 9. CLI Commands

### 9.1 Command Semantics

- **resolve** returns the effective alias/backend/model/config decision.
- **validate** checks whether that decision is usable.
- **list** shows configured aliases and optionally filters by validation/client.
- **status** reports backend/runtime status for an alias.
- **start** starts or warms the backend runtime only.
- **stop** stops the backend runtime according to lifecycle policy.
- **unload** frees model memory where supported, without necessarily killing the
  runtime.
- **run** prepares the selected client command/env/config for execution.

**`run` must not silently perform unrelated lifecycle actions** unless the
selected lifecycle policy explicitly requires them.

bridgeV002 still owns tmux orchestration. Therefore `model-allocator run` may
return or execute a client launch command, but it must **not** own bridge step
ordering, deliverable routing, tmux target selection, or flow control.

### 9.2 V1 Commands

```
model-allocator resolve  --role <role_key> --client <client>
model-allocator validate --alias <model_alias> --client <client>
model-allocator list     --only-ok --client <client>
model-allocator status   --alias <model_alias>
model-allocator start    --alias <model_alias>
model-allocator stop     --alias <model_alias>
model-allocator run      --role <role_key> --client <client>
```

### 9.3 Later Commands

```
model-allocator env           --role <role_key> --client claude-code
model-allocator render-config --role <role_key> --client opencode
model-allocator unload        --alias <model_alias>
model-allocator preflight     --role <role_key> --client <client>
```

---

## 10. Validation Behavior

bridgeV002 may only use allocator aliases with:

```
validation_status == OK
```

### 10.1 Validation Checks

Validation must check:

- alias exists
- runtime profile exists
- backend adapter exists
- required environment variables are present
- real model is available or reachable
- API base is reachable where applicable
- client compatibility is declared
- start support exists
- stop/unload support is known
- context value is valid
- GPU policy is valid
- port/PID policy is valid for llama.cpp

### 10.2 Example Output

```
OK
Logical model: imple-fast
Backend: ollama
Runtime: local
Real model: qwen36-27b-q4km:latest
API base: http://127.0.0.1:11434
Client support:
  opencode: OK
  claude-code: OK
Lifecycle:
  start: OK
  stop: OK
  unload: OK
Context: 131072
GPU policy: cuda0
Warnings: none
Errors: none
```

### 10.3 Stored / Displayed Validation Data

bridgeV002 or Model Allocator may store/display:

- `last_validated_at`
- `validation_status`
- `logical_model_alias`
- `resolved_backend`
- `resolved_real_model`
- `resolved_api_base`
- `client_support`
- `warnings`
- `errors`

Validation-cache storage location is **intentionally not decided** by this
scope document. It may live in allocator state or Father DB, depending on the
implementation handoff.

---

## 11. Lifecycle Responsibilities

Start, stop, unload, context, and offload live in **Model Allocator**, not
bridgeV002.

| Operation | Definition |
|-----------|------------|
| **start** | Start or warm up the model/runtime |
| **stop** | Stop the model/runtime completely where supported |
| **unload** | Free model memory/VRAM where supported |
| **context** | Context size / KV cache / keep_alive / ctx flags |
| **offload** | Runtime placement of model/context between GPU/RAM/CPU where supported |

### 11.1 Backend Examples

| Backend | Start | Stop / Unload | Status | Context | Offload |
|---------|-------|---------------|--------|---------|---------|
| Ollama | warm up via API / `ollama run` + `keep_alive` | `ollama stop <real_model>` | `ollama ps` / API | `num_ctx` / model param | limited, model/runtime/env dependent |
| llama.cpp | `llama-server --model ... --ctx-size ... --n-gpu-layers ...` | stop PID / managed process | health endpoint / port / PID | `--ctx-size` | `--n-gpu-layers`, `--tensor-split`, KV cache type, mmap/mlock, threads |
| Cloud / OpenAI-compatible / Minimax | validate API base + key | no-op | API reachability | model max context / request settings | N/A |

---

## 12. Lifecycle Policies

Keep V1 simple.

| Policy | Behavior |
|--------|----------|
| `persistent` | Keep model warm/running |
| `stop_after_step` | Stop/unload after the step is done |
| `shared_runtime` | Leave runtime running because multiple roles/steps share it |
| `cloud_noop` | Cloud backend; no local start/stop |

```yaml
models:
  junior-fast:
    runtime_profile: local_ollama_cuda0
    real_model: qwen3:14b
    lifecycle_policy: stop_after_step
    context: 32768
  imple-main:
    runtime_profile: local_ollama_cuda0
    real_model: qwen36-27b-q4km:latest
    lifecycle_policy: persistent
    context: 131072
  llama-test:
    runtime_profile: local_llamacpp_cuda0
    model_path: /models/test.gguf
    lifecycle_policy: stop_after_step
    context: 65536
    gpu_layers: 35
```

---

## 13. Client Adapters

Model Allocator understands **backend adapters** and **client adapters**
separately.

**Backend adapters:** `ollama`, `llama_cpp`, `openai_compatible`, `minimax`
**Client adapters:** `opencode`, `claude-code`

### 13.1 OpenCode Adapter

The OpenCode adapter may:

- resolve model/API base
- generate or render OpenCode config if needed
- expose a run command
- support local Ollama, llama.cpp, and cloud-compatible APIs

```
model-allocator render-config --role imple01 --client opencode
model-allocator run           --role imple01 --client opencode
```

### 13.2 Claude Code Adapter

The Claude Code adapter may:

- resolve model/API base
- export required environment variables
- set compatible model name
- apply Claude Code-specific wrapper behavior
- optionally disable adaptive thinking if configured

Example environment:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:11434
export ANTHROPIC_AUTH_TOKEN=dummy
export CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1
export MODEL_ALLOCATOR_ACTIVE_MODEL=qwen36-27b-q4km:latest
```

Commands:

```
model-allocator env --role imple01 --client claude-code
model-allocator run --role imple01 --client claude-code
```

---

## 14. bridgeV002 Integration Summary

See Section 5 for verified impact.

**Integration surface recap:**

Role fields:

```
role_key
default_runtime / client
default_model_source
default_model_alias
```

Step fields:

```
step_key
role_key
model_source
model_alias
```

**bridgeV002 behavior:**

```
if model_source == "model_allocator":
    call model-allocator validate/resolve
    require validation_status == "OK"
    call model-allocator run/start/status/stop as needed
elif model_source in ("direct_ollama", "direct_cloud", "direct_llama_cpp"):
    use existing bridgeV002 behavior
else:
    use existing inheritance/system-default behavior
```

**bridgeV002 display example:**

```
Role: imple01
Step: sim01
Client: opencode
Model source: Model Allocator
Logical alias: imple-fast
Validation: OK
Resolved backend: ollama
Resolved real model: qwen36-27b-q4km:latest
Context: 131072
Lifecycle policy: persistent
```

---

## 15. What Must NOT Live in bridgeV002

bridgeV002 must **not** store backend-specific runtime commands:

- `ollama stop` command
- llama.cpp server command
- llama.cpp `gpu_layers` / `tensor_split`
- Claude Code environment variable details
- OpenCode generated config details
- backend-specific offload logic
- PID files
- port allocation logic

Those belong in **Model Allocator** or **backend/client adapters**.

bridgeV002 stores/selects **only**:

- role
- step
- client
- `model_source`
- `model_alias`
- inheritance/override behavior

---

## 16. Versioned Scope

### 16.1 V1A — Minimal Allocator Proof

Implement only:

- standalone Python CLI
- allocator-local YAML/JSON config loading
- logical aliases
- runtime profiles
- `resolve`
- `validate`
- `list`
- `status`
- local Ollama backend adapter
- **no bridgeV002 schema change yet**
- **no Claude Code/OpenCode launch integration yet**

**Goal:** prove that an alias can resolve and validate correctly.

### 16.2 V1B — First bridgeV002 Pilot

Add:

- `run`
- `start`
- `stop`
- one client adapter first, preferably OpenCode
- one pilot role/step using `model_allocator`
- versioned SQL migration for nullable model-source fields
- existing `direct_*` path unchanged

**Goal:** prove opt-in bridgeV002 use on one non-critical role/step.

### 16.3 V2

Add:

- Claude Code client adapter hardening
- OpenAI-compatible cloud adapter
- Minimax profile support
- llama.cpp server adapter
- llama.cpp PID files
- llama.cpp port allocation
- llama.cpp health checks
- context/offload flags
- generated OpenCode config files
- `unload` command
- `preflight` command

### 16.4 V3

Add:

- bridgeV002 UI integration
- role/step dropdown for allocator aliases
- validation status cards
- runtime status cards
- CUDA0/CUDA1 policy display
- persistent last validation result
- optional FastAPI service
- integration with AI PC Resource WebUI

### 16.5 Avoid in V1

V1 must avoid:

- full WebUI
- heavy database dependency
- advanced scheduling
- complex multi-user permissions
- automatic model benchmarking
- advanced llama.cpp tuning
- broad replacement of existing bridgeV002 model logic
- removal of `command_builder.py`
- global switch to allocator

---

## 17. Hard Design Rules

These are **hard constraints**. Implementation handoffs must not violate them.

1. Model Allocator must be **optional per role and per step**.
2. Model Allocator must **not** be forced globally.
3. Step-level model config **overrides** role-level model config.
4. Role-level model config **overrides** system default.
5. bridgeV002 may only use allocator aliases with `validation_status == OK`.
6. bridgeV002 must display **both** logical alias and resolved real model.
7. Backend-specific start/stop/unload/context/offload belongs in Model Allocator.
8. Concrete backend commands belong in **backend adapters**.
9. OpenCode and Claude Code differences belong in **client adapters**.
10. V1 must stay small: **CLI first**, service later only if needed.
11. Existing `direct_*` model sources must remain supported.
12. `machine.json`, environment variables, and database values remain usable and **feed** allocator resolution.
13. Schema changes in Father must use versioned SQL migrations, **not** `init_db.py` edits.
14. No hardcoded `/home/svend/...` paths in committed reusable code.
15. `inherit_from_role` is valid **only** on `bridge_flow_steps.model_source`.
16. bridgeV002 must retain tmux orchestration ownership.
17. Model Allocator must **not** own flow order, deliverable routing, or bridge step execution.
18. Model/runtime config must be **structured data**, not opaque shell strings.
19. Secrets must be referenced by **environment variable name only**.
20. V1 must prove alias validation **before** bridgeV002 integration.

---

## 18. Expected Benefit

The user selects:

```
imple-fast
imple-main
review-cloud
junior-cuda1
llama-test
```

instead of manually managing:

- Ollama model name
- llama.cpp model path
- server port
- API base
- API key environment variable
- Claude Code env vars
- OpenCode config files
- context size
- offload flags
- stop/unload commands

Once an allocator alias validates as **OK**, bridgeV002 can point to it and
know the backend/client/runtime combination works.

---

## 19. Final Architecture Summary

```
Model Allocator   = validated runtime/model alias layer
bridgeV002        = role/client/step orchestration layer
Backend adapters  = concrete model runtime commands
Client adapters   = OpenCode / Claude Code launch/config behavior
```

---

## 20. Open Items

These are **not decided** by this scope doc. They are flagged for the
Architect/Implementer when the first implementation handoff is written.

1. **Validation-cache storage location** — allocator state file vs. Father DB.
2. **Alias config file format** — YAML vs JSON for `models.yaml` / `roles.yaml`.
   Spec uses YAML examples; CLI may load both later.
3. **Executable install path** — `scripts/model-allocator` wrapper +
   `src/model_allocator/` importable package; define how it lands on `PATH`
   for bridgeV002 subprocess calls.
4. **Client field on `bridge_roles`** — current `default_runtime` may already
   serve as client. Confirm whether this spec maps `client` to
   `default_runtime`, or whether a new `client` column is needed.
5. **`dispatch.py` post-dispatch hang** — separate known bug where
   `ollama stop` can hang. Allocator `stop` must use timeout/error handling
   and must not inherit this hang.
6. **First migration target** — choose one non-critical pilot role/step for
   `model_source = model_allocator`.
7. **First client adapter** — recommendation: OpenCode first, Claude Code
   after allocator validation is stable.
8. **Machine path strategy** — decide exact config getter/placeholders for
   machine-specific paths before committed code is written.
