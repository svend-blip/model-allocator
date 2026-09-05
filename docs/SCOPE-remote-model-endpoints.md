# SCOPE Addendum — Remote AI-PC Model Endpoints

**Placement (Human, 2026-09-05):** This addendum is recorded here as future
scope for the **Model Allocator** project. It is explicitly **NOT** to run in
the DPMtF `9000` FlowRunner flow ("kør det ikke i flow 9000"). Implement it only
after the currently planned/existing Model Allocator runs, unless an existing
run already provides the required abstraction. Keep it architecturally separate
from the Lightworker capability (see Lightworker Separation below). Not yet
committed — pending Human review.

## Mission

Extend Model Allocator so a model running on another AI-PC can be used by DPMtF
in the same way as an externally hosted model. The remote AI-PC exposes the
model through an HTTP API, preferably an OpenAI-compatible endpoint. DPMtF must
continue to reference **model aliases only**; Model Allocator resolves the alias
to the configured remote endpoint. This addendum is intentionally small — **no
load balancing, distributed scheduling, or automatic routing.**

## Target Architecture

```
DPMtF → model alias → Model Allocator → fixed remote endpoint
      → Remote AI-PC → model runtime → model
```

Example:

```
Reviewer Step → reviewer-remote → Model Allocator
  → http://remote-ai-pc:8090/v1 → FreeToken → Qwen model
```

The remote machine may be reached over LAN or Tailscale.

## In Scope

1. **Remote model endpoint** — a model alias that resolves to a fixed remote
   HTTP endpoint. Conceptual config (adapt to the EXISTING Model Allocator
   config model in `models.yaml`, do not create a parallel config system):
   ```yaml
   models:
     reviewer-remote:
       provider: openai_compatible
       location: remote
       base_url: http://remote-ai-pc:8090/v1
       model: Qwen3.8-Flash-Next
   ```
2. **OpenAI-compatible API** — initial implementation supports OpenAI-compatible
   endpoints (FreeToken, llama.cpp, SGLang, other compatible runtimes). Do NOT
   build runtime-specific remote implementations where the existing
   OpenAI-compatible interface suffices.
3. **Deterministic routing** — each configured alias resolves to EXACTLY ONE
   endpoint (`reviewer-remote → AI-PC #2 → fixed model endpoint`). No automatic
   selection between machines.
4. **Validation** — Model Allocator validates: remote host reachable; configured
   API endpoint responds; configured model available where practical; failures
   produce a clear diagnostic. A remote endpoint being unavailable must produce
   a normal explicit failure — do NOT auto-redirect to another model/machine.
5. **Existing role/Step Key model selection** — the existing selection mechanism
   must continue to work (`Step Key → model alias → remote endpoint`); DPMtF
   needs no special logic for remote AI-PC models.

## Lightworker Separation

Do NOT merge this with the DPMtF Lightworker concept — they are separate
capabilities:
- **Lightworker** = remote EXECUTION node (repositories, commands, tests,
  builds, tools).
- **Remote Model Endpoint** = remote INFERENCE node (model API request/response).

The same physical AI-PC may eventually provide both, but they remain
architecturally separate. This addendum concerns only the Remote Model Endpoint.

## Out of Scope

Do NOT implement: load balancing; model endpoint pools; automatic failover;
automatic workload distribution; dynamic machine selection; GPU-aware
scheduling; queue-based routing between AI-PCs; distributed inference; model
migration between machines; automatic model replication; cluster management.
(These may be considered separately in the future.)

## Compatibility

Preserve existing Model Allocator behavior for: local models; existing cloud
providers; FreeToken; llama.cpp; Ollama; SGLang; existing model aliases;
role-based model selection; Step Key model selection. Prefer EXTENDING the
existing provider/runtime abstraction rather than a second routing architecture.

## Security

Do NOT expose an unauthenticated inference endpoint directly to the public
Internet. LAN endpoints are acceptable. For cross-network communication, prefer
an existing private network such as Tailscale. Secrets/API tokens, if required,
must use the existing Model Allocator secret/configuration mechanism and must
NOT be committed to the repository.

## Minimal Acceptance Criteria

The feature is complete when:
1. A model runs on AI-PC #2.
2. AI-PC #2 exposes an OpenAI-compatible endpoint.
3. Model Allocator on AI-PC #1 has an alias pointing to that endpoint.
4. Model Allocator can validate the remote endpoint.
5. DPMtF can assign that alias to a role or Step Key.
6. The role sends a normal model request through Model Allocator.
7. The response returns to DPMtF normally.
8. Stopping the remote model produces a clear failure rather than silent
   rerouting.
9. Existing local and cloud model configurations continue to work.

## Suggested Implementation Order

Implement only after currently planned/existing Model Allocator runs unless an
existing run already provides the abstraction. Keep the change as small as
possible:
1. Inspect existing Model Allocator provider/runtime abstractions.
2. Reuse the existing OpenAI-compatible client where possible.
3. Add configurable remote `base_url` support.
4. Add remote endpoint validation.
5. Add one test configuration.
6. Add integration tests.
7. Verify DPMtF role/Step Key selection end-to-end.

Do NOT refactor unrelated Model Allocator components merely to implement this.

## Example Initial Deployment

```
AI-PC #1: DPMtF + Model Allocator
   │ Tailscale/LAN
   ▼
AI-PC #2: FreeToken or llama.cpp + OpenAI-compatible API + local model
```

A selected DPMtF role uses AI-PC #2 while the rest of the flow uses models
configured elsewhere. From DPMtF's perspective, the remote model behaves like
any other Model Allocator alias.

## Future Direction — NOT Part of This Scope

The architecture should not PREVENT future support for: multiple remote AI-PCs →
endpoint pools → availability-aware routing → load balancing → distributed
scheduling. None of these are implemented as part of this addendum. For now:
**one alias → one fixed endpoint → one remote AI-PC.**
