# SCOPE Addendum — Queued Remote Inference Nodes

**Placement (Human, 2026-09-05):** Model Allocator scope, run AUTONOMOUSLY with
commit + push, explicitly **NOT** in the DPMtF `9000` FlowRunner flow. Builds on
[SCOPE-remote-model-endpoints.md](SCOPE-remote-model-endpoints.md) (alias → fixed
remote OpenAI-compatible endpoint). Keep it deliberately simple. Cloud models
must keep working exactly as today. Keep inference nodes and Lightworkers
conceptually separate.

## Mission

Extend Model Allocator so DPMtF can use dedicated AI-PCs as **remote inference
nodes**. A node may keep ONE model permanently loaded (warm) and process
requests from DPMtF or other authorized clients. Each node uses a simple FIFO
queue and processes ONLY ONE request at a time. DPMtF continues to select models
through Model Allocator aliases — an alias resolves to a local model, a remote
inference node, or a cloud provider, with no separate DPMtF mechanism for remote
models. The DPMtF machine need not have a large GPU; its job is orchestration,
governance, flow execution, and Model Allocator.

## Architecture

```
            DPMtF → Model Allocator
                 ┌────────┼────────┐
                 ▼        ▼        ▼
              AI-PC #1  AI-PC #2  Cloud
                 │        │
              FIFO queue FIFO queue
                 │        │
              FreeToken  FreeToken
                 │        │
              Model A    Model B   (each always loaded)
```

## Remote Inference Node

An AI-PC that exposes a model service over the network (LAN/Tailscale). Initial
runtimes: FreeToken, llama.cpp, SGLang, other OpenAI-compatible runtimes —
prefer an OpenAI-compatible interface.

- **Warm model:** the node may keep one model permanently loaded (start →
  FreeToken → model loads → stays loaded → process jobs). Do NOT unload/reload
  between normal requests.
- **Single-worker FIFO queue:** one request processed at a time; others QUEUED
  in order. Minimum job states: QUEUED, RUNNING, COMPLETED, FAILED. This is a
  serialized inference queue, NOT batch inference.
- **Context isolation:** the model stays loaded, but request context MUST NOT
  leak between independent jobs (create context → process → return → release/
  reset → next). Independent users/flows/roles/jobs must not inherit prior
  context. Persistent conversational sessions are not required.
- **Multiple clients:** more than one authorized client may use the same node;
  requests queue when busy (more clients → longer waits, expected). Do NOT solve
  with load balancing in this scope.
- **Basic access control:** do not expose the queue publicly unprotected. Use
  the private-network approach (Tailscale + a simple auth/API token); be able to
  identify/authenticate the calling client. Do NOT build complex user management.

## Model Allocator Responsibility

Model Allocator knows: model alias, node/endpoint, runtime/provider, model name,
connection/authentication info. Concept (adapt to the EXISTING config
architecture — runtime_profiles + models; do NOT create a parallel config):

```yaml
models:
  implementer-local:  { provider: remote_openai_compatible, endpoint: http://ai-pc-1:8090, model: coder-model }
  reviewer-local:     { provider: remote_openai_compatible, endpoint: http://ai-pc-2:8090, model: reviewer-model }
  supervisor-cloud:   { provider: existing_cloud_provider,   model: cloud-model }
```

Model Allocator simply ROUTES each configured alias to its fixed destination;
the FIFO queue lives on the NODE (in front of the runtime), not in the
allocator's routing.

## Cloud Models

Cloud support stays intact; cloud requests do NOT use the inference-node FIFO
queue. A single DPMtF flow may freely combine cloud and remote models.

## Lightworker Separation

Inference Node = model inference / warm model / inference queue. Lightworker =
commands / repositories / tests / builds / tools. Same physical AI-PC may run
both, but they stay separate services.

## Out of Scope

load balancing; automatic failover; GPU scheduling; dynamic node selection;
distributed inference; multi-node model execution; Kubernetes; Ray; Slurm;
automatic model migration/replication; complex priority queues; automatic cloud
fallback; cluster management. (Later, maybe.)

## Minimum Acceptance Criteria

1. An AI-PC keeps a model loaded (FreeToken or another supported runtime).
2. Multiple authorized clients can submit inference requests to it.
3. Only one request is processed at a time.
4. Additional requests wait in FIFO order.
5. Each request has isolated context.
6. The model remains loaded between requests.
7. Model Allocator addresses the remote model through a normal alias.
8. DPMtF can assign that alias to a role or Step Key.
9. Two different remote AI-PCs can be configured independently.
10. DPMtF can keep using cloud models alongside the remote AI-PC models.
11. Existing local and cloud Model Allocator behavior is not broken.

## Initial Design Rule

First version: one inference node = one fixed endpoint + one warm model + one
FIFO queue + one active inference request. Multiple nodes operate independently;
no load balancing between them; the allocator routes each alias to its fixed
destination. Objective: let DPMtF run on a machine without a large GPU while
using warm models hosted by one or more dedicated AI-PCs, alongside normal cloud
models.
