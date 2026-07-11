# ONYX Lite — local deployment (optional runtime for model-allocator V5)

ONYX (github.com/onyx-dot-app/onyx) runs as an OPTIONAL local runtime.
Nothing in bridgeV002 or model-allocator requires it — only aliases that
explicitly point at the `onyx` backend use it.

## Layout

- Upstream checkout (sparse): `~/onyx` (deployment/docker_compose only)
- Stack: ONYX **Lite** overlay — Postgres + api_server + web_server only
  (no Vespa/Redis/MinIO/model servers; connectors and RAG indexing are
  disabled in Lite; chat + assistants + file uploads work)
- Local override (this dir): direct ports, nginx bypassed
  - API:    http://127.0.0.1:9162  (adapter endpoint)
  - Web UI: http://127.0.0.1:9163

## Start / stop

```bash
cd ~/onyx/deployment/docker_compose
docker compose -f docker-compose.yml -f docker-compose.onyx-lite.yml \
               -f docker-compose.local-ports.yml up -d
docker compose -f docker-compose.yml -f docker-compose.onyx-lite.yml \
               -f docker-compose.local-ports.yml down   # fully optional stack
```

Containers use `restart: unless-stopped` — the stack survives reboots
while docker is enabled. `docker compose down` removes it cleanly; the
rest of the ecosystem is unaffected (optionality guarantee).

## Configuration decisions (2026-07-11)

- `AUTH_TYPE=basic` (the old `disabled` mode no longer exists). First
  registered user is admin: credentials in `~/.bashrc`
  (`ONYX_EMAIL` / `ONYX_ADMIN_PASSWORD`). API keys are Business-tier
  gated, so the adapter authenticates via login + session cookie.
- LLM provider: **local Ollama via its OpenAI-compatible endpoint**
  (`http://172.17.0.1:11434/v1`, provider type `openai`). The native
  litellm `ollama` provider type drops qwen3.6's thinking-format answers
  ("LLM packet is empty") — the /v1 route handles it correctly.
  Default model: `qwen3.6:27b-q4_K_M`. Zero cloud cost.
- Assistant: persona `allocator-plain` (id 1) — no tools, no memory.
  User flags `use_memories`/`enable_memory_tool` are disabled (models
  otherwise emit memory-tool JSON as text and the run dies).
- `.env` lives in `~/onyx/deployment/docker_compose/.env` (on-machine
  only, never committed; contains postgres password).

## Upgrade path (deferred by design)

Standard mode (connectors + RAG indexing) = remove the Lite overlay and
start the full stack (Vespa/OpenSearch + model servers; several GB RAM).
Candidate first connector: DPMtF governance docs. Not needed for the
V5 adapter/runner/MCP phases.
