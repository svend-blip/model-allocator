# Model Allocator — V1A Minimal Allocator Proof

Standalone runtime/model-alias layer for the DPMtF ecosystem.

## V1A scope

- Logical model aliases resolve to runtime profiles, backends, and real models.
- Local Ollama backend adapter (read-only: status / availability / reachability).
- CLI commands: `resolve`, `validate`, `list`, `status`.
- YAML/JSON config loading with environment-variable resolution.

Out of scope in V1A: `start`, `stop`, `unload`, `run`, client launch adapters,
bridgeV002 schema changes, llama.cpp / cloud adapters, FastAPI service.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m model_allocator --help
python3 -m model_allocator validate --alias imple-fast --client opencode
python3 -m unittest discover tests/
```

## Config files

- `models.example.yaml` — logical aliases
- `roles.example.yaml` — role-to-alias mappings
- `runtime_profiles.example.yaml` — runtime profiles
- `machine.example.json` — illustrative machine profile fields

Example files may contain commented machine-specific paths. Reusable code does
not hardcode absolute paths.
