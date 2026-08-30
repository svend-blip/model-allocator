# Validate JSON Output Implementation Plan

> **For agentic workers:** Execute tasks in order. Steps use checkbox (`- [ ]`) syntax for tracking. Do not skip verification steps.

**Goal:** Add `--json` to `model-allocator validate` (structured output with `warnings`/`errors` as arrays) and migrate Father's `/allocator/validate` endpoint off its fragile regex/text parser, with a one-release text fallback.

**Architecture:** `Validator.validate(alias, client)` (src/model_allocator/validator.py:23-85) already returns a structured dict; `cmd_validate` (src/model_allocator/cli.py:74-82) currently throws that structure away by printing `format_output()` text, which Father re-parses with `_parse_allocator_validate_text` (routers/bridge.py:986-1042). This plan prints the dict as JSON behind a flag, keeps the text output byte-identical as default, and switches Father to `--json` with fallback to the text parser while both repos' commits land independently.

**Tech Stack:** Python 3.10+ stdlib (json, argparse), pyyaml (only runtime dep), pytest capsys tests (tests/test_config_cli.py conventions), FastAPI + subprocess on the Father side.

## Cold-Start Context

- model-allocator is a standalone Python CLI at `/home/svend/model-allocator` that resolves logical model aliases → runtime commands for ollama / llama.cpp / opencode / claude-code / onyx.
- Config = `models.yaml` + `runtime_profiles.yaml` + `roles.yaml` at the repo root, loaded by `src/model_allocator/config_loader.py`.
- Source layout: `src/model_allocator/` (cli.py, validator.py, resolver.py, ...) and `src/model_allocator/adapters/`.
- The Father project `/home/svend/DPMtF-WebUI` consumes it via subprocess shell-out to `/home/svend/model-allocator/scripts/model-allocator` (bash wrapper → `python3 -m model_allocator`). `routers/bridge.py` exposes `/api/bridge-v2/allocator/*` proxy endpoints; `static/js/dpmtf-app.js` renders the results.
- Run tests: `cd /home/svend/model-allocator && python3 -m pytest` → **95 passed** in ~5s (baseline).
- Exit codes (cli.py:24-27): `EXIT_OK=0`, `EXIT_ERROR=1`, `EXIT_WARNING=2`, `EXIT_USAGE=64`. Note argparse itself exits with code 2 on unknown flags — this collides with EXIT_WARNING and matters for the Father fallback (see edge case 4).

## Global Constraints

- `python3 -m py_compile <file>` MUST pass on every touched Python file.
- Single runtime dependency stays `pyyaml` (`mcp` optional extra). NO new dependencies without explicit Human approval (none needed here — `json` is stdlib).
- All 95 existing tests stay green after every task.
- TDD: failing test → implement → green. Allocator CLI tests: pytest style with `capsys` and tmp-dir configs (see tests/test_config_cli.py). Father endpoint tests: `tests/test_allocator_config_endpoints.py` pattern (monkeypatch `bridge.subprocess.run`, `client` fixture from tests/conftest.py).
- Backwards compatibility: default (no-flag) `validate` text output stays byte-identical — Father instances running older Father code keep parsing it. The Father migration includes a fallback so it also works against an allocator without `--json`.
- Git policy: the Human approves all commits in BOTH repos. Tasks end with `git add <files>` and STOP.

## Edge Cases a Weaker Model Would Miss

1. **`warnings`/`errors` must be arrays in JSON.** `format_output` (validator.py:206-207) comma-joins them (`', '.join(warnings)`), which is exactly the lossy seam this plan kills — a warning containing a comma (e.g. `Ollama API base unreachable: <urlopen error [Errno 111] Connection refused>`) shreds into fragments in the text parser. The JSON path must serialize `result["warnings"]` / `result["errors"]` lists directly, never the joined strings.
2. **Exit-code semantics must be preserved in `--json` mode.** `cmd_validate` returns 1 on ERROR, 2 on WARNING, 0 on OK (cli.py:78-82). Keep that identical in JSON mode — scripts (and the acceptance criteria) rely on it. Father itself does NOT check the returncode of `validate` today (routers/bridge.py:1096-1108 parses stdout regardless) — do not "fix" that as a side effect; response shape is the contract.
3. **stdout purity in `--json` mode.** The only bytes on stdout must be the JSON document. `cmd_validate` currently prints nothing else, and `Validator` performs its network probes silently — but keep it that way: any future diagnostic in the validate path must go to `sys.stderr`. Do not add banners/log lines to stdout.
4. **Old-allocator fallback trips on argparse exit code 2, not a clean error.** Against a pre-flag allocator, `validate ... --json` makes argparse raise `SystemExit(2)` with EMPTY stdout and a usage message on stderr. Exit code 2 == `EXIT_WARNING`, so the Father fallback must key on "stdout does not parse as JSON", never on the return code.
5. **Key-name mismatch: `gpu_policy` vs `resolved_gpu`.** The validator dict has `resolved_gpu` (validator.py:46); Father's response and the JS both use `gpu_policy` (routers/bridge.py:1116, dpmtf-app.js:517-519). Map `resolved_gpu` → `gpu_policy` in the Father endpoint; do NOT rename the validator key (other consumers/tests use it).
6. **`last_validated_at` is a datetime-derived ISO string already** (validator.py:25 uses `.isoformat()`), but use `json.dumps(..., default=str)` anyway to match the envelope style of `resolve`/`status`/`list` (cli.py:70, 100, 132 all use `json.dumps(result, indent=2, default=str)`) and to survive future non-serializable values.
7. **Deterministic tests need aliases that skip network probes.** `validate` on an ollama alias probes `http://127.0.0.1:11434` (may or may not be running on the dev machine → flaky assertions). Two deterministic cases: (a) unknown alias → early return, zero network (validator.py:36-41); (b) minimax alias with `claude-code` client and no `default_api_base`/env → hard ERROR from `_check_client_backend_compatibility` (validator.py:90-99) and the openai adapter raises "API base not configured" locally without any socket (openai_compatible.py:29-30). Use only these in tests.
8. **Two-repo landing order.** The allocator commit must land first; the Father change is safe against both allocator versions because of the fallback, but the reverse order would leave Father calling a flag that makes every validate fall back (works, but double subprocess per click). Note the order in the handoff.

---

### Task 1: TDD — `--json` flag on `validate`

**Files:**
- Test: Create `/home/svend/model-allocator/tests/test_validate_json.py` (new file; pytest style like tests/test_config_cli.py).
- Modify: `/home/svend/model-allocator/src/model_allocator/cli.py` — `cmd_validate` (lines 74-82) and the `p_validate` parser block (lines 531-534).

**Interfaces:**
- Consumes: `args.json: bool` (argparse `store_true`), `Validator.validate(alias, client) -> dict` (unchanged).
- Produces: stdout = `json.dumps(result, indent=2, default=str)` when `--json`; unchanged `validator.format_output(result)` text otherwise. Return codes unchanged: 0 OK / 2 WARNING / 1 ERROR.

- [ ] Step 1: Create `tests/test_validate_json.py` with exactly this content:

```python
"""Tests for `validate --json` (structured output, PLAN-validate-json-output)."""

import json
from pathlib import Path

import yaml

from model_allocator.cli import main


def _seed(tmp_path: Path) -> Path:
    (tmp_path / "models.yaml").write_text(
        yaml.safe_dump({"models": {
            "review-cloud": {
                "runtime_profile": "cloud_minimax",
                "real_model": "minimax-m3",
                "lifecycle_policy": "cloud_noop",
                "clients": {"opencode": True, "claude-code": False},
            },
        }}), encoding="utf-8")
    (tmp_path / "runtime_profiles.yaml").write_text(
        yaml.safe_dump({"runtime_profiles": {
            "cloud_minimax": {
                "backend": "openai_compatible",
                "api_base_env": "MINIMAX_API_BASE_TEST_UNSET",
                "api_key_env": "MINIMAX_API_KEY_TEST_UNSET",
                "provider": "minimax",
            },
        }}), encoding="utf-8")
    (tmp_path / "roles.yaml").write_text(yaml.safe_dump({"roles": {}}), encoding="utf-8")
    return tmp_path


def test_validate_json_unknown_alias_is_structured_error(tmp_path, capsys):
    d = _seed(tmp_path)
    rc = main(["--config-dir", str(d), "validate", "--alias", "ghost",
               "--client", "opencode", "--json"])
    assert rc == 1  # EXIT_ERROR preserved in JSON mode
    out = capsys.readouterr().out
    data = json.loads(out)  # stdout must be pure JSON
    assert data["validation_status"] == "ERROR"
    assert isinstance(data["errors"], list) and data["errors"]
    assert "ghost" in data["errors"][0]
    assert isinstance(data["warnings"], list)


def test_validate_json_incompatible_client_error_array_not_joined(tmp_path, capsys):
    d = _seed(tmp_path)
    rc = main(["--config-dir", str(d), "validate", "--alias", "review-cloud",
               "--client", "claude-code", "--json"])
    assert rc == 1
    data = json.loads(capsys.readouterr().out)
    assert data["validation_status"] == "ERROR"
    assert data["logical_model_alias"] == "review-cloud"
    assert data["resolved_backend"] == "openai_compatible"
    # arrays, not comma-joined strings
    assert isinstance(data["errors"], list)
    assert any("incompatible" in e.lower() for e in data["errors"])


def test_validate_default_text_output_unchanged(tmp_path, capsys):
    d = _seed(tmp_path)
    rc = main(["--config-dir", str(d), "validate", "--alias", "ghost",
               "--client", "opencode"])
    assert rc == 1
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "ERROR"                      # first line = status word
    assert lines[1] == "Logical model: ghost"       # byte-identical text format
    assert any(line.startswith("Errors: ") for line in lines)
```

- [ ] Step 2: Run and confirm failure:
```bash
cd /home/svend/model-allocator && python3 -m pytest tests/test_validate_json.py -v
```
Expected: the two `--json` tests fail with `SystemExit: 2` (unrecognized arguments: --json); the text test passes.

- [ ] Step 3: In `cli.py`, add the flag to the parser. After line 533 (`p_validate.add_argument("--client", ...)`) insert:

```python
    p_validate.add_argument(
        "--json",
        action="store_true",
        help="Print the full validation result as JSON (warnings/errors as arrays)",
    )
```

- [ ] Step 4: Replace `cmd_validate` (cli.py:74-82) with:

```python
def cmd_validate(args: argparse.Namespace) -> int:
    validator = Validator(config_dir=_config_dir(args))
    result = validator.validate(args.alias, args.client)
    if getattr(args, "json", False):
        # Same envelope style as resolve/status/list. stdout stays pure JSON;
        # warnings/errors remain ARRAYS (the text formatter comma-joins them,
        # which is exactly the lossy seam --json exists to avoid).
        print(json.dumps(result, indent=2, default=str))
    else:
        print(validator.format_output(result))
    if result["validation_status"] == "ERROR":
        return EXIT_ERROR
    if result["validation_status"] == "WARNING":
        return EXIT_WARNING
    return EXIT_OK
```

- [ ] Step 5: Verify:
```bash
python3 -m py_compile src/model_allocator/cli.py && python3 -m pytest tests/test_validate_json.py -v && python3 -m pytest 2>&1 | tail -1
```
Expected: 3/3 in the new file; full suite `98 passed`.

- [ ] Step 6: Manual smoke against the live repo config (network probes may add warnings — only check shape):
```bash
./scripts/model-allocator validate --alias imple01-claude --client claude-code --json | python3 -m json.tool > /dev/null && echo JSON_OK
./scripts/model-allocator validate --alias imple01-claude --client claude-code | head -2
```
Expected: `JSON_OK`, then two text lines starting with the status word and `Logical model: imple01-claude`.

- [ ] Step 7: Stage and STOP — await Human commit approval:
```bash
git add src/model_allocator/cli.py tests/test_validate_json.py
git status --short
```
Suggested commit message: `[V5.1] validate --json: structured output with warnings/errors arrays, exit codes preserved`

---

### Task 2: CROSS-REPO — migrate Father `/allocator/validate` to `--json` with text fallback

> **Cross-repo task in `/home/svend/DPMtF-WebUI` — Father's governance requires Human approval for `routers/` changes; only the Human commits. Prepare the change, verify, present the diff, and STOP.**

**Files:**
- Modify: `/home/svend/DPMtF-WebUI/routers/bridge.py` — endpoint `bridge_v2_allocator_validate` (lines 1081-1122). KEEP `_parse_allocator_validate_text` (lines 986-1042) for the fallback window.
- Test: `/home/svend/DPMtF-WebUI/tests/test_allocator_config_endpoints.py` — append new tests (same monkeypatch-subprocess pattern as the existing tests in that file).

**Interfaces:**
- Consumes: allocator stdout. New-allocator: JSON dict with keys `validation_status`, `resolved_backend`, `resolved_real_model`, `resolved_gpu`, `warnings` (list), `errors` (list), `logical_model_alias`, `last_validated_at`, `client_support`, `resolved_api_base`, `resolved_context`. Old-allocator: argparse SystemExit(2) with empty stdout → re-run without `--json` → text.
- Produces: HTTP JSON response with UNCHANGED keys (verified against the JS consumer `static/js/dpmtf-app.js` — `createModelSourceControl` validate handler lines 416-445 and `renderAllocatorStatusCard`/`updateValidationSection` lines 494-536 read exactly: `validation_status`, `resolved_backend`, `resolved_real_model`, `warnings[]`, `errors[]`, `gpu_policy`): `{validation_status, resolved_backend, resolved_real_model, warnings, errors, gpu_policy, raw_output}`. **No JS change is required** because the response contract is preserved.

- [ ] Step 1: Add failing Father tests. Append to `/home/svend/DPMtF-WebUI/tests/test_allocator_config_endpoints.py`:

```python
def test_validate_uses_json_flag_and_maps_gpu_policy(client, monkeypatch):
    captured = {}
    allocator_json = json.dumps({
        "validation_status": "WARNING",
        "logical_model_alias": "imple01-claude",
        "resolved_backend": "ollama",
        "resolved_real_model": "qwen3-coder:30b-256k",
        "resolved_gpu": "cuda0",
        "warnings": ["Context not declared", "Ollama API base unreachable: x, y"],
        "errors": [],
    })

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return _completed(stdout=allocator_json, rc=2)

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    resp = client.post("/api/bridge-v2/allocator/validate",
                       json={"alias": "imple01-claude", "client": "claude-code"})
    assert resp.status_code == 200
    assert "--json" in captured["cmd"]
    body = resp.json()
    assert body["validation_status"] == "WARNING"
    assert body["gpu_policy"] == "cuda0"          # mapped from resolved_gpu
    # arrays survive intact — including the comma inside one warning
    assert body["warnings"] == ["Context not declared",
                                "Ollama API base unreachable: x, y"]


def test_validate_falls_back_to_text_parse_for_old_allocator(client, monkeypatch):
    calls = []
    legacy_text = (
        "ERROR\n"
        "Logical model: ghost\n"
        "Backend: N/A\n"
        "Warnings: none\n"
        "Errors: Alias 'ghost' not found\n"
    )

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        if "--json" in cmd:
            # old allocator: argparse rejects the flag -> rc 2, empty stdout
            return _completed(stdout="", stderr="usage: model-allocator ...", rc=2)
        return _completed(stdout=legacy_text, rc=1)

    monkeypatch.setattr(bridge.subprocess, "run", fake_run)
    resp = client.post("/api/bridge-v2/allocator/validate",
                       json={"alias": "ghost", "client": "opencode"})
    assert resp.status_code == 200
    assert len(calls) == 2                        # json attempt + text fallback
    body = resp.json()
    assert body["validation_status"] == "ERROR"
    assert body["errors"] == ["Alias 'ghost' not found"]
```

- [ ] Step 2: Run and confirm the first test FAILS (no `--json` in the command yet):
```bash
cd /home/svend/DPMtF-WebUI && python3 -m pytest tests/test_allocator_config_endpoints.py -k validate -v
```

- [ ] Step 3: Replace the body of `bridge_v2_allocator_validate` (routers/bridge.py lines 1081-1122) with:

```python
@router.post("/allocator/validate")
async def bridge_v2_allocator_validate(request: Request):
    """Validate an allocator alias/client by shelling out to model-allocator."""
    data = await request.json()
    alias = data.get("alias")
    client = data.get("client")
    if not alias or not client:
        raise HTTPException(status_code=400, detail="alias and client are required")

    allocator_script = os.path.join(
        config.get_project_path("model-allocator"),
        "scripts",
        "model-allocator",
    )
    base_cmd = [allocator_script, "validate", "--alias", alias, "--client", client]
    try:
        result = subprocess.run(
            base_cmd + ["--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        raw_output = result.stdout.strip()
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            # Fallback window: allocator without --json support rejects the
            # flag via argparse (rc 2, empty stdout). Re-run in text mode and
            # use the legacy parser. Remove once the allocator flag has been
            # deployed for one release (follow-up: delete
            # _parse_allocator_validate_text and this branch).
            result = subprocess.run(
                base_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            raw_output = result.stdout.strip()
            parsed = _parse_allocator_validate_text(raw_output)

        return {
            "validation_status": parsed.get("validation_status", "UNKNOWN"),
            "resolved_backend": parsed.get("resolved_backend"),
            "resolved_real_model": parsed.get("resolved_real_model"),
            "warnings": parsed.get("warnings", []),
            "errors": parsed.get("errors", []),
            # allocator JSON uses resolved_gpu; legacy text parser emits gpu_policy
            "gpu_policy": parsed.get("gpu_policy") or parsed.get("resolved_gpu"),
            "raw_output": raw_output,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=502, detail="model-allocator validate timed out after 30s")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"model-allocator validate error: {exc}")
```

- [ ] Step 4: Verify Father:
```bash
cd /home/svend/DPMtF-WebUI && python3 -m py_compile routers/bridge.py && python3 -m pytest tests/test_allocator_config_endpoints.py -v && node --check static/js/dpmtf-app.js
```
Expected: all endpoint tests pass (existing + 2 new); `node --check` clean (JS untouched — contract preserved).

- [ ] Step 5: Live smoke (requires the app on port 9130 and the allocator Task 1 present in the working tree):
```bash
curl -s -X POST http://localhost:9130/api/bridge-v2/allocator/validate \
  -H 'Content-Type: application/json' \
  -d '{"alias": "imple01-claude", "client": "claude-code"}' | python3 -m json.tool
```
Expected: JSON with `validation_status`, `warnings` as an array (not comma-shredded), `gpu_policy: "cuda0"`, and `raw_output` beginning with `{` (proving the JSON path was taken).

- [ ] Step 6: Present the diff (`git -C /home/svend/DPMtF-WebUI diff routers/bridge.py tests/test_allocator_config_endpoints.py`) to the Human and STOP — Father's governance: only the Human stages/commits `routers/` changes. State the landing order: allocator commit first, Father commit second.

---

### Task 3: Follow-up marker — legacy text-parser removal (documentation step, no code)

**Files:** none in this cycle.

- [ ] Step 1: In the handoff/summary to the Human, record this follow-up verbatim: "After the allocator `--json` commit and the Father migration have both been deployed for one release cycle, delete `_parse_allocator_validate_text` (routers/bridge.py:986-1042) and the fallback re-run branch in `bridge_v2_allocator_validate`, and simplify `test_validate_falls_back_to_text_parse_for_old_allocator` into a 502-on-garbage test." Do NOT delete it now — the whole point of the fallback is that the two repos' commits land independently.

## Acceptance Criteria

1. `cd /home/svend/model-allocator && python3 -m pytest` → `98 passed` (95 baseline + 3 new), 0 failures.
2. `./scripts/model-allocator validate --alias no-such-alias --client opencode --json; echo "rc=$?"` → pure-JSON stdout with `"validation_status": "ERROR"`, `"errors"` as a non-empty array, and `rc=1`.
3. `./scripts/model-allocator validate --alias no-such-alias --client opencode | head -1` → exactly `ERROR` (default text output byte-identical to pre-change).
4. `./scripts/model-allocator validate --alias imple01-claude --client claude-code --json | python3 -c "import json,sys; d=json.load(sys.stdin); print(type(d['warnings']).__name__, type(d['errors']).__name__)"` → `list list`.
5. `cd /home/svend/DPMtF-WebUI && python3 -m pytest tests/test_allocator_config_endpoints.py -q` → all pass, including `test_validate_uses_json_flag_and_maps_gpu_policy` and `test_validate_falls_back_to_text_parse_for_old_allocator`.
6. `grep -c "innerHTML" /home/svend/DPMtF-WebUI/static/js/dpmtf-app.js` unchanged from baseline and no JS file modified (`git -C /home/svend/DPMtF-WebUI status --short static/` → empty).
7. Live: the Bridge Roles UI "Validate" button renders the same fields as before (Validation status, GPU Policy, Model, warnings, errors) — verified via the curl in Task 2 Step 5 returning `gpu_policy` non-null for `imple01-claude`.
8. Allocator repo staged files: `src/model_allocator/cli.py`, `tests/test_validate_json.py` only; nothing committed in either repo without the Human.
