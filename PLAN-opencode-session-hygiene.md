# OpenCode Session Hygiene Implementation Plan

> **For agentic workers:** Execute tasks in order. Steps use checkbox (`- [ ]`) syntax for tracking. Do not skip verification steps.

**Goal:** Stop OpenCode roles from silently running the wrong model: remove the dead `--model` flag, guarantee `opencode.json`'s `model` field is refreshed on every `run`, and detect/archive stale sessions that pin an outdated model.

**Architecture:** OpenCode's TUI ignores `--model` when it resumes a session; only the `model` field in the role's `opencode.json` (and the model pinned inside the session record) counts. Live incident: an allocator-launched OpenCode session resumed an old session and used Kimi/OpenRouter instead of the configured qwen3-coder/Ollama. This plan (a) drops the misleading `--model` argv from `build_opencode_command`, (b) makes `run --client opencode` internally render+merge `opencode.json` before emitting the shell string, and (c) adds a session-hygiene module that reads OpenCode's session store (verified on disk: SQLite) to warn about — and optionally archive — sessions pinned to a different provider/model pair.

**Tech Stack:** Python 3.10+ stdlib (`sqlite3`, `json`, `argparse`, `tempfile`), pyyaml (only runtime dep), pytest + unittest (both styles exist in tests/).

## Cold-Start Context

- model-allocator is a standalone Python CLI at `/home/svend/model-allocator` resolving model aliases → runtime commands for ollama / llama.cpp / opencode / claude-code / onyx. Config = `models.yaml` + `runtime_profiles.yaml` + `roles.yaml` at repo root; source under `src/model_allocator/` with adapters in `src/model_allocator/adapters/`.
- The Father project `/home/svend/DPMtF-WebUI` consumes it via subprocess to `scripts/model-allocator`: `scripts/bridgeV002/start_coding.py` lines 220-243 call `render-config --output ~/.config/opencode-roles/<config_dir>/opencode.json`, then lines 245-257 call `run --role X --client opencode` and inject the printed shell string into tmux.
- Run tests: `cd /home/svend/model-allocator && python3 -m pytest` → **95 passed** ~5s (baseline).
- **On-disk evidence gathered 2026-07-12 (OpenCode 1.17.18 at `/home/svend/.opencode/bin/opencode`):** session storage is a SQLite database `~/.local/share/opencode/opencode.db` (WAL mode; `.db-shm`/`.db-wal` siblings present; 261 sessions live). Schema (relevant columns): `session(id TEXT PK, project_id, directory TEXT, title, model TEXT, time_updated INTEGER, time_archived INTEGER, ...)` where `model` is a JSON string `{"id": "<modelID>", "providerID": "<providerID>"}` (live sample: `{"id":"qwen3.6-27b-48k","providerID":"ollama"}`); `project(id TEXT PK, worktree TEXT, ...)` maps project hash → directory. There is NO per-project session directory to rotate — the "clear stale session" operation is therefore implemented as **archive rows in the DB (set `time_archived`) after taking a file backup**, which preserves transcripts.

## Global Constraints

- `python3 -m py_compile <file>` MUST pass on every touched Python file.
- Single runtime dependency stays `pyyaml` (`sqlite3` is stdlib; `mcp` optional extra). NO new dependencies without explicit Human approval.
- All 95 existing tests stay green (4 adapter tests are deliberately REWRITTEN in Task 1 because their assertions encode the dead flag; count stays, meaning changes).
- TDD: failing test → implement → green. Adapter tests: `tests/test_v2.py` `TestOpenCodeAdapter` (unittest). New hygiene tests: new pytest file.
- Backwards compatibility: `run` stdout stays a single tmux-safe shell line (Father injects it opaquely — verified start_coding.py:258-270 does not parse it). `render-config` CLI contract unchanged. All new diagnostics from `run` go to **stderr** (stdout purity).
- Git policy: the Human approves commits. Tasks end with `git add <files>` and STOP.

## Edge Cases a Weaker Model Would Miss

1. **`opencode.json` is shared state — merge, never overwrite.** The file may carry a human's own `$schema`, `permission`, `mcp`, and other-provider blocks. `_merge_opencode_config` (src/model_allocator/cli.py:338-356) already preserves them — REUSE it; do not write `build_opencode_config` output directly.
2. **Model comparison must use the (providerID, modelID) pair, not display name.** The session record stores `{"id","providerID"}`; `display_name` (used in provider `models.<id>.name`) is cosmetic. Two aliases can share a display name while pointing at different backends.
3. **Deleting session rows loses transcripts.** The user may need them. Archive (`time_archived = <epoch-ms>`) instead of DELETE, and take a `sqlite3` backup file first. OpenCode hides archived sessions from resumption, which is all we need.
4. **The DB is live and WAL-mode — a plain `shutil.copy` of `opencode.db` alone can be inconsistent.** Use `sqlite3.Connection.backup()` (checkpoints WAL into the copy) for the backup, and `busy_timeout` on the write connection so a running OpenCode doesn't cause immediate `database is locked` failures.
5. **Read-only detection must not create/lock the DB.** Open with URI `file:<path>?mode=ro`. A missing DB means OpenCode never ran → zero stale sessions → success, not an error.
6. **`run` output purity.** `cmd_run` prints the shell string on stdout; Father captures stdout (`result.stdout.strip()`, start_coding.py:258). Any render/hygiene message printed to stdout would be injected into tmux as part of the command. Everything new goes to `sys.stderr`.
7. **Tests must not touch the real `~/.config/opencode-roles` or the real `opencode.db`.** `run --client opencode` now writes a config file: tests set `OPENCODE_ROLES_CONFIG_BASE`, `MODEL_ALLOCATOR_STATE_DIR`, and `MODEL_ALLOCATOR_OPENCODE_DB` to tmp paths. The env override `MODEL_ALLOCATOR_OPENCODE_DB` exists precisely as the test seam (and as an operator escape hatch for non-XDG setups).
8. **`OPENCODE_ROLES_CONFIG_BASE` default contains a literal `$HOME`** (`opencode.py:22` — it is rendered into the tmux env for the SHELL to expand). When the allocator itself writes the file, expand with `os.path.expandvars` + `os.path.expanduser` first.
9. **XDG variance:** the data dir is `$XDG_DATA_HOME/opencode` when set, else `~/.local/share/opencode`. Honor `XDG_DATA_HOME`.
10. **Father's V2.3 DIRECT-role path depends on `--model` in `command_builder.py` argv** (start_coding.py:326-328 extracts the model string from the element after `"--model"` to write it into opencode.json). That is Father's own builder, NOT this repo's adapter. Do NOT propose dropping `--model` in Father's `command_builder.py` without reworking that extraction — it is explicitly out of scope here (see Task 6).

---

### Task 1: Drop the dead `--model` argv from `build_opencode_command`

**Files:**
- Test: `/home/svend/model-allocator/tests/test_v2.py` — class `TestOpenCodeAdapter`, tests `test_ollama_prefix` (lines 157-160), `test_openrouter_prefix` (162-166), `test_minimax_bare` (168-172), `test_llama_cpp_prefix` (174-183); class `TestCliV2` `test_run_openrouter` (394-397).
- Modify: `/home/svend/model-allocator/src/model_allocator/adapters/opencode.py` — `build_opencode_command` (lines 50-66; the dead flag is line 65).

**Interfaces:**
- Consumes: `resolved: dict`, `config_dir: str` (unchanged signature `build_opencode_command(resolved, config_dir) -> dict`).
- Produces: `{"env": {OPENCODE_CONFIG_DIR, OPENCODE_CONFIG}, "argv": [opencode_bin]}` — no `--model`. `_model_arg` (lines 30-47) STAYS: `build_opencode_config` uses it for the `model` field, and Task 3 uses it for pair comparison.

- [ ] Step 1: Rewrite the four adapter tests so they pin the NEW contract (replace the existing four methods in `TestOpenCodeAdapter` wholesale):

```python
    @patch("model_allocator.adapters.opencode.shutil.which", side_effect=_fake_which)
    def test_command_has_no_model_flag(self, _mock):
        cmd = opencode.build_opencode_command({"backend": "ollama", "real_model": "qwen"}, "r")
        # OpenCode ignores --model on session resumption (live incident:
        # Kimi/OpenRouter instead of qwen3-coder/Ollama). The flag is dead
        # and misleading; opencode.json's model field is the only channel.
        self.assertNotIn("--model", cmd["argv"])
        self.assertEqual(cmd["argv"], ["/usr/bin/opencode"])
        self.assertIn("OPENCODE_CONFIG", cmd["env"])

    def test_model_arg_ollama_prefix(self):
        self.assertEqual(
            opencode._model_arg({"backend": "ollama", "real_model": "qwen"}),
            "ollama/qwen")

    def test_model_arg_openrouter_prefix(self):
        resolved = {"backend": "openai_compatible", "provider": "openrouter", "real_model": "qwen"}
        self.assertEqual(opencode._model_arg(resolved), "openrouter/qwen")

    def test_model_arg_minimax_bare(self):
        resolved = {"backend": "openai_compatible", "provider": "minimax", "real_model": "minimax-m3"}
        self.assertEqual(opencode._model_arg(resolved), "minimax-m3")

    def test_model_arg_llama_cpp_prefix(self):
        resolved = {
            "backend": "llama_cpp",
            "provider": "llama-turbo",
            "real_model": "/models/model.gguf",
            "opencode_model_id": "qwen36-35b-turbo262k",
        }
        self.assertEqual(opencode._model_arg(resolved), "llama-turbo/qwen36-35b-turbo262k")
```

- [ ] Step 2: Run and confirm `test_command_has_no_model_flag` FAILS (argv still `[bin, "--model", "ollama/qwen"]`):
```bash
cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py::TestOpenCodeAdapter -v
```

- [ ] Step 3: In `opencode.py`, replace `build_opencode_command` (lines 50-66) with:

```python
def build_opencode_command(resolved: dict, config_dir: str) -> dict[str, Any]:
    """Build an OpenCode command object.

    NOTE: no ``--model`` argv. The OpenCode TUI ignores --model when it
    resumes a session (verified live, OpenCode 1.17.x); the only reliable
    channel is the ``model`` field in the role's opencode.json, which
    ``cmd_run`` refreshes via render-config before every run.
    """
    real_model = resolved.get("real_model", "")
    if not real_model:
        raise ValueError("Resolved alias is missing real_model")

    opencode_bin = _resolve_opencode_bin()

    return {
        "env": _config_env(config_dir),
        "argv": [opencode_bin],
    }
```

- [ ] Step 4: Verify adapter tests green, then run the FULL suite and expect `TestCliV2.test_run_openrouter` still green (it only asserts exit code):
```bash
python3 -m py_compile src/model_allocator/adapters/opencode.py && python3 -m pytest 2>&1 | tail -1
```
Expected: `96 passed` (95 − 4 rewritten + 5 new = 96).

---

### Task 2: `run --client opencode` auto-refreshes `opencode.json`

**Files:**
- Test: `/home/svend/model-allocator/tests/test_v2.py` — `TestCliV2` (add one test; adjust `test_run_openrouter` env).
- Modify: `/home/svend/model-allocator/src/model_allocator/cli.py` — extract a helper next to `cmd_render_config` (lines 359-392) and call it from `cmd_run` (lines 140-171).

**Interfaces:**
- New helper: `_write_opencode_json(resolved: dict, output_path: Path) -> None` — renders `opencode.build_opencode_config(resolved)`, merges with existing file via `_merge_opencode_config`, writes atomically via `_atomic_write_json`. Raises on invalid existing JSON / IO errors; callers decide fatality (render-config: error+exit; run: warn on stderr and continue, matching Father's own warning-and-continue at start_coding.py:240-243).

- [ ] Step 1: Add a failing test to `TestCliV2` (uses a tmp config base so nothing touches the real home dir):

```python
    @patch("model_allocator.adapters.opencode.shutil.which", side_effect=_fake_which)
    def test_run_opencode_refreshes_opencode_json(self, _mock):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as base:
            env = {
                "OPENCODE_ROLES_CONFIG_BASE": base,
                "MODEL_ALLOCATOR_STATE_DIR": base,
            }
            buf = io.StringIO()
            with patch.dict(os.environ, env):
                with contextlib.redirect_stdout(buf):
                    code = cli.main([
                        "--config-dir", str(self.cfg_dir),
                        "run", "--role", "openrouter-test", "--client", "opencode",
                    ])
            self.assertEqual(code, cli.EXIT_OK)
            cfg_path = Path(base) / "openrouter-test" / "opencode.json"
            self.assertTrue(cfg_path.exists())
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(data["model"], "openrouter/qwen3-30b-a3b")
            # stdout purity: exactly one line, the shell string, no --model
            out_lines = [l for l in buf.getvalue().splitlines() if l.strip()]
            self.assertEqual(len(out_lines), 1)
            self.assertNotIn("--model", out_lines[0])
```

- [ ] Step 2: Also update the existing `test_run_openrouter` (lines 394-397) to run inside the same two env overrides (wrap the `cli.main` call in `with patch.dict(os.environ, {"OPENCODE_ROLES_CONFIG_BASE": tmp, "MODEL_ALLOCATOR_STATE_DIR": tmp})` using a `tempfile.TemporaryDirectory()` — otherwise it would now write into the real `~/.config/opencode-roles/openrouter-test/`).

- [ ] Step 3: Run and confirm the new test FAILS (no file written yet).

- [ ] Step 4: In `cli.py`, add the helper directly above `cmd_render_config` (after `_merge_opencode_config`, line 356):

```python
def _opencode_json_path(config_dir: str) -> Path:
    """Filesystem path of a role's opencode.json (shell-expanded)."""
    config_base = os.environ.get("OPENCODE_ROLES_CONFIG_BASE", "$HOME/.config/opencode-roles")
    raw = f"{config_base}/{config_dir}/opencode.json"
    return Path(os.path.expanduser(os.path.expandvars(raw)))


def _write_opencode_json(resolved: dict, output_path: Path) -> None:
    """Render + merge + atomically write opencode.json for an opencode alias."""
    config = opencode.build_opencode_config(resolved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provider_key = resolved.get("opencode_provider_name") or resolved.get("provider", "")
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        config = _merge_opencode_config(existing, config, provider_key)
    _atomic_write_json(output_path, config)
```

`cli.py` has no `Path`-independent import gap (`from pathlib import Path` is already at line 8) but DOES need `os`: add `import os` to the imports block at the top (cli.py currently imports argparse/json/sys — verify and add `import os` after `import json`).

- [ ] Step 5: Rewire `cmd_render_config` (lines 374-389) to use the helper — replace its `if args.output:` block body with:

```python
    if args.output:
        output_path = Path(args.output)
        try:
            _write_opencode_json(resolved, output_path)
        except json.JSONDecodeError as exc:
            print(f"ERROR: existing opencode.json is not valid JSON: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"Config written to {output_path}")
        return EXIT_OK
```

(Behavior identical: same merge, same atomic write, same messages — the existing render-config tests `test_render_config_output_fresh_path` and `test_render_config_output_merges_existing` in tests/test_v2.py:408-457 must still pass unmodified.)

- [ ] Step 6: In `cmd_run` (lines 154-157), extend the opencode branch:

```python
        if args.client == "opencode":
            config_dir = resolved.get("config_dir") or args.role
            command_object = opencode.build_opencode_command(resolved, config_dir)
            # OpenCode honors only opencode.json's model field (and pins the
            # model inside resumed sessions). Refresh the file on EVERY run so
            # a fresh session always starts on the resolved model. Failures
            # warn on stderr and never block the run (stdout stays pure).
            json_path = _opencode_json_path(config_dir)
            try:
                _write_opencode_json(resolved, json_path)
            except Exception as exc:
                print(f"WARNING: failed to refresh opencode.json at {json_path}: {exc}",
                      file=sys.stderr)
```

- [ ] Step 7: Verify:
```bash
python3 -m py_compile src/model_allocator/cli.py && python3 -m pytest 2>&1 | tail -1
```
Expected: `97 passed`.

---

### Task 3: Session-hygiene module — detect and archive stale-model sessions

**Files:**
- Create: `/home/svend/model-allocator/src/model_allocator/opencode_state.py`
- Test: Create `/home/svend/model-allocator/tests/test_opencode_hygiene.py` (pytest style).

**Interfaces:**
- `expected_pair(resolved: dict) -> tuple[str, str]` — (providerID, modelID) the session SHOULD be pinned to, derived from the same mapping as `opencode._model_arg`.
- `stale_sessions(db_path: Path, project_dir: str, provider_id: str, model_id: str) -> list[dict]` — read-only; each dict: `{"id", "title", "provider_id", "model_id", "updated"}`.
- `archive_stale_sessions(db_path: Path, session_ids: list[str], backup_dir: Path) -> dict` — `{"archived": int, "backup": str}`; backup BEFORE write.
- `record_rendered(state_dir, config_dir, provider_id, model_id) -> None` / `last_rendered(state_dir, config_dir) -> dict | None` — allocator-side memory of the last-rendered pair.

- [ ] Step 1: Create `tests/test_opencode_hygiene.py`:

```python
"""Tests for OpenCode session hygiene (PLAN-opencode-session-hygiene)."""

import json
import sqlite3
import time
from pathlib import Path

from model_allocator import opencode_state as ocs


def _make_fake_db(path: Path, rows):
    """Minimal replica of the verified OpenCode 1.17.x schema subset."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, "
        "directory TEXT NOT NULL, title TEXT, model TEXT, "
        "time_updated INTEGER NOT NULL, time_archived INTEGER)")
    now_ms = int(time.time() * 1000)
    for i, (sid, directory, provider, model, archived) in enumerate(rows):
        model_json = json.dumps({"id": model, "providerID": provider}) if model else None
        conn.execute(
            "INSERT INTO session VALUES (?, 'p', ?, ?, ?, ?, ?)",
            (sid, directory, f"t{i}", model_json, now_ms + i,
             now_ms if archived else None))
    conn.commit()
    conn.close()


def test_expected_pair_ollama():
    resolved = {"backend": "ollama", "real_model": "qwen3-coder:30b-256k"}
    assert ocs.expected_pair(resolved) == ("ollama", "qwen3-coder:30b-256k")


def test_expected_pair_llama_cpp_uses_provider_and_model_id():
    resolved = {"backend": "llama_cpp", "provider": "llama-turbo",
                "real_model": "/models/x.gguf", "opencode_model_id": "qwen36-turbo"}
    assert ocs.expected_pair(resolved) == ("llama-turbo", "qwen36-turbo")


def test_stale_sessions_pair_compare_and_filters(tmp_path):
    db = tmp_path / "opencode.db"
    _make_fake_db(db, [
        ("s-ok", "/proj", "ollama", "qwen3-coder:30b-256k", False),   # matches
        ("s-stale", "/proj", "openrouter", "kimi-k2.7", False),        # wrong pair
        ("s-other-dir", "/elsewhere", "openrouter", "kimi-k2.7", False),
        ("s-archived", "/proj", "openrouter", "kimi-k2.7", True),     # already archived
        ("s-no-model", "/proj", "", None, False),                      # never pinned
    ])
    stale = ocs.stale_sessions(db, "/proj", "ollama", "qwen3-coder:30b-256k")
    assert [s["id"] for s in stale] == ["s-stale"]
    assert stale[0]["provider_id"] == "openrouter"


def test_stale_sessions_missing_db_is_empty(tmp_path):
    assert ocs.stale_sessions(tmp_path / "nope.db", "/proj", "ollama", "m") == []


def test_archive_stale_sessions_backs_up_and_archives(tmp_path):
    db = tmp_path / "opencode.db"
    _make_fake_db(db, [("s-stale", "/proj", "openrouter", "kimi", False)])
    backup_dir = tmp_path / "backups"
    result = ocs.archive_stale_sessions(db, ["s-stale"], backup_dir)
    assert result["archived"] == 1
    assert Path(result["backup"]).exists()
    conn = sqlite3.connect(str(db))
    (archived,) = conn.execute(
        "SELECT time_archived FROM session WHERE id='s-stale'").fetchone()
    conn.close()
    assert archived is not None
    # transcript row still present (archived, not deleted)


def test_record_and_last_rendered_roundtrip(tmp_path):
    ocs.record_rendered(tmp_path, "imple01", "ollama", "qwen3-coder:30b-256k")
    state = ocs.last_rendered(tmp_path, "imple01")
    assert state["provider_id"] == "ollama"
    assert state["model_id"] == "qwen3-coder:30b-256k"
    assert ocs.last_rendered(tmp_path, "never-rendered") is None
```

- [ ] Step 2: Run — all fail with `ModuleNotFoundError: model_allocator.opencode_state`.

- [ ] Step 3: Create `src/model_allocator/opencode_state.py`:

```python
"""OpenCode session hygiene (PLAN-opencode-session-hygiene).

On-disk evidence (2026-07-12, OpenCode 1.17.18): session state is a SQLite
database at ``~/.local/share/opencode/opencode.db`` (WAL mode). Relevant
schema subset::

    session(id TEXT PK, project_id, directory TEXT, title, model TEXT,
            time_updated INTEGER, time_archived INTEGER, ...)
    -- model is a JSON string: {"id": "<modelID>", "providerID": "<providerID>"}

Resumed sessions keep their pinned model and IGNORE ``--model`` /
``opencode.json`` — the root cause of the live wrong-model incident
(Kimi/OpenRouter instead of qwen3-coder/Ollama). This module detects
sessions whose pinned (providerID, modelID) pair differs from the resolved
alias, and can archive them (``time_archived``) after a backup — it never
DELETEs transcripts.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path


def opencode_db_path() -> Path:
    """Locate OpenCode's session database (env override > XDG > default)."""
    override = os.environ.get("MODEL_ALLOCATOR_OPENCODE_DB")
    if override:
        return Path(override)
    data_home = os.environ.get("XDG_DATA_HOME", "")
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base / "opencode" / "opencode.db"


def state_dir() -> Path:
    """Allocator state dir — same convention as the llama.cpp adapter."""
    return Path(os.environ.get("MODEL_ALLOCATOR_STATE_DIR", tempfile.gettempdir()))


def expected_pair(resolved: dict) -> tuple[str, str]:
    """(providerID, modelID) a fresh session should be pinned to.

    Mirrors adapters.opencode._model_arg's provider/model mapping, but keeps
    the pair split so comparisons never depend on display names or on
    slash-splitting model ids that themselves contain slashes.
    """
    backend = resolved.get("backend")
    provider = resolved.get("provider", "")
    real_model = resolved.get("real_model", "")
    if backend == "ollama":
        return ("ollama", real_model)
    if backend == "llama_cpp":
        provider_name = resolved.get("opencode_provider_name") or provider or "llama-local"
        model_id = resolved.get("opencode_model_id") or real_model or "model"
        return (provider_name, model_id)
    if backend == "openai_compatible":
        if provider == "openrouter":
            return ("openrouter", real_model)
        # Built-in providers (e.g. minimax): OpenCode stores its own
        # providerID; compare model id only by returning provider as-is.
        return (provider, real_model)
    return (provider, real_model)


def stale_sessions(db_path: Path, project_dir: str,
                   provider_id: str, model_id: str) -> list[dict]:
    """Unarchived sessions in *project_dir* pinned to a DIFFERENT pair.

    Read-only (URI mode=ro). A missing/unreadable DB means OpenCode never
    ran here -> no stale sessions -> empty list, never an exception.
    """
    if not Path(db_path).exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            "SELECT id, title, model, time_updated FROM session "
            "WHERE directory = ? AND time_archived IS NULL AND model IS NOT NULL "
            "ORDER BY time_updated DESC",
            (project_dir,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    stale = []
    for sid, title, model_json, updated in rows:
        try:
            pinned = json.loads(model_json)
        except (TypeError, ValueError):
            continue
        pair = (pinned.get("providerID", ""), pinned.get("id", ""))
        expected = (provider_id, model_id)
        # Built-in providers report their own providerID; when our expected
        # provider is empty, compare the model id only.
        matches = pair == expected if provider_id else pair[1] == model_id
        if not matches:
            stale.append({
                "id": sid,
                "title": title,
                "provider_id": pair[0],
                "model_id": pair[1],
                "updated": updated,
            })
    return stale


def archive_stale_sessions(db_path: Path, session_ids: list[str],
                           backup_dir: Path) -> dict:
    """Archive sessions (set time_archived) after backing up the DB file.

    Backup uses sqlite3's backup API so the WAL is checkpointed into the
    copy (a bare file copy of a live WAL database can be inconsistent).
    """
    if not session_ids:
        return {"archived": 0, "backup": None}
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"opencode-db-backup-{stamp}.db"

    src = sqlite3.connect(str(db_path), timeout=5)
    try:
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
        src.execute("PRAGMA busy_timeout = 5000")
        now_ms = int(time.time() * 1000)
        placeholders = ",".join("?" for _ in session_ids)
        cur = src.execute(
            f"UPDATE session SET time_archived = ? WHERE id IN ({placeholders}) "
            "AND time_archived IS NULL",
            [now_ms, *session_ids],
        )
        src.commit()
        return {"archived": cur.rowcount, "backup": str(backup_path)}
    finally:
        src.close()


def _state_file(base_dir: Path, config_dir: str) -> Path:
    return Path(base_dir) / f"model-allocator-opencode-{config_dir}.json"


def record_rendered(base_dir: Path, config_dir: str,
                    provider_id: str, model_id: str) -> None:
    path = _state_file(base_dir, config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider_id": provider_id,
        "model_id": model_id,
        "rendered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def last_rendered(base_dir: Path, config_dir: str) -> dict | None:
    path = _state_file(base_dir, config_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
```

- [ ] Step 4: Verify:
```bash
python3 -m py_compile src/model_allocator/opencode_state.py && python3 -m pytest tests/test_opencode_hygiene.py -v && python3 -m pytest 2>&1 | tail -1
```
Expected: 6/6 new; full suite `103 passed`.

---

### Task 4: Wire drift warning into `run` + new `opencode-clean` subcommand

**Files:**
- Test: append to `/home/svend/model-allocator/tests/test_opencode_hygiene.py`.
- Modify: `/home/svend/model-allocator/src/model_allocator/cli.py` — `cmd_run` opencode branch (extended in Task 2), new `cmd_opencode_clean`, parser additions in `build_parser`.

**Interfaces:**
- `run` (opencode): after the opencode.json refresh, computes `expected_pair(resolved)`, warns on stderr when `stale_sessions(...)` for the project dir (`--project-dir`, default `os.getcwd()`) is non-empty or when the pair changed since `last_rendered`, then `record_rendered(...)`. Never changes stdout or exit code.
- `opencode-clean --role R --client opencode [--project-dir DIR] [--archive]`: dry-run by default (list stale, exit 2 if any); `--archive` backs up + archives (exit 0). JSON report on stdout.

- [ ] Step 1: Add failing CLI tests to `tests/test_opencode_hygiene.py`:

```python
from unittest.mock import patch

import yaml

from model_allocator.cli import main


def _seed_cfg(tmp_path: Path) -> Path:
    (tmp_path / "models.yaml").write_text(yaml.safe_dump({"models": {
        "imple01-local": {
            "runtime_profile": "local_ollama",
            "real_model": "qwen3-coder:30b-256k",
            "context": 131072,
            "clients": {"opencode": True},
        }}}), encoding="utf-8")
    (tmp_path / "runtime_profiles.yaml").write_text(yaml.safe_dump({"runtime_profiles": {
        "local_ollama": {"backend": "ollama",
                         "default_api_base": "http://127.0.0.1:11434"}}}), encoding="utf-8")
    (tmp_path / "roles.yaml").write_text(yaml.safe_dump({"roles": {
        "imple01": {"default_alias": "imple01-local", "config_dir": "imple01",
                    "client_aliases": {"opencode": "imple01-local"}}}}), encoding="utf-8")
    return tmp_path


def test_run_warns_on_stale_session(tmp_path, capsys, monkeypatch):
    (tmp_path / "cfg").mkdir()
    cfg = _seed_cfg(tmp_path / "cfg")
    db = tmp_path / "opencode.db"
    _make_fake_db(db, [("s-stale", str(tmp_path / "proj"), "openrouter", "kimi", False)])
    monkeypatch.setenv("MODEL_ALLOCATOR_OPENCODE_DB", str(db))
    monkeypatch.setenv("MODEL_ALLOCATOR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("OPENCODE_ROLES_CONFIG_BASE", str(tmp_path / "ocroles"))
    with patch("model_allocator.adapters.opencode.shutil.which",
               return_value="/usr/bin/opencode"):
        rc = main(["--config-dir", str(cfg), "run", "--role", "imple01",
                   "--client", "opencode", "--project-dir", str(tmp_path / "proj")])
    captured = capsys.readouterr()
    assert rc == 0
    assert "--model" not in captured.out
    assert "stale OpenCode session" in captured.err
    assert "kimi" in captured.err


def test_opencode_clean_dry_run_and_archive(tmp_path, capsys, monkeypatch):
    (tmp_path / "cfg").mkdir()
    cfg = _seed_cfg(tmp_path / "cfg")
    db = tmp_path / "opencode.db"
    _make_fake_db(db, [("s-stale", str(tmp_path / "proj"), "openrouter", "kimi", False)])
    monkeypatch.setenv("MODEL_ALLOCATOR_OPENCODE_DB", str(db))
    monkeypatch.setenv("MODEL_ALLOCATOR_STATE_DIR", str(tmp_path / "state"))

    rc = main(["--config-dir", str(cfg), "opencode-clean", "--role", "imple01",
               "--project-dir", str(tmp_path / "proj")])
    report = json.loads(capsys.readouterr().out)
    assert rc == 2                       # stale found, dry-run
    assert report["stale"][0]["id"] == "s-stale"
    assert report["archived"] == 0

    rc = main(["--config-dir", str(cfg), "opencode-clean", "--role", "imple01",
               "--project-dir", str(tmp_path / "proj"), "--archive"])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert report["archived"] == 1
    assert Path(report["backup"]).exists()
```

- [ ] Step 2: Run — failures: unknown `--project-dir`, unknown command `opencode-clean`.

- [ ] Step 3: Wire into `cmd_run`. Extend the opencode branch from Task 2 (after the `_write_opencode_json` try/except) with:

```python
            from model_allocator import opencode_state as ocs
            provider_id, model_id = ocs.expected_pair(resolved)
            project_dir = getattr(args, "project_dir", None) or os.getcwd()
            stale = ocs.stale_sessions(ocs.opencode_db_path(), project_dir,
                                       provider_id, model_id)
            if stale:
                newest = stale[0]
                print(
                    f"WARNING: {len(stale)} stale OpenCode session(s) in "
                    f"{project_dir} pinned to {newest['provider_id']}/"
                    f"{newest['model_id']} (expected {provider_id}/{model_id}). "
                    "Resuming one will IGNORE the configured model — start a "
                    "fresh session, or run: model-allocator opencode-clean "
                    f"--role {args.role} --project-dir {project_dir} --archive",
                    file=sys.stderr,
                )
            ocs.record_rendered(ocs.state_dir(), config_dir, provider_id, model_id)
```

- [ ] Step 4: Add `cmd_opencode_clean` to `cli.py` (place after `cmd_run`):

```python
def cmd_opencode_clean(args: argparse.Namespace) -> int:
    """List or archive OpenCode sessions pinned to an outdated model."""
    from model_allocator import opencode_state as ocs

    resolver = Resolver(config_dir=_config_dir(args))
    try:
        resolved = resolver.resolve_role_client(args.role, args.client)
    except ResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    provider_id, model_id = ocs.expected_pair(resolved)
    project_dir = args.project_dir or os.getcwd()
    db_path = ocs.opencode_db_path()
    stale = ocs.stale_sessions(db_path, project_dir, provider_id, model_id)

    report = {
        "project_dir": project_dir,
        "expected": {"provider_id": provider_id, "model_id": model_id},
        "stale": stale,
        "archived": 0,
        "backup": None,
    }
    if stale and args.archive:
        result = ocs.archive_stale_sessions(
            db_path, [s["id"] for s in stale], ocs.state_dir())
        report["archived"] = result["archived"]
        report["backup"] = result["backup"]
    print(json.dumps(report, indent=2, default=str))
    if stale and not args.archive:
        return EXIT_WARNING
    return EXIT_OK
```

- [ ] Step 5: Parser wiring in `build_parser`. Add to `p_run` (after its `--client` argument, cli.py:568):

```python
    p_run.add_argument(
        "--project-dir",
        default=None,
        help="Project directory whose OpenCode sessions are checked for model drift (default: cwd)",
    )
```

and register the new subcommand (place after the `p_run` block):

```python
    p_clean = sub.add_parser(
        "opencode-clean",
        help="List/archive OpenCode sessions pinned to a model that differs from the resolved alias",
    )
    p_clean.add_argument("--role", required=True, help="Role key")
    p_clean.add_argument("--client", default="opencode", help="Client key (default opencode)")
    p_clean.add_argument("--project-dir", default=None, help="Project directory (default: cwd)")
    p_clean.add_argument("--archive", action="store_true",
                         help="Archive stale sessions (time_archived) after a DB backup; default is dry-run")
    p_clean.set_defaults(func=cmd_opencode_clean)
```

- [ ] Step 6: Verify:
```bash
python3 -m py_compile src/model_allocator/cli.py && python3 -m pytest 2>&1 | tail -1
```
Expected: `105 passed`. Then stage and STOP — await Human commit approval:
```bash
git add src/model_allocator/adapters/opencode.py src/model_allocator/cli.py src/model_allocator/opencode_state.py tests/test_v2.py tests/test_opencode_hygiene.py
git status --short
```
Suggested commit message: `[V5.2] OpenCode session hygiene — drop dead --model, auto render-config on run, stale-session detect/archive (opencode-clean)`

---

### Task 5: Manual live-check checklist (perform with the Human, after commit)

**Files:** none. This is verification, not code.

- [ ] Step 1: Pick the allocator pilot role (imple01, `model_source = model_allocator`, alias `imple01-local` → `qwen3-coder:30b-256k` on Ollama).
- [ ] Step 2: Dry-run the hygiene check first:
```bash
cd /home/svend/DPMtF-WebUI && /home/svend/model-allocator/scripts/model-allocator opencode-clean --role imple01 --project-dir /home/svend/DPMtF-WebUI
```
Expected: JSON report; exit 2 if stale sessions exist (then re-run with `--archive`), exit 0 otherwise.
- [ ] Step 3: Start the role via Father exactly as production does (kill+recreate sessions first — stale tmux sessions cause token overflow):
```bash
cd /home/svend/DPMtF-WebUI && python3 scripts/bridgeV002/start_coding.py strict_review
```
Expected console: `Regenerated opencode.json at /home/svend/.config/opencode-roles/imple01/opencode.json` and `(model_allocator) ... Command sent to session.`
- [ ] Step 4: In the tmux session, ask the model: `Reply with exactly the model name and provider you are running as.` Confirm it identifies as the configured Ollama qwen3-coder model, NOT a cloud model.
- [ ] Step 5: Cross-check server-side: `curl -s http://127.0.0.1:11434/api/ps | python3 -m json.tool | grep name` → shows `qwen3-coder:30b-256k` loaded.
- [ ] Step 6: Confirm `cat /home/svend/.config/opencode-roles/imple01/opencode.json | python3 -c "import json,sys; print(json.load(sys.stdin)['model'])"` → `ollama/qwen3-coder:30b-256k`.

---

### Task 6: Cross-repo notes (Father) — verified, mostly no-op

> **Cross-repo notes for `/home/svend/DPMtF-WebUI` — any change there requires Human approval; only the Human commits.**

- [ ] Step 1: **No Father change is required for Task 1** (verified 2026-07-12): `start_coding.py` does NOT pass `--model` to the allocator — it calls `run --role X --client Y` (lines 245-257) and injects the returned string opaquely (lines 258-270). After Task 1 that string simply no longer contains `--model`.
- [ ] Step 2: Father's own `render-config --output` call (start_coding.py:220-243) becomes redundant once `run` auto-refreshes, but keep it: it is harmless (idempotent merge) and keeps Father working against an older allocator. Record as optional cleanup for a later Father commit.
- [ ] Step 3: **Do NOT remove `--model` from Father's `command_builder.py` opencode builders** (lines 204-284): the DIRECT-role path in start_coding.py:326-328 extracts the model string from the argv element after `"--model"` to write opencode.json (V2.3 mechanism). Removing the flag there without reworking that extraction breaks direct roles. Out of scope; note it in the handoff so nobody "cleans it up" casually.

## Acceptance Criteria

1. `cd /home/svend/model-allocator && python3 -m pytest` → `105 passed`, 0 failures.
2. `python3 -m py_compile src/model_allocator/cli.py src/model_allocator/adapters/opencode.py src/model_allocator/opencode_state.py` → exit 0.
3. `OPENCODE_ROLES_CONFIG_BASE=/tmp/oc-test MODEL_ALLOCATOR_STATE_DIR=/tmp/oc-test ./scripts/model-allocator run --role imple01 --client opencode` → single stdout line containing `OPENCODE_CONFIG=` and the opencode binary, containing NO `--model`; `/tmp/oc-test/imple01/opencode.json` exists with `"model": "ollama/qwen3-coder:30b-256k"`.
4. `./scripts/model-allocator opencode-clean --role imple01 --project-dir /home/svend/DPMtF-WebUI; echo rc=$?` → JSON report on stdout; `rc=0` when no stale sessions, `rc=2` when stale sessions are listed (dry-run never writes).
5. With `--archive` and stale sessions present: report shows `"archived": N >= 1` and a `"backup"` path that exists; `sqlite3 ~/.local/share/opencode/opencode.db "SELECT COUNT(*) FROM session"` is UNCHANGED from before (rows archived, never deleted).
6. Grep proof the dead flag is gone from the adapter: `grep -n '"--model"' src/model_allocator/adapters/opencode.py` → no matches.
7. Manual live check (Task 5) performed once with the Human: role session self-identifies as the configured Ollama model.
8. `git status --short` shows exactly the five files from Task 4 Step 6 staged; nothing committed.
