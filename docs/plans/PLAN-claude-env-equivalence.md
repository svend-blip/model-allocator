# Claude Code Env Equivalence Implementation Plan

> **For agentic workers:** Execute tasks in order. Steps use checkbox (`- [ ]`) syntax for tracking. Do not skip verification steps.

**Goal:** Make `build_claude_code_command` emit the full env/argv surface that Father's `command_builder.py` produces, so allocator-launched Claude Code roles are drop-in equivalent to Machine-Profile-launched roles.

**Architecture:** The allocator resolves an alias (models.yaml + runtime_profiles.yaml merge via `Resolver.resolve_alias`) into a `resolved` dict; `adapters/claude_code.py:build_claude_code_command(resolved)` turns it into `{"env": {...}, "argv": [...]}`. This plan adds four config-sourced fields (`max_output_tokens`, `disable_adaptive_thinking`, `claude_binary`, `claude_extra_args`), one always-set env var (`MODEL_ALLOCATOR_ACTIVE_MODEL`), and a `--max-output-tokens` CLI passthrough so Father can inject its per-role DB value without editing YAML.

**Tech Stack:** Python 3.10+ stdlib, pyyaml (only runtime dep), unittest (tests/test_v2.py conventions), argparse CLI.

## Cold-Start Context

- model-allocator is a standalone Python CLI at `/home/svend/model-allocator` that resolves logical model aliases into runtime commands for ollama / llama.cpp / opencode / claude-code / onyx backends.
- Config lives at the repo root: `models.yaml` (aliases), `runtime_profiles.yaml` (backends), `roles.yaml` (role → alias mapping). `src/model_allocator/config_loader.py` resolves `${ENV_VAR}` references at load time.
- Source layout: `src/model_allocator/` (cli.py, resolver.py, validator.py, renderer.py, config_loader.py, config_writer.py) and `src/model_allocator/adapters/` (claude_code.py, opencode.py, llama_cpp.py, ollama.py, openai_compatible.py, onyx.py, headless.py).
- The Father project (`/home/svend/DPMtF-WebUI`) consumes this repo by shelling out to `/home/svend/model-allocator/scripts/model-allocator` (a bash wrapper that runs `python3 -m model_allocator`): from `routers/bridge.py` (`/allocator/*` endpoints), `scripts/bridgeV002/start_coding.py` (run/render-config), and `scripts/bridgeV002/dispatch.py` (stop).
- Run tests: `cd /home/svend/model-allocator && python3 -m pytest` → **95 passed** in ~5s (baseline before this plan).
- Entry point: `model-allocator run --role <role> --client claude-code` prints a tmux-safe shell string (env assignments + argv, rendered by `src/model_allocator/renderer.py:render_tmux_shell_string`).

## Global Constraints

- `python3 -m py_compile <file>` MUST pass on every touched Python file before a task is complete.
- Single runtime dependency stays `pyyaml` (`mcp` is an optional extra). NO new dependencies — adding one requires explicit Human approval and is out of scope here.
- All 95 existing tests stay green: `python3 -m pytest` must show `>= 95 passed, 0 failed` after every task.
- TDD: write the failing test first, run it to see it fail, implement, run again to see it pass. Claude Code adapter tests live in `tests/test_v2.py` class `TestClaudeCodeAdapter` (unittest style, `@patch` on `shutil.which`).
- Backwards compatibility: existing CLI output formats consumed by Father keep working. New env vars in `run` output are additive (Father treats the string as opaque); existing keys/argv order for existing configs must not change except as specified.
- Git policy: the Human approves commits. Never run `git commit` or `git push`. Tasks end with `git add <files>` and STOP.

## Edge Cases a Weaker Model Would Miss

1. **Env values must be strings.** `max_output_tokens` arrives as YAML int (`32768`). `renderer.render_tmux_shell_string` calls `str(value)` so an int would render, but the command-object contract (and Father's `command_builder.py` which uses string values like `"32768"`) is `dict[str, str]`. Always `str()` at the adapter boundary.
2. **Absent optional fields must OMIT the env var entirely.** `CLAUDE_CODE_MAX_OUTPUT_TOKENS=''` (empty string) breaks Claude Code (it parses the value as a number). Never emit an empty-string value for the new vars. Note the contrast: `ANTHROPIC_API_KEY=""` for cloud backends is *deliberately* empty (prevents direct-Anthropic fallback) — do not "fix" that.
3. **`disable_adaptive_thinking: false` in YAML must NOT emit the var.** PyYAML gives you Python `False`; `if "disable_adaptive_thinking" in resolved:` would emit `...=1` behavior wrongly if you only test key presence. Test truthiness, and emit the literal string `"1"` when truthy (matching `docs/governance-templates-v2/11_SCOPE.md` §13.2 example `export CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1`).
4. **`claude_extra_args` ordering matters.** Father's `command_builder.py:155` builds `[claude_bin, *runtime_cfg.get("extra_args", []), "--model", model]` — extra args (`--bare` on the live machine profile) come BEFORE `--model`. Mirror that exact order.
5. **Resolver merge precedence is alias > profile.** `Resolver.resolve_alias` (src/model_allocator/resolver.py:25-55) copies every alias field, then backfills profile fields not already set. So `max_output_tokens` on an alias overrides the same key on its profile automatically — do NOT add special merge code.
6. **The CLI override must win over YAML.** Father's `start_coding.py` applies `extra_env` LAST ("last assignment wins", command_builder.py:64-68). The allocator equivalent: `--max-output-tokens` overwrites `resolved["max_output_tokens"]` after resolution, before the adapter builds env.
7. **`claude_binary` may contain `${HOME}`.** `config_loader.resolve_env` resolves `${...}` at load time, so the adapter receives an absolute path. Validate it with the same isfile+X_OK check already used for `CLAUDE_BIN` (claude_code.py:35-42). Precedence must be: explicit `CLAUDE_BIN` env var > `claude_binary` config field > bare `"claude"` on PATH — the env var is the operator's runtime escape hatch.
8. **`env` subcommand shares the adapter.** `cmd_env` (cli.py:308-325) prints `export K=V` lines from the same command object — new env vars appear there automatically. Don't duplicate logic; just extend the adapter.
9. **`MODEL_ALLOCATOR_ACTIVE_MODEL` carries the alias name, not the real model.** SCOPE §13.2's example shows a model-looking value, but the useful invariant for debugging is "which allocator alias launched this session". Use `resolved["alias"]`; when `resolved` has no alias key (direct adapter unit-test dicts), omit the var rather than emitting an empty string.

---

### Task 1: Diff audit — command_builder.py vs claude_code.py (COMPLETED — results below)

**Files:** none modified (audit only). Sources read: `/home/svend/DPMtF-WebUI/scripts/bridgeV002/command_builder.py` (claude builders, lines 125-201), `/home/svend/DPMtF-WebUI/scripts/bridgeV002/start_coding.py` (extra_env, lines 294-305), `/home/svend/DPMtF-WebUI/profiles/machine.local.json` (live values), `/home/svend/model-allocator/src/model_allocator/adapters/claude_code.py`.

The audit has already been performed against the live sources (2026-07-12). The table below is the authoritative gap list — implement exactly these gaps, nothing more.

| # | Element | Father `command_builder.py` (claude builders) | Allocator `claude_code.py` | Gap? |
|---|---------|----------------------------------------------|-----------------------------|------|
| 1 | argv[0] claude binary | Absolute path from Machine Profile `binaries.claude` = `/home/svend/.local/bin/claude`, validated isfile+X_OK (`_resolve_binary`, lines 89-105) | `CLAUDE_BIN` env or `shutil.which("claude")` (lines 34-42) — fails if claude not on tmux PATH | **YES — add `claude_binary` config field** |
| 2 | argv extra args | `*runtime_cfg.get("extra_args", [])` before `--model` (lines 155, 200); live value `["--bare"]` (machine.local.json:32-34) | Not supported | **YES — add `claude_extra_args` config field** |
| 3 | argv `--model <model>` | Yes (lines 155, 200) | Yes (line 68) | No |
| 4 | env `ANTHROPIC_BASE_URL` | Provider `endpoint` (lines 138, 150 / 178, 192) | `api_base_env` env var or `default_api_base` (lines 46-48, 57-62) | No (equivalent) |
| 5 | env `ANTHROPIC_AUTH_TOKEN` | `"ollama"` (line 151) or `"$OPENROUTER_API_KEY"` shell ref (line 193) | `"ollama"` (line 49) or `f"${api_key_env}"` (line 63) | No (equivalent) |
| 6 | env `ANTHROPIC_API_KEY=""` | OpenRouter builder only (line 196) | Cloud branch only (line 64) | No (equivalent) |
| 7 | env runtime `default_env` | `dict(runtime_cfg.get("default_env", {}))` (lines 148, 190); live value `{}`, example profiles set `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | Not supported | Covered by gap 9 (max_output_tokens as first-class field) |
| 8 | env provider `env` | `env.update(provider_cfg.get("env", {}))` (lines 149, 191); live `providers.local_ollama.env` = `{"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32768"}` (machine.local.json:50-52) | Not supported | Covered by gap 9 |
| 9 | env `CLAUDE_CODE_MAX_OUTPUT_TOKENS` per-role | `start_coding.py:301-304` passes `extra_env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(role["max_output_tokens"])}` from `bridge_roles.max_output_tokens` (migration `scripts/db/004_role_runtime_config.sql`), applied LAST (command_builder.py:64-68) | Not supported | **YES — `max_output_tokens` config field + `--max-output-tokens` CLI override** |
| 10 | env `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING` | Settable via profile/provider env dicts (mechanism exists); required by allocator SCOPE §13.2 (docs/governance-templates-v2/11_SCOPE.md:654) | Not supported | **YES — `disable_adaptive_thinking` config field** |
| 11 | env `MODEL_ALLOCATOR_ACTIVE_MODEL` | Not set by command_builder; required by allocator SCOPE §13.2 (11_SCOPE.md:655) | Not supported | **YES — always emit `MODEL_ALLOCATOR_ACTIVE_MODEL=<alias>`** |
| 12 | OpenRouter API key presence check at build time | `command_builder.py:183-188` raises if `$OPENROUTER_API_KEY` unset | Not checked (only endpoint checked, line 61) | Out of scope here — `validator._validate_openai_compatible` already warns on missing key; do not add a hard failure |

- [x] Step 1: Audit performed; table above is the result. No code changes in this task.

---

### Task 2: TDD — extend `build_claude_code_command` with the four fields + MODEL_ALLOCATOR_ACTIVE_MODEL

**Files:**
- Test: `/home/svend/model-allocator/tests/test_v2.py` — add tests inside class `TestClaudeCodeAdapter` (after `test_rejects_minimax`, line 146-153).
- Modify: `/home/svend/model-allocator/src/model_allocator/adapters/claude_code.py` (whole file is 69 lines; changes in lines 34-68).

**Interfaces:**
- Consumes: `resolved: dict` from `Resolver.resolve_alias` / `resolve_role_client`. New optional keys read: `max_output_tokens` (int|str), `disable_adaptive_thinking` (bool), `claude_binary` (str), `claude_extra_args` (list[str]), plus existing `alias` (str).
- Produces: `{"env": dict[str, str], "argv": list[str]}` — signature `build_claude_code_command(resolved: dict) -> dict[str, Any]` unchanged.

- [ ] Step 1: Add failing tests to `tests/test_v2.py`, inside `TestClaudeCodeAdapter` (append after `test_rejects_minimax`):

```python
    @patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which)
    def test_equivalence_fields_emitted(self, _mock):
        resolved = {
            "alias": "imple01-claude",
            "backend": "ollama",
            "real_model": "qwen3-coder:30b-256k",
            "api_base_env": "OLLAMA_BASE_URL",
            "default_api_base": "http://127.0.0.1:11434",
            "max_output_tokens": 32768,
            "disable_adaptive_thinking": True,
            "claude_extra_args": ["--bare"],
        }
        cmd = claude_code.build_claude_code_command(resolved)
        # env values are strings; int max_output_tokens is stringified
        self.assertEqual(cmd["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "32768")
        self.assertEqual(cmd["env"]["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"], "1")
        self.assertEqual(cmd["env"]["MODEL_ALLOCATOR_ACTIVE_MODEL"], "imple01-claude")
        # extra args come BEFORE --model, mirroring command_builder.py:155
        self.assertEqual(
            cmd["argv"],
            ["/usr/bin/claude", "--bare", "--model", "qwen3-coder:30b-256k"],
        )

    @patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which)
    def test_absent_optional_fields_omit_env_vars(self, _mock):
        resolved = {
            "alias": "imple01-claude",
            "backend": "ollama",
            "real_model": "qwen3-coder:30b-256k",
            "default_api_base": "http://127.0.0.1:11434",
            "disable_adaptive_thinking": False,
        }
        cmd = claude_code.build_claude_code_command(resolved)
        self.assertNotIn("CLAUDE_CODE_MAX_OUTPUT_TOKENS", cmd["env"])
        # False must NOT emit the var (YAML false -> Python False)
        self.assertNotIn("CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING", cmd["env"])
        self.assertEqual(cmd["argv"], ["/usr/bin/claude", "--model", "qwen3-coder:30b-256k"])

    def test_no_alias_omits_active_model_var(self):
        resolved = {
            "backend": "ollama",
            "real_model": "qwen3-coder:30b-256k",
            "default_api_base": "http://127.0.0.1:11434",
        }
        with patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which):
            cmd = claude_code.build_claude_code_command(resolved)
        self.assertNotIn("MODEL_ALLOCATOR_ACTIVE_MODEL", cmd["env"])

    def test_claude_binary_config_field(self):
        resolved = {
            "alias": "imple01-claude",
            "backend": "ollama",
            "real_model": "qwen3-coder:30b-256k",
            "default_api_base": "http://127.0.0.1:11434",
            "claude_binary": "/opt/claude/bin/claude",
        }
        with patch("model_allocator.adapters.claude_code.os.path.isfile", return_value=True), \
             patch("model_allocator.adapters.claude_code.os.access", return_value=True):
            cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(cmd["argv"][0], "/opt/claude/bin/claude")

    def test_claude_bin_env_overrides_config_field(self):
        resolved = {
            "alias": "imple01-claude",
            "backend": "ollama",
            "real_model": "qwen3-coder:30b-256k",
            "default_api_base": "http://127.0.0.1:11434",
            "claude_binary": "/opt/claude/bin/claude",
        }
        with patch.dict(os.environ, {"CLAUDE_BIN": "/env/claude"}), \
             patch("model_allocator.adapters.claude_code.os.path.isfile", return_value=True), \
             patch("model_allocator.adapters.claude_code.os.access", return_value=True):
            cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(cmd["argv"][0], "/env/claude")
```

- [ ] Step 2: Run the new tests and confirm they FAIL:
```bash
cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py -k "equivalence_fields or omit_env_vars or active_model_var or claude_binary or claude_bin_env" -v
```
Expected: 5 failures (`KeyError: 'CLAUDE_CODE_MAX_OUTPUT_TOKENS'`, argv mismatches, etc.).

- [ ] Step 3: Implement in `src/model_allocator/adapters/claude_code.py`. Replace lines 34-42 (binary resolution) with:

```python
    claude_bin = os.environ.get("CLAUDE_BIN", "")
    if not claude_bin:
        claude_bin = resolved.get("claude_binary", "") or "claude"
    if os.path.isabs(claude_bin):
        if not (os.path.isfile(claude_bin) and os.access(claude_bin, os.X_OK)):
            raise ValueError(f"Claude binary not found: {claude_bin}")
    else:
        resolved_bin = shutil.which(claude_bin)
        if resolved_bin is None:
            raise ValueError(f"Claude binary not found on PATH: {claude_bin}")
        claude_bin = resolved_bin
```

- [ ] Step 4: In the same file, replace the final return block (lines 66-69) with:

```python
    # Equivalence with Father's command_builder.py + SCOPE section 13.2:
    # optional per-alias/per-profile env fields. Absent fields OMIT the
    # env var entirely (empty-string values break Claude Code).
    max_output_tokens = resolved.get("max_output_tokens")
    if max_output_tokens not in (None, ""):
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    if resolved.get("disable_adaptive_thinking"):
        env["CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING"] = "1"
    alias_name = resolved.get("alias", "")
    if alias_name:
        env["MODEL_ALLOCATOR_ACTIVE_MODEL"] = alias_name

    extra_args = [str(a) for a in (resolved.get("claude_extra_args") or [])]

    return {
        "env": env,
        "argv": [claude_bin, *extra_args, "--model", real_model],
    }
```

- [ ] Step 5: Update the module docstring of `build_claude_code_command` (lines 11-20) — append to the docstring env list:

```python
             CLAUDE_CODE_MAX_OUTPUT_TOKENS  = str(max_output_tokens)   (if configured)
             CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING = "1"               (if configured truthy)
             MODEL_ALLOCATOR_ACTIVE_MODEL   = <alias name>             (if alias known)
      - argv: [claude_bin, *claude_extra_args, "--model", <real_model>]
```

- [ ] Step 6: Verify:
```bash
cd /home/svend/model-allocator && python3 -m py_compile src/model_allocator/adapters/claude_code.py && python3 -m pytest tests/test_v2.py -v 2>&1 | tail -5
```
Expected: all tests pass (previous TestClaudeCodeAdapter tests `test_ollama_backend`, `test_openrouter_backend`, `test_rejects_minimax` unchanged and green — they don't set the new fields, so output for them is unchanged apart from nothing: they pass no `alias`, so `MODEL_ALLOCATOR_ACTIVE_MODEL` is omitted and existing assertions hold).

- [ ] Step 7: Full suite: `python3 -m pytest` — expected `100 passed` (95 + 5 new).

---

### Task 3: `--max-output-tokens` CLI passthrough on `run` and `env`

**Files:**
- Test: `/home/svend/model-allocator/tests/test_v2.py` — class `TestCliV2` (starts line 321).
- Modify: `/home/svend/model-allocator/src/model_allocator/cli.py` — `cmd_run` (lines 140-171), `cmd_env` (lines 308-325), parser (`p_run` lines 566-569, `p_env` lines 592-595).

**Interfaces:**
- Consumes: `args.max_output_tokens: int | None` (argparse).
- Produces: overrides `resolved["max_output_tokens"]` after resolution, before adapter dispatch — so it wins over YAML (same "last assignment wins" semantics as Father's `extra_env`).

- [ ] Step 1: Add a failing test to `TestCliV2` in `tests/test_v2.py` (append after `test_run_claude`, line 389-392). It captures stdout the same way `test_config_show_prints_json` in tests/test_config_cli.py does, but TestCliV2 is unittest-based, so use `contextlib.redirect_stdout`:

```python
    @patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which)
    def test_run_claude_max_output_tokens_override(self, _mock):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main([
                "--config-dir", str(self.cfg_dir),
                "run", "--role", "claude-test", "--client", "claude-code",
                "--max-output-tokens", "81920",
            ])
        self.assertEqual(code, cli.EXIT_OK)
        out = buf.getvalue()
        self.assertIn("CLAUDE_CODE_MAX_OUTPUT_TOKENS=81920", out)
        self.assertIn("MODEL_ALLOCATOR_ACTIVE_MODEL=imple01-claude", out)
```

- [ ] Step 2: Run and confirm failure:
```bash
python3 -m pytest tests/test_v2.py -k max_output_tokens_override -v
```
Expected: `SystemExit: 2` (argparse: unrecognized arguments: --max-output-tokens).

- [ ] Step 3: In `cli.py`, add the flag to both subparsers. After `p_run.add_argument("--client", ...)` (line 568) insert:

```python
    p_run.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override CLAUDE_CODE_MAX_OUTPUT_TOKENS for this run (wins over YAML)",
    )
```

and after `p_env.add_argument("--client", ...)` (line 594) insert:

```python
    p_env.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override CLAUDE_CODE_MAX_OUTPUT_TOKENS (wins over YAML)",
    )
```

- [ ] Step 4: In `cmd_run` (cli.py:140-171), immediately after the `resolved = resolver.resolve_role_client(...)` try/except block (after line 147), insert:

```python
    if getattr(args, "max_output_tokens", None):
        # CLI override wins over YAML (mirrors Father extra_env "last wins").
        resolved["max_output_tokens"] = args.max_output_tokens
```

- [ ] Step 5: In `cmd_env` (cli.py:308-325), insert the identical 3-line block after its `resolved = resolver.resolve_role_client(...)` try/except (after line 315).

- [ ] Step 6: Verify:
```bash
python3 -m py_compile src/model_allocator/cli.py && python3 -m pytest tests/test_v2.py -v 2>&1 | tail -3 && python3 -m pytest 2>&1 | tail -1
```
Expected: full suite `101 passed`.

---

### Task 4: Document the YAML fields in live + example config

**Files:**
- Modify: `/home/svend/model-allocator/models.yaml` (alias `imple01-claude`, lines 20-27), `/home/svend/model-allocator/runtime_profiles.yaml` (profile `local_ollama_cuda0`, lines 2-6), `/home/svend/model-allocator/models.example.yaml`, `/home/svend/model-allocator/runtime_profiles.example.yaml`.

These are data-only edits; the resolver's generic merge (resolver.py:47-54) already carries any alias/profile field through — no code change needed.

- [ ] Step 1: In `runtime_profiles.yaml`, extend `local_ollama_cuda0` (keep existing keys, append the new ones):

```yaml
  local_ollama_cuda0:
    backend: ollama
    api_base_env: OLLAMA_BASE_URL
    default_api_base: http://127.0.0.1:11434
    gpu: cuda0
    # Claude Code equivalence (PLAN-claude-env-equivalence):
    # absolute binary path (env-ref, resolved by config_loader) + tmux-session
    # extra args matching Father machine.local.json runtimes.claude.extra_args.
    claude_binary: ${HOME}/.local/bin/claude
    claude_extra_args: ["--bare"]
    max_output_tokens: 32768
```

(The live Father Machine Profile sets `binaries.claude = /home/svend/.local/bin/claude`, `extra_args = ["--bare"]`, and `providers.local_ollama.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "32768"` — these values mirror it without hardcoding `/home/svend` thanks to the `${HOME}` env reference.)

- [ ] Step 2: In `models.yaml`, extend alias `imple01-claude` to demonstrate an alias-level override (alias wins over profile):

```yaml
  imple01-claude:
    runtime_profile: local_ollama_cuda0
    real_model: qwen3-coder:30b-256k
    context: 131072
    lifecycle_policy: stop_after_step
    max_output_tokens: 32768
    disable_adaptive_thinking: false
    clients:
      opencode: false
      claude-code: true
```

- [ ] Step 3: Mirror both blocks into `models.example.yaml` and `runtime_profiles.example.yaml` (same keys, add a one-line comment `# optional Claude Code equivalence fields` above them).

- [ ] Step 4: Verify config still loads and the run command reflects it:
```bash
cd /home/svend/model-allocator && ./scripts/model-allocator run --role claude-test --client claude-code
```
Expected output (single line, order of env vars: ANTHROPIC first, then the new vars): contains `ANTHROPIC_BASE_URL=http://127.0.0.1:11434`, `ANTHROPIC_AUTH_TOKEN=ollama`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=32768`, `MODEL_ALLOCATOR_ACTIVE_MODEL=imple01-claude`, and argv ending `/.local/bin/claude --bare --model qwen3-coder:30b-256k`. `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING` must be ABSENT (configured `false`).

- [ ] Step 5: `python3 -m pytest 2>&1 | tail -1` — still `101 passed` (tests use their own tmp configs, not the repo YAML).

---

### Task 5: Version + USER_AGENT hygiene

**Files:**
- Modify: `/home/svend/model-allocator/pyproject.toml` (line 3: `version = "0.1.0"`), `/home/svend/model-allocator/src/model_allocator/adapters/onyx.py` (line 42), `/home/svend/model-allocator/src/model_allocator/adapters/openai_compatible.py` (line 33).

The repo is functionally at V5 (README version history) while pyproject says 0.1.0, the onyx adapter says 0.3, and the openai_compatible adapter says 0.2.0. Align all three to `0.5.0` (V5 = 0.5.x; the changes in this plan are additive within it).

- [ ] Step 1: `pyproject.toml` line 3 → `version = "0.5.0"`.
- [ ] Step 2: `onyx.py` line 42 → `USER_AGENT = "model-allocator/0.5 (onyx-adapter)"`.
- [ ] Step 3: `openai_compatible.py` line 33 → `req.add_header("User-Agent", "model-allocator/0.5.0")`.
- [ ] Step 4: Verify no test asserts the old strings (verified 2026-07-12: no test references USER_AGENT or the version), then run:
```bash
python3 -m py_compile src/model_allocator/adapters/onyx.py src/model_allocator/adapters/openai_compatible.py && python3 -m pytest 2>&1 | tail -1
```
Expected: `101 passed`.

- [ ] Step 5: Stage files and STOP — await Human commit approval:
```bash
cd /home/svend/model-allocator && git add src/model_allocator/adapters/claude_code.py src/model_allocator/cli.py tests/test_v2.py models.yaml runtime_profiles.yaml models.example.yaml runtime_profiles.example.yaml pyproject.toml src/model_allocator/adapters/onyx.py src/model_allocator/adapters/openai_compatible.py
git status --short
```
Suggested commit message (matches existing style, e.g. `[V2.2] Model-specific opencode.json — ...`):
`[V5.1] Claude Code env equivalence — max_output_tokens/adaptive-thinking/active-model/binary+extra-args, --max-output-tokens passthrough`

---

### Task 6: CROSS-REPO — Father passes per-role max_output_tokens to the allocator

> **Cross-repo task in `/home/svend/DPMtF-WebUI` — Father's governance requires Human approval; only the Human commits. Prepare the diff, stage nothing without explicit instruction, and present it.**

**Files:** Modify `/home/svend/DPMtF-WebUI/scripts/bridgeV002/start_coding.py` — the allocator `run` invocation at lines 245-257. (The role dict already carries `max_output_tokens`: it is selected in both role queries at lines 58 and 74 and mapped at line 103; the DB column comes from `scripts/db/004_role_runtime_config.sql`.)

- [ ] Step 1: Replace the subprocess call at start_coding.py lines 245-257:

Current code (verify before editing):
```python
            try:
                result = subprocess.run(
                    [
                        model_allocator_path,
                        "run",
                        "--role", role["role_key"],
                        "--client", role["default_runtime"],
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
```

New code:
```python
            try:
                run_cmd = [
                    model_allocator_path,
                    "run",
                    "--role", role["role_key"],
                    "--client", role["default_runtime"],
                ]
                # Per-role override from bridge_roles.max_output_tokens
                # (migration 004). Passed unconditionally when set: the
                # allocator applies it only where it matters (claude-code
                # env); other clients ignore the resolved field.
                if role.get("max_output_tokens"):
                    run_cmd += ["--max-output-tokens", str(role["max_output_tokens"])]
                result = subprocess.run(
                    run_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
```

- [ ] Step 2: Compatibility ordering: the Father change must land only AFTER the allocator change (Task 3) is committed — an old allocator errors with exit 2 on the unknown flag. The `if role.get("max_output_tokens")` guard means roles without the DB value keep the old command line, but roles WITH the value would fail against an old allocator. State this in the handoff to the Human.

- [ ] Step 3: Verify: `python3 -m py_compile /home/svend/DPMtF-WebUI/scripts/bridgeV002/start_coding.py` → no output. Then present the diff to the Human and STOP (Father git policy: only the Human commits).

## Acceptance Criteria

1. `cd /home/svend/model-allocator && python3 -m pytest` → `101 passed` (95 baseline + 6 new), 0 failures.
2. `python3 -m py_compile src/model_allocator/adapters/claude_code.py src/model_allocator/cli.py src/model_allocator/adapters/onyx.py src/model_allocator/adapters/openai_compatible.py` → exit 0, no output.
3. `./scripts/model-allocator run --role claude-test --client claude-code` → one line containing `CLAUDE_CODE_MAX_OUTPUT_TOKENS=32768`, `MODEL_ALLOCATOR_ACTIVE_MODEL=imple01-claude`, `--bare --model qwen3-coder:30b-256k`; NOT containing `CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING`.
4. `./scripts/model-allocator run --role claude-test --client claude-code --max-output-tokens 81920` → line contains `CLAUDE_CODE_MAX_OUTPUT_TOKENS=81920` (CLI wins over YAML 32768).
5. `./scripts/model-allocator env --role claude-test --client claude-code` → contains the lines `export CLAUDE_CODE_MAX_OUTPUT_TOKENS=32768` and `export MODEL_ALLOCATOR_ACTIVE_MODEL=imple01-claude`.
6. `grep -n 'version = ' pyproject.toml` → `version = "0.5.0"`; `grep -n USER_AGENT src/model_allocator/adapters/onyx.py | head -1` → contains `model-allocator/0.5`.
7. `git status --short` in /home/svend/model-allocator shows only the files listed in Task 5 Step 5 as staged; nothing committed.
8. (Cross-repo, after Human approves both commits) In Father: `python3 -m py_compile scripts/bridgeV002/start_coding.py` → exit 0; a role with `bridge_roles.max_output_tokens = 81920` and `model_source = model_allocator` produces a tmux start command containing `CLAUDE_CODE_MAX_OUTPUT_TOKENS=81920`.
