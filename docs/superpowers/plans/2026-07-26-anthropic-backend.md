# Anthropic Native API Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native Anthropic API backend (`anthropic`) to model-allocator so Claude Code can use Fable 5 directly via `api.anthropic.com`.

**Architecture:** New `anthropic` backend type alongside existing `ollama`/`llama_cpp`/`openai_compatible`/`onyx`. A thin `AnthropicAdapter` handles lifecycle (cloud_noop — start/stop/status check credentials only). Claude Code adapter sets `ANTHROPIC_API_KEY` without `ANTHROPIC_BASE_URL`, letting Claude Code use its default endpoint.

**Tech Stack:** Python 3, no new dependencies.

## Global Constraints

- No new Python dependencies without Human approval
- `python3 -m py_compile <file>` must pass before signaling completion
- Follow existing adapter patterns (same method signatures as `OpenAICompatibleAdapter`)
- `ANTHROPIC_API_KEY` env var for API key (standard Claude Code convention)
- Backend lifecycle: `cloud_noop` (no local process to manage)

---

### Task 1: Schema — Register `anthropic` backend

**Files:**
- Modify: `src/model_allocator/schema.py:33`

**Interfaces:**
- Produces: `BACKENDS = ("ollama", "llama_cpp", "openai_compatible", "onyx", "anthropic")`

- [ ] **Step 1: Add `"anthropic"` to BACKENDS**

```python
BACKENDS = ("ollama", "llama_cpp", "openai_compatible", "onyx", "anthropic")
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile src/model_allocator/schema.py`
Expected: no output (success)

- [ ] **Step 3: Commit**

```bash
git add src/model_allocator/schema.py
git commit -m "[anthropic] register anthropic backend in schema BACKENDS"
```

---

### Task 2: AnthropicAdapter — New lifecycle adapter

**Files:**
- Create: `src/model_allocator/adapters/anthropic.py`

**Interfaces:**
- Produces: `AnthropicAdapter(api_key_env: str)` with methods:
  - `are_credentials_present() -> dict` — `{"present": bool, "error": str | None}`
  - `status() -> dict` — `{"reachable": bool, "credentials_present": bool, "error": str | None}`
  - `start() -> dict` — `{"started": bool, "error": str | None}`
  - `stop() -> dict` — `{"stopped": bool, "error": str | None}`
  - `unload() -> dict` — `{"unloaded": bool, "error": str | None}`

- [ ] **Step 1: Write the failing test in `tests/test_v2.py`**

Add to imports at top of file:
```python
from model_allocator.adapters import anthropic as anthropic_adapter
```

Add new test class before `if __name__ == "__main__":`:
```python
class TestAnthropicAdapter(unittest.TestCase):
    def test_credentials_present_when_key_set(self):
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env="TEST_ANTHROPIC_KEY")
        with patch.dict(os.environ, {"TEST_ANTHROPIC_KEY": "sk-ant-12345"}):
            result = adapter.are_credentials_present()
        self.assertTrue(result["present"])
        self.assertIsNone(result["error"])

    def test_credentials_missing_when_key_not_set(self):
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env="MISSING_KEY")
        result = adapter.are_credentials_present()
        self.assertFalse(result["present"])
        self.assertIsNotNone(result["error"])

    def test_start_succeeds_when_credentials_present(self):
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env="TEST_ANTHROPIC_KEY")
        with patch.dict(os.environ, {"TEST_ANTHROPIC_KEY": "sk-ant-12345"}):
            result = adapter.start()
        self.assertTrue(result["started"])
        self.assertIsNone(result["error"])

    def test_start_fails_when_credentials_missing(self):
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env="MISSING_KEY")
        result = adapter.start()
        self.assertFalse(result["started"])
        self.assertIsNotNone(result["error"])

    def test_stop_is_noop(self):
        adapter = anthropic_adapter.AnthropicAdapter()
        result = adapter.stop()
        self.assertTrue(result["stopped"])
        self.assertIsNone(result["error"])

    def test_unload_is_noop(self):
        adapter = anthropic_adapter.AnthropicAdapter()
        result = adapter.unload()
        self.assertTrue(result["unloaded"])
        self.assertIsNone(result["error"])

    def test_status_reports_credentials(self):
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env="TEST_ANTHROPIC_KEY")
        with patch.dict(os.environ, {"TEST_ANTHROPIC_KEY": "sk-ant-12345"}):
            result = adapter.status()
        self.assertTrue(result["reachable"])
        self.assertTrue(result["credentials_present"])
        self.assertIsNone(result["error"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py::TestAnthropicAdapter -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model_allocator.adapters.anthropic'`

- [ ] **Step 3: Create `src/model_allocator/adapters/anthropic.py`**

```python
"""Anthropic native API adapter for model-allocator.

Handles runtime lifecycle for the anthropic backend. This is a cloud
backend — start/stop/unload are credential-check-only (cloud_noop).
"""

from __future__ import annotations

import os
from typing import Any


class AnthropicAdapter:
    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY"):
        self.api_key_env = api_key_env

    def are_credentials_present(self) -> dict:
        value = os.environ.get(self.api_key_env, "")
        if not value:
            return {
                "present": False,
                "error": f"Environment variable '{self.api_key_env}' is not set",
            }
        return {"present": True, "error": None}

    def status(self) -> dict:
        credentials = self.are_credentials_present()
        return {
            "reachable": True,
            "credentials_present": credentials["present"],
            "error": credentials.get("error"),
        }

    def start(self) -> dict[str, Any]:
        status = self.status()
        if not status["credentials_present"]:
            return {"started": False, "error": status["error"]}
        return {"started": True, "error": None}

    def stop(self) -> dict[str, Any]:
        return {"stopped": True, "error": None}

    def unload(self) -> dict[str, Any]:
        return {"unloaded": True, "error": None}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py::TestAnthropicAdapter -v`
Expected: 7 passed

- [ ] **Step 5: Verify syntax**

Run: `python3 -m py_compile src/model_allocator/adapters/anthropic.py`
Expected: no output (success)

- [ ] **Step 6: Commit**

```bash
git add src/model_allocator/adapters/anthropic.py tests/test_v2.py
git commit -m "[anthropic] add AnthropicAdapter for cloud_noop lifecycle"
```

---

### Task 3: Claude Code adapter — Handle `anthropic` backend

**Files:**
- Modify: `src/model_allocator/adapters/claude_code.py:29-31`

**Interfaces:**
- Consumes: `resolved["backend"] == "anthropic"`, `resolved["api_key_env"]`
- Produces: env dict with `ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY`, no `ANTHROPIC_BASE_URL`, no `ANTHROPIC_AUTH_TOKEN`

- [ ] **Step 1: Write the failing test in `tests/test_v2.py`**

Add to `TestClaudeCodeAdapter` class:
```python
    @patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which)
    def test_anthropic_backend(self, _mock):
        resolved = {
            "backend": "anthropic",
            "real_model": "claude-fable-5",
            "api_key_env": "ANTHROPIC_API_KEY",
        }
        cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(cmd["argv"], ["/usr/bin/claude", "--model", "claude-fable-5"])
        self.assertEqual(cmd["env"]["ANTHROPIC_API_KEY"], "$ANTHROPIC_API_KEY")
        self.assertNotIn("ANTHROPIC_BASE_URL", cmd["env"])
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", cmd["env"])

    @patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which)
    def test_anthropic_backend_with_max_output_tokens(self, _mock):
        resolved = {
            "backend": "anthropic",
            "real_model": "claude-fable-5",
            "api_key_env": "ANTHROPIC_API_KEY",
            "max_output_tokens": 65536,
        }
        cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(cmd["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "65536")

    @patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which)
    def test_anthropic_backend_with_alias_metadata(self, _mock):
        resolved = {
            "backend": "anthropic",
            "real_model": "claude-fable-5",
            "api_key_env": "ANTHROPIC_API_KEY",
            "alias": "fable5",
        }
        cmd = claude_code.build_claude_code_command(resolved)
        self.assertEqual(cmd["env"]["MODEL_ALLOCATOR_ACTIVE_MODEL"], "fable5")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py::TestClaudeCodeAdapter::test_anthropic_backend -v`
Expected: FAIL with `ValueError: Backend 'anthropic' is not supported by the Claude Code adapter`

- [ ] **Step 3: Add `anthropic` branch to `build_claude_code_command()`**

In `claude_code.py`, change the backend check at line 29-31 from:
```python
    backend = resolved.get("backend")
    if backend not in ("ollama", "openai_compatible"):
        raise ValueError(f"Backend '{backend}' is not supported by the Claude Code adapter")
```
to:
```python
    backend = resolved.get("backend")
    if backend not in ("ollama", "openai_compatible", "anthropic"):
        raise ValueError(f"Backend '{backend}' is not supported by the Claude Code adapter")
```

After the existing `if backend == "ollama":` block and before the `else:` block (which handles `openai_compatible`), insert the anthropic branch. The current structure is:

```python
    if backend == "ollama":
        ...
        env["ANTHROPIC_BASE_URL"] = endpoint
        env["ANTHROPIC_AUTH_TOKEN"] = "ollama"
    else:
        # openai_compatible branch
        ...
        env["ANTHROPIC_BASE_URL"] = endpoint
        env["ANTHROPIC_AUTH_TOKEN"] = f"${api_key_env}"
        env["ANTHROPIC_API_KEY"] = ""
```

Change to:

```python
    if backend == "ollama":
        api_base_env = resolved.get("api_base_env", "OLLAMA_BASE_URL")
        endpoint = os.environ.get(api_base_env, "") or resolved.get("default_api_base", "http://127.0.0.1:11434")
        env["ANTHROPIC_BASE_URL"] = endpoint
        env["ANTHROPIC_AUTH_TOKEN"] = "ollama"
    elif backend == "anthropic":
        api_key_env = resolved.get("api_key_env", "ANTHROPIC_API_KEY")
        env["ANTHROPIC_API_KEY"] = f"${api_key_env}"
    else:
        # openai_compatible branch (unchanged)
        api_base_env = resolved.get("api_base_env")
        ...
```

- [ ] **Step 4: Run all Claude Code adapter tests**

Run: `cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py::TestClaudeCodeAdapter -v`
Expected: 6 passed (3 existing + 3 new)

- [ ] **Step 5: Verify syntax**

Run: `python3 -m py_compile src/model_allocator/adapters/claude_code.py`
Expected: no output (success)

- [ ] **Step 6: Commit**

```bash
git add src/model_allocator/adapters/claude_code.py tests/test_v2.py
git commit -m "[anthropic] handle anthropic backend in Claude Code adapter"
```

---

### Task 4: Validator — Validate `anthropic` backend

**Files:**
- Modify: `src/model_allocator/validator.py:87-98` (compatibility check)
- Modify: `src/model_allocator/validator.py:65-79` (backend dispatch)

**Interfaces:**
- Consumes: `AnthropicAdapter` from Task 2
- Produces: `_validate_anthropic(resolved, client, result)` method

- [ ] **Step 1: Write the failing test in `tests/test_v2.py`**

Add to `SAMPLE_CONFIG` in the `models` dict:
```python
        "fable5": {
            "runtime_profile": "cloud_anthropic",
            "real_model": "claude-fable-5",
            "context": 200000,
            "lifecycle_policy": "cloud_noop",
            "clients": {"opencode": False, "claude-code": True},
        },
```

Add to `SAMPLE_CONFIG` in the `runtime_profiles` dict:
```python
        "cloud_anthropic": {
            "backend": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "provider": "anthropic",
        },
```

Add to `TestValidatorV2` class:
```python
    def test_anthropic_with_claude_is_ok(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            result = self.v.validate("fable5", "claude-code")
        self.assertEqual(result["validation_status"], "OK")

    def test_anthropic_with_opencode_is_error(self):
        result = self.v.validate("fable5", "opencode")
        self.assertEqual(result["validation_status"], "ERROR")
        self.assertTrue(any("incompatible" in e.lower() for e in result["errors"]))

    def test_anthropic_missing_key_is_warning(self):
        # Ensure ANTHROPIC_API_KEY is not set
        with patch.dict(os.environ, {}, clear=False):
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            result = self.v.validate("fable5", "claude-code")
        self.assertIn(result["validation_status"], ("WARNING", "OK"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py::TestValidatorV2::test_anthropic_with_claude_is_ok -v`
Expected: FAIL with "Backend adapter 'anthropic' is not implemented"

- [ ] **Step 3: Update `_check_client_backend_compatibility()`**

In `validator.py`, change the claude-code compatibility block (lines 90-99) from:
```python
        if client == "claude-code":
            if backend == "ollama":
                return
            if backend == "openai_compatible" and provider != "minimax":
                return
            result["errors"].append(
                f"Client 'claude-code' is incompatible with backend '{backend}' (provider '{provider}')"
            )
            result["validation_status"] = "ERROR"
            return
```
to:
```python
        if client == "claude-code":
            if backend == "ollama":
                return
            if backend == "openai_compatible" and provider != "minimax":
                return
            if backend == "anthropic":
                return
            result["errors"].append(
                f"Client 'claude-code' is incompatible with backend '{backend}' (provider '{provider}')"
            )
            result["validation_status"] = "ERROR"
            return
```

- [ ] **Step 4: Add `_validate_anthropic()` method and dispatch**

Add to the backend dispatch in `validate()` (after the `elif backend == "onyx":` block at line 72):
```python
        elif backend == "anthropic":
            self._validate_anthropic(resolved, client, result)
```

Add the method to the `Validator` class (after `_validate_onyx`):
```python
    def _validate_anthropic(self, resolved: dict, client: str, result: dict) -> None:
        from model_allocator.adapters import anthropic as anthropic_adapter

        api_key_env = resolved.get("api_key_env", "ANTHROPIC_API_KEY")
        adapter = anthropic_adapter.AnthropicAdapter(api_key_env=api_key_env)
        credentials = adapter.are_credentials_present()
        if not credentials["present"]:
            result["warnings"].append(credentials["error"])
            result["client_support"][client] = "NO_CREDENTIALS"
```

- [ ] **Step 5: Run all validator tests**

Run: `cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py::TestValidatorV2 -v`
Expected: 6 passed (3 existing + 3 new)

- [ ] **Step 6: Verify syntax**

Run: `python3 -m py_compile src/model_allocator/validator.py`
Expected: no output (success)

- [ ] **Step 7: Commit**

```bash
git add src/model_allocator/validator.py tests/test_v2.py
git commit -m "[anthropic] add validator support for anthropic backend"
```

---

### Task 5: CLI — Route `anthropic` backend

**Files:**
- Modify: `src/model_allocator/cli.py:43-61` (`_get_backend_adapter`)
- Modify: `src/model_allocator/cli.py:254-263` (`cmd_start`)
- Modify: `src/model_allocator/cli.py:285-293` (`cmd_stop`)
- Modify: `src/model_allocator/cli.py:316-324` (`cmd_unload`)
- Modify: `src/model_allocator/cli.py:146-163` (`cmd_status`)
- Modify: `src/model_allocator/cli.py:356-364` (`cmd_preflight`)

**Interfaces:**
- Consumes: `AnthropicAdapter` from Task 2

- [ ] **Step 1: Write the failing test in `tests/test_v2.py`**

Add to `TestCliV2._write_config()`, in the `models.yaml` section:
```python
            "  fable5:\n"
            "    runtime_profile: cloud_anthropic\n"
            "    real_model: claude-fable-5\n"
            "    context: 200000\n"
            "    lifecycle_policy: cloud_noop\n"
            "    clients:\n"
            "      opencode: false\n"
            "      claude-code: true\n"
```

Add to `TestCliV2._write_config()`, in the `runtime_profiles.yaml` section:
```python
            "  cloud_anthropic:\n"
            "    backend: anthropic\n"
            "    api_key_env: ANTHROPIC_API_KEY\n"
            "    provider: anthropic\n"
```

Add to `TestCliV2._write_config()`, in the `roles.yaml` section:
```python
            "  fable5-role:\n"
            "    default_alias: fable5\n"
            "    config_dir: fable5-role\n"
            "    client_aliases:\n"
            "      claude-code: fable5\n"
```

Add test methods to `TestCliV2`:
```python
    @patch("model_allocator.adapters.claude_code.shutil.which", side_effect=_fake_which)
    def test_run_fable5_claude(self, _mock):
        code = cli.main(["--config-dir", str(self.cfg_dir), "run", "--role", "fable5-role", "--client", "claude-code"])
        self.assertEqual(code, cli.EXIT_OK)

    def test_start_anthropic_alias(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            code = cli.main(["--config-dir", str(self.cfg_dir), "start", "--alias", "fable5"])
        self.assertEqual(code, cli.EXIT_OK)

    def test_start_anthropic_alias_missing_key(self):
        # Ensure key is not set
        with patch.dict(os.environ, {}, clear=False):
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            code = cli.main(["--config-dir", str(self.cfg_dir), "start", "--alias", "fable5"])
        self.assertEqual(code, cli.EXIT_WARNING)

    def test_stop_anthropic_alias(self):
        code = cli.main(["--config-dir", str(self.cfg_dir), "stop", "--alias", "fable5"])
        self.assertEqual(code, cli.EXIT_OK)

    def test_unload_anthropic_alias(self):
        code = cli.main(["--config-dir", str(self.cfg_dir), "unload", "--alias", "fable5"])
        self.assertEqual(code, cli.EXIT_OK)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py::TestCliV2::test_run_fable5_claude -v`
Expected: FAIL (either ValueError in adapter or ERROR exit code)

- [ ] **Step 3: Update `_get_backend_adapter()`**

Add import at top of `cli.py`:
```python
from model_allocator.adapters import anthropic as anthropic_adapter
```

Add to `_get_backend_adapter()` after the `if backend == "onyx":` block:
```python
    if backend == "anthropic":
        return anthropic_adapter.AnthropicAdapter(
            api_key_env=resolved.get("api_key_env", "ANTHROPIC_API_KEY"),
        )
```

- [ ] **Step 4: Update `cmd_start()`**

Add after `elif backend == "openai_compatible":` block:
```python
    elif backend == "anthropic":
        result = adapter.start()
```

- [ ] **Step 5: Update `cmd_stop()`**

Add after `elif backend == "openai_compatible":` block:
```python
    elif backend == "anthropic":
        result = adapter.stop()
```

- [ ] **Step 6: Update `cmd_unload()`**

Add after `elif backend == "openai_compatible":` block:
```python
    elif backend == "anthropic":
        result = adapter.unload()
```

- [ ] **Step 7: Update `cmd_status()`**

The status report dispatch is at lines 146-153. Add after `elif backend == "llama_cpp":`:
```python
    elif backend == "anthropic":
        report.update(adapter.status())
```

The exit-code logic is at lines 159-162. Add after the `llama_cpp` check:
```python
    if backend == "anthropic" and not report.get("credentials_present", False):
        return EXIT_WARNING
```

- [ ] **Step 8: Update `cmd_preflight()`**

Add after `elif backend == "openai_compatible":` block:
```python
    elif backend == "anthropic":
        start_result = adapter.start()
```

- [ ] **Step 9: Run all CLI tests**

Run: `cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py::TestCliV2 -v`
Expected: all tests pass (existing + 5 new)

- [ ] **Step 10: Verify syntax**

Run: `python3 -m py_compile src/model_allocator/cli.py`
Expected: no output (success)

- [ ] **Step 11: Commit**

```bash
git add src/model_allocator/cli.py tests/test_v2.py
git commit -m "[anthropic] route anthropic backend in CLI (start/stop/status/unload/preflight)"
```

---

### Task 6: Config — Add `cloud_anthropic` profile and `fable5` alias

**Files:**
- Modify: `runtime_profiles.yaml`
- Modify: `models.yaml`

**Interfaces:**
- Produces: `cloud_anthropic` runtime profile, `fable5` model alias

- [ ] **Step 1: Add `cloud_anthropic` to `runtime_profiles.yaml`**

Add after the `cloud_openrouter` block:
```yaml
  cloud_anthropic:
    backend: anthropic
    api_key_env: ANTHROPIC_API_KEY
    provider: anthropic
```

- [ ] **Step 2: Add `fable5` to `models.yaml`**

Add after the `imple-pay` block:
```yaml
  fable5:
    runtime_profile: cloud_anthropic
    real_model: claude-fable-5
    context: 200000
    lifecycle_policy: cloud_noop
    clients:
      claude-code: true
```

- [ ] **Step 3: Run doctor to verify config**

Run: `cd /home/svend/model-allocator && python3 -m model_allocator doctor --json`
Expected: no errors for `cloud_anthropic` or `fable5`

- [ ] **Step 4: Run full test suite**

Run: `cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py -v`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add runtime_profiles.yaml models.yaml
git commit -m "[anthropic] add cloud_anthropic profile and fable5 alias"
```

---

### Task 7: Integration — End-to-end smoke test

**Files:**
- No file changes — verification only

- [ ] **Step 1: Resolve the alias**

Run: `cd /home/svend/model-allocator && python3 -m model_allocator resolve --role fable5-role --client claude-code 2>&1 || true`
Expected: JSON output with `backend: anthropic`, `real_model: claude-fable-5`

- [ ] **Step 2: Validate the alias**

Run: `cd /home/svend/model-allocator && python3 -m model_allocator validate --alias fable5 --client claude-code`
Expected: OK or WARNING (depending on whether ANTHROPIC_API_KEY is set)

- [ ] **Step 3: Render the shell command**

Run: `cd /home/svend/model-allocator && python3 -m model_allocator run --role fable5-role --client claude-code`
Expected: shell string with `ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY` and `claude --model claude-fable-5`, no `ANTHROPIC_BASE_URL`

- [ ] **Step 4: Run full test suite one final time**

Run: `cd /home/svend/model-allocator && python3 -m pytest tests/test_v2.py -v`
Expected: all tests pass

- [ ] **Step 5: Commit any remaining changes**

```bash
git status
# Only commit if there are unexpected changes
```
