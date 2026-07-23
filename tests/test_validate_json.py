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


def test_validate_json_minimax_claude_error_is_structured(tmp_path, capsys):
    d = _seed(tmp_path)
    rc = main(["--config-dir", str(d), "validate", "--alias", "review-cloud",
               "--client", "claude-code", "--json"])
    assert rc == 1  # ERROR: claude-code incompatible with minimax
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["validation_status"] == "ERROR"
    assert isinstance(data["errors"], list) and data["errors"]
    assert any("claude-code" in e for e in data["errors"])


def test_validate_text_default_unchanged(tmp_path, capsys):
    """Default (no --json) output must stay byte-identical (text format)."""
    d = _seed(tmp_path)
    rc = main(["--config-dir", str(d), "validate", "--alias", "ghost",
               "--client", "opencode"])
    assert rc == 1
    out = capsys.readouterr().out
    # Text output starts with the status word, not JSON
    assert out.startswith("ERROR")
    # Must NOT be valid JSON (it's text)
    import json as _json
    try:
        _json.loads(out)
        assert False, "text output should not parse as JSON"
    except _json.JSONDecodeError:
        pass


def test_validate_json_warnings_are_array_not_joined(tmp_path, capsys):
    """Warnings must be an array, not a comma-joined string."""
    d = _seed(tmp_path)
    # review-cloud with opencode client: validates but may have warnings
    # about unreachable API base (env var unset) — warnings must be a list
    rc = main(["--config-dir", str(d), "validate", "--alias", "review-cloud",
               "--client", "opencode", "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data["warnings"], list)
    # Each warning must be a string, not fragments from a split
    for w in data["warnings"]:
        assert isinstance(w, str)
