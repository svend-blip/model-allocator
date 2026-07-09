import json
from pathlib import Path

import yaml

from model_allocator.cli import main


def _seed(tmp_path: Path) -> Path:
    (tmp_path / "models.yaml").write_text(
        yaml.safe_dump({"models": {"a1": {"runtime_profile": "p1", "real_model": "m"}}}),
        encoding="utf-8")
    (tmp_path / "roles.yaml").write_text(
        yaml.safe_dump({"roles": {}}), encoding="utf-8")
    (tmp_path / "runtime_profiles.yaml").write_text(
        yaml.safe_dump({"runtime_profiles": {"p1": {"backend": "ollama"}}}), encoding="utf-8")
    return tmp_path


def test_config_show_prints_json(tmp_path, capsys):
    d = _seed(tmp_path)
    rc = main(["--config-dir", str(d), "config", "show"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "a1" in out["aliases"]
    assert out["profiles"]["p1"]["backend"] == "ollama"


def test_config_set_alias_writes(tmp_path, capsys):
    d = _seed(tmp_path)
    payload = json.dumps({"runtime_profile": "p1", "real_model": "m2"})
    rc = main(["--config-dir", str(d), "config", "set-alias", "--name", "a2", "--json", payload])
    assert rc == 0
    assert "a2" in yaml.safe_load((d / "models.yaml").read_text())["models"]


def test_config_set_alias_bad_profile_returns_error(tmp_path, capsys):
    d = _seed(tmp_path)
    payload = json.dumps({"runtime_profile": "ghost", "real_model": "m"})
    rc = main(["--config-dir", str(d), "config", "set-alias", "--name", "bad", "--json", payload])
    assert rc == 1
    assert "error" in json.loads(capsys.readouterr().err)


def test_config_delete_role(tmp_path):
    d = _seed(tmp_path)
    (d / "roles.yaml").write_text(
        yaml.safe_dump({"roles": {"r1": {"default_alias": "a1", "config_dir": "r1"}}}),
        encoding="utf-8")
    rc = main(["--config-dir", str(d), "config", "delete-role", "--name", "r1"])
    assert rc == 0
    assert yaml.safe_load((d / "roles.yaml").read_text())["roles"] == {}
