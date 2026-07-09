import copy
from pathlib import Path

import pytest

from model_allocator import config_writer as cw


BASE_MODELS = {
    "models": {
        "imple-fast": {"runtime_profile": "local_ollama_cuda0", "real_model": "qwen:latest"}
    }
}
BASE_ROLES = {
    "roles": {
        "imple01": {"default_alias": "imple-fast", "config_dir": "imple01",
                    "client_aliases": {"opencode": "imple-fast"}}
    }
}
BASE_PROFILES = {
    "runtime_profiles": {"local_ollama_cuda0": {"backend": "ollama"}}
}


def _seed(tmp_path: Path) -> Path:
    import yaml
    (tmp_path / "models.yaml").write_text(yaml.safe_dump(BASE_MODELS), encoding="utf-8")
    (tmp_path / "roles.yaml").write_text(yaml.safe_dump(BASE_ROLES), encoding="utf-8")
    (tmp_path / "runtime_profiles.yaml").write_text(yaml.safe_dump(BASE_PROFILES), encoding="utf-8")
    return tmp_path


def test_load_raw_returns_three_sections(tmp_path):
    d = _seed(tmp_path)
    raw = cw.load_raw(d)
    assert set(raw) == {"aliases", "roles", "profiles"}
    assert "imple-fast" in raw["aliases"]
    assert "imple01" in raw["roles"]
    assert "local_ollama_cuda0" in raw["profiles"]


def test_load_raw_does_not_resolve_env(tmp_path):
    import yaml
    d = _seed(tmp_path)
    (d / "models.yaml").write_text(
        yaml.safe_dump({"models": {"llama": {"runtime_profile": "local_ollama_cuda0",
                                             "model_path": "${MODEL_ROOT_GGUF}/x.gguf"}}}),
        encoding="utf-8")
    raw = cw.load_raw(d)
    assert raw["aliases"]["llama"]["model_path"] == "${MODEL_ROOT_GGUF}/x.gguf"


def test_set_alias_upserts_and_roundtrips(tmp_path):
    d = _seed(tmp_path)
    cw.set_alias(d, "new-alias", {"runtime_profile": "local_ollama_cuda0", "real_model": "m:1"})
    raw = cw.load_raw(d)
    assert raw["aliases"]["new-alias"]["real_model"] == "m:1"
    assert "imple-fast" in raw["aliases"]  # existing preserved


def test_set_alias_rejects_unknown_profile(tmp_path):
    d = _seed(tmp_path)
    with pytest.raises(cw.ConfigWriteError):
        cw.set_alias(d, "bad", {"runtime_profile": "nope", "real_model": "m"})


def test_set_alias_writes_backup(tmp_path):
    d = _seed(tmp_path)
    cw.set_alias(d, "new-alias", {"runtime_profile": "local_ollama_cuda0", "real_model": "m"})
    assert (d / "models.yaml.bak").exists()


def test_delete_alias_refused_when_referenced(tmp_path):
    d = _seed(tmp_path)
    with pytest.raises(cw.ConfigWriteError):
        cw.delete_alias(d, "imple-fast")  # referenced by role imple01


def test_delete_alias_removes_unreferenced(tmp_path):
    d = _seed(tmp_path)
    cw.set_alias(d, "temp", {"runtime_profile": "local_ollama_cuda0", "real_model": "m"})
    cw.delete_alias(d, "temp")
    assert "temp" not in cw.load_raw(d)["aliases"]


def test_set_role_rejects_dangling_alias(tmp_path):
    d = _seed(tmp_path)
    with pytest.raises(cw.ConfigWriteError):
        cw.set_role(d, "r2", {"default_alias": "ghost", "config_dir": "r2"})


def test_set_role_upserts(tmp_path):
    d = _seed(tmp_path)
    cw.set_role(d, "r2", {"default_alias": "imple-fast", "config_dir": "r2",
                          "client_aliases": {"opencode": "imple-fast"}})
    assert "r2" in cw.load_raw(d)["roles"]


def test_delete_role_removes(tmp_path):
    d = _seed(tmp_path)
    cw.delete_role(d, "imple01")
    assert "imple01" not in cw.load_raw(d)["roles"]
