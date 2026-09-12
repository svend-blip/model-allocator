"""Tests for the `freetoken-qwen38-flash-next-abliterated` alias.

The runtime was validated OUTSIDE the allocator (systemd unit
freetoken-qwen38-flash-next-abliterated.service, FreeToken 0.1.2, port 8091,
262144 context, a single worker). The alias must reproduce that unit's argv
flag for flag, and the allocator must treat the server as ready only when
/v1/models actually exposes Qwen3.8-Flash-Next-Abliterated-NVFP4.

Everything here is deterministic: the HTTP boundary is the routing table
from the sibling test module, the executable is a stub. No FreeToken, no
GPU, no systemd, no network.
"""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from model_allocator import cli, schema
from model_allocator.adapters import freetoken as ft
from model_allocator.adapters.freetoken import (
    MODEL_CACHE_INCOMPLETE, MODEL_IDENTITY_MISMATCH, RUNTIME_NOT_READY,
    FreeTokenAdapter,
)
from model_allocator.resolver import Resolver
from tests.test_freetoken_flash_next import _Runtime, _fake_executable

REPO_ROOT = Path(__file__).resolve().parent.parent
ALIAS = "freetoken-qwen38-flash-next-abliterated"
SIBLING = "freetoken-qwen38-flash-next"
PROFILE = "freetoken_qwen38_cuda0"
MODEL_DIR = "/data/ai-data/models/qwen38-flash-next-abliterated-ftw-fixed"
SERVED = "Qwen3.8-Flash-Next-Abliterated-NVFP4"
CONTEXT = 262144
PORT = 8091

# The validated unit's ExecStart, minus the executable and the `serve` verb.
UNIT_ARGV = [
    "--model", MODEL_DIR,
    "--served-model-name", SERVED,
    "--host", "127.0.0.1",
    "--port", "8091",
    "--gpu", "0",
    "--max-seq-len-override", "262144",
    "--kv-reserve-tokens", "262144",
    "--max-prefill-length", "8192",
    "--moe-strategy", "offload",
    "--ple-backend", "disk",
    "--moe-cache-auto",
    "--quant-backend", "moe.nvfp4=triton",
    "--max-running-requests", "1",
    "--sampling-defaults", "model",
    "--reasoning-parser", "qwen3",
    "--tool-call-parser", "qwen3_coder",
]


def _load(name: str) -> dict:
    return yaml.safe_load((REPO_ROOT / name).read_text())


def _shipped_alias() -> dict:
    models = _load("models.yaml")
    profiles = _load("runtime_profiles.yaml")
    alias = dict(models["models"][ALIAS])
    profile = dict(profiles["runtime_profiles"][alias["runtime_profile"]])
    return {**profile, **alias, "alias": ALIAS, "backend": profile["backend"]}


def _pairs(argv: list[str]) -> set[tuple[str, str]]:
    """Flags as (flag, value) pairs; a bare flag pairs with ''."""
    out, i = set(), 0
    while i < len(argv):
        flag = argv[i]
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out.add((flag, argv[i + 1]))
            i += 2
        else:
            out.add((flag, ""))
            i += 1
    return out


class TestAliasConfiguration(unittest.TestCase):
    def test_alias_carries_the_validated_facts(self):
        alias = _load("models.yaml")["models"][ALIAS]
        self.assertEqual(alias["runtime_profile"], PROFILE)
        self.assertEqual(alias["model_path"], MODEL_DIR)
        self.assertEqual(alias["real_model"], SERVED)
        self.assertEqual(alias["served_model_name"], SERVED)
        self.assertEqual(alias["port"], PORT)
        self.assertEqual(alias["context"], CONTEXT)
        self.assertEqual(alias["max_running_requests"], 1)
        self.assertEqual(alias["qualification"]["concurrency"], 1)
        self.assertEqual(alias["qualification"]["runtime_version"], "0.1.2")

    def test_reasoning_effort_defaults_to_low_per_request(self):
        alias = _load("models.yaml")["models"][ALIAS]
        self.assertEqual(alias["reasoning_effort"], "low")
        self.assertEqual(alias["qualification"]["reasoning_efforts"],
                         ["low", "medium", "xhigh"])
        self.assertEqual(alias["qualification"]["reasoning_effort_default"], "xhigh")
        # Never a launch flag: the server default stays whatever it is.
        self.assertNotIn("reasoning_effort", dict(ft._VALUE_FLAGS))
        self.assertNotIn("reasoning_effort", dict(ft._BOOL_FLAGS))

    def test_alias_and_profile_validate_without_warnings(self):
        models = _load("models.yaml")["models"]
        profiles = _load("runtime_profiles.yaml")["runtime_profiles"]
        issues = schema.validate_alias(ALIAS, models[ALIAS], profiles, {})
        self.assertEqual([i.message for i in issues], [])

    def test_sibling_alias_is_untouched(self):
        """Additive: the existing Flash-Next alias keeps its port and model."""
        models = _load("models.yaml")["models"]
        self.assertEqual(models[SIBLING]["port"], 8090)
        self.assertEqual(models[SIBLING]["real_model"], "Qwen3.8-Flash-Next-NVFP4")
        self.assertEqual(models[SIBLING]["runtime_profile"], PROFILE)

    def test_resolver_exposes_the_local_endpoint(self):
        resolved = Resolver(config_dir=str(REPO_ROOT)).resolve_alias(ALIAS)
        self.assertEqual(resolved["backend"], "freetoken")
        self.assertEqual(resolved["port"], PORT)
        self.assertEqual(resolved["real_model"], SERVED)
        self.assertEqual(resolved["context"], CONTEXT)
        self.assertEqual(resolved["gpu"], "cuda0")
        self.assertEqual(resolved["reasoning_effort"], "low")
        self.assertEqual(resolved["start_timeout"], 1200)


class TestLaunch(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.executable = _fake_executable(tempfile.mkdtemp())

    def _adapter(self, **overrides):
        resolved = _shipped_alias()
        resolved["executable"] = self.executable
        resolved.update(overrides)
        return FreeTokenAdapter(resolved, state_dir=self.state_dir)

    def test_argv_reproduces_the_validated_unit(self):
        argv = self._adapter().build_argv()
        self.assertEqual(argv[:2], [self.executable, "serve"])
        rest = argv[2:]
        # The adapter spells the checkpoint flag --model-path; the unit uses
        # --model (a FreeToken alias of the same option). Normalise that one.
        rest = ["--model" if t == "--model-path" else t for t in rest]
        self.assertEqual(_pairs(rest), _pairs(UNIT_ARGV))
        self.assertEqual(len(rest), len(UNIT_ARGV), "no flag may be doubled")

    def test_single_worker_is_pinned(self):
        argv = self._adapter().build_argv()
        self.assertEqual(argv[argv.index("--max-running-requests") + 1], "1")

    @staticmethod
    def _ftw_checkpoint(directory: str, truncate: str | None = None) -> None:
        """Build a FreeToken-native checkpoint to shape: the manifest names
        two shards with byte sizes and one side file."""
        shards = [("freetoken-00000.ftw", 4096), ("freetoken-00001.ftw", 2048)]
        for name, size in shards:
            data = b"x" * (size - 1 if name == truncate else size)
            Path(directory, name).write_bytes(data)
        Path(directory, "ple-table-00000.safetensors").write_bytes(b"p")
        manifest = {"format": "freetoken_weight", "version": 1,
                    "shards": [{"file": n, "global_off": 0, "nbytes": s}
                               for n, s in shards],
                    "side_files": ["ple-table-00000.safetensors"]}
        Path(directory, "freetoken_weight.json").write_text(json.dumps(manifest))

    def test_ftw_checkpoint_passes_the_preflight(self):
        """The FTW layout has no safetensors index; the preflight verifies
        it by freetoken_weight.json instead of refusing to start."""
        with tempfile.TemporaryDirectory() as checkpoint:
            self._ftw_checkpoint(checkpoint)
            report = self._adapter(model_path=checkpoint).checkpoint_preflight()
        self.assertEqual(report["kind"], "local_directory")
        self.assertTrue(report["ok"], report["error"])
        self.assertTrue(report["checked"])
        self.assertEqual(report["shards"], 2)
        self.assertTrue(report["index"].endswith("freetoken_weight.json"))

    def test_truncated_ftw_shard_is_reported_incomplete(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            self._ftw_checkpoint(checkpoint, truncate="freetoken-00001.ftw")
            report = self._adapter(model_path=checkpoint).checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertEqual(report["code"], MODEL_CACHE_INCOMPLETE)
        self.assertEqual(report["incomplete"], ["freetoken-00001.ftw"])

    def test_missing_ftw_shard_is_reported_missing(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            self._ftw_checkpoint(checkpoint)
            os.remove(Path(checkpoint, "freetoken-00000.ftw"))
            report = self._adapter(model_path=checkpoint).checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertEqual(report["missing"], ["freetoken-00000.ftw"])

    def test_directory_without_any_index_still_refuses(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            report = self._adapter(model_path=checkpoint).checkpoint_preflight()
        self.assertFalse(report["ok"])
        self.assertEqual(report["code"], MODEL_CACHE_INCOMPLETE)

    def test_expected_model_name_is_the_abliterated_one(self):
        self.assertEqual(self._adapter().expected_model_name(), SERVED)


class TestReadiness(unittest.TestCase):
    def setUp(self):
        self.state_dir = tempfile.mkdtemp()
        self.executable = _fake_executable(tempfile.mkdtemp())

    def _adapter(self):
        resolved = _shipped_alias()
        resolved["executable"] = self.executable
        return FreeTokenAdapter(resolved, state_dir=self.state_dir)

    def test_ready_when_models_exposes_the_abliterated_model(self):
        runtime = _Runtime(model=SERVED, context=CONTEXT)
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen):
            readiness = self._adapter().readiness(alive=True)
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["served"], SERVED)
        self.assertEqual(readiness["context"], CONTEXT)
        self.assertTrue(any(u.endswith("/v1/models") for u in runtime.calls),
                        "readiness must consult /v1/models, not only /health")

    def test_active_unit_serving_the_wrong_model_is_not_ready(self):
        """`systemctl is-active` is not proof: the port answering with the
        non-abliterated model is an identity mismatch, never READY."""
        runtime = _Runtime(model="Qwen3.8-Flash-Next-NVFP4", context=CONTEXT)
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen):
            readiness = self._adapter().readiness(alive=True)
        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["code"], MODEL_IDENTITY_MISMATCH)

    def test_frontend_up_but_still_loading_is_not_ready(self):
        runtime = _Runtime(model=SERVED, state="loading")
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen):
            readiness = self._adapter().readiness(alive=True)
        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["code"], RUNTIME_NOT_READY)

    def test_unit_started_server_on_the_port_is_adopted(self):
        """A server the systemd unit started (not allocator-owned) that serves
        this exact model is reusable, so `start` adopts it instead of
        launching a second one onto the card."""
        runtime = _Runtime(model=SERVED, context=CONTEXT)
        adapter = self._adapter()
        with patch.object(ft.urllib.request, "urlopen", runtime.urlopen), \
                patch.object(adapter, "_port_open", return_value=True):
            occupancy = adapter.inspect_port()
        self.assertEqual(occupancy["kind"], "external_freetoken")
        self.assertTrue(occupancy["reusable"])
        self.assertEqual(occupancy["model"], SERVED)


class TestStartTimeout(unittest.TestCase):
    def test_alias_start_timeout_is_used_when_the_cli_gives_none(self):
        args = argparse.Namespace(timeout=None)
        self.assertEqual(cli._start_timeout(args, {"start_timeout": 1200}), 1200)

    def test_explicit_cli_timeout_wins(self):
        args = argparse.Namespace(timeout=45)
        self.assertEqual(cli._start_timeout(args, {"start_timeout": 1200}), 45)

    def test_historical_default_when_neither_is_set(self):
        args = argparse.Namespace(timeout=None)
        self.assertEqual(cli._start_timeout(args, {}), cli.DEFAULT_START_TIMEOUT)
        self.assertEqual(cli.DEFAULT_START_TIMEOUT, 120)

    def test_start_timeout_is_a_known_freetoken_alias_field(self):
        self.assertIs(schema.FREETOKEN_ALIAS_FIELDS["start_timeout"], int)


if __name__ == "__main__":
    unittest.main()
