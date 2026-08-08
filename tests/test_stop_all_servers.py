"""stop --all-servers: local VRAM reclaim.

Contract (born from the 2026-08-03..07 incident where a resident Laguna
llama.cpp server held 29 GB VRAM and starved the trade engine's daily
runs for five days):

- ONLY local server backends (llama_cpp, sglang) are stopped.
- Ollama aliases are never touched (Ollama evicts by itself), nor are
  cloud/external/onyx aliases — and remote machines are never contacted.
- Aliases sharing one server port stop it once.
- A failed stop yields EXIT_WARNING, not a crash.
"""
import argparse
import json
import unittest
from unittest.mock import patch

from model_allocator import cli


def _args(**overrides):
    ns = argparse.Namespace(alias=None, all_servers=True, timeout=30,
                            config_dir=None)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class FakeAdapter:
    def __init__(self, port=None, stopped=True):
        if port is not None:
            self.port = port
        self._stopped = stopped
        self.stop_calls = 0

    def stop(self, timeout=30):
        self.stop_calls += 1
        return {"stopped": self._stopped, "error": None}


class FakeResolver:
    def __init__(self, aliases):
        self._aliases = aliases

    def list_aliases(self):
        return list(self._aliases)

    def resolve_alias(self, alias):
        return self._aliases[alias]


class StopAllServersTests(unittest.TestCase):
    def _run(self, aliases, adapters, capsys=None):
        """Run _stop_all_local_servers with fakes; return (exit, results)."""
        def fake_adapter(resolved):
            return adapters[resolved["alias"]]

        printed = []
        with patch.object(cli, "Resolver",
                          lambda config_dir=None: FakeResolver(aliases)), \
             patch.object(cli, "_get_backend_adapter", fake_adapter), \
             patch("builtins.print",
                   lambda *a, **k: printed.append(a[0] if a else "")):
            code = cli._stop_all_local_servers(_args())
        results = json.loads(printed[-1]) if printed else []
        return code, results

    def test_only_local_server_backends_are_stopped(self):
        aliases = {
            "laguna-local": {"alias": "laguna-local", "backend": "llama_cpp"},
            "qwen-sglang": {"alias": "qwen-sglang", "backend": "sglang"},
            "imple01-local": {"alias": "imple01-local", "backend": "ollama"},
            "opus5": {"alias": "opus5", "backend": "anthropic"},
            "freebuff-cli": {"alias": "freebuff-cli", "backend": "external"},
        }
        adapters = {
            "laguna-local": FakeAdapter(port=8080),
            "qwen-sglang": FakeAdapter(port=30000),
            # Non-server aliases get adapters too; they must never be asked.
            "imple01-local": FakeAdapter(),
            "opus5": FakeAdapter(),
            "freebuff-cli": FakeAdapter(),
        }
        code, results = self._run(aliases, adapters)
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual({r["alias"] for r in results},
                         {"laguna-local", "qwen-sglang"})
        self.assertEqual(adapters["imple01-local"].stop_calls, 0)
        self.assertEqual(adapters["opus5"].stop_calls, 0)
        self.assertEqual(adapters["freebuff-cli"].stop_calls, 0)

    def test_shared_port_is_stopped_once(self):
        aliases = {
            "sg-a": {"alias": "sg-a", "backend": "sglang"},
            "sg-b": {"alias": "sg-b", "backend": "sglang"},
        }
        shared = FakeAdapter(port=30000)
        adapters = {"sg-a": shared, "sg-b": shared}
        code, results = self._run(aliases, adapters)
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(shared.stop_calls, 1)
        shared_entries = [r for r in results if r.get("shared_port")]
        self.assertEqual(len(shared_entries), 1)

    def test_failed_stop_is_warning_not_crash(self):
        aliases = {
            "laguna-local": {"alias": "laguna-local", "backend": "llama_cpp"},
        }
        adapters = {"laguna-local": FakeAdapter(port=8080, stopped=False)}
        code, results = self._run(aliases, adapters)
        self.assertEqual(code, cli.EXIT_WARNING)
        self.assertFalse(results[0]["stopped"])

    def test_alias_and_all_servers_are_mutually_exclusive(self):
        with patch("builtins.print"):
            code = cli.cmd_stop(_args(alias="laguna-local"))
        self.assertEqual(code, cli.EXIT_ERROR)

    def test_stop_without_alias_or_all_servers_errors(self):
        with patch("builtins.print"):
            code = cli.cmd_stop(_args(all_servers=False))
        self.assertEqual(code, cli.EXIT_ERROR)


if __name__ == "__main__":
    unittest.main()
