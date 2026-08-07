"""llama.cpp must serve the model under the name the client asks for.

Without `--alias`, llama.cpp names the model after the GGUF file. `/v1/models`
returned `qwen2.5-coder-14b-instruct-q4_K_M.gguf` while the opencode config,
built from `real_model`, told OpenCode to request the same name without the
extension. Two strings assembled in different places from different sources,
with no reason to agree -- and the mismatch only shows on the first request.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_allocator.adapters.llama_cpp import LlamaCppAdapter  # noqa: E402


def _argv(**over):
    resolved = {
        "alias": "imple01-3060-lcpp",
        "backend": "llama_cpp",
        "real_model": "qwen2.5-coder-14b-instruct-q4_K_M",
        "model_path": "/home/svend/models/qwen2.5-coder-14b-instruct-q4_K_M.gguf",
        "server_bin_path": "/bin/sh",
        "context": 32768,
        "default_port": 8080,
    }
    resolved.update(over)
    return LlamaCppAdapter(resolved)._build_argv()


class ServedNameTests(unittest.TestCase):

    def test_the_served_name_is_the_model_the_config_names(self):
        argv = _argv()
        self.assertIn("--alias", argv)
        self.assertEqual(argv[argv.index("--alias") + 1],
                         "qwen2.5-coder-14b-instruct-q4_K_M")

    def test_it_is_not_the_gguf_filename(self):
        """The defect, named. The filename is what llama.cpp falls back to."""
        argv = _argv()
        self.assertNotIn(".gguf", argv[argv.index("--alias") + 1])

    def test_an_explicit_served_model_name_wins(self):
        argv = _argv(served_model_name="coder")
        self.assertEqual(argv[argv.index("--alias") + 1], "coder")

    def test_no_name_means_no_flag(self):
        """Absent must not become the string 'None' on the command line."""
        argv = _argv(real_model="")
        self.assertNotIn("--alias", argv)


class OneNameEverywhereTests(unittest.TestCase):
    """The server's --alias and the config's requested model id must come
    from ONE function.

    The adapter preferred `served_model_name` first; the config builder's
    llama_cpp branch did not know that key existed. An alias setting only
    `served_model_name` got a server serving one name and a config
    requesting another -- the .gguf mismatch reintroduced, and it only
    shows on the first request, after preflight has passed.
    """

    def test_the_config_requests_the_name_the_server_serves(self):
        resolved = {
            "alias": "x", "backend": "llama_cpp", "provider": "llama-local",
            "real_model": "qwen2.5-coder-14b",
            "model_path": "/m.gguf", "server_bin_path": "/bin/sh",
            "context": 32768, "default_port": 8080,
            "served_model_name": "coder",
        }
        argv = LlamaCppAdapter(resolved)._build_argv()
        served = argv[argv.index("--alias") + 1]
        cfg = opencode.build_opencode_config(resolved)
        provider = next(iter(cfg["provider"].values()))
        self.assertIn(served, provider["models"],
                      f"server serves {served!r}, config requests "
                      f"{list(provider['models'])!r}")
        self.assertTrue(cfg["model"].endswith("/" + served))


from model_allocator.adapters import opencode  # noqa: E402
