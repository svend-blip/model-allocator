"""An ollama role's output budget comes from max_output_tokens.

Every other backend reads it. The ollama branch hardcoded
`output = min(context, 65536)`, so the output budget always equalled the whole
context window -- no headroom for the system prompt or the work by any
accounting, and `max_output_tokens` in models.yaml had no effect at all.

It looked like a deliberate config choice. DPMtF-LightWorker's imple01LW ran
against context 8192 / output 8192 and compacted on nearly every turn, drifting
off the task each time.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_allocator.adapters import opencode  # noqa: E402


def _resolved(**over):
    base = {
        "backend": "ollama",
        "provider": "ollama",
        "real_model": "qwen2.5-coder:14b",
        "context": 65536,
    }
    base.update(over)
    return base


def _limit(resolved):
    cfg = opencode.build_opencode_config(resolved)
    provider = next(iter(cfg["provider"].values()))
    return next(iter(provider["models"].values()))["limit"]


class OllamaOutputBudgetTests(unittest.TestCase):

    def test_max_output_tokens_is_honoured(self):
        self.assertEqual(_limit(_resolved(max_output_tokens=32768))["output"], 32768)

    def test_it_is_not_silently_the_whole_window(self):
        """The defect, named. An output budget equal to the context leaves
        nothing for input."""
        limit = _limit(_resolved(max_output_tokens=32768))
        self.assertLess(limit["output"], limit["context"])

    def test_the_context_is_still_the_configured_one(self):
        self.assertEqual(_limit(_resolved(max_output_tokens=32768))["context"], 65536)

    def test_an_alias_without_the_setting_gets_a_bounded_default(self):
        """Unset must not mean the whole window either."""
        limit = _limit(_resolved())
        self.assertEqual(limit["output"], 8192)
        self.assertLess(limit["output"], limit["context"])


if __name__ == "__main__":
    unittest.main()


def _llama(**over):
    base = {
        "backend": "llama_cpp",
        "provider": "llama-local",
        "real_model": "qwen2.5-coder-14b",
        "context": 32768,
        "default_port": 8080,
    }
    base.update(over)
    return base


class LlamaCppLimitTests(unittest.TestCase):
    """This branch emitted no limit at all.

    A llama.cpp role never told OpenCode its context window, so the client
    fell back to whatever it assumes for an unknown model. Configuring a
    window is pointless if the client is not told.
    """

    def test_the_context_reaches_opencode(self):
        self.assertEqual(_limit(_llama())["context"], 32768)

    def test_the_output_budget_is_not_the_whole_window(self):
        self.assertLess(_limit(_llama())["output"], 32768)

    def test_max_output_tokens_is_honoured(self):
        self.assertEqual(_limit(_llama(max_output_tokens=4096))["output"], 4096)

    def test_an_alias_without_a_context_emits_no_limit(self):
        """Absent is not zero. A limit of 0 would be worse than none."""
        cfg = opencode.build_opencode_config(_llama(context=None))
        provider = next(iter(cfg["provider"].values()))
        self.assertNotIn("limit", next(iter(provider["models"].values())))


class FallbackHeadroomTests(unittest.TestCase):
    """The unset default must also leave headroom.

    The old fallback was `min(context, 8192)` -- which for any window of
    8192 or less meant output == the whole window, the original defect
    surviving in precisely the case that hid it the first time (imple01LW
    ran at 8192/8192 for four executions). The openai branch capped at
    65536 and had it for every window up to 64k.
    """

    def test_a_small_window_keeps_half_for_input(self):
        limit = _limit(_resolved(context=4096))
        self.assertEqual(limit["output"], 2048)
        self.assertLess(limit["output"], limit["context"])

    def test_the_boundary_case_that_hid_the_original_defect(self):
        limit = _limit(_resolved(context=8192))
        self.assertEqual(limit["output"], 4096)

    def test_the_openai_branch_no_longer_gives_the_whole_window(self):
        resolved = {
            "backend": "openai_compatible",
            "provider": "cloud_minimax",
            "real_model": "m",
            "api_base_env": "X_BASE",
            "context": 65536,
        }
        import os
        os.environ.setdefault("X_BASE", "https://api.example")
        cfg = opencode.build_opencode_config(resolved)
        provider = next(iter(cfg["provider"].values()))
        limit = next(iter(provider["models"].values()))["limit"]
        self.assertLess(limit["output"], limit["context"])
