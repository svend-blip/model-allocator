"""onyx-mcp tool-body tests (V5C) — framework-free, no network, no mcp dep.

The FastMCP wrapper itself needs the optional 'mcp' extra and a live
event loop; the tool BODIES (onyx_answer_impl/onyx_status_impl) are plain
functions tested here with a fake adapter. Server wiring is live-validated
separately (see commit message / deploy docs).
"""

import unittest

from model_allocator import mcp_server
from model_allocator.invoke_result import make_invoke_result


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return make_invoke_result(
            provider="onyx",
            text="Answer with [1]",
            citations=[{"citation_number": 1, "title": "Doc",
                        "link": None, "document_id": "d1",
                        "source_type": "file"}],
            elapsed_ms=5.0,
        )

    def status(self):
        return {"reachable": True, "credentials_present": True,
                "api_base": "http://x", "error": None}


class OnyxMcpToolTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeAdapter()
        mcp_server._adapter = self.fake
        mcp_server._alias = "test-alias"
        self.addCleanup(setattr, mcp_server, "_adapter", None)

    def test_answer_returns_envelope_with_alias(self):
        result = mcp_server.onyx_answer_impl("What is X?")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["text"], "Answer with [1]")
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["metadata"]["alias"], "test-alias")
        self.assertEqual(self.fake.calls[0][0], "What is X?")

    def test_persona_override_passthrough(self):
        mcp_server.onyx_answer_impl("q", persona_id=9)
        self.assertEqual(self.fake.calls[0][1], {"persona_id": 9})

    def test_empty_question_rejected(self):
        result = mcp_server.onyx_answer_impl("  ")
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.fake.calls, [])

    def test_adapter_exception_becomes_error_dict(self):
        class Boom:
            def invoke(self, prompt, **kwargs):
                raise RuntimeError("kaputt")

        mcp_server._adapter = Boom()
        result = mcp_server.onyx_answer_impl("q")
        self.assertEqual(result["status"], "error")
        self.assertIn("kaputt", result["error"])

    def test_status_includes_alias(self):
        status = mcp_server.onyx_status_impl()
        self.assertTrue(status["reachable"])
        self.assertEqual(status["alias"], "test-alias")


if __name__ == "__main__":
    unittest.main()
