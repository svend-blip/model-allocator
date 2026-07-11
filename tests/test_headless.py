"""Headless runner tests (V5B) — no network, no tmux.

Covers: idle-batched prompt framing (multi-line paste => one prompt),
EOF handling, the run_loop contract (print answer + write InvokeResult
files), and headless command construction incl. capability gating.
"""

import io
import json
import os
import tempfile
import threading
import time
import unittest

from model_allocator.adapters.headless import build_headless_command
from model_allocator.headless import read_prompt, run_loop
from model_allocator.invoke_result import make_invoke_result


class ReadPromptTests(unittest.TestCase):
    def _pipe_stream(self):
        read_fd, write_fd = os.pipe()
        return os.fdopen(read_fd, "r"), os.fdopen(write_fd, "w")

    def test_multiline_paste_becomes_one_prompt(self):
        reader, writer = self._pipe_stream()
        try:
            def feed():
                writer.write("line one\nline two\nline three\n")
                writer.flush()
            t = threading.Thread(target=feed)
            t.start()
            prompt = read_prompt(reader, idle_seconds=0.3, poll_seconds=0.05)
            t.join()
            self.assertEqual(prompt, "line one\nline two\nline three\n")
        finally:
            reader.close()
            writer.close()

    def test_two_bursts_become_two_prompts(self):
        reader, writer = self._pipe_stream()
        try:
            writer.write("first prompt\n")
            writer.flush()
            first = read_prompt(reader, idle_seconds=0.2, poll_seconds=0.05)
            writer.write("second prompt\n")
            writer.flush()
            second = read_prompt(reader, idle_seconds=0.2, poll_seconds=0.05)
            self.assertEqual(first, "first prompt\n")
            self.assertEqual(second, "second prompt\n")
        finally:
            reader.close()
            writer.close()

    def test_eof_without_content_returns_none(self):
        reader, writer = self._pipe_stream()
        writer.close()
        try:
            self.assertIsNone(read_prompt(reader, idle_seconds=0.2))
        finally:
            reader.close()

    def test_eof_flushes_pending_content(self):
        reader, writer = self._pipe_stream()
        try:
            writer.write("tail prompt\n")
            writer.close()
            prompt = read_prompt(reader, idle_seconds=5.0, poll_seconds=0.05)
            self.assertEqual(prompt, "tail prompt\n")
        finally:
            reader.close()


class RunLoopTests(unittest.TestCase):
    def _run(self, stdin_text, invoke_fn, output_dir=None):
        read_fd, write_fd = os.pipe()
        reader = os.fdopen(read_fd, "r")
        writer = os.fdopen(write_fd, "w")
        out = io.StringIO()

        def feed():
            writer.write(stdin_text)
            writer.flush()
            time.sleep(0.05)
            writer.close()

        t = threading.Thread(target=feed)
        t.start()
        code = run_loop(reader, invoke_fn, out,
                        output_dir=output_dir, idle_seconds=0.2)
        t.join()
        reader.close()
        return code, out.getvalue()

    def test_answer_printed_and_result_written(self):
        calls = []

        def invoke_fn(prompt):
            calls.append(prompt)
            return make_invoke_result(
                provider="onyx", text="THE ANSWER",
                citations=[{"citation_number": 1, "title": "GATES.md",
                            "link": None, "document_id": "d1",
                            "source_type": "file"}],
                elapsed_ms=12.0)

        with tempfile.TemporaryDirectory() as tmp:
            code, output = self._run("what is X?\n", invoke_fn, output_dir=tmp)
            self.assertEqual(code, 0)
            self.assertIn("HEADLESS RUNNER READY", output)
            self.assertIn("THE ANSWER", output)
            self.assertIn("GATES.md", output)
            files = sorted(os.listdir(tmp))
            self.assertIn("latest.json", files)
            invoke_files = [f for f in files if f.startswith("invoke-")]
            self.assertEqual(len(invoke_files), 1)
            saved = json.load(open(os.path.join(tmp, "latest.json")))
            self.assertEqual(saved["text"], "THE ANSWER")
        self.assertEqual(calls, ["what is X?\n"])

    def test_error_result_reported_not_raised(self):
        def invoke_fn(prompt):
            return make_invoke_result(provider="onyx", text="",
                                      error="upstream down", elapsed_ms=1.0)

        code, output = self._run("hello\n", invoke_fn)
        self.assertEqual(code, 0)  # loop survives errors; exits on EOF
        self.assertIn("ERROR: upstream down", output)

    def test_blank_input_ignored(self):
        def invoke_fn(prompt):  # pragma: no cover - must not be called
            raise AssertionError("invoke_fn called for blank input")

        code, output = self._run("\n\n", invoke_fn)
        self.assertEqual(code, 0)
        self.assertIn("EOF", output)


class HeadlessCommandTests(unittest.TestCase):
    RESOLVED = {
        "alias": "company-knowledge",
        "backend": "onyx",
        "capabilities": ["invoke"],
    }

    def test_command_contains_headless_and_alias(self):
        command = build_headless_command(dict(self.RESOLVED), "advisor01")
        argv = command["argv"]
        self.assertIn("headless", argv)
        self.assertIn("--alias", argv)
        self.assertIn("company-knowledge", argv)

    def test_output_dir_and_idle_pass_through(self):
        resolved = dict(self.RESOLVED,
                        headless_output_dir="/tmp/x",
                        headless_idle_seconds=3.5)
        argv = build_headless_command(resolved, "advisor01")["argv"]
        self.assertIn("--output-dir", argv)
        self.assertIn("/tmp/x", argv)
        self.assertIn("--idle-seconds", argv)

    def test_capability_gate(self):
        resolved = dict(self.RESOLVED, capabilities=[])
        with self.assertRaises(ValueError):
            build_headless_command(resolved, "advisor01")


if __name__ == "__main__":
    unittest.main()
