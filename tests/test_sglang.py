"""Tests for SGLang adapter."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_allocator.adapters import sglang as sglang_adapter


class TestSGLangAdapter(unittest.TestCase):
    def setUp(self):
        self.resolved = {
            "alias": "qwen-shared-sglang",
            "model_path": "/home/svend/models/sglang/Qwen3-Coder-30B-A3B-Instruct-AWQ",
            "served_model_name": "qwen-shared",
            "context": 32768,
            "port": 30000,
            "host": "127.0.0.1",
            "venv": "/home/svend/venvs/sglang",
        }
        self.state_dir = tempfile.mkdtemp()

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_argv_assembly(self, _mock_access, _mock_isfile):
        adapter = sglang_adapter.SGLangAdapter(self.resolved, state_dir=self.state_dir)
        argv = adapter._build_argv()
        self.assertIn("sglang.launch_server", argv[2])
        self.assertIn("--model-path", argv)
        self.assertIn("--served-model-name", argv)
        self.assertIn("qwen-shared", argv)
        self.assertIn("--port", argv)
        self.assertIn("30000", argv)
        self.assertIn("--context-length", argv)
        self.assertIn("32768", argv)
        self.assertIn("--tool-call-parser", argv)

    def test_argv_uses_defaults_when_fields_absent(self):
        minimal = {"model_path": "/tmp/model"}
        adapter = sglang_adapter.SGLangAdapter(minimal, state_dir=self.state_dir)
        argv = adapter._build_argv()
        self.assertIn("--served-model-name", argv)
        self.assertIn("qwen-shared", argv)  # default
        self.assertIn("--context-length", argv)
        self.assertIn("32768", argv)  # default

    def test_missing_model_path_raises(self):
        with self.assertRaises(sglang_adapter.SGLangAdapterError):
            sglang_adapter.SGLangAdapter({}, state_dir=self.state_dir)._build_argv()

    def test_finds_free_port_when_not_configured(self):
        minimal = {"model_path": "/tmp/model"}
        adapter = sglang_adapter.SGLangAdapter(minimal, state_dir=self.state_dir)
        port = adapter.port
        self.assertGreater(port, 0)
        self.assertNotEqual(port, 30000)  # not the default when free-port is used

    def test_stop_removes_pid_file(self):
        adapter = sglang_adapter.SGLangAdapter(self.resolved, state_dir=self.state_dir)
        Path(adapter.pid_file).write_text("99999", encoding="utf-8")
        with patch.object(sglang_adapter.SGLangAdapter, "_kill_pid", return_value={"stopped": True, "error": None}):
            result = adapter.stop()
        self.assertTrue(result["stopped"])
        self.assertFalse(os.path.exists(adapter.pid_file))

    def test_stop_no_pid_file_is_noop(self):
        adapter = sglang_adapter.SGLangAdapter(self.resolved, state_dir=self.state_dir)
        result = adapter.stop()
        self.assertTrue(result["stopped"])

    def test_status_no_pid_file(self):
        adapter = sglang_adapter.SGLangAdapter(self.resolved, state_dir=self.state_dir)
        status = adapter.status()
        self.assertFalse(status["running"])
        self.assertIsNone(status["pid"])

    def test_unload_calls_stop(self):
        adapter = sglang_adapter.SGLangAdapter(self.resolved, state_dir=self.state_dir)
        result = adapter.unload()
        self.assertTrue(result["stopped"])


if __name__ == "__main__":
    unittest.main()
