"""`run --config` points OpenCode at a file the caller owns.

Without it, `run` always names the role's shared config under
OPENCODE_ROLES_CONFIG_BASE and refreshes it on the way past. A caller that
built its own config for one run had no way to make it the one read.

DPMtF-LightWorker is that caller. It renders a per-execution config, merges
in its machine's provider endpoint and a permission block confining the role
to that execution's worktree, validates and publishes it -- and then the
launch command named the shared file instead. The rendered config was never
read, so the role ran with no confinement and no endpoint it could reach,
failing with `"undefined/chat/completions" cannot be parsed as a URL`.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model_allocator.adapters import opencode  # noqa: E402


RESOLVED = {
    "real_model": "qwen2.5-coder:14b",
    "backend": "ollama",
    "provider": "ollama",
}


class ConfigPathTests(unittest.TestCase):

    def test_the_given_path_is_what_opencode_reads(self):
        cmd = opencode.build_opencode_command(
            RESOLVED, "imple01",
            config_path="/home/svend/lightworker/opencode/EXEC-008.json")
        self.assertEqual(
            cmd["env"]["OPENCODE_CONFIG"],
            "/home/svend/lightworker/opencode/EXEC-008.json")

    def test_the_config_dir_follows_the_file(self):
        """OpenCode reads both. A CONFIG_DIR still pointing at the role's
        shared directory would have it loading two configs from two places."""
        cmd = opencode.build_opencode_command(
            RESOLVED, "imple01",
            config_path="/home/svend/lightworker/opencode/EXEC-008.json")
        self.assertEqual(
            cmd["env"]["OPENCODE_CONFIG_DIR"],
            "/home/svend/lightworker/opencode")

    def test_the_role_directory_is_not_named_at_all(self):
        """The whole point. Naming it alongside is what made the caller's
        file decorative."""
        cmd = opencode.build_opencode_command(
            RESOLVED, "imple01",
            config_path="/tmp/mine/opencode.json")
        self.assertNotIn("opencode-roles", str(cmd["env"]))

    def test_without_it_nothing_changes(self):
        """Every existing caller passes no config path and must keep the
        shared role directory it has always had."""
        cmd = opencode.build_opencode_command(RESOLVED, "imple01")
        self.assertIn("opencode-roles/imple01", cmd["env"]["OPENCODE_CONFIG_DIR"])
        self.assertTrue(
            cmd["env"]["OPENCODE_CONFIG"].endswith(
                "opencode-roles/imple01/opencode.json"))


if __name__ == "__main__":
    unittest.main()
