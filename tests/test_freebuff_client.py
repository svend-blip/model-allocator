"""freebuff: a self-managing coding TUI as a first-class allocator client.

The alternative was a parallel launch mechanism in the bridge for roles
that are 'just a program'. One start path is worth a thin adapter: roles.yaml
-> alias -> client, same as every other client, and if freebuff ever takes
an allocator-selected model or environment the place already exists.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model_allocator.adapters import freebuff


class FreebuffAdapterTests(unittest.TestCase):

    def test_the_command_is_just_the_binary(self):
        import unittest.mock as m
        with m.patch("shutil.which", return_value="/usr/bin/freebuff"):
            cmd = freebuff.build_freebuff_command({"alias": "freebuff-cli"})
        self.assertEqual(cmd["argv"], ["freebuff"])
        self.assertEqual(cmd["env"], {})

    def test_a_missing_binary_is_an_error_not_a_broken_session(self):
        """Failing at run-time beats sending a command the shell cannot
        find into a role's pane."""
        import unittest.mock as m
        with m.patch("shutil.which", return_value=None):
            with self.assertRaises(ValueError):
                freebuff.build_freebuff_command({})

    def test_freebuff_bin_overrides(self):
        import os
        import unittest.mock as m
        with m.patch.dict(os.environ, {"FREEBUFF_BIN": "/opt/fb/freebuff"}):
            cmd = freebuff.build_freebuff_command({})
        self.assertEqual(cmd["argv"], ["/opt/fb/freebuff"])


if __name__ == "__main__":
    unittest.main()
