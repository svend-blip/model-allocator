"""Package entry point for `python3 -m model_allocator`."""

from __future__ import annotations

import sys

from model_allocator.cli import main

if __name__ == "__main__":
    sys.exit(main())
