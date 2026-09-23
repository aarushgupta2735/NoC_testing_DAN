"""
Convenience wrapper: `python scripts/cli.py ...` runs the CLI without
requiring `pip install -e .` first (useful during development). The
real CLI logic lives in lmga_igsa.cli, and is what `pyproject.toml`'s
[project.scripts] entry point calls once the package is installed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.lmga_igsa.cli import main

if __name__ == "__main__":
    main()