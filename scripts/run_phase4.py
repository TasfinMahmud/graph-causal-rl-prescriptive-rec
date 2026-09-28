#!/usr/bin/env python3
"""Thin entry point for Phase 4. See `gcrl --help` for all options."""
import sys

from gcrl.cli import main

if __name__ == "__main__":
    sys.exit(main(["phase4"] + sys.argv[1:]))
