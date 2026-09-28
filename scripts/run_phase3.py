#!/usr/bin/env python3
"""Thin entry point for Phase 3. See `gcrl --help` for all options."""
import sys

from gcrl.cli import main

if __name__ == "__main__":
    sys.exit(main(["phase3"] + sys.argv[1:]))
