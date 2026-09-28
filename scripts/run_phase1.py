#!/usr/bin/env python3
"""Thin entry point for Phase 1. See `gcrl --help` for all options."""
import sys

from gcrl.cli import main

if __name__ == "__main__":
    sys.exit(main(["phase1"] + sys.argv[1:]))
