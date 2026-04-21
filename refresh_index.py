"""Compatibility shim — delegates to `research-admin refresh`.

Kept so existing scheduled tasks (e.g. Windows Task Scheduler) don't break.
For new usage, prefer:

    research-admin refresh [--force|--check]
"""
import sys

from src.research_library.cli import main

if __name__ == "__main__":
    # Forward argparse flags to the refresh subcommand.
    argv = ["refresh"] + sys.argv[1:]
    sys.exit(main(argv))
