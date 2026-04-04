"""
Refresh the library index when new papers have been added to Zotero.

Usage:
    python refresh_index.py              # rebuild only if paper count changed
    python refresh_index.py --force      # always rebuild
    python refresh_index.py --check      # report status without rebuilding

Scheduling (Windows Task Scheduler):
    Action: Start a program
    Program: C:\\path\\to\\venv\\Scripts\\python.exe
    Arguments: C:\\Users\\james\\PycharmProjects\\ResearchLibrary\\refresh_index.py
    Start in: C:\\Users\\james\\PycharmProjects\\ResearchLibrary

    Suggested trigger: Daily at a time you're unlikely to be using the app,
    or "When the computer is idle".
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CACHE_PATH = BASE_DIR / ".cache" / "library_index.json"
LOG_PATH = BASE_DIR / ".cache" / "refresh_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def get_current_paper_count() -> int:
    """Fetch the total number of non-attachment items from Zotero (fast, no pagination)."""
    from src.research_library.zotero_client import get_client
    zot = get_client()
    # top() with limit=1 just to read the total-results header
    zot.items(limit=1, itemType="-attachment || note")
    return int(zot.request.headers.get("Total-Results", 0))


def get_cached_count() -> int | None:
    """Return the total_fetched value from the cached index, or None if no cache."""
    if CACHE_PATH.exists():
        index = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return index.get("total_fetched")
    return None


def get_cached_generated_at() -> str | None:
    if CACHE_PATH.exists():
        index = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return index.get("generated_at")
    return None


def rebuild():
    """Run the full index rebuild pipeline."""
    from src.research_library.analysis import build_index

    def on_progress(msg):
        log.info(msg)

    log.info("Starting full index rebuild...")
    index = build_index(progress_callback=on_progress)
    log.info(
        f"Index rebuilt: {index['unique_papers']} unique papers, "
        f"{len(index['categories'])} categories, "
        f"{index['duplicates_removed']} duplicates removed."
    )
    return index


def main():
    parser = argparse.ArgumentParser(description="Refresh the ResearchLibrary index.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--force", action="store_true", help="Rebuild regardless of paper count")
    group.add_argument("--check", action="store_true", help="Check for new papers without rebuilding")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("ResearchLibrary index refresh check")

    try:
        current = get_current_paper_count()
    except Exception as e:
        log.error(f"Could not reach Zotero API: {e}")
        sys.exit(1)

    cached_count = get_cached_count()
    cached_at = get_cached_generated_at()

    if cached_count is None:
        log.info(f"No existing index. Current library: {current} items.")
    else:
        log.info(
            f"Cached index: {cached_count} items (built {cached_at[:19].replace('T', ' ')})"
        )
        log.info(f"Current library: {current} items")

    if args.check:
        if cached_count is None:
            log.info("Status: no index exists — run without --check to build one.")
        elif current != cached_count:
            log.info(f"Status: {current - cached_count:+d} items since last build — rebuild recommended.")
        else:
            log.info("Status: index is up to date.")
        return

    if args.force:
        log.info("--force specified, rebuilding...")
        rebuild()
    elif cached_count is None:
        log.info("No cache found, building index for the first time...")
        rebuild()
    elif current != cached_count:
        log.info(f"Paper count changed ({cached_count} -> {current}), rebuilding index...")
        rebuild()
    else:
        log.info("Library unchanged — skipping rebuild.")

    log.info("Done.")


if __name__ == "__main__":
    main()
