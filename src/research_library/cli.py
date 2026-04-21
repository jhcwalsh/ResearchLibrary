"""Unified admin CLI for ResearchLibrary.

Subcommands:
    refresh   Rebuild the library index if Zotero paper count changed
    embed     Recompute Voyage embeddings from the cached index
    links     Regenerate paper_index.md
    download  Attempt to fetch PDFs for items missing attachments
    all       Run refresh, embed, links in order

Run `research-admin <command> --help` for per-subcommand options.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from .analysis import CACHE_PATH as INDEX_CACHE_PATH


BASE_DIR = Path(__file__).parent.parent.parent
LOG_PATH = INDEX_CACHE_PATH.parent / "refresh_log.txt"


def _setup_logging() -> logging.Logger:
    INDEX_CACHE_PATH.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("research-admin")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_current_paper_count() -> int:
    """Fetch total non-attachment items from Zotero (no pagination)."""
    from .zotero_client import get_client
    zot = get_client()
    zot.items(limit=1, itemType="-attachment || note")
    return int(zot.request.headers.get("Total-Results", 0))


def _cached_total() -> int | None:
    if INDEX_CACHE_PATH.exists():
        return json.loads(INDEX_CACHE_PATH.read_text(encoding="utf-8")).get("total_fetched")
    return None


def _cached_generated_at() -> str | None:
    if INDEX_CACHE_PATH.exists():
        return json.loads(INDEX_CACHE_PATH.read_text(encoding="utf-8")).get("generated_at")
    return None


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_refresh(args, log: logging.Logger) -> int:
    from .analysis import build_index

    try:
        current = _get_current_paper_count()
    except Exception as e:
        log.error(f"Could not reach Zotero API: {e}")
        return 1

    cached_count = _cached_total()
    cached_at = _cached_generated_at()

    if cached_count is None:
        log.info(f"No existing index. Current library: {current} items.")
    else:
        log.info(f"Cached index: {cached_count} items (built {cached_at[:19].replace('T', ' ')})")
        log.info(f"Current library: {current} items")

    if args.check:
        if cached_count is None:
            log.info("Status: no index exists — run `research-admin refresh` to build one.")
        elif current != cached_count:
            log.info(f"Status: {current - cached_count:+d} items since last build — rebuild recommended.")
        else:
            log.info("Status: index is up to date.")
        return 0

    should_rebuild = args.force or cached_count is None or current != cached_count
    if not should_rebuild:
        log.info("Library unchanged — skipping rebuild.")
        return 0

    reason = "forced" if args.force else ("no cache" if cached_count is None else f"{cached_count} -> {current}")
    log.info(f"Rebuilding index ({reason})...")
    index = build_index(progress_callback=log.info)
    log.info(
        f"Index rebuilt: {index['unique_papers']} unique papers, "
        f"{len(index['categories'])} categories, "
        f"{index['duplicates_removed']} duplicates removed."
    )
    return 0


def cmd_embed(args, log: logging.Logger) -> int:
    from .embeddings import build_embeddings, EMBEDDINGS_PATH

    if EMBEDDINGS_PATH.exists() and not args.force:
        log.info(f"Embeddings already exist at {EMBEDDINGS_PATH}. Use --force to rebuild.")
        return 0

    try:
        result = build_embeddings(progress=log.info)
    except FileNotFoundError as e:
        log.error(str(e))
        return 1
    log.info(
        f"Embeddings built: {result['paper_count']} papers, "
        f"{result['category_count']} categories → {result['path']}"
    )
    return 0


def cmd_links(args, log: logging.Logger) -> int:
    from .links import generate_links
    try:
        result = generate_links()
    except FileNotFoundError as e:
        log.error(str(e))
        return 1
    log.info(
        f"Wrote {result['written']}: {result['total_papers']} papers "
        f"({result['direct_links']} direct, {result['scholar_links']} Scholar)"
    )
    return 0


def cmd_download(args, log: logging.Logger) -> int:
    from .downloads import download_missing
    try:
        result = download_missing(dry_run=not args.download, limit=args.limit, progress=log.info)
    except ImportError as e:
        log.error(str(e))
        return 1
    log.info(
        f"Done: {result['needed_pdf']} needed PDFs; "
        f"{result['attached']} attached, {result['failed']} failed "
        f"(dry_run={result['dry_run']})"
    )
    return 0


def cmd_all(args, log: logging.Logger) -> int:
    refresh_args = argparse.Namespace(force=args.force, check=False)
    rc = cmd_refresh(refresh_args, log)
    if rc:
        return rc
    rc = cmd_embed(argparse.Namespace(force=True), log)
    if rc:
        return rc
    return cmd_links(argparse.Namespace(), log)


# ── Entry point ───────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-admin", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_refresh = sub.add_parser("refresh", help="Rebuild the index if paper count changed")
    group = p_refresh.add_mutually_exclusive_group()
    group.add_argument("--force", action="store_true", help="Rebuild unconditionally")
    group.add_argument("--check", action="store_true", help="Only report status")
    p_refresh.set_defaults(func=cmd_refresh)

    p_embed = sub.add_parser("embed", help="Compute Voyage embeddings from the cached index")
    p_embed.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    p_embed.set_defaults(func=cmd_embed)

    p_links = sub.add_parser("links", help="Regenerate paper_index.md")
    p_links.set_defaults(func=cmd_links)

    p_download = sub.add_parser("download", help="Try to fetch missing PDFs from Zotero items")
    p_download.add_argument("--download", action="store_true", help="Actually attach PDFs (default is dry-run)")
    p_download.add_argument("--limit", type=int, default=None, help="Max items to process")
    p_download.set_defaults(func=cmd_download)

    p_all = sub.add_parser("all", help="Run refresh, embed, and links in order")
    p_all.add_argument("--force", action="store_true", help="Force rebuild even if paper count unchanged")
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log = _setup_logging()
    return args.func(args, log)


if __name__ == "__main__":
    sys.exit(main())
