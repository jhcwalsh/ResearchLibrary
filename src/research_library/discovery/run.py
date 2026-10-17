"""The daily discovery run: fetch, dedupe, score, pick, summarise, write the digest.

Never writes to Zotero and never rebuilds the index. A source that fails, or a
summary that fails, is reported and the run carries on.
"""
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from .. import embeddings
from ..analysis import CACHE_PATH as INDEX_CACHE_PATH
from . import scoring, sources, store, summaries

WINDOW_FLOOR_DAYS = 1
WINDOW_CAP_DAYS = 7
FIRST_RUN_DAYS = 2
EXPIRE_DAYS = 14

DEFAULT_FETCHERS = {
    "arxiv": sources.fetch_arxiv,
    "ssrn": sources.fetch_ssrn,
    "openalex": sources.fetch_openalex,
}


def compute_window(last_run: str | None, now: datetime, days: int | None) -> tuple[datetime, datetime]:
    """From the last run to now, clamped to [1, 7] days; `days` overrides; first run looks back 2."""
    if days:
        return now - timedelta(days=days), now
    if not last_run:
        return now - timedelta(days=FIRST_RUN_DAYS), now
    start = datetime.fromisoformat(last_run)
    if start.tzinfo is not None:
        start = start.replace(tzinfo=None)
    start = min(start, now - timedelta(days=WINDOW_FLOOR_DAYS))
    start = max(start, now - timedelta(days=WINDOW_CAP_DAYS))
    return start, now


def _load_index(index_path: Path | None) -> dict:
    path = Path(index_path) if index_path else INDEX_CACHE_PATH
    if not path.exists():
        raise FileNotFoundError(f"No library index at {path}. Run `research-admin refresh` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _failed_summary(exc: Exception) -> dict:
    return {"summary": None, "why_it_fits": "", "recommendation": "maybe", "reason": f"summary failed: {exc}"}


def discover(days: int | None = None, force_queries: bool = False, dry_run: bool = False, progress=None,
             client=None, now: datetime | None = None, root: Path | None = None, index_path: Path | None = None,
             library: dict | None = None, embed=None, fetchers: dict | None = None, sleep=time.sleep) -> dict:
    """Run one discovery pass. Returns counts plus the selected candidates."""
    step = progress or (lambda msg: None)
    now = now or datetime.now()
    fetchers = fetchers or DEFAULT_FETCHERS

    index = _load_index(index_path)
    categories = index["categories"]
    papers_by_key = {p["key"]: p for c in categories.values() for p in c["papers"]}
    library = library or embeddings.load_embeddings()
    if library is None:
        raise FileNotFoundError(f"No embeddings at {embeddings.EMBEDDINGS_PATH}. Run `research-admin embed` first.")

    follow = store.load_follow(root)
    ledger = store.load_ledger(root)
    followed = store.followed_topics(follow, list(categories))
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    topic_queries = summaries.ensure_queries(client, follow, categories, followed, force=force_queries, progress=step)
    window_from, window_to = compute_window(follow.get("last_run"), now, days)
    step(f"Window {window_from:%Y-%m-%d %H:%M} to {window_to:%Y-%m-%d %H:%M}; {len(followed)} topics followed")

    # ── Fetch ────────────────────────────────────────────────────────────────
    raw: list[dict] = []
    fetched: dict[str, int] = {}
    mailto = os.environ.get("UNPAYWALL_EMAIL", "")
    calls = {
        "arxiv": lambda: fetchers["arxiv"](window_from=window_from, window_to=window_to, topic_queries=topic_queries, sleep=sleep),
        "ssrn": lambda: fetchers["ssrn"](bindings=follow.get("ssrn_bindings") or [], since=window_from.date(), sleep=sleep),
        "openalex": lambda: fetchers["openalex"](topic_queries=topic_queries, since=window_from.date(), mailto=mailto),
    }
    for name, call in calls.items():
        if name == "openalex" and not mailto:
            step("  OpenAlex skipped: set UNPAYWALL_EMAIL in .env to identify us to its API")
            fetched[name] = 0
            continue
        try:
            found = call()
        except Exception as exc:  # a dead source must not kill the run
            step(f"  {name} failed: {exc}")
            found = []
        fetched[name] = len(found)
        raw.extend(found)
        step(f"  {name}: {len(found)} papers")

    # ── Dedupe, score, pick ──────────────────────────────────────────────────
    cands = sources.drop_owned(sources.merge_candidates(raw), index, ledger)
    step(f"{len(raw)} fetched, {len(cands)} new")
    if cands:
        vectors = scoring.embed_candidates(cands, embed=embed)
        scoring.score_candidates(cands, vectors, library, followed)
    selected = scoring.select_candidates(cands, follow)
    by_topic: dict[str, int] = {}
    for c in selected:
        by_topic[c["topic"]] = by_topic.get(c["topic"], 0) + 1
    step(f"{len(selected)} selected: " + ", ".join(f"{t} {n}" for t, n in by_topic.items()))

    result = {"fetched": fetched, "new": len(cands), "scored": len(cands), "shown": len(selected),
              "by_topic": by_topic, "digest": None, "expired": 0, "selected": selected}
    if dry_run:
        step("dry run, nothing written")
        return result

    # ── Summarise and write ──────────────────────────────────────────────────
    for c in selected:
        nearest = [papers_by_key[n["key"]] for n in c.get("nearest", []) if n["key"] in papers_by_key]
        try:
            c.update(summaries.summarize_candidate(client, c, nearest))
        except Exception as exc:
            step(f"  summary failed for {c['title'][:60]}: {exc}")
            c.update(_failed_summary(exc))

    today = now.date().isoformat()
    stats = {"fetched": fetched, "new": len(cands), "scored": len(cands), "shown": len(selected)}
    digest = store.load_digest(today, root)
    if digest:
        known = {c["id"] for c in digest["candidates"]}
        digest["candidates"].extend(c for c in selected if c["id"] not in known)
        old = digest.get("stats") or {}
        digest["stats"] = {
            "fetched": {k: old.get("fetched", {}).get(k, 0) + v for k, v in fetched.items()},
            **{k: old.get(k, 0) + stats[k] for k in ("new", "scored", "shown")},
        }
        digest["generated_at"] = now.isoformat(timespec="seconds")
    else:
        digest = {"date": today, "generated_at": now.isoformat(timespec="seconds"),
                  "window": {"from": window_from.isoformat(timespec="seconds"), "to": window_to.isoformat(timespec="seconds")},
                  "stats": stats, "candidates": selected, "sent": []}
    store.save_digest(digest, root)
    store.record_pending(digest, root)
    result["digest"] = today
    result["expired"] = store.expire_pending(now.date(), EXPIRE_DAYS, root)
    # Re-read before saving: Desktop may have changed follow settings while we were fetching.
    latest = store.load_follow(root)
    latest["queries"] = follow.get("queries", {})
    latest["last_run"] = now.isoformat(timespec="seconds")
    store.save_follow(latest, root)
    return result
