"""Files under .cache/discovery/: follow settings, daily digests and the ledger.

follow.json      which topics are followed (all by default, minus `unfollowed`),
                 cached search queries per topic, thresholds, last run time.
digest-DATE.json everything one run found and scored, plus `sent` candidates
                 from consider_paper. Candidates carry their embedding vector
                 so `keep` can update the index without re-embedding.
seen.json        ledger {candidate id: {status, date, digest, title, zotero_key}}
                 with status pending | kept | dismissed | expired. Only candidates
                 that were put in front of James are recorded.
"""
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from ..analysis import CACHE_PATH as INDEX_CACHE_PATH, _title_key
from ..zotero_client import _normalize_doi

DISCOVERY_DIR = INDEX_CACHE_PATH.parent / "discovery"

DEFAULTS = {
    "unfollowed": [],
    "queries": {},
    "ssrn_bindings": [203],
    "min_fit": 0.62,
    "min_fit_title_only": 0.55,
    "per_topic": 3,
    "max_total": 15,
    "last_run": None,
}


def _root(root: Path | None) -> Path:
    return Path(root) if root else DISCOVERY_DIR


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Candidate ids ─────────────────────────────────────────────────────────────

def candidate_id(cand: dict) -> str:
    """Stable 12-hex id: DOI, else arXiv id, else SSRN id, else normalised title."""
    doi = _normalize_doi(cand.get("doi") or "").lower()
    if doi:
        seed = "doi:" + doi
    elif cand.get("arxiv_id"):
        seed = "arxiv:" + str(cand["arxiv_id"])
    elif cand.get("ssrn_id"):
        seed = "ssrn:" + str(cand["ssrn_id"])
    else:
        seed = "title:" + _title_key(cand.get("title") or "")
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]


# ── Follow settings ───────────────────────────────────────────────────────────

def load_follow(root: Path | None = None) -> dict:
    data = _read_json(_root(root) / "follow.json", {})
    return {**DEFAULTS, **data}


def save_follow(follow: dict, root: Path | None = None) -> None:
    _write_json(_root(root) / "follow.json", follow)


def followed_topics(follow: dict, index_topics: list[str]) -> list[str]:
    """Index order, minus the topics James has unfollowed."""
    unfollowed = set(follow.get("unfollowed") or [])
    return [t for t in index_topics if t not in unfollowed]


# ── Ledger ────────────────────────────────────────────────────────────────────

def load_ledger(root: Path | None = None) -> dict:
    return _read_json(_root(root) / "seen.json", {})


def save_ledger(ledger: dict, root: Path | None = None) -> None:
    _write_json(_root(root) / "seen.json", ledger)


def set_status(cand_id: str, status: str, root: Path | None = None, zotero_key: str | None = None) -> None:
    ledger = load_ledger(root)
    entry = ledger.setdefault(cand_id, {"title": "", "digest": None, "zotero_key": None})
    entry["status"] = status
    entry["date"] = _now()
    if zotero_key:
        entry["zotero_key"] = zotero_key
    save_ledger(ledger, root)


def expire_pending(today: date, days: int = 14, root: Path | None = None) -> int:
    """Pending candidates from digests older than `days` become `expired`."""
    ledger = load_ledger(root)
    expired = 0
    for entry in ledger.values():
        if entry.get("status") != "pending" or not entry.get("digest"):
            continue
        age = (today - date.fromisoformat(entry["digest"])).days
        if age > days:
            entry["status"] = "expired"
            entry["date"] = _now()
            expired += 1
    if expired:
        save_ledger(ledger, root)
    return expired


# ── Digests ───────────────────────────────────────────────────────────────────

def digest_path(date_str: str, root: Path | None = None) -> Path:
    return _root(root) / f"digest-{date_str}.json"


def load_digest(date_str: str, root: Path | None = None) -> dict | None:
    return _read_json(digest_path(date_str, root), None)


def save_digest(digest: dict, root: Path | None = None) -> None:
    _write_json(digest_path(digest["date"], root), digest)


def digest_dates(root: Path | None = None) -> list[str]:
    """Digest dates on disk, newest first."""
    base = _root(root)
    if not base.exists():
        return []
    dates = [p.stem[len("digest-"):] for p in base.glob("digest-*.json")]
    return sorted(dates, reverse=True)


def record_pending(digest: dict, root: Path | None = None) -> None:
    """Every candidate in the digest enters the ledger as pending, unless already known."""
    ledger = load_ledger(root)
    for cand in list(digest.get("candidates", [])) + list(digest.get("sent", [])):
        if cand["id"] in ledger:
            continue
        ledger[cand["id"]] = {"status": "pending", "date": _now(), "digest": digest["date"],
                              "title": cand.get("title", ""), "zotero_key": None}
    save_ledger(ledger, root)


def _candidates_in(digest: dict) -> list[dict]:
    out = []
    for cand in list(digest.get("candidates", [])) + list(digest.get("sent", [])):
        out.append({**cand, "digest": digest["date"]})
    return out


def pending_candidates(root: Path | None = None) -> list[dict]:
    """Candidates still awaiting a decision, newest digest first."""
    ledger = load_ledger(root)
    pending_ids = {cid for cid, e in ledger.items() if e.get("status") == "pending"}
    out = []
    for date_str in digest_dates(root):
        digest = load_digest(date_str, root) or {}
        out.extend(c for c in _candidates_in(digest) if c["id"] in pending_ids)
    return out


def find_candidate(cand_id: str, root: Path | None = None) -> dict | None:
    """The candidate dict (with its digest date) for an id the ledger knows."""
    entry = load_ledger(root).get(cand_id)
    if not entry:
        return None
    dates = [entry["digest"]] if entry.get("digest") else digest_dates(root)
    for date_str in dates:
        digest = load_digest(date_str, root)
        if not digest:
            continue
        for cand in _candidates_in(digest):
            if cand["id"] == cand_id:
                return cand
    return None
