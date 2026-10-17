"""The six discovery tools as Claude Desktop sees them. tools.py dispatches here."""
from .. import analysis
from . import cards, keep, store

TOOL_NAMES = {"get_new_papers", "keep_new_papers", "dismiss_new_papers", "consider_paper",
              "get_followed_topics", "set_followed_topics"}


def _load_index() -> dict | None:
    return analysis.load_cached_index()


def _index_and_papers() -> tuple[dict, dict]:
    index = _load_index() or {"categories": {}}
    papers = {p["key"]: p for c in index["categories"].values() for p in c.get("papers", [])}
    return index, papers


def handle(name: str, arguments: dict) -> str | None:
    """Text for one of the discovery tools, or None if the name is not ours."""
    if name not in TOOL_NAMES:
        return None
    arguments = arguments or {}
    if name == "get_new_papers":
        return _get_new_papers(arguments.get("date"))
    if name == "keep_new_papers":
        results = keep.keep_candidates(list(arguments.get("ids") or []), attach_pdf=bool(arguments.get("attach_pdf", True)))
        return cards.format_keep_results(results)
    if name == "dismiss_new_papers":
        results = keep.dismiss_candidates(list(arguments.get("ids") or []), all_pending=bool(arguments.get("all", False)))
        return cards.format_dismiss_results(results)
    if name == "consider_paper":
        return _consider(arguments.get("reference", ""))
    if name == "get_followed_topics":
        return _followed()
    if name == "set_followed_topics":
        return _set_followed(list(arguments.get("follow") or []), list(arguments.get("unfollow") or []))
    return None


def _get_new_papers(date: str | None) -> str:
    _index, papers = _index_and_papers()
    if date:
        digest = store.load_digest(date)
        if not digest:
            return f"No digest for {date}."
        return cards.format_digest(digest, papers)
    dates = store.digest_dates()
    latest = store.load_digest(dates[0]) if dates else None
    return cards.format_inbox(store.pending_candidates(), latest, papers)


def _consider(reference: str) -> str:
    _index, papers = _index_and_papers()
    try:
        cand = keep.consider_reference(reference)
    except LookupError as exc:
        return str(exc)
    if cand.get("owned"):
        key = f" (Zotero {cand['zotero_key']})" if cand.get("zotero_key") else ""
        return f"Already in the library: {cand.get('title', '')}{key}"
    return "Added to today's inbox:\n\n" + cards.format_candidate(cand, papers) + "\n\nSay keep to file it, or dismiss."


def _topics_and_counts() -> tuple[list[str], dict[str, int]]:
    index, _papers = _index_and_papers()
    topics = list(index["categories"])
    counts = {t: len(index["categories"][t].get("papers", [])) for t in topics}
    return topics, counts


def _followed() -> str:
    topics, counts = _topics_and_counts()
    follow = store.load_follow()
    followed = store.followed_topics(follow, topics)
    unfollowed = [t for t in topics if t not in followed]
    return cards.format_followed(followed, unfollowed, counts)


def _set_followed(follow_names: list[str], unfollow_names: list[str]) -> str:
    topics, counts = _topics_and_counts()
    known = set(topics)
    unknown = [t for t in follow_names + unfollow_names if t not in known]
    follow = store.load_follow()
    unfollowed = set(follow.get("unfollowed") or [])
    unfollowed |= {t for t in unfollow_names if t in known}
    unfollowed -= {t for t in follow_names if t in known}
    follow["unfollowed"] = sorted(unfollowed)
    store.save_follow(follow)
    followed = store.followed_topics(follow, topics)
    text = cards.format_followed(followed, [t for t in topics if t not in followed], counts)
    if unknown:
        text = "Unknown topics: " + ", ".join(unknown) + "\n" + text
    return text
