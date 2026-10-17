"""Plain-text cards for Claude Desktop: the inbox, a single candidate, keep results, followed topics."""
from ..contents import short_cite


def _nearest_names(cand: dict, papers_by_key: dict) -> list[str]:
    return [short_cite(papers_by_key[n["key"]]) for n in cand.get("nearest", []) if n.get("key") in papers_by_key]


def format_candidate(cand: dict, papers_by_key: dict, n: int | None = None) -> str:
    """One card: id and title, a facts line, then summary, fit, nearest owned papers and the link."""
    prefix = f"{n}. " if n is not None else ""
    lines = [f"{prefix}[{cand['id']}] {cand.get('title', '')}"]
    facts = [short_cite(cand), cand.get("venue") or cand.get("source", ""), cand.get("published", "")]
    facts = [f for f in facts if f]
    if cand.get("fit") is not None:
        facts.append(f"fit {cand['fit']:.2f}")
    if cand.get("recommendation"):
        facts.append(f"recommend: {cand['recommendation']}")
    lines.append(" · ".join(facts))

    extra = cand.get("extra") or {}
    reason = cand.get("reason") or ""
    if cand.get("summary"):
        lines.append(f"Summary: {cand['summary']}")
    elif not cand.get("abstract"):
        where = "the SSRN feed" if cand.get("source") == "ssrn" or cand.get("ssrn_id") else "this source"
        details = []
        if extra.get("pages") is not None:
            details.append(f"{extra['pages']} pages")
        if extra.get("downloads") is not None:
            details.append(f"{extra['downloads']} downloads")
        lines.append(f"No abstract on {where}" + (f" ({', '.join(details)})." if details else "."))
    elif reason.startswith("summary failed"):
        lines.append(f"Summary unavailable ({reason}).")
    if cand.get("why_it_fits"):
        lines.append(f"Why it fits: {cand['why_it_fits']}")
    names = _nearest_names(cand, papers_by_key)
    if names:
        lines.append("Nearest owned: " + "; ".join(names))
    if reason and not reason.startswith("summary failed"):
        lines.append(f"Reason: {reason}")
    if cand.get("url"):
        lines.append(cand["url"])
    return lines[0] + "\n" + "\n".join("  " + line for line in lines[1:])


def _sections(cands: list[dict], papers_by_key: dict) -> str:
    by_topic: dict[str, list[dict]] = {}
    for c in cands:
        by_topic.setdefault(c.get("topic") or "Unsorted", []).append(c)
    parts = []
    for topic, items in by_topic.items():
        cards = "\n\n".join(format_candidate(c, papers_by_key) for c in items)
        parts.append(f"## {topic}\n\n{cards}")
    return "\n\n".join(parts)


def format_inbox(cands: list[dict], latest: dict | None, papers_by_key: dict) -> str:
    """Everything pending, newest digest first, grouped by topic."""
    if not cands:
        if latest is None:
            return "No digest has been written yet. Run `research-admin discover` to make one."
        return ("No pending papers. The daily run writes a digest each morning; "
                "run `research-admin discover` to make one now.")
    header = f"Inbox: {len(cands)} pending papers"
    if latest:
        scored = (latest.get("stats") or {}).get("scored", 0)
        header += f" · latest digest {latest.get('date')}, {scored} candidates scored"
    hint = "Say which ids to keep or dismiss."
    return f"{header}\n{hint}\n\n{_sections(cands, papers_by_key)}"


def format_digest(digest: dict, papers_by_key: dict) -> str:
    """One day's digest as written, whatever has happened to its candidates since."""
    cands = list(digest.get("candidates", [])) + list(digest.get("sent", []))
    stats = digest.get("stats") or {}
    header = f"Digest {digest.get('date')}: {len(cands)} papers shown, {stats.get('scored', 0)} candidates scored"
    if not cands:
        return header + ". Nothing cleared the fit floor that day."
    return f"{header}\n\n{_sections(cands, papers_by_key)}"


def format_keep_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        line = f"- [{r['id']}] {r.get('title') or '(unknown)'}: {r['status']}"
        if r.get("zotero_key"):
            line += f" · Zotero {r['zotero_key']}"
        if r.get("status") in ("kept", "already_owned"):
            line += f" · PDF {r.get('pdf')} · index {r.get('index')}"
        if r.get("message"):
            line += f" · {r['message']}"
        lines.append(line)
    return "\n".join(lines) or "Nothing to do."


def format_dismiss_results(results: list[dict]) -> str:
    return "\n".join(f"- [{r['id']}] {r.get('title') or '(unknown)'}: {r['status']}" for r in results) or "Nothing to do."


def format_followed(followed: list[str], unfollowed: list[str], counts: dict[str, int]) -> str:
    total = len(followed) + len(unfollowed)
    lines = [f"Following {len(followed)} of {total} topics:"]
    lines += [f"- {t} ({counts.get(t, 0)} papers)" for t in followed]
    if unfollowed:
        lines.append("Not following:")
        lines += [f"- {t} ({counts.get(t, 0)} papers)" for t in unfollowed]
    return "\n".join(lines)
