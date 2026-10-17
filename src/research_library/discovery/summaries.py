"""The two Haiku calls in discovery: search phrases per topic, and a card per candidate.

Queries are cached in follow.json keyed by the topic's fingerprint (the same one
contents.py uses), so they cost one call per topic until the topic changes.
"""
import json

from ..contents import short_cite, topic_fingerprint
from ..llm import HAIKU, response_text

QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "why_it_fits": {"type": "string"},
        "recommendation": {"type": "string", "enum": ["keep", "maybe", "skip"]},
        "reason": {"type": "string"},
    },
    "required": ["summary", "why_it_fits", "recommendation", "reason"],
    "additionalProperties": False,
}

MAX_QUERIES = 6
SAMPLE_TITLES = 15


def _ask(client, prompt: str, schema: dict) -> dict:
    response = client.messages.create(
        model=HAIKU,
        max_tokens=2000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response_text(response))


# ── Search phrases ────────────────────────────────────────────────────────────

def generate_queries(client, name: str, cat: dict) -> list[str]:
    """4 to 6 short search phrases that would find new working papers in this topic."""
    subs = ", ".join((cat.get("subcategories") or {}).keys()) or "none"
    titles = "\n".join(f"- {p['title']}" for p in cat["papers"][:SAMPLE_TITLES])
    prompt = f"""A personal research library on investing and finance has a topic area called "{name}".
Its subtopics: {subs}.
Sample paper titles:
{titles}

Write 4 to 6 search phrases, each 2 to 4 words, that a researcher would type into arXiv or SSRN to
find new working papers in this area. Use the field's own vocabulary (e.g. "subscription lines",
"regime switching"), avoid generic words like "finance" or "model" on their own, and do not repeat
the same idea twice."""
    data = _ask(client, prompt, QUERY_SCHEMA)
    queries = [q.strip() for q in data.get("queries", []) if q and q.strip()]
    return queries[:MAX_QUERIES]


def ensure_queries(client, follow: dict, categories: dict, followed: list[str], force: bool = False,
                   progress=None) -> dict[str, list[str]]:
    """Queries for every followed topic, regenerating only when the topic changed (or on force)."""
    cache = follow.setdefault("queries", {})
    out: dict[str, list[str]] = {}
    for name in followed:
        cat = categories[name]
        fp = topic_fingerprint(cat)
        cached = cache.get(name) or {}
        if not force and cached.get("fingerprint") == fp and cached.get("queries"):
            out[name] = cached["queries"]
            continue
        if progress:
            progress(f"  writing search phrases for {name}")
        out[name] = generate_queries(client, name, cat)
        cache[name] = {"fingerprint": fp, "queries": out[name]}
    return out


# ── Candidate cards ───────────────────────────────────────────────────────────

def _nearest_lines(nearest_papers: list[dict]) -> str:
    lines = []
    for p in nearest_papers:
        abstract = (p.get("abstract") or "").replace("\n", " ")[:400]
        lines.append(f"- {short_cite(p)} | {p.get('title', '')}" + (f" | {abstract}" if abstract else ""))
    return "\n".join(lines) or "- (none)"


def summarize_candidate(client, cand: dict, nearest_papers: list[dict]) -> dict:
    """Summary (when there is an abstract), why it fits, and a keep/maybe/skip call."""
    extra = cand.get("extra") or {}
    if cand.get("abstract"):
        body = f"Abstract: {cand['abstract']}"
        summary_rule = "- summary: up to 120 words on what the paper does, its main finding and why it matters."
    else:
        details = []
        if extra.get("affiliations"):
            details.append("affiliations: " + ", ".join(map(str, extra["affiliations"])))
        if extra.get("pages") is not None:
            details.append(f"{extra['pages']} pages")
        if extra.get("downloads") is not None:
            details.append(f"{extra['downloads']} downloads so far")
        body = "No abstract is available for this paper, only the title" + (" and: " + "; ".join(details) if details else ".")
        summary_rule = "- summary: leave empty (there is no abstract to summarise)."
    prompt = f"""You screen new papers for a personal research library on investing and finance.

Candidate:
Title: {cand.get('title', '')}
Authors: {', '.join(cand.get('authors') or []) or 'unknown'}
Source: {cand.get('venue', '')} · {cand.get('published', '')}
{body}

The library papers it sits closest to:
{_nearest_lines(nearest_papers)}

Write, for an investment professional, plainly:
{summary_rule}
- why_it_fits: one or two sentences on how it relates to the nearest library papers, naming them as
  "Author Year". Do not invent findings that are not in the text above.
- recommendation: keep, maybe or skip, judged on whether it adds something to this library.
- reason: one sentence."""
    data = _ask(client, prompt, SUMMARY_SCHEMA)
    return {
        "summary": (data.get("summary") or "").strip() or None if cand.get("abstract") else None,
        "why_it_fits": (data.get("why_it_fits") or "").strip(),
        "recommendation": data.get("recommendation") or "maybe",
        "reason": (data.get("reason") or "").strip(),
    }
