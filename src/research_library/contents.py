"""Cited topic summaries for the app's Contents page.

Each topic (an index category) gets an overview and a note per subtopic that
cite its papers by Zotero key, e.g. [AB12CD34]. Sonnet writes them once and
they are cached in .cache/topic_summaries.json; a topic is only rewritten when
its papers or subtopics change. render_citations turns the keys into links.
"""
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .analysis import CACHE_PATH as INDEX_CACHE_PATH
from .links import _extract_year, paper_link
from .llm import SONNET, response_text

SUMMARIES_PATH = INDEX_CACHE_PATH.parent / "topic_summaries.json"
ABSTRACT_CHARS = 600

# One or more 8-character Zotero keys in square brackets: [AB12CD34] or [AB12CD34, EF56GH78]
_CITE_RE = re.compile(r"\s?\[([A-Z0-9]{8}(?:\s*[,;]\s*[A-Z0-9]{8})*)\]")

SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "subtopics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "summary": {"type": "string"}},
                "required": ["name", "summary"],
                "additionalProperties": False,
            },
        },
        "key_papers": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overview", "subtopics", "key_papers"],
    "additionalProperties": False,
}


# ── Citations ─────────────────────────────────────────────────────────────────

def short_cite(p: dict) -> str:
    """'Smith 2020', 'Smith & Jones 2020' or 'Smith et al. 2020'."""
    year = _extract_year(p.get("year") or "") or "n.d."
    surnames = [a.split(",")[0].strip() for a in p.get("authors") or [] if a.strip()]
    if not surnames:
        return f"{(p.get('title') or 'Untitled')[:40]} {year}"
    if len(surnames) == 1:
        who = surnames[0]
    elif len(surnames) == 2:
        who = f"{surnames[0]} & {surnames[1]}"
    else:
        who = f"{surnames[0]} et al."
    return f"{who} {year}"


def render_citations(text: str, papers_by_key: dict) -> str:
    """Replace [KEY] / [KEY1, KEY2] with linked short citations; drop unknown keys."""
    def link(match):
        keys = re.split(r"\s*[,;]\s*", match.group(1))
        cites = [f"[{short_cite(papers_by_key[k])}]({paper_link(papers_by_key[k])})"
                 for k in keys if k in papers_by_key]
        return f" ({'; '.join(cites)})" if cites else ""
    return _CITE_RE.sub(link, text)


# ── Writing summaries ─────────────────────────────────────────────────────────

def topic_fingerprint(cat: dict) -> str:
    """Changes when the topic's papers or subtopic grouping change."""
    keys = sorted(p["key"] for p in cat["papers"])
    subs = {name: sorted(ks) for name, ks in (cat.get("subcategories") or {}).items()}
    blob = json.dumps({"keys": keys, "subs": subs}, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _paper_line(p: dict) -> str:
    abstract = (p.get("abstract") or "").replace("\n", " ")[:ABSTRACT_CHARS]
    return f"[{p['key']}] {short_cite(p)} | {p['title']}" + (f" | {abstract}" if abstract else "")


def _prompt(name: str, cat: dict) -> str:
    by_key = {p["key"]: p for p in cat["papers"]}
    sections, grouped = [], set()
    for sub, keys in (cat.get("subcategories") or {}).items():
        lines = [_paper_line(by_key[k]) for k in keys if k in by_key]
        grouped.update(keys)
        if lines:
            sections.append(f"## {sub}\n" + "\n".join(lines))
    rest = [_paper_line(p) for p in cat["papers"] if p["key"] not in grouped]
    if rest:
        sections.append("## Other papers\n" + "\n".join(rest))
    return f"""You are writing the contents page of a personal research library on investing and finance.
Topic: "{name}" ({len(cat['papers'])} papers). The papers are listed below, grouped by subtopic,
each as [KEY] citation | title | abstract (abstract may be missing).

Write:
- overview: 150-250 words on what this body of work covers, where the papers agree, where they
  disagree, and the main open questions. Cite papers inline by key in square brackets, e.g. [AB12CD34]
  or [AB12CD34, EF56GH78]. Cite only keys from the list; every claim about a paper needs its key.
- subtopics: for each subtopic heading below (use the exact heading text, skip "Other papers"),
  2-3 sentences on what its papers contribute, with inline key citations.
- key_papers: the 3-6 keys a newcomer to this topic should read first.

Write plainly for an investment professional. Don't invent findings that aren't in the titles or abstracts.

{chr(10).join(sections)}"""


def summarize_topic(client, name: str, cat: dict) -> dict:
    """Ask Sonnet for one topic's cited overview, subtopic notes and key papers."""
    response = client.messages.create(
        model=SONNET,
        max_tokens=16000,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": _prompt(name, cat)}],
    )
    data = json.loads(response_text(response))
    valid = {p["key"] for p in cat["papers"]}
    return {
        "overview": data["overview"],
        "subtopics": {s["name"]: s["summary"] for s in data["subtopics"]},
        "key_papers": [k for k in data["key_papers"] if k in valid],
    }


def load_topic_summaries(path: Path | None = None) -> dict:
    """{topic name: summary dict}; empty when nothing has been written yet."""
    path = path or SUMMARIES_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("topics", {})


def build_topic_summaries(force: bool = False, progress=None, index_path: Path | None = None,
                          out_path: Path | None = None, client=None, max_workers: int = 4) -> dict:
    """Write summaries for new or changed topics. Returns {written, skipped, failed}."""
    index_path = index_path or INDEX_CACHE_PATH
    out_path = out_path or SUMMARIES_PATH
    step = progress or (lambda msg: None)
    if not index_path.exists():
        raise FileNotFoundError(f"No library index at {index_path}. Run `research-admin refresh` first.")
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    categories = json.loads(index_path.read_text(encoding="utf-8"))["categories"]
    existing = load_topic_summaries(out_path)
    topics = {name: existing[name] for name in categories if name in existing}  # drops removed topics

    todo = [name for name, cat in categories.items()
            if force or topics.get(name, {}).get("fingerprint") != topic_fingerprint(cat)]
    step(f"{len(todo)} of {len(categories)} topics need a summary")

    lock = threading.Lock()
    failed = []

    def save():
        out_path.write_text(json.dumps({"topics": topics}, indent=2, ensure_ascii=False), encoding="utf-8")

    def run(name):
        summary = summarize_topic(client, name, categories[name])
        summary.update(fingerprint=topic_fingerprint(categories[name]), model=SONNET,
                       generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        return summary

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run, name): name for name in todo}
        for future in as_completed(futures):
            name = futures[future]
            try:
                summary = future.result()
            except Exception as e:
                step(f"  FAIL {name}: {e}")
                failed.append(name)
                topics.pop(name, None)  # never show a summary written for an older paper set
                continue
            with lock:
                topics[name] = summary
                save()  # keep finished topics even if a later one crashes the run
            step(f"  done {name}")

    save()
    return {"written": len(todo) - len(failed), "skipped": len(categories) - len(todo),
            "failed": sorted(failed)}
