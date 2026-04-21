"""
Library indexing pipeline: fetch all papers → deduplicate → categorise → summarise → cache.
"""
import json
import re
import os
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

CACHE_PATH = Path(__file__).parent.parent.parent / ".cache" / "library_index.json"


# ── Deduplication ─────────────────────────────────────────────────────────────

def _title_key(title: str) -> str:
    """Normalised title for dedup comparison."""
    return re.sub(r"[^\w\s]", "", title.lower()).strip()[:60]


def deduplicate(papers: list[dict]) -> list[dict]:
    """
    Remove duplicate papers. Strategy:
      1. Exact DOI match → keep the copy with an abstract (or the first seen).
      2. Normalised title match (first 60 chars, lowercase, no punctuation) → same rule.
    """
    seen_dois: dict[str, int] = {}   # doi → index in `unique`
    seen_titles: dict[str, int] = {} # title_key → index in `unique`
    unique: list[dict] = []

    for p in papers:
        doi = p["doi"].strip().lower() if p["doi"] else None
        tkey = _title_key(p["title"])

        # Check DOI collision
        if doi and doi in seen_dois:
            existing = unique[seen_dois[doi]]
            # Upgrade to this copy if it has an abstract and the existing one doesn't
            if p["abstract"] and not existing["abstract"]:
                unique[seen_dois[doi]] = p
            continue

        # Check title collision
        if tkey in seen_titles:
            existing = unique[seen_titles[tkey]]
            if p["abstract"] and not existing["abstract"]:
                unique[seen_titles[tkey]] = p
            continue

        # New unique paper
        idx = len(unique)
        unique.append(p)
        if doi:
            seen_dois[doi] = idx
        seen_titles[tkey] = idx

    return unique


# ── Categorisation ────────────────────────────────────────────────────────────

def categorize_papers(papers: list[dict]) -> dict[str, list[str]]:
    """
    Use Claude Haiku to assign each paper to a subject category.
    Returns {category_name: [paper_key, ...]}
    """
    client = anthropic.Anthropic()

    # Build a compact listing: key: title | abstract[:150]
    lines = []
    for p in papers:
        abstract_preview = (p["abstract"] or "")[:150].replace("\n", " ")
        lines.append(f'{p["key"]}: {p["title"]} | {abstract_preview}')
    listing = "\n".join(lines)

    prompt = f"""You are organising an academic research library with {len(papers)} papers.

Analyse the titles and abstract previews below and:
1. Identify 12–20 coherent, specific subject categories that capture the main research themes.
   Good categories are specific (e.g. "Regime-Switching & Macro Models") not generic (e.g. "Finance").
   Every paper must fit into exactly one category. Do NOT create an "Uncategorised" category.
2. Assign every single paper key to exactly one category. No paper may be omitted.

Return ONLY valid JSON in this exact structure (no markdown, no explanation):
{{
  "categories": ["Category A", "Category B", ...],
  "assignments": {{"PAPERKEY1": "Category A", "PAPERKEY2": "Category B", ...}}
}}

Papers:
{listing}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    data = json.loads(raw)
    assignments: dict[str, str] = data["assignments"]

    # Invert: category → [keys]
    categories: dict[str, list[str]] = {}
    for key, cat in assignments.items():
        categories.setdefault(cat, []).append(key)

    # Move any self-labelled "Uncategorised" papers into real categories
    stragglers = categories.pop("Uncategorised", [])

    # Also catch any papers missing from assignments entirely
    assigned_keys = set(assignments.keys())
    stragglers.extend(p["key"] for p in papers if p["key"] not in assigned_keys)

    if stragglers:
        real_cats = [c for c in categories.keys()]
        straggler_papers = [p for p in papers if p["key"] in stragglers]
        straggler_lines = "\n".join(
            f'{p["key"]}: {p["title"]} | {(p["abstract"] or "")[:100]}'
            for p in straggler_papers
        )
        cats_list = "\n".join(f"- {c}" for c in real_cats)
        rescue_prompt = f"""Assign each paper below to the single best-matching category from this list.
Return ONLY valid JSON: {{"PAPERKEY": "Category Name", ...}}

Categories:
{cats_list}

Papers:
{straggler_lines}"""
        rescue_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": rescue_prompt}],
        )
        rescue_raw = rescue_response.content[0].text.strip()
        if rescue_raw.startswith("```"):
            rescue_raw = re.sub(r"^```[a-z]*\n?", "", rescue_raw)
            rescue_raw = re.sub(r"\n?```$", "", rescue_raw)
        rescue_assignments = json.loads(rescue_raw)
        for key, cat in rescue_assignments.items():
            if cat in categories:
                categories[cat].append(key)
            else:
                categories[real_cats[0]].append(key)  # last-resort fallback

    return categories


# ── Subcategorisation ─────────────────────────────────────────────────────────

SUBCATEGORY_THRESHOLD = 20


def _subcategorize(cat_name: str, papers: list[dict]) -> dict[str, list[str]]:
    """
    Ask Claude Sonnet to cluster a large category into 3–6 subcategories.
    Returns {subcategory_name: [paper_key, ...]}. Adapted from the standalone
    suggest_subcategories.py script.
    """
    client = anthropic.Anthropic()

    lines = []
    for p in papers:
        abstract = (p.get("abstract") or "")[:120].replace("\n", " ")
        lines.append(f'{p["key"]}: {p["title"]} | {abstract}')

    prompt = f"""You are sub-categorising a group of {len(papers)} academic papers \
that all belong to the parent category "{cat_name}".

Identify 3–6 specific, meaningful subcategories that carve this group into coherent clusters.
Then assign every paper key to exactly one subcategory.

Rules:
- Subcategory names should be specific and descriptive (not generic like "Other" or "Miscellaneous")
- Every paper must be assigned — no paper left out
- Subcategory names should be shorter than the parent category name where possible

Return ONLY valid JSON:
{{
  "subcategories": ["Sub A", "Sub B", ...],
  "assignments": {{"PAPERKEY": "Sub A", ...}}
}}

Papers:
{chr(10).join(lines)}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {}
        data = json.loads(m.group(0))

    assignments = data.get("assignments", {})
    subcats: dict[str, list[str]] = {}
    for key, sub in assignments.items():
        subcats.setdefault(sub, []).append(key)

    # Rescue any unassigned paper into the largest subcategory so no paper is lost.
    assigned = set(assignments.keys())
    unassigned = [p["key"] for p in papers if p["key"] not in assigned]
    if unassigned and subcats:
        biggest = max(subcats, key=lambda s: len(subcats[s]))
        subcats[biggest].extend(unassigned)

    return subcats


# ── Summarisation ─────────────────────────────────────────────────────────────

def summarize_category(category: str, papers: list[dict]) -> str:
    """Write a 2–3 sentence summary of a category using paper titles and abstracts."""
    client = anthropic.Anthropic()

    paper_lines = []
    for p in papers:
        authors = ", ".join(p["authors"][:2]) + (" et al." if len(p["authors"]) > 2 else "")
        abstract = (p["abstract"] or "")[:300].replace("\n", " ")
        paper_lines.append(f'- {p["title"]} ({authors}, {p["year"] or "n.d."}): {abstract}')

    listing = "\n".join(paper_lines)

    prompt = f"""Write a 2–3 sentence academic summary of the following group of {len(papers)} papers \
in the category "{category}".

Focus on: what unifies them, the main methodological approaches, and the key research questions addressed.
Be specific and informative, not generic.

Papers:
{listing}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_index(progress_callback=None) -> dict:
    """
    Full pipeline: fetch → deduplicate → categorise → summarise → cache.
    progress_callback(message: str) is called at each stage if provided.
    Returns the index dict.
    """
    from .zotero_client import get_all_papers

    def progress(msg):
        if progress_callback:
            progress_callback(msg)

    progress("Fetching all papers from Zotero...")
    all_papers = get_all_papers()
    progress(f"Fetched {len(all_papers)} items. Deduplicating...")

    unique = deduplicate(all_papers)
    duplicates_removed = len(all_papers) - len(unique)
    progress(f"{len(unique)} unique papers ({duplicates_removed} duplicates removed). Categorising...")

    categories = categorize_papers(unique)
    progress(f"Identified {len(categories)} categories. Summarising each...")

    # Build a key→paper lookup
    paper_by_key = {p["key"]: p for p in unique}

    index_categories = {}
    for i, (cat_name, keys) in enumerate(sorted(categories.items())):
        progress(f"Summarising '{cat_name}' ({i+1}/{len(categories)})...")
        cat_papers = [paper_by_key[k] for k in keys if k in paper_by_key]
        summary = summarize_category(cat_name, cat_papers)

        subcategories: dict[str, list[str]] = {}
        if len(cat_papers) >= SUBCATEGORY_THRESHOLD:
            progress(f"  Subcategorising '{cat_name}' ({len(cat_papers)} papers)...")
            subcategories = _subcategorize(cat_name, cat_papers)

        index_categories[cat_name] = {
            "summary": summary,
            "subcategories": subcategories,
            "papers": [
                {
                    "key": p["key"],
                    "title": p["title"],
                    "authors": p["authors"],
                    "year": p["year"],
                    "abstract": p["abstract"],
                    "doi": p["doi"],
                }
                for p in sorted(cat_papers, key=lambda x: x["year"] or "0000", reverse=True)
            ],
        }

    index = {
        "generated_at": datetime.now().isoformat(),
        "total_fetched": len(all_papers),
        "unique_papers": len(unique),
        "duplicates_removed": duplicates_removed,
        "categories": index_categories,
    }

    # Cache to disk
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(index, indent=2))
    progress("Index saved to cache.")
    return index


def load_cached_index() -> dict | None:
    """Load index from cache if it exists."""
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return None
