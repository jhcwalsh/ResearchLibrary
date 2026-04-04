"""
For each category with >20 papers, ask Claude to suggest subcategories.
Prints a structured report without modifying the cache.
"""
import json
import re
from pathlib import Path
import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

CACHE_PATH = Path(".cache/library_index.json")
THRESHOLD = 20

index = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
client = anthropic.Anthropic()

large_cats = {
    k: v for k, v in index["categories"].items()
    if len(v["papers"]) >= THRESHOLD
}
print(f"Analysing {len(large_cats)} categories with {THRESHOLD}+ papers...\n")

results = {}

for cat_name, cat_data in sorted(large_cats.items(), key=lambda x: -len(x[1]["papers"])):
    papers = cat_data["papers"]
    print(f"  [{len(papers):3d}] {cat_name}...", end="", flush=True)

    lines = []
    for p in papers:
        abstract = (p.get("abstract") or "")[:120].replace("\n", " ")
        lines.append(f'{p["key"]}: {p["title"]} | {abstract}')

    prompt = f"""You are sub-categorising a group of {len(papers)} academic finance/investment papers \
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

    if not raw:
        print(f"\n  WARNING: empty response for {cat_name}, skipping")
        results[cat_name] = {"(no subcategories suggested)": papers}
        continue

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # Try to extract JSON object from within the response
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
        else:
            print(f"\n  WARNING: could not parse JSON for {cat_name}: {e}")
            print(f"  Raw (first 300 chars): {raw[:300]}")
            results[cat_name] = {"(parse error)": papers}
            continue
    assignments = data["assignments"]

    # Group papers by subcategory
    subcats: dict[str, list] = {}
    paper_by_key = {p["key"]: p for p in papers}
    for key, sub in assignments.items():
        subcats.setdefault(sub, []).append(paper_by_key.get(key, {"key": key, "title": key}))

    # Any unassigned papers
    assigned_keys = set(assignments.keys())
    unassigned = [p for p in papers if p["key"] not in assigned_keys]
    if unassigned:
        subcats.setdefault("_Unassigned", []).extend(unassigned)

    results[cat_name] = subcats
    # Save incrementally so a later crash doesn't lose work
    out_path = Path("subcategory_suggestions.json")
    out_path.write_text(json.dumps(
        {cat: {sub: [p["key"] for p in pps] for sub, pps in subcats.items()}
         for cat, subcats in results.items()},
        indent=2
    ), encoding="utf-8")
    print(f" {len(subcats)} subcategories")

# ── Print report ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUBCATEGORY SUGGESTIONS")
print("=" * 70)

for cat_name, subcats in sorted(results.items(), key=lambda x: -sum(len(v) for v in x[1].values())):
    total = sum(len(v) for v in subcats.values())
    print(f"\n{'-'*70}")
    print(f"  {cat_name}  ({total} papers)")
    print(f"{'-'*70}")
    for sub_name, sub_papers in sorted(subcats.items(), key=lambda x: -len(x[1])):
        print(f"\n  [{len(sub_papers):3d}]  {sub_name}")
        for p in sub_papers[:5]:
            year = p.get("year", "") or ""
            year_str = f" ({year})" if year else ""
            print(f"          - {p['title'][:65]}{year_str}")
        if len(sub_papers) > 5:
            print(f"          ... and {len(sub_papers)-5} more")

# Save for reference
out_path = Path("subcategory_suggestions.json")
out_path.write_text(json.dumps(
    {cat: {sub: [p["key"] for p in papers] for sub, papers in subcats.items()}
     for cat, subcats in results.items()},
    indent=2
), encoding="utf-8")
print(f"\n\nFull assignments saved to {out_path}")
