"""Generate a markdown file with links to all papers in the index."""
import json
import re
import urllib.parse
from pathlib import Path

CACHE_PATH = Path("C:/Users/james/PycharmProjects/ResearchLibrary/.cache/library_index.json")
OUTPUT_PATH = Path("C:/Users/james/PycharmProjects/ResearchLibrary/paper_index.md")

index = json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def extract_year(raw: str) -> str:
    """Extract 4-digit year from any date string."""
    m = re.search(r"\b(19|20)\d{2}\b", raw or "")
    return m.group(0) if m else ""


def scholar_url(title: str, authors: list) -> str:
    query = title
    if authors:
        query += " " + authors[0].split(",")[0]
    return "https://scholar.google.com/scholar?q=" + urllib.parse.quote(query)


lines = [
    "# Research Library Index\n",
    f"{index['unique_papers']} papers · {len(index['categories'])} categories · "
    f"Generated {index['generated_at'][:10]}\n",
    "Links: **direct** (DOI/URL) where available, **[Scholar]** search otherwise.\n",
    "---",
]

direct_links = 0
scholar_links = 0

for cat_name, cat_data in sorted(index["categories"].items(), key=lambda x: -len(x[1]["papers"])):
    papers = cat_data["papers"]
    lines.append(f"\n## {cat_name} ({len(papers)} papers)\n")
    lines.append(f"*{cat_data['summary']}*\n")

    for p in papers:
        authors = p.get("authors", [])
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."

        # Fix year — may be stored with month/day or as "January 2024" etc.
        year = extract_year(p.get("year", "") or "")
        year_str = f" ({year})" if year else ""

        doi = (p.get("doi") or "").strip()
        url = (p.get("url") or "").strip()

        if doi:
            link = f"https://doi.org/{doi}"
            badge = ""
            direct_links += 1
        elif url:
            link = url
            badge = ""
            direct_links += 1
        else:
            link = scholar_url(p["title"], authors)
            badge = r" \[Scholar\]"
            scholar_links += 1

        author_line = f"  *{author_str}*" if author_str else ""
        lines.append(f"- [{p['title']}]({link}){year_str}{badge}{author_line}")

OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

print(f"Written : {OUTPUT_PATH}")
print(f"Total   : {index['unique_papers']} papers")
print(f"Direct  : {direct_links}  (DOI or URL)")
print(f"Scholar : {scholar_links}  (search fallback)")
