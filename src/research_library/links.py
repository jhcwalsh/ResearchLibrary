"""Generate paper_index.md from the cached library index."""
import json
import re
import urllib.parse
from pathlib import Path

from .analysis import CACHE_PATH as INDEX_CACHE_PATH

DEFAULT_OUTPUT_PATH = INDEX_CACHE_PATH.parent.parent / "paper_index.md"


def _extract_year(raw: str) -> str:
    m = re.search(r"\b(19|20)\d{2}\b", raw or "")
    return m.group(0) if m else ""


def _scholar_url(title: str, authors: list) -> str:
    query = title
    if authors:
        query += " " + authors[0].split(",")[0]
    return "https://scholar.google.com/scholar?q=" + urllib.parse.quote(query)


def generate_links(output_path: Path | None = None) -> dict:
    """Write a markdown index with clickable links for every paper. Returns
    {written, direct_links, scholar_links, total_papers}."""
    if not INDEX_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No library index at {INDEX_CACHE_PATH}. Run `research-admin refresh` first."
        )
    output_path = output_path or DEFAULT_OUTPUT_PATH
    index = json.loads(INDEX_CACHE_PATH.read_text(encoding="utf-8"))

    lines = [
        "# Research Library Index\n",
        f"{index['unique_papers']} papers · {len(index['categories'])} categories · "
        f"Generated {index['generated_at'][:10]}\n",
        "Links: **direct** (DOI/URL) where available, **[Scholar]** search otherwise.\n",
        "---",
    ]

    direct_links = 0
    scholar_links = 0

    for cat_name, cat_data in sorted(index["categories"].items(),
                                     key=lambda x: -len(x[1]["papers"])):
        papers = cat_data["papers"]
        lines.append(f"\n## {cat_name} ({len(papers)} papers)\n")
        lines.append(f"*{cat_data['summary']}*\n")

        for p in papers:
            authors = p.get("authors", [])
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            year = _extract_year(p.get("year", "") or "")
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
                link = _scholar_url(p["title"], authors)
                badge = r" \[Scholar\]"
                scholar_links += 1

            author_line = f"  *{author_str}*" if author_str else ""
            lines.append(f"- [{p['title']}]({link}){year_str}{badge}{author_line}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "written": str(output_path),
        "direct_links": direct_links,
        "scholar_links": scholar_links,
        "total_papers": index["unique_papers"],
    }
