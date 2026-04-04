"""Print subcategory suggestions from saved JSON."""
import json
from pathlib import Path

CACHE_PATH = Path(".cache/library_index.json")
SUBS_PATH = Path("subcategory_suggestions.json")

index = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
subs = json.loads(SUBS_PATH.read_text(encoding="utf-8"))

# Rebuild paper lookup
all_papers = {}
for cat_data in index["categories"].values():
    for p in cat_data["papers"]:
        all_papers[p["key"]] = p

for cat_name, subcats in sorted(subs.items(), key=lambda x: -sum(len(v) for v in x[1].values())):
    total = sum(len(v) for v in subcats.values())
    print(f"\n{'='*68}")
    print(f"  {cat_name}  ({total} papers  ->  {len(subcats)} subcategories)")
    print(f"{'='*68}")
    for sub_name, keys in sorted(subcats.items(), key=lambda x: -len(x[1])):
        print(f"\n  [{len(keys):3d}]  {sub_name}")
        for key in keys[:5]:
            p = all_papers.get(key, {})
            title = p.get("title", key)[:63]
            year = p.get("year", "") or ""
            year_str = f" ({year})" if year else ""
            print(f"          - {title}{year_str}")
        if len(keys) > 5:
            print(f"          ... and {len(keys)-5} more")
