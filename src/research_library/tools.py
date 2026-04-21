import json
import mcp.types as types
from . import zotero_client as zc
from .analysis import CACHE_PATH as INDEX_CACHE_PATH


TOOLS: list[types.Tool] = [
    types.Tool(
        name="search_papers",
        description="Search the Zotero library by keyword. Searches titles, authors, abstracts, and full-text where indexed.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "limit": {"type": "integer", "default": 20, "description": "Max results (default 20, max 100)"},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="get_paper",
        description="Get full metadata for a specific paper by its Zotero item key.",
        inputSchema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Zotero item key (e.g. 'A1B2C3D4')"},
            },
            "required": ["key"],
        },
    ),
    types.Tool(
        name="list_collections",
        description="List all collections (folders) in the Zotero library with their names and item counts.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="get_collection_papers",
        description="Get papers within a named Zotero collection. Supports partial name matching.",
        inputSchema={
            "type": "object",
            "properties": {
                "collection_name": {"type": "string", "description": "Name (or partial name) of the collection"},
                "limit": {"type": "integer", "default": 50, "description": "Max results"},
            },
            "required": ["collection_name"],
        },
    ),
    types.Tool(
        name="get_recent_papers",
        description="Get the most recently added papers in the library.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20, "description": "Number of papers to return"},
            },
        },
    ),
    types.Tool(
        name="get_annotations",
        description="Get all notes and annotations (highlights, comments) attached to a paper.",
        inputSchema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Zotero item key of the parent paper"},
            },
            "required": ["key"],
        },
    ),
    types.Tool(
        name="get_tags",
        description="List all tags used in the library. Useful for discovering how the library is organized.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="get_papers_by_tag",
        description="Get papers that have a specific tag.",
        inputSchema={
            "type": "object",
            "properties": {
                "tag": {"type": "string", "description": "Tag to filter by (exact match)"},
                "limit": {"type": "integer", "default": 50, "description": "Max results"},
            },
            "required": ["tag"],
        },
    ),
    types.Tool(
        name="get_fulltext",
        description="Get the full text of a paper from Zotero's indexed content. Returns the complete text if Zotero has indexed the PDF attachment.",
        inputSchema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Zotero item key of the parent paper"},
                "char_limit": {"type": "integer", "default": 40000, "description": "Max characters to return"},
            },
            "required": ["key"],
        },
    ),
    types.Tool(
        name="get_fulltext_batch",
        description="Get full text for multiple papers at once. Useful for answering questions across a set of papers.",
        inputSchema={
            "type": "object",
            "properties": {
                "keys": {"type": "array", "items": {"type": "string"}, "description": "List of Zotero item keys"},
                "max_papers": {"type": "integer", "default": 5, "description": "Max papers to fetch full text for"},
                "char_limit": {"type": "integer", "default": 3000, "description": "Max characters per paper"},
            },
            "required": ["keys"],
        },
    ),
    types.Tool(
        name="semantic_search",
        description=(
            "Semantic (embedding-based) search over the library using Voyage AI. "
            "Finds papers by meaning rather than keyword match — prefer this over search_papers "
            "for conceptual queries. Requires the embeddings cache (built via `research-admin embed`)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query"},
                "limit": {"type": "integer", "default": 20, "description": "Max results"},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="list_categories",
        description=(
            "List all categories (and their subcategories) from the Claude-generated library index. "
            "Each category has a short summary and paper count. Useful for orienting to the shape of the library."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="get_category_papers",
        description=(
            "Get papers within a named category from the cached library index. "
            "Optionally filter to a single subcategory. Reads the cache directly (no Zotero round-trip)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Category name (exact match from list_categories)"},
                "subcategory": {"type": "string", "description": "Optional subcategory name to filter by"},
                "limit": {"type": "integer", "default": 50, "description": "Max papers to return"},
            },
            "required": ["category"],
        },
    ),
    types.Tool(
        name="ask_library",
        description=(
            "Answer a natural-language question using the categorised, semantically-indexed library. "
            "Picks the most relevant categories, ranks papers within them, and has Claude draft a "
            "cited answer. The flagship research-library tool — prefer it for broad or thematic questions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to answer using the library"},
                "top_k_cats": {"type": "integer", "default": 3, "description": "How many categories to draw from"},
                "top_k_papers": {"type": "integer", "default": 8, "description": "How many papers to include in context"},
                "use_fulltext": {"type": "boolean", "default": False, "description": "Fetch full text for the top 3 papers (slower, more thorough)"},
            },
            "required": ["question"],
        },
    ),
]


def _text(content: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=content)]


def _load_cached_index() -> dict | None:
    if not INDEX_CACHE_PATH.exists():
        return None
    return json.loads(INDEX_CACHE_PATH.read_text(encoding="utf-8"))


def _hydrate_paper(key: str, index: dict | None) -> dict | None:
    """Look up a paper key in the cached index first (fast), fall back to Zotero."""
    if index is not None:
        for cat_data in index.get("categories", {}).values():
            for p in cat_data.get("papers", []):
                if p.get("key") == key:
                    return p
    return zc.get_paper(key)


async def handle_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_papers":
        results = zc.search_papers(arguments["query"], arguments.get("limit", 20))
        if not results:
            return _text("No papers found.")
        return _text(json.dumps(results, indent=2))

    if name == "get_paper":
        paper = zc.get_paper(arguments["key"])
        if not paper:
            return _text(f"No paper found with key '{arguments['key']}'.")
        return _text(json.dumps(paper, indent=2))

    if name == "list_collections":
        cols = zc.list_collections()
        if not cols:
            return _text("No collections found.")
        return _text(json.dumps(cols, indent=2))

    if name == "get_collection_papers":
        papers = zc.get_collection_papers(arguments["collection_name"], arguments.get("limit", 50))
        if not papers:
            return _text(f"No papers found in collection matching '{arguments['collection_name']}'.")
        return _text(json.dumps(papers, indent=2))

    if name == "get_recent_papers":
        papers = zc.get_recent_papers(arguments.get("limit", 20))
        return _text(json.dumps(papers, indent=2))

    if name == "get_annotations":
        annotations = zc.get_annotations(arguments["key"])
        if not annotations:
            return _text("No notes or annotations found for this item.")
        return _text(json.dumps(annotations, indent=2))

    if name == "get_tags":
        tags = zc.get_tags()
        return _text(json.dumps(tags, indent=2))

    if name == "get_papers_by_tag":
        papers = zc.get_papers_by_tag(arguments["tag"], arguments.get("limit", 50))
        if not papers:
            return _text(f"No papers found with tag '{arguments['tag']}'.")
        return _text(json.dumps(papers, indent=2))

    if name == "get_fulltext":
        ft = zc.get_fulltext(arguments["key"], arguments.get("char_limit", 40000))
        if not ft["content"]:
            return _text(f"Full text not available: {ft['error']}")
        note = f" [truncated from {ft['total_chars']:,} chars]" if ft["truncated"] else ""
        return _text(f"{ft['content']}{note}")

    if name == "get_fulltext_batch":
        fake_papers = [{"key": k} for k in arguments["keys"]]
        papers = zc.get_fulltext_batch(fake_papers, arguments.get("max_papers", 5), arguments.get("char_limit", 3000))
        out = []
        for p in papers:
            ft = p.get("fulltext", {})
            if ft.get("content"):
                out.append({"key": p["key"], "content": ft["content"], "truncated": ft["truncated"]})
            else:
                out.append({"key": p["key"], "error": ft.get("error", "unknown")})
        return _text(json.dumps(out, indent=2))

    if name == "semantic_search":
        return await _handle_semantic_search(arguments)

    if name == "list_categories":
        return _handle_list_categories()

    if name == "get_category_papers":
        return _handle_get_category_papers(arguments)

    if name == "ask_library":
        return await _handle_ask_library(arguments)

    return _text(f"Unknown tool: {name}")


# ── Index-aware / semantic handlers ───────────────────────────────────────────

async def _handle_semantic_search(arguments: dict) -> list[types.TextContent]:
    from . import embeddings
    try:
        hits = embeddings.search(arguments["query"], top_k=arguments.get("limit", 20))
    except FileNotFoundError as e:
        return _text(str(e))
    index = _load_cached_index()
    papers = []
    for h in hits:
        paper = _hydrate_paper(h["key"], index)
        if paper:
            papers.append({**paper, "score": round(h["score"], 4), "category": h["category"]})
    if not papers:
        return _text("No papers found.")
    return _text(json.dumps(papers, indent=2))


def _handle_list_categories() -> list[types.TextContent]:
    index = _load_cached_index()
    if not index:
        return _text("No library index cached. Run `research-admin refresh` first.")
    out = []
    for name, data in sorted(index["categories"].items(),
                             key=lambda x: -len(x[1].get("papers", []))):
        subs = data.get("subcategories") or {}
        out.append({
            "name": name,
            "summary": data.get("summary", ""),
            "paper_count": len(data.get("papers", [])),
            "subcategories": [
                {"name": sn, "paper_count": len(keys)}
                for sn, keys in sorted(subs.items(), key=lambda kv: -len(kv[1]))
            ],
        })
    return _text(json.dumps(out, indent=2))


def _handle_get_category_papers(arguments: dict) -> list[types.TextContent]:
    index = _load_cached_index()
    if not index:
        return _text("No library index cached. Run `research-admin refresh` first.")
    cat_name = arguments["category"]
    cat_data = index["categories"].get(cat_name)
    if cat_data is None:
        available = ", ".join(sorted(index["categories"].keys())[:10])
        return _text(f"Category '{cat_name}' not found. Known categories include: {available}...")

    papers = cat_data.get("papers", [])
    subcategory = arguments.get("subcategory")
    if subcategory:
        sub_keys = (cat_data.get("subcategories") or {}).get(subcategory)
        if sub_keys is None:
            subs = list((cat_data.get("subcategories") or {}).keys())
            return _text(f"Subcategory '{subcategory}' not found in '{cat_name}'. "
                         f"Known subcategories: {', '.join(subs) or '(none)'}")
        sub_key_set = set(sub_keys)
        papers = [p for p in papers if p["key"] in sub_key_set]

    papers = papers[: arguments.get("limit", 50)]
    return _text(json.dumps(papers, indent=2))


async def _handle_ask_library(arguments: dict) -> list[types.TextContent]:
    from . import embeddings
    import anthropic

    question = arguments["question"]
    top_k_cats = arguments.get("top_k_cats", 3)
    top_k_papers = arguments.get("top_k_papers", 8)
    use_fulltext = arguments.get("use_fulltext", False)

    try:
        ranking = embeddings.search_by_category(
            question, top_k_cats=top_k_cats, top_k_papers=top_k_papers
        )
    except FileNotFoundError as e:
        return _text(str(e))

    index = _load_cached_index()
    hydrated = []
    for hit in ranking["papers"]:
        paper = _hydrate_paper(hit["key"], index)
        if paper:
            hydrated.append({**paper, "score": hit["score"], "category": hit["category"]})

    if not hydrated:
        return _text("No relevant papers found for this question.")

    # Optionally fetch full text for the top 3
    fulltext_map: dict[str, str] = {}
    if use_fulltext:
        top_three = hydrated[:3]
        batch = zc.get_fulltext_batch(top_three, max_papers=3, char_limit=4000)
        for p in batch:
            ft = p.get("fulltext", {})
            if ft.get("content"):
                fulltext_map[p["key"]] = ft["content"]

    # Build context for Claude
    parts = []
    for p in hydrated:
        header = (
            f"[{p['key']}] {p['title']} ({p.get('year') or 'n.d.'}) — "
            f"category: {p['category']}"
        )
        body = fulltext_map.get(p["key"]) or p.get("abstract") or "(no abstract)"
        parts.append(f"{header}\n{body}")
    context = "\n\n---\n\n".join(parts)

    cat_list = ", ".join(f"{c['name']} ({c['score']:.2f})" for c in ranking["categories"])
    prompt = (
        f"Question: {question}\n\n"
        f"The following papers were drawn from these library categories (by semantic relevance): "
        f"{cat_list}.\n\n"
        f"Answer the question using only these papers. Cite specific papers by their key in square "
        f"brackets, e.g. [ABC123]. If the papers do not address the question, say so.\n\n"
        f"Papers:\n{context}"
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text

    payload = {
        "answer": answer,
        "categories_consulted": ranking["categories"],
        "papers_consulted": [
            {"key": p["key"], "title": p["title"], "year": p.get("year"),
             "category": p["category"], "score": round(p["score"], 4)}
            for p in hydrated
        ],
        "used_fulltext_for": list(fulltext_map.keys()),
    }
    return _text(json.dumps(payload, indent=2))
