import json
import mcp.types as types
from . import zotero_client as zc


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
]


def _text(content: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=content)]


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

    return _text(f"Unknown tool: {name}")
