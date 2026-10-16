"""Claude model IDs and response reading shared by the app, index builder and MCP tools.

The 5.5 models think by default, so a reply can start with a `thinking` block
(empty text under the default display) and content[0] is not the answer.
Thinking also counts toward max_tokens, so callers leave room for it.
"""

HAIKU = "claude-haiku-5-5"    # quick extraction and classification
SONNET = "claude-sonnet-5-5"  # summaries and cited answers


class RefusalError(RuntimeError):
    """Claude declined the request (stop_reason == "refusal")."""


def response_text(message) -> str:
    """All text blocks of a reply, joined; raises if Claude declined or wrote no text."""
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) or "unspecified"
        raise RefusalError(f"Claude declined this request (category: {category})")
    text = "".join(b.text for b in message.content if b.type == "text")
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason: {message.stop_reason})")
    return text
