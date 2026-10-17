"""Tests for the discovery cards and the six MCP tools."""
import pytest

from research_library import tools
from research_library.discovery import cards, handlers, store


def _cand(title, topic="Big", abstract="abs", **kw):
    c = {"source": "arxiv", "title": title, "authors": ["Doe, Jane", "Roe, Rick"], "year": "2026", "published": "2026-10-15",
         "abstract": abstract, "doi": "", "url": "https://arxiv.org/abs/1", "pdf_url": "", "arxiv_id": title, "ssrn_id": "",
         "venue": "arXiv q-fin.PM", "extra": {"pages": 52, "downloads": 11},
         "topic": topic, "fit": 0.7123, "nearest": [{"key": "B1", "score": 0.8, "category": "Big"}], "vector": [1, 0],
         "summary": "S", "why_it_fits": "W", "recommendation": "keep", "reason": "R"}
    c.update(kw)
    c["id"] = store.candidate_id(c)
    return c


PAPERS = {"B1": {"key": "B1", "title": "Owned", "authors": ["Smith, A"], "year": "2020"}}


def test_format_candidate_and_inbox():
    c = _cand("New A")
    s = _cand("SSRN one", topic="Small", abstract="", source="ssrn", summary=None)
    card = cards.format_candidate(c, PAPERS)
    assert card.startswith(f"[{c['id']}] New A") and "Doe & Roe 2026 · arXiv q-fin.PM · 2026-10-15 · fit 0.71 · recommend: keep" in card
    assert "Summary: S" in card and "Nearest owned: Smith 2020" in card and "https://arxiv.org/abs/1" in card
    assert "No abstract on the SSRN feed (52 pages, 11 downloads)" in cards.format_candidate(s, PAPERS)
    text = cards.format_inbox([c, s], {"date": "2026-10-16", "stats": {"scored": 212}}, PAPERS)
    assert text.startswith("Inbox: 2 pending papers") and "212 candidates scored" in text and text.index("## Big") < text.index("## Small")
    assert "No pending papers" in cards.format_inbox([], {"date": "2026-10-16", "stats": {}}, PAPERS)
    assert "No digest has been written yet" in cards.format_inbox([], None, PAPERS)


@pytest.fixture
def disc(tmp_path, monkeypatch):
    root = tmp_path / "disc"
    monkeypatch.setattr(store, "DISCOVERY_DIR", root)
    index = {"generated_at": "x", "unique_papers": 1, "categories": {
        "Big": {"summary": "s", "subcategories": {}, "papers": [PAPERS["B1"]]},
        "Small": {"summary": "s", "subcategories": {}, "papers": []}}}
    monkeypatch.setattr(handlers, "_load_index", lambda: index)
    return root


async def test_get_new_papers_before_any_digest(disc):
    out = await tools.handle_tool("get_new_papers", {})
    assert "No digest has been written yet" in out[0].text


async def test_inbox_keep_and_dismiss_through_handle_tool(disc, monkeypatch):
    c = _cand("New A")
    d = {"date": "2026-10-16", "generated_at": "x", "window": {}, "stats": {"scored": 5}, "candidates": [c], "sent": []}
    store.save_digest(d, disc)
    store.record_pending(d, disc)
    out = (await tools.handle_tool("get_new_papers", {}))[0].text
    assert "Inbox: 1 pending papers" in out and c["id"] in out
    kept = []
    monkeypatch.setattr(handlers.keep, "keep_candidates",
                        lambda ids, attach_pdf=True, **kw: kept.append((ids, attach_pdf)) or [
                            {"id": ids[0], "title": "New A", "status": "kept", "zotero_key": "K", "pdf": "attached", "index": "updated", "message": ""}])
    out = (await tools.handle_tool("keep_new_papers", {"ids": [c["id"]], "attach_pdf": False}))[0].text
    assert kept == [([c["id"]], False)] and "kept" in out and "K" in out
    out = (await tools.handle_tool("dismiss_new_papers", {"ids": ["zzz"]}))[0].text
    assert "unknown" in out


async def test_followed_topics_tools(disc):
    out = (await tools.handle_tool("set_followed_topics", {"unfollow": ["Small", "Nope"]}))[0].text
    assert "Unknown topics: Nope" in out
    out = (await tools.handle_tool("get_followed_topics", {}))[0].text
    assert "Big" in out and "Small" in out and store.load_follow(disc)["unfollowed"] == ["Small"]
    out = (await tools.handle_tool("set_followed_topics", {"follow": ["Small"]}))[0].text
    assert store.load_follow(disc)["unfollowed"] == []


def test_tools_are_registered():
    names = {t.name for t in tools.TOOLS}
    assert {"get_new_papers", "keep_new_papers", "dismiss_new_papers", "consider_paper", "get_followed_topics", "set_followed_topics"} <= names
    assert "WRITE TOOL" in next(t for t in tools.TOOLS if t.name == "keep_new_papers").description


def _raise(exc):
    raise exc


async def test_consider_paper_reports_fetch_failures_in_a_sentence(disc, monkeypatch):
    monkeypatch.setattr(handlers.keep, "consider_reference", lambda ref, **kw: _raise(RuntimeError("HTTP 403 Forbidden")))
    out = (await tools.handle_tool("consider_paper", {"reference": "https://example.org/x"}))[0].text
    assert out.startswith("Could not") and "403" in out and "Traceback" not in out
