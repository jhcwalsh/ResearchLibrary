# Daily Paper Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A daily job that finds new arXiv and SSRN papers in the topics James follows, scores them against his library, and puts the best few in an inbox he approves from Claude Desktop, where `keep` files them in Zotero and the cached index.

**Architecture:** A new `research_library.discovery` package: `store` (files under `.cache/discovery/`), `sources` (arXiv, SSRN, OpenAlex fetch and parse, dedupe), `scoring` (Voyage embeddings, per-topic fit, selection), `summaries` (Haiku queries and cards), `run` (the daily job), `keep` (Zotero add, index and embedding append, dismiss, consider), `cards` (text for Desktop) and `handlers` (the six MCP tools). `tools.py` dispatches to `handlers`, `cli.py` gains `discover`, `zotero_client` gains preprint support. Candidates are plain dicts, like papers everywhere else in this codebase.

**Tech Stack:** Python 3.10+, `requests`, `numpy`, `voyageai` (voyage-finance-2), `anthropic` (Haiku 5.5 via `llm.HAIKU`), `pyzotero`, `mcp`, pytest. Standard-library XML parsing for arXiv.

**Spec:** `docs/superpowers/specs/2026-10-16-daily-paper-discovery-design.md`

## Global Constraints

- Nothing is written to Zotero except in `keep_candidates`. The daily job never writes to Zotero.
- The daily job never calls `build_index`, `build_embeddings` or `build_topic_summaries`.
- Model calls: `model=HAIKU`, explicit `output_config={"effort": "low", ...}`, JSON schema output, `max_tokens` of at least 2000, text read with `llm.response_text`.
- Every network and model call is behind an injectable seam and mocked in tests: `sources._http_get`, `embed=` parameters, `client=` parameters, `zc=`/`fetch_pdf=`/`attach=` parameters.
- Files live under `.cache/discovery/`; every store function takes `root: Path | None = None` defaulting to `store.DISCOVERY_DIR`.
- Candidate dict keys, exactly: `id, source, title, authors, year, published, abstract, doi, url, pdf_url, arxiv_id, ssrn_id, venue, extra`, then after scoring `topic, fit, nearest, vector`, then after summarising `summary, why_it_fits, recommendation, reason`. `authors` are `"Last, First"` strings. Empty strings, not `None`, for missing text fields.
- Candidate id: first 12 hex chars of sha1 over the normalised DOI, else `arxiv_id`, else `ssrn_id`, else `analysis._title_key(title)`.
- Defaults: `ssrn_bindings=[203]`, `min_fit=0.62`, `min_fit_title_only=0.55`, `per_topic=3`, `max_total=15`, expiry 14 days, window floor 1 day, cap 7 days, first run 2 days.
- Commit style: a plain sentence saying what changed, a short body saying why, the `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer. Before every commit append a line to today's entry in `SESSION_NOTES.md` and stage it (Task 1 creates the entry).
- Do not run `pip install -e .`: Claude Desktop holds `research-library.exe` open and the install half-fails. Nothing in this plan needs it.
- Discovery modules import their siblings as modules (`from . import store, sources, scoring, summaries, keep`) and call through them (`summaries.summarize_candidate(...)`), and `cli.cmd_discover` calls `run.discover` through the module at call time. The tests monkeypatch those attributes. Store functions resolve `root = root or DISCOVERY_DIR` inside the function body, never in a default argument, so a patched `DISCOVERY_DIR` is honoured.

## Review Focus

Inputs the spec implies but no happy-path test covers, most likely first. Each has a test pinned to the owning task.

1. `keep_new_papers` with an id that is unknown, or already kept, must not create a Zotero item and must say so (Task 8).
2. `keep` when the candidate's topic no longer exists in the index (after a rebuild renamed categories) must still add to Zotero and report that the index was not updated (Task 8).
3. A Haiku failure on one candidate must not lose the digest: that card gets empty summary fields and `reason` starting `summary failed`, and the run continues (Task 7).
4. An SSRN paper with a missing or unparseable `approved_date` is skipped, not fatal (Task 3).
5. `get_new_papers` before any digest exists returns a friendly sentence, not a traceback (Task 9).

---

### Task 1: Preprint support in `zotero_client`

**Files:**
- Modify: `src/research_library/zotero_client.py` (`create_paper` at ~323, `_build_item_payload` at ~375)
- Test: `tests/test_zotero_write.py`

**Interfaces:**
- Produces: `create_paper(..., repository: str | None = None, archive_id: str | None = None)` and `_build_item_payload(..., repository=None, archive_id=None)`. For `item_type == "preprint"` the payload carries `repository` and `archiveID` and none of `publicationTitle`, `volume`, `issue`, `pages`.

- [ ] **Step 1: Write the failing tests** in `tests/test_zotero_write.py`, in a new class next to the existing payload tests:

```python
class TestPreprintPayload:
    def test_preprint_uses_repository_and_archive_id(self):
        p = _build_item_payload(item_type="preprint", title="T", authors=["Doe, Jane"], year="2026",
                                doi="10.48550/arXiv.2610.00001", journal=None, volume=None, issue=None,
                                pages=None, url="https://arxiv.org/abs/2610.00001", abstract="A",
                                tags=["discovered"], collection_keys=None,
                                repository="arXiv", archive_id="arXiv:2610.00001")
        assert p["itemType"] == "preprint"
        assert p["repository"] == "arXiv" and p["archiveID"] == "arXiv:2610.00001"
        assert p["DOI"] == "10.48550/arXiv.2610.00001"
        for field in ("publicationTitle", "volume", "issue", "pages"):
            assert field not in p

    def test_journal_article_payload_unchanged_by_new_kwargs(self):
        p = _build_item_payload(item_type="journalArticle", title="T", authors=[], year=None, doi=None,
                                journal="J", volume=None, issue=None, pages=None, url=None, abstract=None,
                                tags=None, collection_keys=None)
        assert p["publicationTitle"] == "J" and "repository" not in p
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_zotero_write.py -k Preprint -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'repository'`

- [ ] **Step 3: Implement.** Add `repository: str | None = None, archive_id: str | None = None` to both signatures, pass them through from `create_paper`, and add a `preprint` branch in `_build_item_payload` before the journal branch: `payload["repository"] = repository or journal or ""`, `payload["archiveID"] = archive_id or ""`. Check `create_papers_batch` still calls `_build_item_payload` with keyword arguments only (it does today).

- [ ] **Step 4: Run the write tests**

Run: `python -m pytest tests/test_zotero_write.py -q`
Expected: all PASS

- [ ] **Step 5: Session notes and commit.** Add a new dated entry at the top of `SESSION_NOTES.md`, "2026-10-16 (later still) — Daily paper discovery (in progress)", with a Done bullet for this task and a Why line pointing at the spec. Then:

```bash
git add src/research_library/zotero_client.py tests/test_zotero_write.py SESSION_NOTES.md docs/superpowers/specs/2026-10-16-daily-paper-discovery-design.md docs/superpowers/plans/2026-10-16-daily-paper-discovery.md
git commit -m "Let create_paper file preprints with a repository and archive id" -m "Zotero rejects journal-only fields on preprint items, and the discovery feature files arXiv and SSRN papers as preprints. Also adds the discovery design and plan." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Discovery store

**Files:**
- Create: `src/research_library/discovery/__init__.py` (empty), `src/research_library/discovery/store.py`
- Test: `tests/test_discovery_store.py`

**Interfaces:**
- Produces, all in `store.py`:
  - `DISCOVERY_DIR = INDEX_CACHE_PATH.parent / "discovery"` (import `CACHE_PATH as INDEX_CACHE_PATH` from `..analysis`)
  - `DEFAULTS: dict` = `{"unfollowed": [], "queries": {}, "ssrn_bindings": [203], "min_fit": 0.62, "min_fit_title_only": 0.55, "per_topic": 3, "max_total": 15, "last_run": None}`
  - `candidate_id(cand: dict) -> str`
  - `load_follow(root=None) -> dict` (DEFAULTS merged under what the file holds; no file means DEFAULTS), `save_follow(follow, root=None) -> None`
  - `followed_topics(follow: dict, index_topics: list[str]) -> list[str]` (index order, minus `unfollowed`)
  - `load_ledger(root=None) -> dict`, `save_ledger(ledger, root=None) -> None`
  - `digest_path(date_str, root=None) -> Path` (`digest-YYYY-MM-DD.json`), `load_digest(date_str, root=None) -> dict | None`, `save_digest(digest, root=None) -> None` (uses `digest["date"]`), `digest_dates(root=None) -> list[str]` newest first
  - `record_pending(digest: dict, root=None) -> None`: every candidate in `digest["candidates"]` and `digest["sent"]` becomes a ledger entry `{"status": "pending", "date": <iso now>, "digest": digest["date"], "title": ..., "zotero_key": None}` unless already present
  - `pending_candidates(root=None) -> list[dict]`: candidates whose ledger status is `pending`, newest digest first, each with an added key `"digest"` = digest date
  - `find_candidate(cand_id, root=None) -> dict | None`: the candidate dict (with `"digest"`) from whichever digest holds it, found via the ledger
  - `set_status(cand_id, status, root=None, zotero_key=None) -> None`
  - `expire_pending(today: date, days: int = 14, root=None) -> int`: pending entries whose `digest` date is older than `days` become `expired`; returns the count
- Digest shape: `{"date", "generated_at", "window": {"from", "to"}, "stats": {...}, "candidates": [...], "sent": [...]}`.

- [ ] **Step 1: Write the failing tests** in `tests/test_discovery_store.py`:

```python
from datetime import date
from research_library.discovery import store

def _cand(cid_src, title="A Paper", **kw):
    c = {"source": "arxiv", "title": title, "authors": ["Doe, Jane"], "year": "2026", "published": "2026-10-15",
         "abstract": "", "doi": "", "url": "", "pdf_url": "", "arxiv_id": "", "ssrn_id": "", "venue": "", "extra": {}}
    c.update(kw); c["id"] = store.candidate_id(c); return c

def test_candidate_id_prefers_doi_then_arxiv_then_ssrn_then_title():
    a = store.candidate_id({"doi": "https://doi.org/10.1/X", "arxiv_id": "1", "title": "t"})
    assert a == store.candidate_id({"doi": "10.1/x", "title": "other"}) and len(a) == 12
    assert store.candidate_id({"doi": "", "arxiv_id": "2610.1", "ssrn_id": "9"}) == store.candidate_id({"arxiv_id": "2610.1"})
    assert store.candidate_id({"ssrn_id": "9"}) != store.candidate_id({"title": "9"})
    assert store.candidate_id({"title": "Deep Hedging!"}) == store.candidate_id({"title": "deep hedging"})

def test_follow_defaults_and_roundtrip(tmp_path):
    f = store.load_follow(tmp_path)
    assert f["ssrn_bindings"] == [203] and f["min_fit"] == 0.62 and f["per_topic"] == 3 and f["max_total"] == 15
    f["unfollowed"] = ["ESG"]; store.save_follow(f, tmp_path)
    assert store.load_follow(tmp_path)["unfollowed"] == ["ESG"]
    assert store.followed_topics(f, ["PE", "ESG", "Risk"]) == ["PE", "Risk"]

def test_digest_ledger_pending_and_find(tmp_path):
    d1 = {"date": "2026-10-15", "generated_at": "x", "window": {}, "stats": {}, "candidates": [_cand(1, "One", arxiv_id="1")], "sent": []}
    d2 = {"date": "2026-10-16", "generated_at": "x", "window": {}, "stats": {}, "candidates": [_cand(2, "Two", arxiv_id="2")], "sent": [_cand(3, "Three", ssrn_id="3")]}
    for d in (d1, d2):
        store.save_digest(d, tmp_path); store.record_pending(d, tmp_path)
    assert store.digest_dates(tmp_path) == ["2026-10-16", "2026-10-15"]
    pend = store.pending_candidates(tmp_path)
    assert [c["title"] for c in pend] == ["Two", "Three", "One"] and pend[0]["digest"] == "2026-10-16"
    one = d1["candidates"][0]["id"]
    store.set_status(one, "kept", tmp_path, zotero_key="ZK")
    assert store.load_ledger(tmp_path)[one]["zotero_key"] == "ZK"
    assert [c["title"] for c in store.pending_candidates(tmp_path)] == ["Two", "Three"]
    assert store.find_candidate(one, tmp_path)["title"] == "One"
    assert store.find_candidate("nope", tmp_path) is None

def test_expire_pending(tmp_path):
    d = {"date": "2026-09-01", "generated_at": "x", "window": {}, "stats": {}, "candidates": [_cand(1, "Old", arxiv_id="1")], "sent": []}
    store.save_digest(d, tmp_path); store.record_pending(d, tmp_path)
    assert store.expire_pending(date(2026, 10, 16), 14, tmp_path) == 1
    assert store.load_ledger(tmp_path)[d["candidates"][0]["id"]]["status"] == "expired"
    assert store.pending_candidates(tmp_path) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_discovery_store.py -q`
Expected: FAIL with `ModuleNotFoundError: research_library.discovery`

- [ ] **Step 3: Implement `store.py`** per the Interfaces block. JSON files written with `ensure_ascii=False`, `indent=2`, UTF-8. `candidate_id` reads every field with `.get` (callers may pass partial dicts), normalises the DOI with `zotero_client._normalize_doi(...).lower()` and the title with `analysis._title_key`.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_discovery_store.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit** (append a Done line to today's notes entry first)

```bash
git add src/research_library/discovery tests/test_discovery_store.py SESSION_NOTES.md
git commit -m "Add the discovery store for follow settings, digests and the ledger" -m "Files under .cache/discovery hold what the daily job finds and what James has acted on." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Source parsers (arXiv, SSRN, OpenAlex)

**Files:**
- Create: `src/research_library/discovery/sources.py`, `tests/fixtures/arxiv_sample.xml`, `tests/fixtures/ssrn_sample.json`, `tests/fixtures/openalex_sample.json`
- Test: `tests/test_discovery_sources.py`

**Interfaces:**
- Produces in `sources.py`:
  - `last_first(name: str) -> str`: `"Jane Q. Doe"` → `"Doe, Jane Q."`; a single token is returned unchanged; a name already containing a comma is returned stripped.
  - `parse_arxiv_atom(xml_text: str) -> list[dict]`: one candidate per `<entry>` with `source="arxiv"`, `arxiv_id` without version (`2610.22144`), `url` = the `<id>`, `pdf_url` = the link with `title="pdf"`, `doi` = `10.48550/arXiv.<id>` unless the entry carries an `<arxiv:doi>`, `venue` = `"arXiv " + primary category term`, `published` = date part of `<published>`, `year` = its first 4 chars, title and abstract whitespace-collapsed.
  - `parse_ssrn_page(data: dict) -> list[dict]`: one candidate per paper with `source="ssrn"`, `ssrn_id=str(id)`, `url`, `published` = ISO date parsed from `"16 Oct 2026"`, `venue="SSRN"`, `abstract=""`, `extra={"affiliations": [...], "pages": page_count, "downloads": downloads}`; a paper whose `approved_date` is missing or unparseable is skipped.
  - `rebuild_abstract(inverted: dict | None) -> str`
  - `parse_openalex_work(work: dict) -> dict`: `source="openalex"`, `doi` without the `https://doi.org/` prefix, `url` = `primary_location.landing_page_url` or the DOI URL, `pdf_url` = `open_access.oa_url` or `""`, `venue` = `primary_location.source.display_name` or `""`, authors from `authorships[].author.display_name` via `last_first`, `arxiv_id` from an arxiv landing page URL if present.

- [ ] **Step 1: Create the fixtures.** `arxiv_sample.xml`: a real-shaped Atom feed (namespaces `http://www.w3.org/2005/Atom`, `http://arxiv.org/schemas/atom`, `http://a9.com/-/spec/opensearch/1.1/`) with two entries: id `http://arxiv.org/abs/2610.22144v1`, title split over two lines with extra spaces, published `2026-10-15T16:15:47Z`, two authors `Ada Lovelace` and `Charles Babbage`, primary category `q-fin.PM`, a pdf link `https://arxiv.org/pdf/2610.22144v1`, an abstract; and id `http://arxiv.org/abs/2610.00002v2` with `<arxiv:doi>10.1000/abc</arxiv:doi>` and primary category `cs.LG`. `ssrn_sample.json`: `{"total": 3, "papers": [...]}` with three papers: id 7620718 approved `16 Oct 2026` with two authors (`first_name`/`last_name`), affiliations, `page_count: 52`, `downloads: 11`; id 7619999 approved `01 Oct 2026`; id 7610000 with `approved_date: ""`. `openalex_sample.json`: `{"results": [...]}` with one work carrying a DOI `https://doi.org/10.2139/ssrn.1234`, `abstract_inverted_index` for the sentence `"We study private equity cash flows."`, authorships with two display names, `primary_location.source.display_name` `"SSRN Electronic Journal"`, `open_access.oa_url`; and one arXiv-hosted work (`landing_page_url` `https://arxiv.org/abs/2610.00002v2`) with no abstract.

- [ ] **Step 2: Write the failing tests** in `tests/test_discovery_sources.py`:

```python
import json
from pathlib import Path
from research_library.discovery import sources
FIX = Path(__file__).parent / "fixtures"

def test_last_first():
    assert sources.last_first("Jane Q. Doe") == "Doe, Jane Q."
    assert sources.last_first("Plato") == "Plato"
    assert sources.last_first(" Doe, Jane ") == "Doe, Jane"

def test_parse_arxiv_atom():
    cands = sources.parse_arxiv_atom((FIX / "arxiv_sample.xml").read_text(encoding="utf-8"))
    assert len(cands) == 2
    c = cands[0]
    assert c["source"] == "arxiv" and c["arxiv_id"] == "2610.22144"
    assert c["doi"] == "10.48550/arXiv.2610.22144" and c["pdf_url"] == "https://arxiv.org/pdf/2610.22144v1"
    assert c["authors"] == ["Lovelace, Ada", "Babbage, Charles"]
    assert c["published"] == "2026-10-15" and c["year"] == "2026" and c["venue"] == "arXiv q-fin.PM"
    assert "  " not in c["title"] and "\n" not in c["title"]
    assert cands[1]["doi"] == "10.1000/abc"

def test_parse_ssrn_page_skips_bad_dates():
    cands = sources.parse_ssrn_page(json.loads((FIX / "ssrn_sample.json").read_text(encoding="utf-8")))
    assert [c["ssrn_id"] for c in cands] == ["7620718", "7619999"]
    c = cands[0]
    assert c["source"] == "ssrn" and c["published"] == "2026-10-16" and c["year"] == "2026"
    assert c["abstract"] == "" and c["venue"] == "SSRN" and c["extra"]["pages"] == 52 and c["extra"]["downloads"] == 11
    assert c["authors"][0].count(",") == 1

def test_openalex_abstract_and_work():
    assert sources.rebuild_abstract({"We": [0], "study": [1], "it.": [2]}) == "We study it."
    assert sources.rebuild_abstract(None) == ""
    works = json.loads((FIX / "openalex_sample.json").read_text(encoding="utf-8"))["results"]
    a, b = (sources.parse_openalex_work(w) for w in works)
    assert a["source"] == "openalex" and a["doi"] == "10.2139/ssrn.1234"
    assert a["abstract"] == "We study private equity cash flows." and a["venue"] == "SSRN Electronic Journal"
    assert a["pdf_url"].startswith("http") and len(a["authors"]) == 2 and "," in a["authors"][0]
    assert b["arxiv_id"] == "2610.00002" and b["abstract"] == ""
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/test_discovery_sources.py -q`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError`

- [ ] **Step 4: Implement the parsers** in `sources.py` with `xml.etree.ElementTree` for arXiv and `datetime.strptime(s, "%d %b %Y")` for SSRN. Whitespace collapse: `" ".join(text.split())`.

- [ ] **Step 5: Run to verify they pass**

Run: `python -m pytest tests/test_discovery_sources.py -q`
Expected: 4 passed

- [ ] **Step 6: Commit** (notes line first)

```bash
git add src/research_library/discovery/sources.py tests/fixtures tests/test_discovery_sources.py SESSION_NOTES.md
git commit -m "Parse arXiv, SSRN and OpenAlex results into discovery candidates" -m "One candidate shape for all three sources so scoring and the inbox never care where a paper came from." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Fetching, merging and the owned check

**Files:**
- Modify: `src/research_library/discovery/sources.py`
- Test: `tests/test_discovery_sources.py`

**Interfaces:**
- Consumes: parsers from Task 3, `store.candidate_id`.
- Produces in `sources.py`:
  - `USER_AGENT = "research-library/0.1 (+https://github.com/jhcwalsh)"`
  - `_http_get(url: str, headers: dict | None = None, timeout: int = 30) -> bytes` using `requests`; the single network seam, raises on non-2xx.
  - `ARXIV_FIREHOSE = "cat:q-fin.*"`, `ARXIV_KEYWORD_CATS = "(cat:q-fin.* OR cat:econ.* OR cat:stat.ML OR cat:cs.LG)"`
  - `arxiv_query_url(search_query: str, start: int, max_results: int) -> str`
  - `fetch_arxiv(window_from: datetime, window_to: datetime, topic_queries: dict[str, list[str]], pause: float = 3.0, sleep=time.sleep) -> list[dict]`: the firehose query (`ARXIV_FIREHOSE AND submittedDate:[FROM TO TO]`, `FROM`/`TO` as `YYYYMMDDHHMM`, paged by 200 until a page is short) plus one request per topic: `ARXIV_KEYWORD_CATS AND (all:"q1" OR all:"q2" ...) AND submittedDate:[...]`, 100 results; `sleep(pause)` between requests.
  - `fetch_ssrn(bindings: list[int], since: date, max_pages: int = 5, sleep=time.sleep) -> list[dict]`: `https://api.ssrn.com/content/v1/bindings/{b}/papers?index={i}&count=200&sort=0`, stop paging when the last paper on a page is older than `since` or the page is short; keep only papers with `published >= since`.
  - `fetch_openalex(topic_queries: dict[str, list[str]], since: date, mailto: str) -> list[dict]`: one request per topic to `https://api.openalex.org/works` with `filter=title_and_abstract.search:("q1" OR "q2"),from_publication_date:YYYY-MM-DD`, `per_page=50`, `select=id,doi,title,publication_date,primary_location,type,authorships,abstract_inverted_index,open_access`, `mailto`.
  - `merge_candidates(cands: list[dict]) -> list[dict]`: sets `id` on each via `store.candidate_id`, then merges duplicates by DOI, arXiv id, SSRN id and title key in that order of preference; when two merge, keep the one with an abstract, and fill empty fields from the other.
  - `drop_owned(cands: list[dict], index: dict, ledger: dict) -> list[dict]`: removes candidates whose normalised DOI or title key matches an index paper, or whose id is in the ledger.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_discovery_sources.py`):

```python
from datetime import date, datetime

def _fake_http(monkeypatch, routes):
    calls = []
    def fake(url, headers=None, timeout=30):
        calls.append(url)
        for prefix, body in routes.items():
            if url.startswith(prefix):
                return body if isinstance(body, bytes) else body.encode("utf-8")
        raise AssertionError(f"unexpected url {url}")
    monkeypatch.setattr(sources, "_http_get", fake)
    return calls

def test_fetch_arxiv_builds_firehose_and_topic_queries(monkeypatch):
    calls = _fake_http(monkeypatch, {"http://export.arxiv.org/api/query": (FIX / "arxiv_sample.xml").read_text(encoding="utf-8")})
    out = sources.fetch_arxiv(datetime(2026, 10, 14, 6, 30), datetime(2026, 10, 16, 6, 30),
                              {"PE": ["private equity", "buyout returns"]}, pause=0, sleep=lambda s: None)
    assert len(calls) == 2
    assert "cat:q-fin.*" in calls[0] and "submittedDate" in calls[0] and "202610140630" in calls[0]
    assert "all:%22private+equity%22" in calls[1] or 'all:"private equity"' in calls[1]
    assert len(out) == 4  # both fixture entries, twice; merge happens later

def test_fetch_ssrn_pages_until_window_edge(monkeypatch):
    page = (FIX / "ssrn_sample.json").read_text(encoding="utf-8")
    calls = _fake_http(monkeypatch, {"https://api.ssrn.com/content/v1/bindings/203/papers": page})
    out = sources.fetch_ssrn([203], since=date(2026, 10, 10), sleep=lambda s: None)
    assert len(calls) == 1 and "index=0" in calls[0] and "count=200" in calls[0]
    assert [c["ssrn_id"] for c in out] == ["7620718"]

def test_fetch_openalex_query_shape(monkeypatch):
    calls = _fake_http(monkeypatch, {"https://api.openalex.org/works": (FIX / "openalex_sample.json").read_text(encoding="utf-8")})
    out = sources.fetch_openalex({"PE": ["private equity"]}, since=date(2026, 10, 14), mailto="me@example.org")
    assert len(out) == 2 and "mailto=me%40example.org" in calls[0]
    assert "from_publication_date%3A2026-10-14" in calls[0] and "title_and_abstract.search" in calls[0]

def test_merge_and_drop_owned():
    a = {"source": "arxiv", "title": "Deep Hedging", "doi": "10.1/x", "abstract": "", "arxiv_id": "1", "ssrn_id": "", "authors": [], "year": "", "published": "", "url": "", "pdf_url": "u", "venue": "", "extra": {}}
    b = {**a, "source": "openalex", "abstract": "Full abstract", "pdf_url": ""}
    c = {**a, "doi": "", "arxiv_id": "", "ssrn_id": "9", "title": "Deep Hedging!"}
    d = {**a, "doi": "10.9/owned", "arxiv_id": "", "title": "Owned Paper"}
    merged = sources.merge_candidates([a, b, c, d])
    assert len(merged) == 2
    kept = next(m for m in merged if m["doi"] == "10.1/x")
    assert kept["abstract"] == "Full abstract" and kept["pdf_url"] == "u"
    index = {"categories": {"T": {"papers": [{"key": "K1", "title": "Owned paper", "doi": ""}]}}}
    out = sources.drop_owned(merged, index, ledger={})
    assert [m["doi"] for m in out] == ["10.1/x"]
    assert sources.drop_owned(out, {"categories": {}}, ledger={out[0]["id"]: {"status": "dismissed"}}) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_discovery_sources.py -q`
Expected: the four new tests FAIL with `AttributeError`

- [ ] **Step 3: Implement** per the Interfaces block. Use `urllib.parse.urlencode` for query strings (arXiv accepts `+` for spaces and `%22` for quotes). In `fetch_ssrn`, parse each page with `parse_ssrn_page` and compare `published` strings to `since.isoformat()`.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_discovery_sources.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit** (notes line first)

```bash
git add src/research_library/discovery/sources.py tests/test_discovery_sources.py SESSION_NOTES.md
git commit -m "Fetch new papers from arXiv, SSRN and OpenAlex and drop what the library owns" -m "Three sources behind one HTTP seam, merged by DOI, arXiv id, SSRN id and title so a paper is scored once." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Scoring and selection

**Files:**
- Create: `src/research_library/discovery/scoring.py`
- Test: `tests/test_discovery_scoring.py`

**Interfaces:**
- Consumes: `embeddings._embed_all`, `embeddings._normalize`, `embeddings._paper_text`, `embeddings.load_embeddings()` shape `{"keys", "vectors", "paper_cats", "cat_names", "cat_vectors"}`.
- Produces in `scoring.py`:
  - `embed_candidates(cands: list[dict], embed=None) -> np.ndarray`: texts via `_paper_text(cand)`, `embed` defaults to `lambda texts: _embed_all(texts, input_type="document")`, result normalised; shape `(len(cands), dim)`.
  - `score_candidates(cands: list[dict], vectors: np.ndarray, library: dict, followed: list[str]) -> None`: in place sets `topic` (followed topic with the highest per-topic fit, or `None` if no followed topic has papers), `fit` (that per-topic fit, rounded to 4 dp), `nearest` (global top 3 as `[{"key", "score", "category"}]`), `vector` (list of floats). Per-topic fit = mean of the top 3 cosines among that topic's library papers, or of all of them when the topic has fewer than 3.
  - `select_candidates(cands: list[dict], follow: dict) -> list[dict]`: drops `topic is None`; floor is `min_fit_title_only` when `abstract == ""` else `min_fit`; within each topic sort by fit descending and keep up to `per_topic`; then round-robin across topics (topics ordered by their best fit) taking one per topic per round until `max_total`.

- [ ] **Step 1: Write the failing tests** in `tests/test_discovery_scoring.py`:

```python
import numpy as np
from research_library.discovery import scoring

def _lib():
    # 4-dim unit vectors; Big has 3 papers near e1, Small has 1 paper at e2
    v = np.array([[1,0,0,0],[.9,.1,0,0],[.8,.2,0,0],[0,1,0,0]], dtype=np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return {"keys": ["B1","B2","B3","S1"], "vectors": v, "paper_cats": ["Big","Big","Big","Small"], "cat_names": ["Big","Small"], "cat_vectors": v[:2]}

def _c(title, abstract="abs"):
    return {"title": title, "abstract": abstract, "id": title}

def test_embed_candidates_uses_paper_text_and_normalises():
    seen = []
    def fake_embed(texts):
        seen.extend(texts); return np.array([[3.0, 4.0, 0, 0]] * len(texts), dtype=np.float32)
    vecs = scoring.embed_candidates([_c("T", "A")], embed=fake_embed)
    assert seen == ["T\n\nA"] and np.allclose(np.linalg.norm(vecs, axis=1), 1.0)

def test_score_assigns_small_topic_by_per_topic_fit():
    cands = [_c("near e2"), _c("near e1")]
    vecs = np.array([[0.1, 1, 0, 0], [1, 0.05, 0, 0]], dtype=np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    scoring.score_candidates(cands, vecs, _lib(), followed=["Big", "Small"])
    assert cands[0]["topic"] == "Small" and cands[1]["topic"] == "Big"
    assert cands[0]["nearest"][0]["key"] == "S1" and len(cands[0]["nearest"]) == 3
    assert 0 < cands[1]["fit"] <= 1 and len(cands[1]["vector"]) == 4

def test_score_ignores_unfollowed_topics():
    cands = [_c("near e2")]
    vecs = np.array([[0, 1, 0, 0]], dtype=np.float32)
    scoring.score_candidates(cands, vecs, _lib(), followed=["Big"])
    assert cands[0]["topic"] == "Big"
    scoring.score_candidates(cands, vecs, _lib(), followed=[])
    assert cands[0]["topic"] is None

def test_select_applies_floors_per_topic_cap_and_round_robin():
    follow = {"min_fit": 0.6, "min_fit_title_only": 0.5, "per_topic": 2, "max_total": 3}
    cands = [
        {"id": "a", "topic": "Big", "fit": 0.9, "abstract": "x"},
        {"id": "b", "topic": "Big", "fit": 0.8, "abstract": "x"},
        {"id": "c", "topic": "Big", "fit": 0.7, "abstract": "x"},   # cut by per_topic
        {"id": "d", "topic": "Small", "fit": 0.55, "abstract": ""},  # title-only, above its floor
        {"id": "e", "topic": "Small", "fit": 0.55, "abstract": "x"}, # below min_fit
        {"id": "f", "topic": None, "fit": 0.99, "abstract": "x"},
    ]
    assert [c["id"] for c in scoring.select_candidates(cands, follow)] == ["a", "d", "b"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_discovery_scoring.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `scoring.py`.** Similarity matrix `S = vectors @ library["vectors"].T`; per-topic fit from `S[:, idx_t]` with `np.sort(..., axis=1)[:, -3:].mean(axis=1)` (or fewer columns when the topic is small).

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_discovery_scoring.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit** (notes line first)

```bash
git add src/research_library/discovery/scoring.py tests/test_discovery_scoring.py SESSION_NOTES.md
git commit -m "Score discovery candidates by per-topic fit and pick a balanced shortlist" -m "Per-topic fit instead of nearest-neighbour voting, so the two largest topics do not take every slot." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Haiku queries and summaries

**Files:**
- Create: `src/research_library/discovery/summaries.py`
- Test: `tests/test_discovery_summaries.py`

**Interfaces:**
- Consumes: `llm.HAIKU`, `llm.response_text`, `contents.topic_fingerprint`, `contents.short_cite`.
- Produces in `summaries.py`:
  - `QUERY_SCHEMA` (object with `queries: array of string`, required, no additional properties) and `SUMMARY_SCHEMA` (object with `summary: string`, `why_it_fits: string`, `recommendation: enum keep|maybe|skip`, `reason: string`, all required, no additional properties).
  - `generate_queries(client, name: str, cat: dict) -> list[str]`: Haiku, effort low, `max_tokens=2000`; prompt gives the topic name, subtopic names and up to 15 titles and asks for 4 to 6 search phrases of 2 to 4 words that would find new working papers in this area; returns 4 to 6 stripped phrases (truncate at 6).
  - `ensure_queries(client, follow: dict, categories: dict, followed: list[str], force: bool = False, progress=None) -> dict[str, list[str]]`: for each followed topic, reuse `follow["queries"][name]["queries"]` when its stored `fingerprint` equals `topic_fingerprint(cat)` and not `force`, else call `generate_queries` and store `{"fingerprint", "queries"}`; returns `{name: queries}`.
  - `summarize_candidate(client, cand: dict, nearest_papers: list[dict]) -> dict`: Haiku, effort low, `max_tokens=2000`, `SUMMARY_SCHEMA`; the prompt carries title, authors, venue, date, abstract (or, when empty, affiliations, page count and downloads from `extra` and the sentence that no abstract is available), and the nearest owned papers as `short_cite | title | abstract[:400]`; returns `{"summary": str | None, "why_it_fits", "recommendation", "reason"}` with `summary=None` when `cand["abstract"] == ""`.

- [ ] **Step 1: Write the failing tests** in `tests/test_discovery_summaries.py` (reuse the `_fake_client` pattern from `tests/test_contents.py`, which returns a thinking block then a text block):

```python
import json
from types import SimpleNamespace
from research_library.discovery import summaries
from research_library.contents import topic_fingerprint

def _client(payload, sent):
    msg = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="", signature="s"),
                                   SimpleNamespace(type="text", text=json.dumps(payload))], stop_reason="end_turn", stop_details=None)
    def create(**kw): sent.append(kw); return msg
    return SimpleNamespace(messages=SimpleNamespace(create=create))

def _cat(keys): return {"summary": "s", "papers": [{"key": k, "title": f"Title {k}", "authors": ["Doe, J"], "year": "2020", "abstract": "a"} for k in keys], "subcategories": {"Sub": list(keys)}}

def test_generate_queries_uses_haiku_schema_and_caps_at_six():
    sent = []
    out = summaries.generate_queries(_client({"queries": [f"q{i}" for i in range(8)]}, sent), "Private Equity", _cat(["A", "B"]))
    assert out == [f"q{i}" for i in range(6)]
    assert sent[0]["model"] == "claude-haiku-5-5" and sent[0]["output_config"]["effort"] == "low"
    assert sent[0]["output_config"]["format"]["type"] == "json_schema" and "Title A" in sent[0]["messages"][0]["content"]

def test_ensure_queries_caches_by_fingerprint():
    sent = []
    cats = {"PE": _cat(["A"]), "Risk": _cat(["B"])}
    follow = {"queries": {"PE": {"fingerprint": topic_fingerprint(cats["PE"]), "queries": ["cached"]}}}
    out = summaries.ensure_queries(_client({"queries": ["fresh", "two", "three", "four"]}, sent), follow, cats, ["PE", "Risk"])
    assert out == {"PE": ["cached"], "Risk": ["fresh", "two", "three", "four"]} and len(sent) == 1
    assert follow["queries"]["Risk"]["fingerprint"] == topic_fingerprint(cats["Risk"])
    summaries.ensure_queries(_client({"queries": ["x", "y", "z", "w"]}, sent), follow, cats, ["PE"], force=True)
    assert follow["queries"]["PE"]["queries"] == ["x", "y", "z", "w"]

def test_summarize_candidate_with_and_without_abstract():
    sent = []
    payload = {"summary": "S", "why_it_fits": "Extends Doe 2020.", "recommendation": "keep", "reason": "R"}
    cand = {"title": "T", "authors": ["Doe, J"], "venue": "arXiv q-fin.PM", "published": "2026-10-15", "abstract": "The abstract.", "extra": {}}
    out = summaries.summarize_candidate(_client(payload, sent), cand, [{"key": "A", "title": "Title A", "authors": ["Doe, J"], "year": "2020", "abstract": "a"}])
    assert out == {"summary": "S", "why_it_fits": "Extends Doe 2020.", "recommendation": "keep", "reason": "R"}
    assert "The abstract." in sent[0]["messages"][0]["content"] and "Doe 2020" in sent[0]["messages"][0]["content"]
    cand2 = {**cand, "abstract": "", "extra": {"affiliations": ["LBS"], "pages": 52, "downloads": 11}}
    out2 = summaries.summarize_candidate(_client(payload, sent), cand2, [])
    assert out2["summary"] is None and "52" in sent[1]["messages"][0]["content"] and "no abstract" in sent[1]["messages"][0]["content"].lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_discovery_summaries.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `summaries.py`** per the Interfaces block, following the call shape in `contents.summarize_topic` (`output_config={"effort": "low", "format": {"type": "json_schema", "schema": ...}}`).

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_discovery_summaries.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit** (notes line first)

```bash
git add src/research_library/discovery/summaries.py tests/test_discovery_summaries.py SESSION_NOTES.md
git commit -m "Have Haiku write per-topic search phrases and candidate summaries" -m "Queries are cached by topic fingerprint so they cost one call per topic until the topic changes." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: The daily run

**Files:**
- Create: `src/research_library/discovery/run.py`
- Test: `tests/test_discovery_run.py`

**Interfaces:**
- Consumes: everything from Tasks 2 to 6; `embeddings.load_embeddings`, `analysis.load_cached_index` (or reading `index_path`).
- Produces in `run.py`:
  - `compute_window(last_run: str | None, now: datetime, days: int | None) -> tuple[datetime, datetime]`: `days` given means `now - days`; else from `last_run` clamped to `[now - 7d, now - 1d]`; no `last_run` means `now - 2d`.
  - `discover(days: int | None = None, force_queries: bool = False, dry_run: bool = False, progress=None, client=None, now: datetime | None = None, root: Path | None = None, index_path: Path | None = None, library: dict | None = None, embed=None, fetchers: dict | None = None, sleep=time.sleep) -> dict`. Steps: load index (raise `FileNotFoundError` with the same wording `contents.py` uses when missing), library embeddings (`library` or `load_embeddings()`), follow and ledger; `followed = store.followed_topics(...)`; `client` defaults to `anthropic.Anthropic()` (imported lazily, as `contents.py` does); `ensure_queries`; window; fetch from `fetchers` (defaults `{"arxiv": sources.fetch_arxiv, "ssrn": sources.fetch_ssrn, "openalex": sources.fetch_openalex}`), each wrapped so an exception is logged via `progress` and counted as 0 fetched; `merge_candidates`, `drop_owned`, `embed_candidates`, `score_candidates`, `select_candidates`; unless `dry_run`, `summarize_candidate` for each selected (an exception yields `summary=None, why_it_fits="", recommendation="maybe", reason="summary failed: <exc>"`), write the digest (`date = now.date().isoformat()`; if a digest for today exists, append to its `candidates` and merge `stats`), `record_pending`, `expire_pending(now.date())`, set `follow["last_run"] = now.isoformat(timespec="seconds")`. Always `save_follow` (queries are a cache) except in `dry_run` when nothing at all is written. Returns `{"fetched": {"arxiv": n, "ssrn": n, "openalex": n}, "new": n, "scored": n, "shown": n, "by_topic": {topic: n}, "digest": date | None, "expired": n, "selected": [cands]}`.
  - OpenAlex `mailto` is `os.environ.get("UNPAYWALL_EMAIL", "")`; when empty the OpenAlex fetcher is skipped with one `progress` line.

- [ ] **Step 1: Write the failing tests** in `tests/test_discovery_run.py`:

```python
import json
import numpy as np
from datetime import datetime
from research_library.discovery import run, store

def test_compute_window():
    now = datetime(2026, 10, 16, 6, 30)
    assert run.compute_window(None, now, None) == (datetime(2026, 10, 14, 6, 30), now)
    assert run.compute_window("2026-10-15T20:00:00", now, None)[0] == datetime(2026, 10, 15, 6, 30)
    assert run.compute_window("2026-09-01T06:30:00", now, None)[0] == datetime(2026, 10, 9, 6, 30)
    assert run.compute_window(None, now, 3)[0] == datetime(2026, 10, 13, 6, 30)

def _setup(tmp_path):
    index = {"generated_at": "x", "unique_papers": 2, "categories": {
        "Big": {"summary": "s", "subcategories": {}, "papers": [{"key": "B1", "title": "Owned", "authors": ["Doe, J"], "year": "2020", "abstract": "a", "doi": "10.9/owned"}]},
        "Small": {"summary": "s", "subcategories": {}, "papers": [{"key": "S1", "title": "Small one", "authors": ["Roe, R"], "year": "2021", "abstract": "b", "doi": ""}]}}}
    ip = tmp_path / "library_index.json"; ip.write_text(json.dumps(index), encoding="utf-8")
    v = np.eye(4, dtype=np.float32)[:2]
    library = {"keys": ["B1", "S1"], "vectors": v, "paper_cats": ["Big", "Small"], "cat_names": ["Big", "Small"], "cat_vectors": v}
    return ip, library

def _cand(title, **kw):
    c = {"source": "arxiv", "title": title, "authors": ["Doe, Jane"], "year": "2026", "published": "2026-10-15", "abstract": "abs",
         "doi": "", "url": "u", "pdf_url": "", "arxiv_id": title, "ssrn_id": "", "venue": "arXiv q-fin.PM", "extra": {}}
    c.update(kw); return c

def test_discover_writes_digest_pending_and_last_run(tmp_path, monkeypatch):
    ip, library = _setup(tmp_path)
    root = tmp_path / "disc"
    fetchers = {"arxiv": lambda **kw: [_cand("New A"), _cand("Owned", doi="10.9/owned")], "ssrn": lambda **kw: [_cand("New S", source="ssrn", abstract="", ssrn_id="7", arxiv_id="")], "openalex": lambda **kw: (_ for _ in ()).throw(RuntimeError("down"))}
    monkeypatch.setattr(run.summaries, "ensure_queries", lambda *a, **k: {"Big": ["q"], "Small": ["q"]})
    calls = []
    def fake_summary(client, cand, nearest):
        calls.append(cand["title"])
        if cand["title"] == "New S": raise RuntimeError("boom")
        return {"summary": "S", "why_it_fits": "W", "recommendation": "keep", "reason": "R"}
    monkeypatch.setattr(run.summaries, "summarize_candidate", fake_summary)
    embed = lambda texts: np.array([[1, 0.2, 0, 0] if "New A" in t else [0.2, 1, 0, 0] for t in texts], dtype=np.float32)
    monkeypatch.setenv("UNPAYWALL_EMAIL", "me@example.org")
    out = run.discover(progress=lambda m: None, client=object(), now=datetime(2026, 10, 16, 6, 30), root=root, index_path=ip, library=library, embed=embed, fetchers=fetchers)
    assert out["fetched"] == {"arxiv": 2, "ssrn": 1, "openalex": 0} and out["new"] == 2 and out["shown"] == 2
    assert out["by_topic"] == {"Big": 1, "Small": 1} and out["digest"] == "2026-10-16"
    digest = store.load_digest("2026-10-16", root)
    titles = {c["title"]: c for c in digest["candidates"]}
    assert titles["New A"]["topic"] == "Big" and titles["New A"]["summary"] == "S" and len(titles["New A"]["vector"]) == 4
    assert titles["New S"]["summary"] is None and titles["New S"]["reason"].startswith("summary failed")
    assert all(e["status"] == "pending" for e in store.load_ledger(root).values()) and len(store.load_ledger(root)) == 2
    assert store.load_follow(root)["last_run"] == "2026-10-16T06:30:00"

def test_dry_run_writes_nothing_and_skips_summaries(tmp_path, monkeypatch):
    ip, library = _setup(tmp_path)
    root = tmp_path / "disc"
    monkeypatch.setattr(run.summaries, "ensure_queries", lambda *a, **k: {"Big": ["q"], "Small": ["q"]})
    monkeypatch.setattr(run.summaries, "summarize_candidate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not summarise")))
    fetchers = {"arxiv": lambda **kw: [_cand("New A")], "ssrn": lambda **kw: [], "openalex": lambda **kw: []}
    out = run.discover(dry_run=True, progress=lambda m: None, client=object(), now=datetime(2026, 10, 16, 6, 30), root=root, index_path=ip, library=library,
                       embed=lambda texts: np.array([[1, 0, 0, 0]] * len(texts), dtype=np.float32), fetchers=fetchers)
    assert out["shown"] == 1 and out["digest"] is None and out["selected"][0]["title"] == "New A"
    assert not root.exists() or not list(root.glob("*.json"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_discovery_run.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `run.py`** per the Interfaces block. Fetchers are called with keyword arguments only: `fetch_arxiv(window_from=, window_to=, topic_queries=, sleep=)`, `fetch_ssrn(bindings=, since=, sleep=)`, `fetch_openalex(topic_queries=, since=, mailto=)`. `nearest_papers` for the summary are looked up from the index by key.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_discovery_run.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit** (notes line first)

```bash
git add src/research_library/discovery/run.py tests/test_discovery_run.py SESSION_NOTES.md
git commit -m "Add the daily discovery run that writes a digest and marks candidates pending" -m "Fetch, dedupe, score, pick, summarise, write. A failing source or a failing summary is reported and never loses the digest." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Keep, dismiss, consider

**Files:**
- Create: `src/research_library/discovery/keep.py`
- Modify: `src/research_library/discovery/sources.py` (add `lookup_reference`)
- Test: `tests/test_discovery_keep.py`

**Interfaces:**
- Consumes: `store.find_candidate`, `store.set_status`, `store.pending_candidates`, `store.load_digest`/`save_digest`/`record_pending`, `zotero_client.create_paper` (Task 1 signature), `downloads.fetch_pdf(doi, url)`, `downloads.attach_pdf_to_item(None, key, title, pdf_bytes)`, `scoring.*`, `summaries.summarize_candidate`.
- Produces:
  - `sources.lookup_reference(reference: str, bindings: list[int]) -> dict`: a candidate (unscored) from an arXiv id or URL (arXiv API `id_list=`), a DOI or doi.org URL (OpenAlex `works/https://doi.org/<doi>`, then Crossref `https://api.crossref.org/works/<doi>` reading `message.title[0]`, `message.author[].family/given`, `message.abstract` stripped of JATS tags, `message.issued`), an SSRN URL or `ssrn:<id>` (scan `fetch_ssrn`-style pages of each binding, up to 3 pages, for that id), or any other URL (fetch the page and read `citation_title`, `citation_author`, `citation_abstract`, `citation_doi`, `citation_pdf_url` meta tags; if a DOI meta is found, resolve through OpenAlex first). Raises `LookupError` with a one-sentence message when nothing usable is found. `source="sent"` on the returned candidate, `extra["reference"] = reference`.
  - `keep.append_to_index(paper: dict, topic: str, index_path: Path | None = None) -> bool`: inserts `{key, title, authors, year, abstract, doi, url}` at the front of `categories[topic]["papers"]`, increments `unique_papers`, returns `False` without writing when the topic is absent.
  - `keep.append_embedding(key: str, vector: list[float], topic: str, emb_path: Path | None = None) -> bool`: appends to `keys`, `vectors` (as float32 row), `paper_cats`; leaves `cat_names`/`cat_vectors`; returns `False` when the file is missing.
  - `keep.keep_candidates(ids: list[str], attach_pdf: bool = True, root=None, index_path=None, emb_path=None, zc=None, fetch_pdf=None, attach=None) -> list[dict]`: per id a result `{"id", "title", "status", "zotero_key", "pdf", "index", "message"}` where `status` is `kept`, `already_owned` (Zotero returned a DOI duplicate), `already_kept` (ledger not pending), `unknown` (no such id) or `error`; `pdf` is `attached`, `not_found`, `failed` or `skipped`; `index` is `updated`, `topic_missing` or `skipped`. Zotero call: `zc.create_paper(title=, authors=, item_type="preprint" if source in ("arxiv","ssrn") or venue in ("SSRN",) or arxiv_id else "journalArticle", year=, doi= or None, url=, abstract=, journal=venue for journal articles, tags=[topic, "discovered"], repository="arXiv"/"SSRN"/None, archive_id="arXiv:<id>"/"SSRN:<id>"/None)`. Only `kept` and `already_owned` update the ledger (with the key) and the index and embeddings.
  - `keep.dismiss_candidates(ids: list[str] | None = None, all_pending: bool = False, root=None) -> list[dict]`: `{"id", "title", "status"}` per candidate; `status` `dismissed`, `unknown`, or `not_pending`.
  - `keep.consider_reference(reference: str, client=None, root=None, index_path=None, library=None, embed=None, bindings=None, now=None) -> dict`: `lookup_reference`, `merge_candidates([cand])`, `drop_owned` (an owned paper returns `{"owned": True, "title", "doi"}` and stores nothing), embed, score against all index topics (followed or not, per spec "ignoring the floor"), summarise, append to today's digest `sent` (creating a digest with empty `candidates` if none exists today), `record_pending`, return the candidate.

- [ ] **Step 1: Write the failing tests** in `tests/test_discovery_keep.py`:

```python
import json
import numpy as np
from datetime import datetime
from types import SimpleNamespace
from research_library.discovery import keep, store, sources

def _index(tmp_path, topics=("Big",)):
    index = {"generated_at": "x", "unique_papers": 1, "categories": {t: {"summary": "s", "subcategories": {}, "papers": [
        {"key": "B1", "title": "Owned", "authors": ["Doe, J"], "year": "2020", "abstract": "a", "doi": "10.9/owned"}]} for t in topics}}
    ip = tmp_path / "library_index.json"; ip.write_text(json.dumps(index), encoding="utf-8")
    ep = tmp_path / "embeddings.npz"
    np.savez(ep, keys=np.array(["B1"]), vectors=np.eye(4, dtype=np.float32)[:1], paper_cats=np.array(["Big"]), cat_names=np.array(["Big"]), cat_vectors=np.eye(4, dtype=np.float32)[:1])
    return ip, ep

def _digest(root, cands):
    d = {"date": "2026-10-16", "generated_at": "x", "window": {}, "stats": {}, "candidates": cands, "sent": []}
    store.save_digest(d, root); store.record_pending(d, root); return d

def _cand(title, **kw):
    c = {"source": "arxiv", "title": title, "authors": ["Doe, Jane"], "year": "2026", "published": "2026-10-15", "abstract": "abs", "doi": "10.48550/arXiv.2610.1",
         "url": "https://arxiv.org/abs/2610.1", "pdf_url": "https://arxiv.org/pdf/2610.1", "arxiv_id": "2610.1", "ssrn_id": "", "venue": "arXiv q-fin.PM", "extra": {},
         "topic": "Big", "fit": 0.7, "nearest": [], "vector": [0, 1, 0, 0]}
    c.update(kw); c["id"] = store.candidate_id(c); return c

class FakeZc:
    def __init__(self, dup=False): self.calls = []; self.dup = dup
    def create_paper(self, **kw): self.calls.append(kw); return {"key": "NEWKEY", "duplicate": self.dup}

def test_keep_creates_preprint_attaches_pdf_and_updates_index(tmp_path):
    ip, ep = _index(tmp_path); root = tmp_path / "disc"
    c = _cand("New A"); _digest(root, [c]); zc = FakeZc()
    out = keep.keep_candidates([c["id"]], root=root, index_path=ip, emb_path=ep, zc=zc,
                               fetch_pdf=lambda doi, url: (b"%PDF-1.4 x", "direct_url"), attach=lambda z, key, title, data: True)
    assert out[0]["status"] == "kept" and out[0]["zotero_key"] == "NEWKEY" and out[0]["pdf"] == "attached" and out[0]["index"] == "updated"
    kw = zc.calls[0]
    assert kw["item_type"] == "preprint" and kw["repository"] == "arXiv" and kw["archive_id"] == "arXiv:2610.1" and kw["tags"] == ["Big", "discovered"]
    index = json.loads(ip.read_text(encoding="utf-8"))
    assert index["categories"]["Big"]["papers"][0] == {"key": "NEWKEY", "title": "New A", "authors": ["Doe, Jane"], "year": "2026", "abstract": "abs", "doi": "10.48550/arXiv.2610.1", "url": "https://arxiv.org/abs/2610.1"}
    assert index["unique_papers"] == 2
    emb = np.load(ep); assert list(emb["keys"]) == ["B1", "NEWKEY"] and emb["vectors"].shape == (2, 4) and list(emb["paper_cats"]) == ["Big", "Big"]
    assert store.load_ledger(root)[c["id"]] ["status"] == "kept"

def test_keep_unknown_and_already_kept_do_not_touch_zotero(tmp_path):
    ip, ep = _index(tmp_path); root = tmp_path / "disc"
    c = _cand("New A"); _digest(root, [c]); store.set_status(c["id"], "kept", root, zotero_key="K"); zc = FakeZc()
    out = keep.keep_candidates(["nope", c["id"]], root=root, index_path=ip, emb_path=ep, zc=zc, fetch_pdf=None, attach=None)
    assert [r["status"] for r in out] == ["unknown", "already_kept"] and zc.calls == []

def test_keep_with_missing_topic_still_adds_to_zotero(tmp_path):
    ip, ep = _index(tmp_path, topics=("Renamed",)); root = tmp_path / "disc"
    c = _cand("New A"); _digest(root, [c]); zc = FakeZc()
    out = keep.keep_candidates([c["id"]], attach_pdf=False, root=root, index_path=ip, emb_path=ep, zc=zc)
    assert out[0]["status"] == "kept" and out[0]["index"] == "topic_missing" and out[0]["pdf"] == "skipped" and len(zc.calls) == 1

def test_keep_reports_duplicate_and_pdf_not_found(tmp_path):
    ip, ep = _index(tmp_path); root = tmp_path / "disc"
    c = _cand("New A"); _digest(root, [c])
    out = keep.keep_candidates([c["id"]], root=root, index_path=ip, emb_path=ep, zc=FakeZc(dup=True), fetch_pdf=lambda doi, url: (None, "failed"), attach=None)
    assert out[0]["status"] == "already_owned" and out[0]["pdf"] == "not_found" and store.load_ledger(root)[c["id"]]["zotero_key"] == "NEWKEY"

def test_dismiss(tmp_path):
    root = tmp_path / "disc"; a, b = _cand("A"), _cand("B", arxiv_id="2610.2", doi="10.48550/arXiv.2610.2"); _digest(root, [a, b])
    assert [r["status"] for r in keep.dismiss_candidates([a["id"], "zzz"], root=root)] == ["dismissed", "unknown"]
    assert [r["id"] for r in keep.dismiss_candidates(all_pending=True, root=root)] == [b["id"]]
    assert store.pending_candidates(root) == []

def test_lookup_reference_routes(monkeypatch):
    seen = []
    def fake(url, headers=None, timeout=30):
        seen.append(url)
        if "export.arxiv.org" in url: return (FIX / "arxiv_sample.xml").read_bytes()
        if "api.openalex.org/works/https://doi.org/" in url: return json.dumps(json.loads((FIX / "openalex_sample.json").read_text(encoding="utf-8"))["results"][0]).encode()
        raise AssertionError(url)
    monkeypatch.setattr(sources, "_http_get", fake)
    assert sources.lookup_reference("https://arxiv.org/abs/2610.22144v1", [203])["arxiv_id"] == "2610.22144" and "id_list=2610.22144" in seen[-1]
    assert sources.lookup_reference("10.2139/ssrn.1234", [203])["doi"] == "10.2139/ssrn.1234"
    import pytest
    with pytest.raises(LookupError): sources.lookup_reference("just some words", [203])

def test_consider_reference_scores_summarises_and_stores_under_sent(tmp_path, monkeypatch):
    from pathlib import Path
    ip, ep = _index(tmp_path); root = tmp_path / "disc"
    monkeypatch.setattr(sources, "lookup_reference", lambda ref, bindings: {**_cand("Sent paper", source="sent", topic=None), "id": ""})
    monkeypatch.setattr(keep.summaries, "summarize_candidate", lambda client, cand, nearest: {"summary": "S", "why_it_fits": "W", "recommendation": "maybe", "reason": "R"})
    library = {"keys": ["B1"], "vectors": np.eye(4, dtype=np.float32)[:1], "paper_cats": ["Big"], "cat_names": ["Big"], "cat_vectors": np.eye(4, dtype=np.float32)[:1]}
    out = keep.consider_reference("https://arxiv.org/abs/2610.1", client=object(), root=root, index_path=ip, library=library,
                                  embed=lambda texts: np.array([[1, 0, 0, 0]] * len(texts), dtype=np.float32), now=datetime(2026, 10, 16, 9, 0))
    assert out["topic"] == "Big" and out["summary"] == "S" and out["source"] == "sent"
    digest = store.load_digest("2026-10-16", root)
    assert digest["sent"][0]["title"] == "Sent paper" and digest["candidates"] == []
    assert store.load_ledger(root)[out["id"]]["status"] == "pending"
```

Add `from pathlib import Path` and `FIX = Path(__file__).parent / "fixtures"` at the top of the file.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_discovery_keep.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `keep.py` and `sources.lookup_reference`** per the Interfaces block. In `keep_candidates`, `zc` defaults to the `zotero_client` module, `fetch_pdf` to `downloads.fetch_pdf` and `attach` to `downloads.attach_pdf_to_item` (imported lazily so the MCP server starts without `requests`... they are core dependencies after Task 10, so a plain import is fine). Wrap the Zotero call in `try/except Exception` → `status="error"`, `message=str(exc)`.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_discovery_keep.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit** (notes line first)

```bash
git add src/research_library/discovery/keep.py src/research_library/discovery/sources.py tests/test_discovery_keep.py SESSION_NOTES.md
git commit -m "Keep, dismiss and consider discovery candidates" -m "Keep is the only Zotero write: it files the paper, tries for a PDF, and folds it into the cached index and embeddings so no rebuild is needed." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Cards and the six MCP tools

**Files:**
- Create: `src/research_library/discovery/cards.py`, `src/research_library/discovery/handlers.py`
- Modify: `src/research_library/tools.py` (append six `types.Tool` entries to `TOOLS`; add one dispatch line at the top of `handle_tool`)
- Test: `tests/test_discovery_tools.py`

**Interfaces:**
- Consumes: `store`, `keep`, `contents.short_cite`, `links.paper_link`, `analysis.load_cached_index`.
- Produces:
  - `cards.format_candidate(cand: dict, papers_by_key: dict, n: int | None = None) -> str`: a block starting `[<id>] <title>` then one line `<authors as short_cite of the candidate> · <venue> · <published> · fit <fit:.2f> · recommend: <recommendation>`, then `Summary: ...` or `No abstract on the SSRN feed (<pages> pages, <downloads> downloads).`, `Why it fits: ...`, `Nearest owned: <short_cite>; <short_cite>; <short_cite>` (from `nearest[].key` via `papers_by_key`, skipping unknown keys), and the link (`url`).
  - `cards.format_inbox(cands: list[dict], latest: dict | None, papers_by_key: dict) -> str`: header `Inbox: N pending papers` plus, when `latest` is given, `latest digest <date>, <stats.scored> candidates scored`; then `## <topic>` sections in order of first appearance with the cards; `cands == []` gives `No pending papers. The daily run writes a digest each morning; run research-admin discover to make one now.` and when `latest is None` the sentence `No digest has been written yet.` instead.
  - `cards.format_keep_results(results: list[dict]) -> str` and `cards.format_followed(followed: list[str], unfollowed: list[str], counts: dict[str, int]) -> str`.
  - `handlers.handle(name: str, arguments: dict) -> str | None`: returns text for the six tool names, `None` otherwise. `get_new_papers(date?)`: pending inbox, or `store.load_digest(date)` rendered the same way when `date` is given (`No digest for <date>.` when missing). `keep_new_papers(ids, attach_pdf=True)`, `dismiss_new_papers(ids?, all=False)`, `consider_paper(reference)` (an owned result renders `Already in the library: <title>`; a `LookupError` renders its message), `get_followed_topics()`, `set_followed_topics(follow=[], unfollow=[])` (unknown names are listed back as `Unknown topics: ...`; the rest update `follow["unfollowed"]`).
  - In `tools.py`: tool descriptions in the voice of the existing ones; `keep_new_papers` and `dismiss_new_papers` carry the `WRITE TOOL` sentence like `add_paper`. Dispatch: `text = discovery_handlers.handle(name, arguments); if text is not None: return _text(text)` placed before the `search_papers` branch.

- [ ] **Step 1: Write the failing tests** in `tests/test_discovery_tools.py`:

```python
import json
import pytest
import numpy as np
from research_library.discovery import store, cards, handlers
from research_library import tools

def _cand(title, topic="Big", abstract="abs", **kw):
    c = {"source": "arxiv", "title": title, "authors": ["Doe, Jane", "Roe, Rick"], "year": "2026", "published": "2026-10-15", "abstract": abstract, "doi": "",
         "url": "https://arxiv.org/abs/1", "pdf_url": "", "arxiv_id": title, "ssrn_id": "", "venue": "arXiv q-fin.PM", "extra": {"pages": 52, "downloads": 11},
         "topic": topic, "fit": 0.7123, "nearest": [{"key": "B1", "score": 0.8, "category": "Big"}], "vector": [1, 0],
         "summary": "S", "why_it_fits": "W", "recommendation": "keep", "reason": "R"}
    c.update(kw); c["id"] = store.candidate_id(c); return c

PAPERS = {"B1": {"key": "B1", "title": "Owned", "authors": ["Smith, A"], "year": "2020"}}

def test_format_candidate_and_inbox():
    c = _cand("New A"); s = _cand("SSRN one", topic="Small", abstract="", source="ssrn", summary=None)
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
    index = {"generated_at": "x", "unique_papers": 1, "categories": {"Big": {"summary": "s", "subcategories": {}, "papers": [PAPERS["B1"]]}, "Small": {"summary": "s", "subcategories": {}, "papers": []}}}
    monkeypatch.setattr(handlers, "_load_index", lambda: index)
    return root

async def test_get_new_papers_before_any_digest(disc):
    out = await tools.handle_tool("get_new_papers", {})
    assert "No digest has been written yet" in out[0].text

async def test_inbox_keep_and_dismiss_through_handle_tool(disc, monkeypatch):
    c = _cand("New A"); d = {"date": "2026-10-16", "generated_at": "x", "window": {}, "stats": {"scored": 5}, "candidates": [c], "sent": []}
    store.save_digest(d, disc); store.record_pending(d, disc)
    out = (await tools.handle_tool("get_new_papers", {}))[0].text
    assert "Inbox: 1 pending papers" in out and c["id"] in out
    kept = []
    monkeypatch.setattr(handlers.keep, "keep_candidates", lambda ids, attach_pdf=True, **kw: kept.append((ids, attach_pdf)) or [{"id": ids[0], "title": "New A", "status": "kept", "zotero_key": "K", "pdf": "attached", "index": "updated", "message": ""}])
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_discovery_tools.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `cards.py`, `handlers.py` and the `tools.py` additions.** `handlers._load_index()` wraps `analysis.load_cached_index()` so tests can replace it; `papers_by_key` is built from it. Store functions are called without `root` so the monkeypatched `DISCOVERY_DIR` applies (the default must be read at call time: use `root = root or DISCOVERY_DIR` inside each function, which Task 2 already does).

- [ ] **Step 4: Run the new tests and the whole suite**

Run: `python -m pytest tests/test_discovery_tools.py -q` then `python -m pytest -q`
Expected: 5 passed, then the full suite green (133 before this plan plus the new tests)

- [ ] **Step 5: Commit** (notes line first)

```bash
git add src/research_library/discovery/cards.py src/research_library/discovery/handlers.py src/research_library/tools.py tests/test_discovery_tools.py SESSION_NOTES.md
git commit -m "Expose the discovery inbox, keep, dismiss, consider and follow tools over MCP" -m "Claude Desktop is the only front end: ask what's new, keep the ones worth having." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: CLI command, dependencies, README

**Files:**
- Modify: `src/research_library/cli.py` (docstring list, `cmd_discover`, parser), `pyproject.toml`, `.env.example`
- Create: `README.md`
- Modify: `docs/superpowers/specs/2026-10-16-daily-paper-discovery-design.md` (one line: run log is the shared `.cache/refresh_log.txt`, not a separate `discover.log`)
- Test: `tests/test_discovery_cli.py`

**Interfaces:**
- Produces: `research-admin discover [--days N] [--force-queries] [--dry-run]` calling `run.discover(days=, force_queries=, dry_run=, progress=log.info)` and logging `Discovery: fetched arxiv N, ssrn N, openalex N; N new; N shown (Topic a, Topic b ...); N expired; digest <date>` (dry run: `dry run, nothing written` instead of the digest), returning 0. `FileNotFoundError` logs the message and returns 1.
- `pyproject.toml`: `requests>=2.32.0` and `beautifulsoup4>=4.12.0` move into `dependencies`; the `download` extra becomes empty or is removed.
- `.env.example`: the `UNPAYWALL_EMAIL` comment gains "Also sent to OpenAlex as the polite-pool contact for research-admin discover."
- `README.md` (about 60 lines): what the project is (Zotero library served to Claude via MCP, Streamlit app, admin CLI); setup (`.venv`, `pip install -e .[dev]` with the Claude Desktop warning from the session notes, `.env` from `.env.example`); Claude Desktop config snippet pointing at `.venv/Scripts/research-library.exe`; the `research-admin` commands in one table (refresh, embed, links, contents, download, discover, all) with one line each; the daily discovery section: what it does, the three tools to use in Desktop, and the scheduled task command:

```powershell
$action  = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c cd /d C:\Users\james\PycharmProjects\ResearchLibrary && .venv\Scripts\research-admin.exe discover' -WorkingDirectory 'C:\Users\james\PycharmProjects\ResearchLibrary'
$trigger = New-ScheduledTaskTrigger -Daily -At 06:30
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "ResearchLibrary Discover" -Action $action -Trigger $trigger -Settings $settings -Description "Daily paper discovery for the research library"
```

- [ ] **Step 1: Write the failing tests** in `tests/test_discovery_cli.py`:

```python
import logging
from research_library import cli

def test_discover_subcommand_parses_and_calls_run(monkeypatch):
    seen = {}
    def fake(**kw): seen.update(kw); return {"fetched": {"arxiv": 1, "ssrn": 2, "openalex": 0}, "new": 3, "scored": 3, "shown": 2, "by_topic": {"PE": 2}, "digest": "2026-10-16", "expired": 0, "selected": []}
    monkeypatch.setattr("research_library.discovery.run.discover", fake)
    args = cli.build_parser().parse_args(["discover", "--days", "3", "--force-queries", "--dry-run"])
    assert args.func(args, logging.getLogger("t")) == 0
    assert seen["days"] == 3 and seen["force_queries"] is True and seen["dry_run"] is True and callable(seen["progress"])

def test_discover_reports_missing_index(monkeypatch):
    monkeypatch.setattr("research_library.discovery.run.discover", lambda **kw: (_ for _ in ()).throw(FileNotFoundError("No library index")))
    args = cli.build_parser().parse_args(["discover"])
    assert args.func(args, logging.getLogger("t")) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_discovery_cli.py -q`
Expected: FAIL with `argparse` error `invalid choice: 'discover'`

- [ ] **Step 3: Implement** `cmd_discover`, the parser entry, the docstring line, the `pyproject.toml` and `.env.example` edits, the README, and the one-line spec correction.

- [ ] **Step 4: Run the tests and check the installed command**

Run: `python -m pytest -q` then `research-admin discover --help`
Expected: suite green; help lists `--days`, `--force-queries`, `--dry-run`

- [ ] **Step 5: Commit** (notes line first)

```bash
git add src/research_library/cli.py pyproject.toml .env.example README.md docs/superpowers/specs/2026-10-16-daily-paper-discovery-design.md tests/test_discovery_cli.py SESSION_NOTES.md
git commit -m "Add research-admin discover, promote requests to a core dependency, write a README" -m "The MCP server now fetches PDFs on keep, so requests can no longer be optional. The README is the first one the repo has had." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Live calibration, scheduled task, wrap-up

**Files:**
- Modify: `src/research_library/discovery/store.py` (only if the floors change), `SESSION_NOTES.md`

No new tests; this task verifies the system against the real world.

- [ ] **Step 1: Dry run over three days**

Run: `research-admin discover --days 3 --dry-run`
Expected: a log showing fetched counts for all three sources (arXiv in the tens, SSRN in the hundreds, OpenAlex in the hundreds), no tracebacks, and a selection of up to 15 candidates with topics and fits. Haiku is called only for queries (one per topic, 19 calls), never for summaries.

- [ ] **Step 2: Inspect the fit distribution.** Add a temporary `progress` line or run a scratch script (in the scratchpad, not the repo) that calls `run.discover(dry_run=True, days=3, ...)` and prints, for all scored candidates, `fit`, `topic`, `abstract != ""` and the title, sorted by fit. Judge where relevant papers stop: the floors should sit just under the last candidate you would want to see. Change `DEFAULTS["min_fit"]` and `DEFAULTS["min_fit_title_only"]` in `store.py` only if the evidence says so, and record the numbers in the session notes.

- [ ] **Step 3: Real run**

Run: `research-admin discover --days 2`
Expected: `.cache/discovery/digest-2026-10-16.json` and `seen.json` written, `follow.json` with `last_run` set and 19 query lists; cost on the console under a few cents of Haiku.

- [ ] **Step 4: Check the inbox renders.** Run a one-off `python -c "import asyncio; from research_library.tools import handle_tool; print(asyncio.run(handle_tool('get_new_papers', {}))[0].text)"` and read it: every card has an id, a title, a fit, a nearest-owned line and a link; SSRN cards show the no-abstract sentence.

- [ ] **Step 5: Register the scheduled task** with the PowerShell block from the README, then confirm:

Run: `Get-ScheduledTask -TaskName "ResearchLibrary Discover" | Select State, @{n='Next';e={($_ | Get-ScheduledTaskInfo).NextRunTime}}`
Expected: `Ready`, next run tomorrow 06:30.

- [ ] **Step 6: Finish the session notes entry.** Replace "(in progress)" in the heading; fill Done, Why, Broken / unsure (floors are a first guess; `consider_paper` for generic URLs depends on publisher meta tags; the Contents page shows the "summary predates" caption on topics that gain a kept paper until `research-admin contents` runs), Next (keep a paper from Desktop and confirm it appears in semantic search; the cross-paper linkage map project). Copy the entry to `$CLAUDE_NOTES_DIR/ResearchLibrary.md`.

- [ ] **Step 7: Commit**

```bash
git add SESSION_NOTES.md src/research_library/discovery/store.py
git commit -m "Calibrate discovery fit floors and schedule the daily run" -m "Floors set from a three-day dry run; the Task Scheduler entry runs research-admin discover at 06:30 and catches up after sleep." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
