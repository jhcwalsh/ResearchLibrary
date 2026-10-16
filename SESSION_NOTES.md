# Session notes

## 2026-10-16 — Switched to Claude Haiku 5.5 / Sonnet 5.5

**Done**
- All 7 Claude calls moved from Haiku 4.5 / Sonnet 4.6 to `claude-haiku-5-5` / `claude-sonnet-5-5`
  (`app.py`, `analysis.py`, `tools.py`). New `src/research_library/llm.py` holds the IDs and `response_text()`.
- Effort set explicitly: low for keywords, category rescue and summaries; medium for categorising,
  subcategories and answers. Small `max_tokens` caps (50, 300, 512, 1024) raised to leave room for thinking.
- New `tests/test_llm.py`; the `ask_library` test mocks now return a thinking block first. 124 tests pass.
  One live call to each model checked out.

**Why**
- The 5.5 models think by default, so every `content[0].text` read would have hit an empty thinking block.
- Refusals now raise a clear `RefusalError` instead of a confusing empty answer.

**Broken / unsure**
- No server-side refusal fallback for the Sonnet calls: SDK 0.101 has no `fallbacks` parameter (needs 1.x).
- Categorisation quality on the new models is untested until the next `research-admin refresh --force`.

## 2026-10-15 (later) — PDF downloader fixed

**Done**
- `research-admin download` dry run: 192 papers have a DOI/URL but no PDF (59 with a DOI, 123 URL only).
- `downloads.py`: uploads now go through `zotero_client.attach_pdf`. The old `attachment_simple` call
  sends the full path as the filename, which Zotero rejects, so every upload would have failed.
- Unpaywall contact email now comes from `UNPAYWALL_EMAIL` in `.env` (was `research@example.com`);
  unset means Unpaywall is skipped with one warning. Added to `.env.example`.
- `cli.py`: console logging no longer crashes on titles a cp1252 console can't print (e.g. λ).
- New `tests/test_downloads.py` (8 tests). Downloads must now start with the `%PDF-` signature, so an HTML error page labelled octet-stream is never attached. Installed bs4 directly, not via reinstall, to avoid the exe lock.

- Real run: **25 of 192 PDFs attached** (11 Unpaywall, 10 page scrape, 3 direct, 1 DOI page),
  167 found no PDF, 0 rejected by Zotero. Spot-checked two in Zotero: named files with md5, so stored.

**Why**
- Reuse the August-verified attach path rather than fix a second copy of it.

**Broken / unsure**
- PDFs uploaded through the API have no full text until Zotero desktop syncs and indexes them.
  Embeddings use title + abstract only, so they are unaffected.

## 2026-10-15 — Progress review, index refresh committed

**Done**
- Reviewed the state of the repo. All 111 tests in `tests/` pass.
- Committed the 13 Oct regenerated `paper_index.md`: 543 → 733 papers, 15 → 19 categories.
  Machine Learning & AI in Finance is now the largest category (145 papers).
- Started this file. Pushed the 4 commits that were waiting (voyage-finance-2 switch to full-text fetch fix).

**Why**
- The index rebuild was sitting uncommitted, and nothing since April was on the remote.
- Checked whether the semantic cache was stale before committing: `refresh --check` reports
  817 Zotero items both cached and live, and `embeddings.npz` holds 733 papers / 19 categories,
  all built together at 13 Oct 22:36. So no rebuild was needed.

- Later: reinstalled the package so `research-admin` works, and deleted the five dead root
  `test_*.py` scratch scripts (untracked) plus their `.gitignore` lines.

**Broken / unsure**
- `research-admin.exe` had never been written because Claude Desktop runs `research-library.exe`
  as an MCP server, and Windows locks a running exe, so pip aborted halfway. **Quit Claude Desktop
  before any `pip install -e .`** or the install half-uninstalls and leaves a `~esearch_library` folder.
- This repo has no `.claude/hooks/require-session-notes.sh`, so the notes rule is not enforced here.

**Next**
- The 167 papers with no free PDF are mostly paywalled or book summaries; attach those by hand in Zotero if wanted.

## History

- 2026-08-14 — Nine Zotero write tools (add, tag, collect, attach PDF). A live smoke test found
  three bugs mocks had missed (DOI dedup never fired, `parentid=` keyword, attachment filename path).
  Pinned the flaky fulltext test to known papers; batch fulltext now picks papers that have text.
- 2026-04-21 — Semantic search and index-aware MCP tools, unified `research-admin` CLI,
  embeddings switched to voyage-finance-2.
- 2026-04-04 — Initial Zotero MCP server, Streamlit app, full-text access, library index with
  subcategories, PDF downloader, periodic refresh script.
