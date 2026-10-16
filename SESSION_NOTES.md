# Session notes

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

**Broken / unsure**
- The `research-admin` console script is not in `.venv/Scripts`; use
  `python -m research_library.cli <command>` until the package is reinstalled (`pip install -e .`).
- Root-level `test_*.py` scripts from April are gitignored scratch files and not run by pytest. Probably dead.
- This repo has no `.claude/hooks/require-session-notes.sh`, so the notes rule is not enforced here.

**Next**
- Reinstall the package so `research-admin` works again, then delete the dead root test scripts.
- 450 of 733 papers still link to Scholar rather than a DOI/URL; `research-admin download` (dry-run) shows what PDFs could be fetched.

## History

- 2026-08-14 — Nine Zotero write tools (add, tag, collect, attach PDF). A live smoke test found
  three bugs mocks had missed (DOI dedup never fired, `parentid=` keyword, attachment filename path).
  Pinned the flaky fulltext test to known papers; batch fulltext now picks papers that have text.
- 2026-04-21 — Semantic search and index-aware MCP tools, unified `research-admin` CLI,
  embeddings switched to voyage-finance-2.
- 2026-04-04 — Initial Zotero MCP server, Streamlit app, full-text access, library index with
  subcategories, PDF downloader, periodic refresh script.
