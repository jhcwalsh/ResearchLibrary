# Session notes

## 2026-10-23 — Paper Map and a desktop launcher

**Done**
- `Research Library.bat` in the repo root: copy it to the desktop and double-click. It cds to the
  repo (absolute path, edit `APP_DIR` if the repo moves), checks the venv exists, and runs
  `streamlit run app.py` on the default port 8501 with the browser opening. Tested from the real
  desktop folder (`D:\OneDrive\Desktop`): the app answered HTTP 200.
- New "Paper Map" page in `app.py`, second in the sidebar. A d3 force-directed graph of the whole
  library (or one topic): each paper is a dot coloured by topic and sized by its links; a line joins
  it to its *k* most similar papers above a similarity floor, both from `embeddings.npz`. Hover
  shows title and citation and lights up the neighbours, click opens the paper (DOI, URL, else
  Scholar), drag and zoom work, the legend fades topics, and a text box rings papers matching a
  title or author. Under the map, "Pick a paper" lists its 12 nearest papers with scores and topics.
- `src/research_library/linkmap.py` (`build_link_graph`, `nearest_papers`: pure functions over the
  two caches) and `src/research_library/mapview.py` (the HTML/JS template, d3 7.9.0 from cdnjs,
  light and dark colours via `prefers-color-scheme`). Spec in `docs/superpowers/specs/`.
- `tests/test_linkmap.py` (5 tests). 175 offline tests pass. Checked in Chrome on the dark theme:
  map, hover, highlight and the nearest-papers list.
- 24 Oct: new MCP tool `get_nearest_papers` (key, limit) in `tools.py` wraps `linkmap.nearest_papers`:
  the starting paper, then neighbours best first with score, topic, authors, year and link, plus a note
  that similarity is content not citation. 4 tests in `test_semantic_tools.py`; 179 pass. Claude Desktop
  sees it after a restart. README mentions the map, the launcher and the tool.

**Why**
- "A graphical interface that links the research papers": the session notes already named the
  cross-paper linkage map as the next project, and the embeddings make it free (no model calls).
  Defaults of 4 links per paper and a 0.55 floor give 2082 links over 733 papers with 30 unlinked,
  which reads as clusters rather than a hairball; the sliders let James thin or thicken it.

**Broken / unsure**
- Edges are content similarity, not citations; the caption says so. The map needs internet for
  the d3 script. The iframe picks light or dark from the OS, not from Streamlit's own theme
  setting, so a forced Streamlit theme that differs from the OS would mismatch.
- The faded-neighbour opacity on hover (0.15) is a guess; may want to be lighter on the dark theme.

**Next**
- Expose the topic summaries over MCP too, and try `get_nearest_papers` from Claude Desktop.
- Possibly a 2D layout from the embeddings themselves (UMAP) instead of a force layout, if the
  clusters are worth reading spatially.

## 2026-10-16 (later still) — Daily paper discovery

**Done**
- New `research_library.discovery` package (store, sources, scoring, summaries, run, keep, cards, handlers),
  `research-admin discover`, six MCP tools (get_new_papers, keep_new_papers, dismiss_new_papers, consider_paper,
  get/set_followed_topics), preprint support in `zotero_client`, and the repo's first `README.md`. Design and
  plan under `docs/superpowers/`. 173 tests pass.
- Sources: arXiv (q-fin firehose plus one keyword query per topic, HTTPS with a retry), SSRN's listing feed
  (binding 203, titles only: no source carries SSRN abstracts within days of posting), OpenAlex (abstracts,
  mostly journals). OpenAlex has almost no recent SSRN papers and SSRN alert emails do not reach the connected
  Gmail, so those routes were dropped.
- Scoring: per-topic fit, floors calibrated on a 3-day window. With-abstract candidates that belonged scored
  0.49-0.61 (floor 0.50); title-only ones score higher across the board and the relevant ones sat at 0.78-0.83
  (floor 0.76). Within a topic candidates rank by margin over their own floor so SSRN titles do not crowd out arXiv.
- First real run: 297 fetched, 292 new, 15 shown across 10 topics. The inbox renders through `get_new_papers`
  and the cards read well. Scheduled task "ResearchLibrary Discover" runs daily at 06:30 and catches up after sleep.

- Review fix pass (fresh reviewer, no critical findings): keep marks the ledger right after the Zotero write and survives index or embedding failures, so a retry never double-files; consider_paper answers in a sentence on any failure, strips a `doi:` prefix from page meta tags, and puts a previously dismissed paper back in the inbox; the inbox reads only digests with pending entries and vectors are stored at 6 dp; follow.json persists only explicit settings and is re-read before the run saves it; all JSON and npz writes are atomic. 183 tests pass.
**Why**
- James wants the library to feed itself: a daily digest he approves from Claude Desktop, with kept papers
  folded into Zotero and the cached index without a paid rebuild.

**Broken / unsure**
- Floors come from one 3-day sample; watch the first week. SSRN cards have no abstract, so Haiku can only say
  why the title fits. arXiv keyword queries find little in a 1-day window; the q-fin firehose carries the load.
- `consider_paper` on a generic URL depends on publisher citation meta tags. Keeping a paper changes its topic
  fingerprint, so the Contents page shows the "summary predates" caption until `research-admin contents` runs.
- Not yet tried end to end: keeping a paper from Claude Desktop against the live Zotero library.

**Next**
- Tomorrow in Claude Desktop: "what's new", keep one, check it in Zotero and in semantic search.
- The second project: the cross-paper linkage map and topic summaries over MCP.

## 2026-10-16 (later) — Contents page with cited topic summaries

**Done**
- New `src/research_library/contents.py`: Sonnet 5.5 writes, per index category, a 150-250 word
  overview, a note per subtopic and 3-6 "start with" papers, citing Zotero keys in square brackets.
  Cached in `.cache/topic_summaries.json` (gitignored with the rest of `.cache/`) with a fingerprint,
  so a topic is only rewritten when its papers or subtopic grouping change. Four topics run in parallel;
  the file is saved after each one finishes.
- `research-admin contents [--force]` in `cli.py`; `all` now runs it last.
- `links.py`: `paper_link` helper (DOI, else URL, else Scholar search) shared with the app.
- `app.py`: "Contents" is now the first sidebar mode. Filterable topic list on the left; on the right
  the overview with `[KEY]` citations rendered as linked "Ilmanen et al. 2020" names, the start-with
  list, and an expander per subtopic (newest first). Warns when a summary predates the topic's papers.
- `tests/test_contents.py` (9 tests). 133 tests pass.
- Live run wrote all 19 topics in about 50 seconds, none failed. Checked in a browser: overview,
  subtopic expander, a topic with no subcategories, and the filter all render.

**Why**
- The Library Index page is a flat list; this gives a newcomer a way in to each topic, with every
  claim traceable to a paper.

**Broken / unsure**
- Many citations show "n.d." because the year is missing in Zotero. Papers with no author show the
  title twice ("Margin of Safety · Margin of Safety n.d."). Both are data quality, not code.
- Summary quality has had one read-through, not a careful check against the papers.

**Next**
- Fill in missing years and authors in Zotero for the Titans Brief items, then `research-admin contents --force`.
- Consider an MCP tool that returns a topic summary so Claude Desktop can use them too.

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
