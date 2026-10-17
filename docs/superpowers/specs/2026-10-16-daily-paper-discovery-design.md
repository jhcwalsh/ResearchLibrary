# Daily paper discovery with an approved digest

Date: 2026-10-16. Status: approved in conversation; builder check done and folded in.

## Intent

James wants the research library to work as a second mind: when he has a real task
(say, a private equity cash flow model that takes the current market environment as an
input), Claude in Claude Desktop should reason from everything relevant he already owns.
For that to stay true the library has to keep up with new work. This project is the feeder:
each day it looks for new papers in the areas he follows, summarises them, and puts the
best few in front of him. He approves; nothing enters Zotero without him.

A second project, the cross-paper linkage map and the MCP tools that serve a working
session, is not covered here and comes next.

## Goals

- Every morning a digest of the best new papers (roughly 3 per followed area, 15 at most)
  from arXiv and SSRN, each with a summary where an abstract exists, a fit score, the nearest
  papers he already owns, and a one-word recommendation.
- Approve or dismiss from Claude Desktop in one sentence ("keep the second and fourth").
  Kept papers go into Zotero with the area as a tag and a PDF where one can be found, and are
  folded into the cached index and embeddings straight away so semantic search and the
  Contents page see them without a rebuild.
- Links people send him go through the same scoring and the same keep step.
- He can choose which of the 19 topic areas to follow.

## Non-goals and anti-goals

- Never adds to Zotero on its own. The digest is a proposal; `keep` is the only write.
- Never fetches or scores a paper again once it has been put in front of him: pending, kept
  and dismissed candidates are remembered.
- Never triggers a full index rebuild (that is a paid Haiku and Sonnet pass over the whole
  library, and category names can shift). `research-admin all` stays a manual step.
- No email, no new Streamlit page. Claude Desktop is the only front end.
- Running cost under about ten cents a day.

## Constraints

- Runs on James's Windows PC, where the MCP server, `.cache/library_index.json` and
  `.cache/embeddings.npz` live. The Mac mini does not have this repo.
- Reuse what exists: `embeddings.py` (voyage-finance-2 vectors), `zotero_client.create_paper`,
  `attach_pdf`, `downloads.fetch_pdf`, the `HAIKU` id and `response_text` in `llm.py`, and
  the Zotero write tools in `tools.py`. No new heavy dependency. `requests` and
  `beautifulsoup4` move from the `download` extra to core dependencies because the keep flow
  fetches PDFs inside the MCP server process (both are already installed in the venv, so no
  reinstall is needed; remember that `pip install -e .` needs Claude Desktop closed).
- Parse arXiv's Atom feed with the standard library.
- Model calls follow `llm.py` conventions: explicit effort, `max_tokens` with room for
  thinking, JSON schema output where structure is needed.
- The repo has no README. Create a short one that covers setup, the admin commands and the
  scheduled task.

## What the probes showed (2026-10-16)

- OpenAlex has almost no SSRN papers within weeks of posting (2 in the last 30 days by
  primary source; a few hundred a week by any location against about 5,000 posted), so it
  cannot be the SSRN route on a daily window.
- SSRN's own listing endpoint, `https://api.ssrn.com/content/v1/bindings/{id}/papers`
  with `index`, `count` (up to 200) and `sort=0` (newest first), is public and current to the
  same day. Binding 203 is the finance working-paper stream: about 85 papers a day with
  finance titles. It returns title, authors (first and last name), affiliations, approved date,
  page count, download counts and the paper URL. It does not return abstracts, the detail
  endpoints need a token, and the paper pages return 403 to scripts. So SSRN candidates are
  title-only.
- James's SSRN alert emails do not reach the Gmail account connected here, so an email route
  is out regardless.
- arXiv `q-fin.*` runs about 10 papers a day; the Atom feed carries abstracts and a PDF link.
- Library similarity scale on voyage-finance-2 (title plus abstract, cosine): the mean of a
  paper's three nearest library neighbours has median 0.70 and lower quartile 0.63; random
  pairs sit around 0.38. Starting fit floor 0.62 for candidates with abstracts.

## How it works

### Sources

1. **arXiv API** (`export.arxiv.org/api/query`): everything submitted in the window in the
   `q-fin.*` categories, plus the per-topic keyword searches across `econ.*`, `stat.ML` and
   `cs.LG`. Respect the 3 second gap between requests.
2. **SSRN listing feed**: binding ids from `follow.json` (default `[203]`), paginated
   newest-first until the approved date falls before the window. Title-only.
3. **OpenAlex** (`api.openalex.org/works`): the per-topic keyword searches with
   `from_publication_date` in the window, as a broad second source for journal and preprint
   papers that carry abstracts (rebuild the inverted abstract into text). Send `mailto`;
   reuse `UNPAYWALL_EMAIL` for that.

The window is from the last successful run to now, with a floor of 1 day and a cap of 7.
First run looks back 2 days.

### Search queries per topic

Haiku writes 4 to 6 short search phrases for each followed topic from the topic name, its
subtopic names and a sample of its titles. They are cached in the follow file keyed by the
topic's fingerprint (the same fingerprint `contents.py` uses), so they are only regenerated
when the topic changes or on `--force-queries`.

### Candidates, dedupe, scoring

- Merge results by DOI, then arXiv id, then SSRN id, then normalised title
  (`analysis._title_key`).
- Drop anything already owned: DOI or normalised title matches a paper in the cached index,
  or the id is in the ledger.
- Embed every remaining candidate (title plus abstract, or title alone for SSRN) with
  voyage-finance-2 as a document and compare against all library vectors.
- **Per-topic fit** is the mean cosine of the candidate's three nearest papers *within* that
  topic. **Topic** is the followed topic with the highest per-topic fit, and **fit** is that
  value. This gives small topics a fair chance; majority voting over global neighbours would
  hand almost everything to the two topics that hold 38% of the library. **Nearest papers**
  shown on the card are the global top 3.
- Floors: `min_fit` (default 0.62) for candidates with an abstract and `min_fit_title_only`
  (default 0.55, to be calibrated) for title-only ones.
- Selection: within each followed topic rank by fit and keep up to `per_topic` (default 3)
  above the floor. Then fill `max_total` (default 15) slots round-robin across topics by
  within-topic rank, so every topic with a qualifying paper gets its best one before any topic
  gets its second.

### Summaries

For each selected candidate one Haiku call, low effort, JSON schema:
`summary` (up to 120 words: what it does, main finding, why it matters; for title-only
candidates this field is omitted and the card says the feed carries no abstract),
`why_it_fits` (one or two sentences naming the nearest owned papers by short citation),
`recommendation` (`keep`, `maybe` or `skip`) and `reason` (one sentence).
The prompt includes the nearest owned papers' titles and abstracts, and for SSRN items the
authors, affiliations, page count and download counts, which is all we have.

### Files

All under `.cache/discovery/`:

- `follow.json`: followed topics (default: all topics in the index), per-topic queries with
  their fingerprint, `ssrn_bindings`, `min_fit`, `min_fit_title_only`, `per_topic`,
  `max_total`, `last_run`.
- `digest-YYYY-MM-DD.json`: the day's run. Candidates carry everything needed to keep them
  later: metadata in the index's shape (authors as "Last, First", `url` included), abstract,
  source, PDF URL, assigned topic, fit, nearest papers, summary fields, embedding vector. A
  section `sent` holds ad hoc candidates from `consider_paper`.
- `seen.json`: ledger `{candidate id: {status, date, digest, title, zotero_key}}` with
  status `pending`, `kept`, `dismissed` or `expired`. Low-scoring candidates are not
  recorded; they drop out of the window naturally.
- `discover.log`: run log.

Candidate id is the first 12 hex characters of sha1 over the DOI if present, else the arXiv
id, else the SSRN id, else the normalised title.

### The inbox

Candidates enter the ledger as `pending` when a digest is written. `get_new_papers()`
with no date returns every pending candidate across digests, newest digest first, grouped
by topic, so a morning after a few days away shows everything not yet acted on. Asking for a
date returns that day's digest as written. Pending candidates older than 14 days become
`expired` at the next run and are listed no more; they can still be found with a date.

### Keeping a paper

`keep` does, per candidate, in this order and reporting each step's outcome:

1. `create_paper` with item type `preprint` for arXiv and SSRN items (else `journalArticle`),
   title, authors, year, DOI, URL, abstract, tags `[topic, "discovered"]`. For preprints the
   venue goes in `repository` ("arXiv" or "SSRN") and the id in `archiveID`; this needs a
   preprint branch in `_build_item_payload`, which today sends journal-only fields that Zotero
   rejects for preprints. A DOI duplicate returned by Zotero counts as kept and is reported as
   already owned.
2. Try `downloads.fetch_pdf` with the candidate's PDF URL (arXiv gives one directly) and
   DOI; attach on success. Failure here is reported, never fatal.
3. Append the paper to `library_index.json` under its topic (index paper shape plus `url`,
   so the Contents page links it directly) and append its vector and topic to
   `embeddings.npz`. Do not touch subcategories (the paper shows under "Other papers") and do
   not bump `total_fetched`, so the next manual refresh still notices the library grew. The
   topic's fingerprint changes, so the Contents page shows its "summary predates the latest
   papers" caption until `research-admin contents` rewrites that one topic; that is accurate
   and accepted.
4. Mark the ledger `kept` with the Zotero key.

### Claude Desktop tools (added to `tools.py`)

- `get_new_papers(date?)`: the inbox (or a given day's digest), grouped by topic, each
  candidate with its id, title, authors, source and date, summary or the no-abstract note,
  why it fits, nearest owned papers as short citations, recommendation, link. Says when the
  latest digest was made and how many candidates were scored.
- `keep_new_papers(ids, attach_pdf=true)`: the keep flow above. Write tool, so its
  description tells the agent to confirm first, matching the existing write tools.
- `dismiss_new_papers(ids | all=true)`: marks dismissed.
- `consider_paper(reference)`: a URL, DOI, arXiv id or SSRN id. Fetches metadata from arXiv
  (arXiv ids and URLs), OpenAlex (DOIs), then Crossref as a fallback; SSRN URLs are looked up
  by id on the listing feed's first pages and otherwise need the title typed. Scores and
  summarises exactly as the daily run does, ignoring the floor; stores it under `sent` in
  today's digest as `pending` and returns the same card so `keep_new_papers` works on it.
- `get_followed_topics()` and `set_followed_topics(follow=[], unfollow=[])`: names are
  validated against the index categories.

### CLI and schedule

- `research-admin discover [--days N] [--force-queries] [--dry-run]` runs the daily job and
  logs a one-line summary (fetched per source, new, shown per topic).
- A Windows scheduled task, created once with `schtasks` and documented in the README, runs
  `discover` at 06:30 daily with "run as soon as possible after a missed start" so a sleeping
  PC catches up. It runs `.venv\Scripts\research-admin.exe` with the repo as working
  directory. Nothing else is chained to it.
- `.env.example` notes that `UNPAYWALL_EMAIL` also identifies us to OpenAlex.

## Left to the builder

- The exact arXiv category list and OpenAlex filters, and request pacing.
- Calibrating both fit floors on a real window before the task is scheduled.
- Prompt wording for query generation and summaries.
- Card formatting for Desktop.
- Test fixtures: a small arXiv Atom sample, an SSRN listing sample, an OpenAlex works sample
  with an inverted abstract, synthetic embeddings. Every network and model call is mocked in
  tests; one live dry run before scheduling.

## Decisions made on James's behalf

- Run on the PC via Task Scheduler rather than the Mini, because the cache and the MCP
  server are here and nothing is scheduled anywhere today.
- SSRN via its listing feed, title-only, rather than OpenAlex (too slow) or alert emails
  (not reachable).
- Per-topic fit for assignment and round-robin filling, so small topics are represented.
- An inbox of pending candidates rather than strict per-day digests, with 14-day expiry.
- Incremental index and embedding update on keep, accepting the Contents page caption.
- Followed topics default to all 19; `follow.json` lives in `.cache/discovery/` and is
  regenerated with defaults if deleted.
- Haiku for queries and summaries; Sonnet is not needed here.
