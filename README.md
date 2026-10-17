# Research Library

A personal Zotero library of investing and finance papers, served to Claude.

- **MCP server** (`research-library`): Claude Desktop reads, searches and writes the Zotero library,
  asks questions answered from the papers, and works through the daily discovery inbox.
- **Streamlit app** (`app.py`): search and ask, browse, and a Contents page with cited topic summaries.
- **Admin CLI** (`research-admin`): builds and refreshes the cached index, embeddings and summaries,
  fetches PDFs, and runs daily paper discovery.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
copy .env.example .env   # then fill in the keys
```

`.env` needs Zotero API credentials, an Anthropic key, a Voyage key and a contact email
(`UNPAYWALL_EMAIL`, also used for OpenAlex). `.env.example` shows the shape.

Quit Claude Desktop before any `pip install -e .`: it runs `research-library.exe` as an MCP server,
Windows locks a running exe, and the install aborts halfway.

Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
"research-library": {
  "command": "C:/Users/james/PycharmProjects/ResearchLibrary/.venv/Scripts/research-library.exe"
}
```

## Admin commands

| Command | What it does |
|---|---|
| `research-admin refresh [--force\|--check]` | Rebuild the library index from Zotero when the paper count changed (Haiku categorises, Sonnet summarises) |
| `research-admin embed [--force]` | Recompute voyage-finance-2 embeddings from the cached index |
| `research-admin links` | Regenerate `paper_index.md` |
| `research-admin contents [--force]` | Write the cited topic summaries for the app's Contents page |
| `research-admin download [--download] [--limit N]` | Find free PDFs for papers that have none (dry run by default) |
| `research-admin discover [--days N] [--force-queries] [--dry-run]` | Daily paper discovery (below) |
| `research-admin all [--force]` | refresh, embed, links, contents in order |

Every command logs to `.cache/refresh_log.txt`.

## Daily paper discovery

Each morning `research-admin discover` searches arXiv (all of q-fin plus keyword searches),
SSRN's finance working-paper feed and OpenAlex for papers posted since the last run, drops anything
the library owns, scores each candidate against the library's embeddings, picks about three per
followed topic, has Haiku summarise them, and writes a digest under `.cache/discovery/`. Nothing is
added to Zotero by the run.

In Claude Desktop:

- "What's new?" calls `get_new_papers`: the inbox of pending papers, grouped by topic, each with an
  id, summary, fit score, nearest owned papers and a recommendation.
- "Keep the second and fourth" calls `keep_new_papers`: files them in Zotero as preprints with the
  topic as a tag, tries to attach the PDF, and folds them into the cached index and embeddings.
- `dismiss_new_papers` drops the rest; unanswered papers expire after 14 days.
- "Consider this link" calls `consider_paper` for a DOI, arXiv or SSRN link someone sent.
- `get_followed_topics` / `set_followed_topics` choose which topic areas to search.

Schedule it once with PowerShell (runs at 06:30, catches up if the PC was asleep):

```powershell
$action  = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c cd /d C:\Users\james\PycharmProjects\ResearchLibrary && .venv\Scripts\research-admin.exe discover' -WorkingDirectory 'C:\Users\james\PycharmProjects\ResearchLibrary'
$trigger = New-ScheduledTaskTrigger -Daily -At 06:30
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "ResearchLibrary Discover" -Action $action -Trigger $trigger -Settings $settings -Description "Daily paper discovery for the research library"
```

## Tests

```powershell
python -m pytest -q
```

Tests marked `integration` call the live Zotero API; the rest run offline.

## Notes

`SESSION_NOTES.md` is the running log of what changed and why. Design documents live under
`docs/superpowers/specs/`.
