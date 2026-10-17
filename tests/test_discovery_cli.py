"""Tests for `research-admin discover`."""
import logging

from research_library import cli


def _raise(exc):
    raise exc


def test_discover_subcommand_parses_and_calls_run(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return {"fetched": {"arxiv": 1, "ssrn": 2, "openalex": 0}, "new": 3, "scored": 3, "shown": 2,
                "by_topic": {"PE": 2}, "digest": "2026-10-16", "expired": 0, "selected": []}

    monkeypatch.setattr("research_library.discovery.run.discover", fake)
    args = cli.build_parser().parse_args(["discover", "--days", "3", "--force-queries", "--dry-run"])
    assert args.func(args, logging.getLogger("t")) == 0
    assert seen["days"] == 3 and seen["force_queries"] is True and seen["dry_run"] is True and callable(seen["progress"])


def test_discover_reports_missing_index(monkeypatch):
    monkeypatch.setattr("research_library.discovery.run.discover", lambda **kw: _raise(FileNotFoundError("No library index")))
    args = cli.build_parser().parse_args(["discover"])
    assert args.func(args, logging.getLogger("t")) == 1
