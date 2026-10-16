"""Tests for the PDF downloader: attaching, Unpaywall contact email, console logging."""
import io
import logging
import os
import sys
from unittest.mock import MagicMock

from research_library import cli, downloads


# ── attach_pdf_to_item ─────────────────────────────────────────────────────────

def test_attach_goes_through_attach_pdf_with_a_bare_filename(monkeypatch):
    """Zotero rejects a stored filename that contains a directory, so uploads
    must go through zotero_client.attach_pdf, which sends name and dir apart."""
    seen = {}

    def fake_attach_pdf(item_key, pdf_path):
        seen["key"] = item_key
        seen["path"] = pdf_path
        with open(pdf_path, "rb") as f:
            seen["bytes"] = f.read()
        return {"ok": True, "unchanged": False, "filename": os.path.basename(pdf_path)}

    monkeypatch.setattr(downloads, "attach_pdf", fake_attach_pdf)

    ok = downloads.attach_pdf_to_item(None, "ABCD1234", 'A/B: "Risk" <Parity>?', b"%PDF-1.4 data")

    assert ok is True
    assert seen["key"] == "ABCD1234"
    assert seen["bytes"] == b"%PDF-1.4 data"
    name = os.path.basename(seen["path"])
    assert name.endswith(".pdf")
    assert not any(c in name for c in '\\/:*?"<>|')
    assert not os.path.exists(seen["path"]), "temp file should be cleaned up"


def test_attach_reports_failure_when_zotero_rejects(monkeypatch):
    monkeypatch.setattr(downloads, "attach_pdf",
                        lambda key, path: {"error": "Zotero rejected the attachment"})
    assert downloads.attach_pdf_to_item(None, "ABCD1234", "Title", b"%PDF") is False


def test_attach_reports_failure_when_attach_raises(monkeypatch):
    def boom(key, path):
        raise RuntimeError("network down")
    monkeypatch.setattr(downloads, "attach_pdf", boom)
    assert downloads.attach_pdf_to_item(None, "ABCD1234", "Title", b"%PDF") is False


# ── Unpaywall contact email ────────────────────────────────────────────────────

def test_unpaywall_skipped_without_contact_email(monkeypatch):
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    requests = MagicMock()
    monkeypatch.setattr(downloads, "_lazy_imports", lambda: (requests, None))

    assert downloads.try_unpaywall("10.1000/xyz") is None
    requests.get.assert_not_called()


def test_unpaywall_uses_email_from_env(monkeypatch):
    monkeypatch.setenv("UNPAYWALL_EMAIL", "someone@example.org")
    api = MagicMock(ok=True)
    api.json.return_value = {"best_oa_location": {"url_for_pdf": "https://x.org/p.pdf"}}
    pdf = MagicMock(ok=True, headers={"Content-Type": "application/pdf"}, content=b"%PDF")
    requests = MagicMock()
    requests.get.side_effect = [api, pdf]
    monkeypatch.setattr(downloads, "_lazy_imports", lambda: (requests, None))

    assert downloads.try_unpaywall("10.1000/xyz") == b"%PDF"
    first_url = requests.get.call_args_list[0].args[0]
    assert "email=someone@example.org" in first_url
    assert "example.com" not in first_url


# ── Console logging ────────────────────────────────────────────────────────────

def test_console_logging_survives_characters_the_console_cannot_encode(monkeypatch, tmp_path, capsys):
    """A paper title with a Greek letter crashed logging on a cp1252 console."""
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(cli, "LOG_PATH", tmp_path / "refresh_log.txt")

    log = cli._setup_logging()
    log.info("Optimal λ for shrinkage")
    for h in logging.getLogger().handlers:
        h.flush()

    assert "UnicodeEncodeError" not in capsys.readouterr().err
    assert "Optimal" in console.buffer.getvalue().decode("cp1252")
    assert "Optimal λ" in (tmp_path / "refresh_log.txt").read_text(encoding="utf-8")

    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
        h.close()


# ── fetch_pdf content check ────────────────────────────────────────────────────

def test_fetch_pdf_rejects_bytes_that_are_not_a_pdf(monkeypatch):
    """An octet-stream response can be an HTML error page; never attach that."""
    monkeypatch.setattr(downloads, "try_direct_url", lambda url: b"<html>Access denied</html>")
    monkeypatch.setattr(downloads, "try_unpaywall", lambda doi: None)
    monkeypatch.setattr(downloads, "try_scrape_for_pdf", lambda url: None)
    assert downloads.fetch_pdf("", "https://x.org/p.pdf") == (None, "failed")


def test_fetch_pdf_accepts_real_pdf_bytes(monkeypatch):
    monkeypatch.setattr(downloads, "try_direct_url", lambda url: b"%PDF-1.7\n...")
    assert downloads.fetch_pdf("", "https://x.org/p.pdf") == (b"%PDF-1.7\n...", "direct_url")
