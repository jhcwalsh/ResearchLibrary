"""
Tests for the write helpers in zotero_client.py.

All tests mock the pyzotero client — nothing here touches the live API.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, "src")

from research_library.zotero_client import (
    _normalize_doi,
    _parse_creators,
    _build_item_payload,
    find_by_doi,
    create_paper,
    create_papers_batch,
    add_to_collection,
    remove_from_collection,
    add_tag,
    remove_tag,
    update_paper,
    attach_pdf,
    create_collection,
)


@pytest.fixture(autouse=True)
def _clear_doi_index():
    """The DOI index is cached at module level; keep tests independent."""
    from research_library import zotero_client
    zotero_client._reset_doi_index()
    yield
    zotero_client._reset_doi_index()


def _make_raw_item(key="ABC123", doi="10.1/test", collections=None, tags=None):
    return {
        "data": {
            "key": key,
            "title": "Test Paper",
            "itemType": "journalArticle",
            "date": "2023",
            "abstractNote": "",
            "DOI": doi,
            "url": "",
            "publicationTitle": "Test Journal",
            "creators": [],
            "tags": tags if tags is not None else [],
            "collections": collections if collections is not None else [],
        }
    }


# ── Unit: DOI + creator parsing ───────────────────────────────────────────────

class TestNormalizeDoi:
    def test_strips_doi_org_prefix(self):
        assert _normalize_doi("https://doi.org/10.1/abc") == "10.1/abc"

    def test_strips_dx_doi_org_prefix(self):
        assert _normalize_doi("http://dx.doi.org/10.1/abc") == "10.1/abc"

    def test_case_insensitive_prefix(self):
        assert _normalize_doi("HTTPS://DOI.ORG/10.1/abc") == "10.1/abc"

    def test_bare_doi_unchanged(self):
        assert _normalize_doi("10.1/abc") == "10.1/abc"

    def test_none_and_empty_return_empty(self):
        assert _normalize_doi(None) == ""
        assert _normalize_doi("") == ""


class TestParseCreators:
    def test_last_first_format(self):
        creators = _parse_creators(["Smith, John"])
        assert creators == [{"creatorType": "author", "lastName": "Smith", "firstName": "John"}]

    def test_full_name_format(self):
        creators = _parse_creators(["John Smith"])
        assert creators == [{"creatorType": "author", "name": "John Smith"}]

    def test_empty_entries_skipped(self):
        assert _parse_creators(["", "  ", "Smith, John"]) == [
            {"creatorType": "author", "lastName": "Smith", "firstName": "John"}
        ]


# ── Unit: payload building ────────────────────────────────────────────────────

class TestBuildItemPayload:
    def _build(self, item_type="journalArticle", **overrides):
        kwargs = dict(
            item_type=item_type, title="T", authors=["Smith, John"], year="2023",
            doi="10.1/x", journal="J. Finance", volume="12", issue="3", pages="1-10",
            url=None, abstract=None, tags=["macro"], collection_keys=["COL1"],
        )
        kwargs.update(overrides)
        return _build_item_payload(**kwargs)

    def test_journal_article_uses_doi_and_publication_title(self):
        p = self._build()
        assert p["DOI"] == "10.1/x"
        assert p["publicationTitle"] == "J. Finance"
        assert p["volume"] == "12"
        assert p["issue"] == "3"
        assert p["pages"] == "1-10"

    def test_book_moves_doi_to_extra_and_journal_to_publisher(self):
        p = self._build(item_type="book")
        assert "DOI" not in p
        assert p["extra"] == "DOI: 10.1/x"
        assert p["publisher"] == "J. Finance"
        assert p["numPages"] == "1-10"
        assert "publicationTitle" not in p

    def test_tags_and_collections_shaped_for_zotero(self):
        p = self._build()
        assert p["tags"] == [{"tag": "macro"}]
        assert p["collections"] == ["COL1"]


# ── Unit: find_by_doi ─────────────────────────────────────────────────────────

def _library_mock(items):
    """A client that behaves like the real Zotero API: q-search never matches a
    DOI (the API only indexes title/creator/year + attachment fulltext), while
    everything() returns the full library."""
    mock_zot = MagicMock()
    mock_zot.items.return_value = []
    mock_zot.everything.return_value = items
    return mock_zot


class TestFindByDoi:
    def test_returns_matching_item(self):
        mock_zot = _library_mock([_make_raw_item(doi="10.1/target")])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = find_by_doi("https://doi.org/10.1/target")
        assert result is not None
        assert result["data"]["key"] == "ABC123"

    def test_does_not_rely_on_q_search(self):
        """Regression: the Zotero q-search does not index the DOI field in any
        qmode, so a lookup built on zot.items(q=...) silently never matches."""
        mock_zot = _library_mock([_make_raw_item(key="REAL", doi="10.1/target")])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = find_by_doi("10.1/target")
        assert result is not None and result["data"]["key"] == "REAL"

    def test_returns_none_when_no_exact_match(self):
        mock_zot = _library_mock([_make_raw_item(doi="10.1/other")])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            assert find_by_doi("10.1/target") is None

    def test_matches_regardless_of_doi_url_prefix_and_case(self):
        mock_zot = _library_mock([_make_raw_item(key="K", doi="HTTPS://DOI.ORG/10.1/Target")])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            assert find_by_doi("10.1/target")["data"]["key"] == "K"

    def test_empty_doi_returns_none_without_api_call(self):
        with patch("research_library.zotero_client.get_client") as mock_get:
            assert find_by_doi("") is None
        mock_get.assert_not_called()

    def test_index_is_built_once_across_lookups(self):
        """A full-library scan costs ~20s, so N lookups must not mean N scans."""
        mock_zot = _library_mock([_make_raw_item(doi="10.1/a")])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            find_by_doi("10.1/a")
            find_by_doi("10.1/b")
            find_by_doi("10.1/c")
        assert mock_zot.everything.call_count == 1


# ── Unit: create_paper ────────────────────────────────────────────────────────

class TestCreatePaper:
    def test_dedup_returns_existing_item(self):
        existing = _make_raw_item(doi="10.1/dup")
        with patch("research_library.zotero_client.find_by_doi", return_value=existing), \
             patch("research_library.zotero_client.get_client") as mock_get:
            result = create_paper(title="T", authors=["A B"], doi="10.1/dup")
        assert result["duplicate"] is True
        assert result["key"] == "ABC123"
        mock_get.assert_not_called()

    def test_force_bypasses_dedup(self):
        mock_zot = MagicMock()
        mock_zot.create_items.return_value = {"successful": {"0": _make_raw_item()}, "failed": {}}
        with patch("research_library.zotero_client.find_by_doi") as mock_find, \
             patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = create_paper(title="T", authors=["A B"], doi="10.1/dup", force=True)
        mock_find.assert_not_called()
        assert result["duplicate"] is False

    def test_failed_creation_returns_error(self):
        mock_zot = MagicMock()
        mock_zot.create_items.return_value = {"failed": {"0": {"code": 400, "message": "bad"}}}
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = create_paper(title="T", authors=["A B"])
        assert "error" in result


# ── Unit: create_papers_batch ─────────────────────────────────────────────────

class TestCreatePapersBatch:
    def test_empty_input(self):
        assert create_papers_batch([]) == {"created": [], "duplicates": [], "failed": []}

    def test_mixed_duplicates_and_created(self):
        existing = _make_raw_item(key="DUP1", doi="10.1/dup")
        mock_zot = MagicMock()
        mock_zot.create_items.return_value = {
            "successful": {"0": _make_raw_item(key="NEW1", doi="10.1/new")},
            "failed": {},
        }

        def fake_find(doi):
            return existing if doi == "10.1/dup" else None

        with patch("research_library.zotero_client.find_by_doi", side_effect=fake_find), \
             patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = create_papers_batch([
                {"title": "Dup", "authors": ["A"], "doi": "10.1/dup"},
                {"title": "New", "authors": ["B"], "doi": "10.1/new"},
            ])
        assert [p["key"] for p in result["duplicates"]] == ["DUP1"]
        assert [p["key"] for p in result["created"]] == ["NEW1"]
        assert result["failed"] == []

    def test_duplicate_dois_within_one_batch_collapse(self):
        """Two entries sharing a DOI must not both be created: nothing exists in
        the library yet, so only the library check would let both through."""
        mock_zot = _library_mock([])
        mock_zot.create_items.return_value = {
            "successful": {"0": _make_raw_item(key="NEW1", doi="10.1/same")},
            "failed": {},
        }
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = create_papers_batch([
                {"title": "First", "authors": ["A"], "doi": "10.1/same"},
                {"title": "Second copy", "authors": ["B"], "doi": "https://doi.org/10.1/same"},
            ])
        assert len(mock_zot.create_items.call_args_list[0].args[0]) == 1
        assert len(result["created"]) == 1
        assert len(result["duplicates"]) == 1

    def test_batch_scans_library_only_once(self):
        mock_zot = _library_mock([])
        mock_zot.create_items.return_value = {"successful": {}, "failed": {}}
        papers = [{"title": f"P{i}", "authors": [], "doi": f"10.1/{i}"} for i in range(10)]
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            create_papers_batch(papers)
        assert mock_zot.everything.call_count == 1

    def test_failed_entries_report_original_input(self):
        mock_zot = MagicMock()
        mock_zot.create_items.return_value = {
            "successful": {},
            "failed": {"0": {"code": 400, "message": "bad"}},
        }
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = create_papers_batch([{"title": "Bad", "authors": []}])
        assert result["failed"][0]["input"]["title"] == "Bad"

    def test_chunks_of_50(self):
        mock_zot = MagicMock()
        mock_zot.create_items.return_value = {"successful": {}, "failed": {}}
        papers = [{"title": f"P{i}", "authors": []} for i in range(60)]
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            create_papers_batch(papers)
        assert mock_zot.create_items.call_count == 2
        assert len(mock_zot.create_items.call_args_list[0].args[0]) == 50
        assert len(mock_zot.create_items.call_args_list[1].args[0]) == 10


# ── Unit: collection + tag membership ─────────────────────────────────────────

class TestCollectionMembership:
    def test_add_to_collection(self):
        mock_zot = MagicMock()
        mock_zot.item.return_value = _make_raw_item(collections=["COL1"])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = add_to_collection("ABC123", "COL2")
        assert result == {"ok": True, "already_member": False, "collections": ["COL1", "COL2"]}
        mock_zot.update_item.assert_called_once()

    def test_add_to_collection_idempotent(self):
        mock_zot = MagicMock()
        mock_zot.item.return_value = _make_raw_item(collections=["COL1"])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = add_to_collection("ABC123", "COL1")
        assert result["already_member"] is True
        mock_zot.update_item.assert_not_called()

    def test_remove_from_collection(self):
        mock_zot = MagicMock()
        mock_zot.item.return_value = _make_raw_item(collections=["COL1", "COL2"])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = remove_from_collection("ABC123", "COL1")
        assert result == {"ok": True, "was_member": True, "collections": ["COL2"]}

    def test_remove_from_collection_not_member(self):
        mock_zot = MagicMock()
        mock_zot.item.return_value = _make_raw_item(collections=[])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = remove_from_collection("ABC123", "COL1")
        assert result["was_member"] is False
        mock_zot.update_item.assert_not_called()


class TestTags:
    def test_add_tag(self):
        mock_zot = MagicMock()
        mock_zot.item.return_value = _make_raw_item(tags=[{"tag": "old"}])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = add_tag("ABC123", "new")
        assert result["already_tagged"] is False
        updated = mock_zot.update_item.call_args.args[0]
        assert {"tag": "new"} in updated["data"]["tags"]

    def test_add_tag_idempotent(self):
        mock_zot = MagicMock()
        mock_zot.item.return_value = _make_raw_item(tags=[{"tag": "dup"}])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = add_tag("ABC123", "dup")
        assert result["already_tagged"] is True
        mock_zot.update_item.assert_not_called()

    def test_remove_tag(self):
        mock_zot = MagicMock()
        mock_zot.item.return_value = _make_raw_item(tags=[{"tag": "gone"}, {"tag": "keep"}])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = remove_tag("ABC123", "gone")
        assert result["was_tagged"] is True
        updated = mock_zot.update_item.call_args.args[0]
        assert updated["data"]["tags"] == [{"tag": "keep"}]

    def test_remove_tag_absent(self):
        mock_zot = MagicMock()
        mock_zot.item.return_value = _make_raw_item(tags=[])
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = remove_tag("ABC123", "missing")
        assert result["was_tagged"] is False
        mock_zot.update_item.assert_not_called()


# ── Unit: update_paper / attach_pdf / create_collection ───────────────────────

class TestUpdatePaper:
    def test_maps_friendly_names_to_zotero_fields(self):
        mock_zot = MagicMock()
        mock_zot.item.return_value = _make_raw_item()
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            update_paper("ABC123", year="2024", journal="New Journal", abstract="A.")
        updated = mock_zot.update_item.call_args.args[0]["data"]
        assert updated["date"] == "2024"
        assert updated["publicationTitle"] == "New Journal"
        assert updated["abstractNote"] == "A."

    def test_unknown_fields_ignored(self):
        mock_zot = MagicMock()
        mock_zot.item.return_value = _make_raw_item()
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            update_paper("ABC123", bogus="x")
        updated = mock_zot.update_item.call_args.args[0]["data"]
        assert "bogus" not in updated


class TestAttachPdf:
    def test_missing_file_returns_error_without_api_call(self):
        with patch("research_library.zotero_client.get_client") as mock_get:
            result = attach_pdf("ABC123", "C:/nonexistent/file.pdf")
        assert "error" in result
        mock_get.assert_not_called()

    @staticmethod
    def _autospec_zot(upload_result=None):
        """Bind against pyzotero's real signatures: a plain MagicMock accepts any
        kwargs, so a wrong keyword name passes tests and fails only live."""
        from unittest.mock import create_autospec
        from pyzotero.zotero import Zotero

        mock_zot = create_autospec(Zotero, instance=True)
        mock_zot.item_template.return_value = {"itemType": "attachment",
                                               "linkMode": "imported_file"}
        mock_zot.upload_attachments.return_value = (
            upload_result or {"success": [{"key": "ATT1"}], "failure": [], "unchanged": []}
        )
        return mock_zot

    def test_filename_sent_to_zotero_has_no_directory_path(self, tmp_path):
        """Regression: Zotero rejects a stored-file filename containing a
        directory path with 400, so the path must be split into basedir + name."""
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        mock_zot = self._autospec_zot()
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = attach_pdf("ABC123", str(pdf))

        payload, = mock_zot.upload_attachments.call_args.args
        assert payload[0]["filename"] == "paper.pdf"
        assert os.sep not in payload[0]["filename"]
        assert mock_zot.upload_attachments.call_args.kwargs["parentid"] == "ABC123"
        assert mock_zot.upload_attachments.call_args.kwargs["basedir"] == str(tmp_path)
        assert result["ok"] is True

    def test_upload_failure_is_reported_as_error(self, tmp_path):
        """Regression: pyzotero reports rejections in a 'failure' list rather
        than raising, so returning its dict verbatim reads as success."""
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        mock_zot = self._autospec_zot(
            {"success": [], "failure": [{"title": "paper.pdf"}], "unchanged": []}
        )
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = attach_pdf("ABC123", str(pdf))
        assert "error" in result
        assert result.get("ok") is not True

    def test_already_present_file_reported_as_unchanged(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        mock_zot = self._autospec_zot(
            {"success": [], "failure": [], "unchanged": [{"key": "ATT1"}]}
        )
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = attach_pdf("ABC123", str(pdf))
        assert result["ok"] is True
        assert result["unchanged"] is True


class TestCreateCollection:
    def test_creates_top_level_collection(self):
        mock_zot = MagicMock()
        mock_zot.create_collections.return_value = {"success": {"0": "NEWKEY"}, "failed": {}}
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = create_collection("Macro")
        assert result == {"key": "NEWKEY", "name": "Macro", "parent": None}
        payload = mock_zot.create_collections.call_args.args[0]
        assert "parentCollection" not in payload[0]

    def test_nested_collection_sets_parent(self):
        mock_zot = MagicMock()
        mock_zot.create_collections.return_value = {"success": {"0": "NEWKEY"}, "failed": {}}
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = create_collection("Child", parent_key="PARENT1")
        payload = mock_zot.create_collections.call_args.args[0]
        assert payload[0]["parentCollection"] == "PARENT1"
        assert result["parent"] == "PARENT1"

    def test_failed_creation_returns_error(self):
        mock_zot = MagicMock()
        mock_zot.create_collections.return_value = {"failed": {"0": {"message": "bad"}}}
        with patch("research_library.zotero_client.get_client", return_value=mock_zot):
            result = create_collection("Macro")
        assert "error" in result
