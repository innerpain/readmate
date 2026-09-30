"""D61: multi-format upload validation, storage naming, and format swap."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.ingestion.docling_converter import FORMAT_BY_EXTENSION
from src.storage.document_registry import (
    ACCEPTED_EXTENSIONS,
    DocumentRegistry,
    InvalidUploadError,
)

ZIP_MAGIC = b"PK\x03\x04"


def registry(tmp_path: Path) -> DocumentRegistry:
    return DocumentRegistry(data_dir=tmp_path / "data")


def upload(reg: DocumentRegistry, filename: str, payload: bytes, content_type: str | None = None):
    return reg.register_upload(filename, content_type, io.BytesIO(payload))


class TestWhitelist:
    def test_accepted_formats_are_the_documented_set(self) -> None:
        assert set(ACCEPTED_EXTENSIONS) == {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".md",
            ".html",
            ".htm",
            ".csv",
        }

    def test_every_accepted_format_has_a_docling_backend(self) -> None:
        # The registry whitelist and the converter map must not drift: an
        # accepted upload with no backend would register fine and then fail
        # parsing for a reason the user cannot act on.
        assert set(ACCEPTED_EXTENSIONS) <= set(FORMAT_BY_EXTENSION)

    def test_unsupported_extension_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidUploadError) as err:
            upload(registry(tmp_path), "legacy.doc", b"\xd0\xcf\x11\xe0legacy")
        assert "unsupported_file_type" in str(err.value)

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        reg = registry(tmp_path)
        doc = upload(reg, "Notes.DOCX", ZIP_MAGIC + b"rest", "application/octet-stream")
        assert doc.file_type == ".docx"

    def test_octet_stream_is_universal(self, tmp_path: Path) -> None:
        doc = upload(registry(tmp_path), "a.csv", b"x,y\n1,2\n", "application/octet-stream")
        assert doc.file_type == ".csv"

    def test_wrong_mime_for_extension_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidUploadError) as err:
            upload(registry(tmp_path), "a.docx", ZIP_MAGIC + b"x", "application/pdf")
        assert "unsupported_content_type" in str(err.value)

    def test_windows_csv_mime_accepted(self, tmp_path: Path) -> None:
        # Excel on Windows reports CSV as application/vnd.ms-excel.
        doc = upload(registry(tmp_path), "a.csv", b"x,y\n", "application/vnd.ms-excel")
        assert doc.file_type == ".csv"


class TestMagic:
    def test_pdf_magic_enforced(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidUploadError) as err:
            upload(registry(tmp_path), "fake.pdf", b"not a pdf at all")
        assert "invalid_file_header" in str(err.value)

    def test_ooxml_zip_magic_enforced(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidUploadError) as err:
            upload(registry(tmp_path), "fake.docx", b"%PDF-1.7 actually a pdf")
        assert "invalid_file_header" in str(err.value)

    def test_text_formats_have_no_magic_requirement(self, tmp_path: Path) -> None:
        # A .md may start with anything; unreadable content fails honestly at
        # parse time instead of being guessed at upload time.
        doc = upload(registry(tmp_path), "a.md", "标题\n\n正文".encode("utf-8"))
        assert doc.file_type == ".md"

    def test_empty_file_still_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidUploadError) as err:
            upload(registry(tmp_path), "a.md", b"")
        assert "empty_file" in str(err.value)


class TestStorage:
    def test_stored_name_keeps_extension(self, tmp_path: Path) -> None:
        reg = registry(tmp_path)
        doc = upload(reg, "report.PPTX", ZIP_MAGIC + b"slides", "application/octet-stream")
        assert doc.stored_filename == f"{doc.document_id}.pptx"
        assert (reg.uploads_dir / doc.stored_filename).is_file()

    def test_file_type_survives_reload(self, tmp_path: Path) -> None:
        reg = registry(tmp_path)
        doc = upload(reg, "a.xlsx", ZIP_MAGIC + b"sheet")
        again = DocumentRegistry(data_dir=tmp_path / "data").get_document(doc.document_id)
        assert again.file_type == ".xlsx"

    def test_legacy_record_without_file_type_defaults_to_pdf(self, tmp_path: Path) -> None:
        # Registry JSON written before D61 has no file_type key; every such
        # record was a PDF, so the default must not require a migration.
        reg = registry(tmp_path)
        reg.registry_path.write_text(
            '{"version": 1, "documents": [{"document_id": "d1",'
            ' "original_filename": "old.pdf", "stored_filename": "d1.pdf",'
            ' "file_size_bytes": 10}]}',
            encoding="utf-8",
        )
        assert reg.get_document("d1").file_type == ".pdf"


class TestReplace:
    def test_replace_same_format_keeps_one_file(self, tmp_path: Path) -> None:
        reg = registry(tmp_path)
        doc = upload(reg, "a.pdf", b"%PDF-1.4 one")
        replaced = reg.replace_upload(doc.document_id, "a.pdf", "application/pdf", io.BytesIO(b"%PDF-1.4 two"))
        assert replaced.stored_filename == doc.stored_filename
        assert (reg.uploads_dir / doc.stored_filename).read_bytes() == b"%PDF-1.4 two"

    def test_replace_swapping_format_removes_the_old_file(self, tmp_path: Path) -> None:
        reg = registry(tmp_path)
        doc = upload(reg, "a.pdf", b"%PDF-1.4 original")
        old_path = reg.uploads_dir / doc.stored_filename
        replaced = reg.replace_upload(
            doc.document_id, "a.docx", "application/octet-stream", io.BytesIO(ZIP_MAGIC + b"docx")
        )
        assert replaced.file_type == ".docx"
        assert replaced.stored_filename == f"{doc.document_id}.docx"
        assert not old_path.exists(), "the old format's file must not linger"
        assert (reg.uploads_dir / replaced.stored_filename).is_file()

    def test_replace_rejects_unsupported_extension(self, tmp_path: Path) -> None:
        reg = registry(tmp_path)
        doc = upload(reg, "a.pdf", b"%PDF-1.4 x")
        with pytest.raises(InvalidUploadError):
            reg.replace_upload(doc.document_id, "a.rtf", "text/plain", io.BytesIO(b"{\\rtf1}"))
        # The original must survive a rejected replacement.
        assert (reg.uploads_dir / doc.stored_filename).read_bytes() == b"%PDF-1.4 x"
