"""FE-2: document alias / enable metadata and the retrieval exclusion it drives.

The point of "停用" in a RAG tool is that the file stays yours but stops feeding
answers -- so the tests below check both halves: the metadata round-trips, and a
disabled document disappears from every scope the retriever can use (single
collection, multi-collection union, and the prompt-snapshot digest).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from src.adapter.contracts import COLLECTION_EMPTY, AdapterError
from src.agent.collections import CollectionService
from src.storage.agent_db import AgentDB
from src.storage.document_registry import DocumentNotFoundError, DocumentRegistry

_PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


def _registry(tmp_path: Path) -> DocumentRegistry:
    return DocumentRegistry(data_dir=tmp_path / "data")


def _upload(registry: DocumentRegistry, name: str = "attention.pdf"):
    return registry.register_upload(name, "application/pdf", io.BytesIO(_PDF))


def _collections(db, registry) -> CollectionService:
    return CollectionService(db=db, registry=registry)


def test_new_documents_default_to_enabled_with_no_alias(tmp_path):
    registry = _registry(tmp_path)
    document = _upload(registry)
    assert document.alias == ""
    assert document.enabled is True
    reloaded = registry.get_document(document.document_id)
    assert (reloaded.alias, reloaded.enabled) == ("", True)


def test_alias_and_enabled_round_trip(tmp_path):
    registry = _registry(tmp_path)
    document = _upload(registry)

    renamed = registry.update_document_meta(document.document_id, alias="  注意力论文  ")
    assert renamed.alias == "注意力论文"  # trimmed
    assert renamed.enabled is True

    disabled = registry.update_document_meta(document.document_id, enabled=False)
    assert disabled.enabled is False
    assert disabled.alias == "注意力论文"  # a partial patch keeps the other field

    # and it survives a reload from disk
    reloaded = registry.get_document(document.document_id)
    assert (reloaded.alias, reloaded.enabled) == ("注意力论文", False)


def test_legacy_records_without_the_new_fields_still_load(tmp_path):
    registry = _registry(tmp_path)
    document = _upload(registry)
    path = registry.registry_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["documents"]:
        row.pop("alias", None)
        row.pop("enabled", None)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    reloaded = registry.get_document(document.document_id)
    assert reloaded.alias == ""
    assert reloaded.enabled is True


def test_unknown_document_raises(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(DocumentNotFoundError):
        registry.update_document_meta("missing", enabled=False)


def test_disabled_document_is_excluded_from_the_collection_scope(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    registry = _registry(tmp_path)
    first = _upload(registry, "a.pdf")
    second = _upload(registry, "b.pdf")
    collection_id = db.upsert_collection("papers")
    db.add_documents(collection_id, [first.document_id, second.document_id])
    service = _collections(db, registry)

    assert set(service.document_ids(collection_id)) == {first.document_id, second.document_id}

    registry.update_document_meta(second.document_id, enabled=False)
    assert service.document_ids(collection_id) == [first.document_id]


def test_multi_collection_union_also_drops_disabled_documents(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    registry = _registry(tmp_path)
    first = _upload(registry, "a.pdf")
    second = _upload(registry, "b.pdf")
    third = _upload(registry, "c.pdf")
    left = db.upsert_collection("left")
    right = db.upsert_collection("right")
    db.add_documents(left, [first.document_id, second.document_id])
    db.add_documents(right, [second.document_id, third.document_id])
    service = _collections(db, registry)

    assert service.document_ids_multi([left, right]) == {first.document_id, second.document_id, third.document_id}

    registry.update_document_meta(second.document_id, enabled=False)
    assert service.document_ids_multi([left, right]) == {first.document_id, third.document_id}


def test_disabling_every_document_reports_collection_empty(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    registry = _registry(tmp_path)
    document = _upload(registry)
    collection_id = db.upsert_collection("only")
    db.add_documents(collection_id, [document.document_id])
    service = _collections(db, registry)

    registry.update_document_meta(document.document_id, enabled=False)
    with pytest.raises(AdapterError) as single:
        service.document_ids(collection_id)
    assert single.value.code == COLLECTION_EMPTY
    assert "disabled" in single.value.message

    with pytest.raises(AdapterError) as multi:
        service.document_ids_multi([collection_id])
    assert multi.value.code == COLLECTION_EMPTY
    assert "disabled" in multi.value.message


def test_digest_declares_only_enabled_material(tmp_path):
    db = AgentDB(tmp_path / "app.db")
    registry = _registry(tmp_path)
    first = _upload(registry, "keep.pdf")
    second = _upload(registry, "drop.pdf")
    collection_id = db.upsert_collection("papers")
    db.add_documents(collection_id, [first.document_id, second.document_id])
    service = _collections(db, registry)

    assert "keep.pdf" in service.digest([collection_id])
    assert "drop.pdf" in service.digest([collection_id])

    registry.update_document_meta(second.document_id, enabled=False)
    digest = service.digest([collection_id])
    assert "keep.pdf" in digest
    assert "drop.pdf" not in digest


def test_a_registry_failure_does_not_block_retrieval(tmp_path):
    """A registry hiccup must not turn into "no documents in scope"."""

    db = AgentDB(tmp_path / "app.db")
    registry = _registry(tmp_path)
    document = _upload(registry)
    collection_id = db.upsert_collection("papers")
    db.add_documents(collection_id, [document.document_id])

    class _Broken:
        def list_documents(self):
            raise RuntimeError("registry down")

    service = CollectionService(db=db, registry=_Broken())
    assert service.disabled_document_ids() == set()
    assert service.document_ids(collection_id) == [document.document_id]
