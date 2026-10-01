"""D18 step 1: unchanged documents keep the ``chunks.jsonl`` they already have.

The reuse decision must be provable, not trusted: these tests pin it (and every
fallback) with a fake embedder and a spy around the chunker, so no model is loaded
and the assertions are about *behaviour*, not timing.

Note on ids (``ordered_chunker._stable_chunk_id``): the hash material includes
``revision``, so chunk ids are content-addressed **within one revision**.  A
re-parsed document gets brand-new ids -- which is exactly why the decision turns on
``revision`` and why a reused id can never pair new text with an old vector.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.retrieval.index_builder as index_builder
from src.retrieval.index_builder import IndexBuilder
from src.storage.document_registry import DocumentRegistry

_REAL_CHUNKER = index_builder.chunk_ordered_document


class _FakeEmbedder:
    """Records every embed call; deterministic 8-dim vectors."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        rows = list(texts)
        self.calls.append(rows)
        if not rows:
            return np.empty((0, 8), dtype="float32")
        return np.array(
            [[float((len(text) % 7) or 3), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for text in rows],
            dtype="float32",
        )

    def runtime_config(self) -> dict[str, int]:
        return {"batch_size": 1, "max_seq_length": 512, "token_budget": 512}


class _ChunkerSpy:
    """Counts which documents were actually re-chunked."""

    def __init__(self) -> None:
        self.chunked: list[str] = []

    def __call__(self, ordered, *, filename=None):
        self.chunked.append(str(ordered.get("document_id")))
        return _REAL_CHUNKER(ordered, filename=filename)


@pytest.fixture()
def env(tmp_path: Path, monkeypatch):
    registry = DocumentRegistry(data_dir=tmp_path / "data")
    embedder = _FakeEmbedder()
    spy = _ChunkerSpy()
    monkeypatch.setattr(index_builder, "chunk_ordered_document", spy)
    builder = IndexBuilder(registry, embedder=embedder, index_dir=tmp_path / "chroma")
    return SimpleNamespace(builder=builder, registry=registry, embedder=embedder, spy=spy, tmp=tmp_path)


def _ordered_payload(document_id: str, revision: str, body: str) -> dict:
    sequence = []
    for index, paragraph in enumerate(body.split("\n\n")):
        sequence.append(
            {
                "element_id": f"e{index + 1}",
                "ordinal": index,
                "type": "paragraph",
                "role": "body",
                "index_policy": "embed",
                "page": 1,
                "heading_path_norm": ["1 Retrieval"],
                "text": paragraph,
                "search_text": paragraph,
                "structure": {},
                "atomic": False,
                "parse_status": "parsed",
                "payload": {},
            }
        )
    return {
        "schema": "ordered_document_v1",
        "document_id": document_id,
        "revision": revision,
        "page_kind": "page",
        "parser": {"name": "docling", "version": "2.5.0", "ocr": False},
        "sequence": sequence,
    }


def _write_parsed(env, document_id: str, revision: str, body: str) -> Path:
    out_dir = env.registry.data_dir / "parsed" / document_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ordered.json").write_text(
        json.dumps(_ordered_payload(document_id, revision, body)), encoding="utf-8"
    )
    (out_dir / "quality_report.json").write_text(
        json.dumps({"stats": {"n_elements": 2}, "quality": {}}), encoding="utf-8"
    )
    return out_dir


def _add_document(env, name: str, body: str = "Alpha paragraph about retrieval.\n\nBeta paragraph about indexing."):
    record = env.registry.register_upload(name, "application/pdf", io.BytesIO(b"%PDF-1.4 test payload"))
    _write_parsed(env, record.document_id, record.revision, body)
    return record


def _bump_registry_revision(env, document_id: str) -> None:
    """Move a document's revision the way a re-parse would."""

    payload = json.loads(env.registry.registry_path.read_text(encoding="utf-8"))
    for entry in payload["documents"]:
        if entry["document_id"] == document_id:
            entry["revision"] = "revision-after-reparse"
    env.registry.registry_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _stored_chunk_ids(out_dir: Path) -> set[str]:
    return {
        json.loads(line)["chunk_id"]
        for line in (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_first_build_chunks_every_document(env) -> None:
    first = _add_document(env, "a.pdf")
    second = _add_document(env, "b.pdf")

    result = env.builder.build(skip_parse=True)

    assert sorted(env.spy.chunked) == sorted([first.document_id, second.document_id])
    assert result["reused_documents"] == []
    assert result["chunked_documents"] == 2
    assert result["chunk_count"] == len(env.embedder.calls[-1])  # step 1 keeps embedding everything


def test_second_build_reuses_unchanged_documents(env) -> None:
    """The whole point of step 1: same corpus, no chunker run, identical index."""

    first = _add_document(env, "a.pdf")
    second = _add_document(env, "b.pdf")
    first_pass = env.builder.build(skip_parse=True)
    stored = {
        doc.document_id: _stored_chunk_ids(env.registry.data_dir / "parsed" / doc.document_id)
        for doc in (first, second)
    }
    mtimes = {
        doc.document_id: (env.registry.data_dir / "parsed" / doc.document_id / "chunks.jsonl").stat().st_mtime_ns
        for doc in (first, second)
    }
    chunked_before = list(env.spy.chunked)

    result = env.builder.build(skip_parse=True)

    assert env.spy.chunked == chunked_before  # the chunker was not called again
    assert sorted(result["reused_documents"]) == sorted([first.document_id, second.document_id])
    assert result["reused_chunks"] == first_pass["chunk_count"]
    assert result["chunked_documents"] == 0
    assert result["chunk_count"] == first_pass["chunk_count"]
    for document_id, ids in stored.items():
        assert _stored_chunk_ids(env.registry.data_dir / "parsed" / document_id) == ids
        # not rewritten at all: the artifact is reused, not regenerated
        assert (
            env.registry.data_dir / "parsed" / document_id / "chunks.jsonl"
        ).stat().st_mtime_ns == mtimes[document_id]
    # step 1 does not touch the embedding path: every chunk is still embedded
    assert len(env.embedder.calls[-1]) == first_pass["chunk_count"]


def test_revision_change_rechunks_only_that_document(env) -> None:
    changed = _add_document(env, "a.pdf")
    untouched = _add_document(env, "b.pdf")
    env.builder.build(skip_parse=True)

    _bump_registry_revision(env, changed.document_id)
    _write_parsed(env, changed.document_id, "revision-after-reparse", "Alpha paragraph about retrieval.")
    env.spy.chunked.clear()

    result = env.builder.build(skip_parse=True)

    assert env.spy.chunked == [changed.document_id]
    assert result["reused_documents"] == [untouched.document_id]
    assert result["chunked_documents"] == 1


def test_fingerprint_change_forces_a_full_rechunk(env, monkeypatch) -> None:
    """A chunker change invalidates every stored artifact, by design."""

    first = _add_document(env, "a.pdf")
    second = _add_document(env, "b.pdf")
    env.builder.build(skip_parse=True)
    env.spy.chunked.clear()

    monkeypatch.setattr(index_builder, "CHUNKER_VERSION", "ordered-aware-2")
    result = env.builder.build(skip_parse=True)

    assert sorted(env.spy.chunked) == sorted([first.document_id, second.document_id])
    assert result["reused_documents"] == []


def test_missing_or_corrupt_artifacts_fall_back_to_rechunking(env) -> None:
    missing = _add_document(env, "a.pdf")
    corrupt = _add_document(env, "b.pdf")
    fine = _add_document(env, "c.pdf")
    env.builder.build(skip_parse=True)
    env.spy.chunked.clear()

    (env.registry.data_dir / "parsed" / missing.document_id / "chunks.jsonl").unlink()
    (env.registry.data_dir / "parsed" / corrupt.document_id / "chunks.jsonl").write_text(
        "{not json\n", encoding="utf-8"
    )

    result = env.builder.build(skip_parse=True)

    assert sorted(env.spy.chunked) == sorted([missing.document_id, corrupt.document_id])
    assert result["reused_documents"] == [fine.document_id]
    # the corrupted artifact is repaired by the rebuild
    assert _stored_chunk_ids(env.registry.data_dir / "parsed" / corrupt.document_id)


def test_chunk_count_mismatch_is_not_reused(env) -> None:
    """A truncated or half-written artifact must not be trusted."""

    tampered = _add_document(env, "a.pdf")
    other = _add_document(env, "b.pdf")
    env.builder.build(skip_parse=True)
    env.spy.chunked.clear()

    path = env.registry.data_dir / "parsed" / tampered.document_id / "chunks.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + json.dumps({"chunk_id": "chk_extra"}) + "\n", encoding="utf-8")

    result = env.builder.build(skip_parse=True)

    assert env.spy.chunked == [tampered.document_id]
    assert result["reused_documents"] == [other.document_id]


# ---------------------------------------------------------------- D18 step 2

def _published_vectors(env, chunk_ids):
    from src.retrieval.chroma_store import ChromaVectorStore

    store, _manifest = ChromaVectorStore.load(env.builder.index_dir)
    try:
        return store.vectors_for(list(chunk_ids))
    finally:
        store.close()


def test_second_build_reuses_vectors_without_calling_the_embedder(env) -> None:
    """Step 2: a rebuild of an unchanged corpus costs no embedding at all."""

    first = _add_document(env, "a.pdf")
    second = _add_document(env, "b.pdf")
    first_pass = env.builder.build(skip_parse=True)
    ids = [
        json.loads(line)["chunk_id"]
        for line in (
            env.registry.data_dir / "parsed" / first.document_id / "chunks.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] + [
        json.loads(line)["chunk_id"]
        for line in (
            env.registry.data_dir / "parsed" / second.document_id / "chunks.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors_before = _published_vectors(env, ids)
    embeds_before = len(env.embedder.calls)

    result = env.builder.build(skip_parse=True)

    assert len(env.embedder.calls) == embeds_before  # the embedder was never called
    assert result["reused_vectors"] == first_pass["chunk_count"]
    assert result["embedded_chunks"] == 0
    assert result["chunk_count"] == first_pass["chunk_count"]
    vectors_after = _published_vectors(env, ids)
    for chunk_id in ids:
        assert vectors_after[chunk_id] == vectors_before[chunk_id]


def test_only_the_changed_document_is_embedded_again(env) -> None:
    changed = _add_document(env, "a.pdf")
    untouched = _add_document(env, "b.pdf")
    env.builder.build(skip_parse=True)

    _bump_registry_revision(env, changed.document_id)
    _write_parsed(env, changed.document_id, "revision-after-reparse", "A rewritten paragraph about retrieval.")
    env.embedder.calls.clear()

    result = env.builder.build(skip_parse=True)

    changed_chunks = len(
        [
            line
            for line in (env.registry.data_dir / "parsed" / changed.document_id / "chunks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    )
    assert result["embedded_chunks"] == changed_chunks
    assert result["reused_vectors"] == result["chunk_count"] - changed_chunks
    assert len(env.embedder.calls) == 1
    assert len(env.embedder.calls[0]) == changed_chunks
    assert result["reused_documents"] == [untouched.document_id]


def test_missing_vectors_are_embedded_instead_of_guessed(env, monkeypatch) -> None:
    """A partial lookup must not shift rows: the gaps get a real embed call."""

    first = _add_document(env, "a.pdf")
    second = _add_document(env, "b.pdf")
    env.builder.build(skip_parse=True)

    from src.retrieval.chroma_store import ChromaVectorStore

    original = ChromaVectorStore.vectors_for
    dropped: list[str] = []

    def partial(self, chunk_ids):
        vectors = original(self, chunk_ids)
        if vectors and not dropped:
            first_id = next(iter(vectors))
            dropped.append(first_id)
            vectors.pop(first_id)
        return vectors

    monkeypatch.setattr(ChromaVectorStore, "vectors_for", partial)
    env.embedder.calls.clear()

    result = env.builder.build(skip_parse=True)

    assert dropped  # the fixture actually dropped something
    assert result["embedded_chunks"] == 1
    assert result["reused_vectors"] == result["chunk_count"] - 1
    assert [text for call in env.embedder.calls for text in call]  # something was embedded


def test_unreadable_previous_index_falls_back_to_full_embedding(env, monkeypatch) -> None:
    _add_document(env, "a.pdf")
    _add_document(env, "b.pdf")
    first_pass = env.builder.build(skip_parse=True)

    monkeypatch.setattr(IndexBuilder, "_open_prev_store", lambda self: None)
    env.embedder.calls.clear()

    result = env.builder.build(skip_parse=True)

    assert result["reused_vectors"] == 0
    assert result["embedded_chunks"] == first_pass["chunk_count"]
    assert len(env.embedder.calls[0]) == first_pass["chunk_count"]
