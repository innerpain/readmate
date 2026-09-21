"""D14 (2026-09-20): Chroma publishes one version, atomically, and reclaims the
previous one.

Root cause: ``publish()`` built straight into ``versions/<id>/`` and only
``rmtree``-ed *its own* half-written directory on failure.  A rebuild therefore
left the old version behind forever -- measured: 11 orphans, ~34 MB -- and a
crash mid-build could leave a directory that looked publishable.

The directory machinery is stdlib-only, so it is tested here without chromadb.
The end-to-end publish test is skipped when chromadb is not installed (it is a
runtime dependency, not a test-time one).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval.chroma_store import _gc_versions


def _make_version(root: Path, version_id: str) -> Path:
    version = root / "versions" / version_id
    version.mkdir(parents=True)
    (version / "manifest.json").write_text(json.dumps({"version_id": version_id}), encoding="utf-8")
    return version


def test_gc_reclaims_every_version_but_the_published_one(tmp_path: Path):
    """D14: the product has exactly one version -- the one ``current.json`` names.
    Everything else is a leftover from a rebuild and goes."""

    current = _make_version(tmp_path, "v-new")
    old_a = _make_version(tmp_path, "v-old-a")
    old_b = _make_version(tmp_path, "v-old-b")

    removed = _gc_versions(tmp_path, keep="v-new")

    assert removed == 2
    assert current.is_dir() and (current / "manifest.json").is_file()
    assert not old_a.exists() and not old_b.exists()


def test_gc_never_touches_the_current_version_or_a_staging_dir(tmp_path: Path):
    """Two invariants: ``current.json`` must keep pointing at something, and a
    concurrent build's ``.building-*`` staging directory must not be stolen."""

    _make_version(tmp_path, "v-new")
    staging = tmp_path / "versions" / ".building-abc12345"
    staging.mkdir(parents=True)
    pointer = tmp_path / "current.json"
    pointer.write_text(json.dumps({"version_id": "v-new"}), encoding="utf-8")

    _gc_versions(tmp_path, keep="v-new")

    assert (tmp_path / "versions" / "v-new").is_dir()
    assert staging.is_dir(), "a staging dir belongs to a build in flight"
    assert json.loads(pointer.read_text(encoding="utf-8"))["version_id"] == "v-new"


def test_gc_on_a_missing_versions_dir_is_a_no_op(tmp_path: Path):
    """A version that cannot be removed must never fail a publish."""

    assert _gc_versions(tmp_path / "does-not-exist", keep="v1") == 0
    (tmp_path / "versions").mkdir()
    assert _gc_versions(tmp_path, keep="v1") == 0


# ------------------------------------------------------------------ end to end
chromadb = pytest.importorskip("chromadb", reason="chromadb is a runtime dependency, absent on this host")


class _Chunk:
    def __init__(self, chunk_id: str, text: str) -> None:
        self.page_content = text
        self.metadata = {"chunk_id": chunk_id, "document_id": "d1", "filename": "a.pdf", "page": 1}


def _publish(root: Path, version_text: str):
    import numpy as np

    from src.retrieval.chroma_store import ChromaVectorStore

    chunks = [_Chunk("c1", version_text)]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    return ChromaVectorStore.publish(
        root,
        chunks,
        embeddings,
        embedding_model="fake",
        chunking_config={},
        parser_config={},
        quality_config={},
        documents=["d1"],
    )


def test_publish_replaces_the_previous_version_and_keeps_the_pointer_valid(tmp_path: Path):
    first = _publish(tmp_path, "first build")
    assert (tmp_path / "versions" / first["version_id"]).is_dir()

    second = _publish(tmp_path, "second build")
    versions = sorted(entry.name for entry in (tmp_path / "versions").iterdir())

    # Exactly one version survives, and it is the one the pointer names.
    assert versions == [second["version_id"]]
    assert json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))["version_id"] == second["version_id"]
    # No staging directory was left behind either.
    assert not any(name.startswith(".") for name in versions)

    from src.retrieval.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(tmp_path)
    assert store.version_id == second["version_id"]
    assert store.manifest["chunk_count"] == 1
