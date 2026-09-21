"""Batch sizing and safety caps for the embedding generator.

These cover the failure that exhausted the GPU during an index rebuild: the
embedder was constructed with code defaults (batch 32) while
``EMBEDDING_BATCH_SIZE`` said 8, so long elements were padded into batches that
needed well over 6 GiB of activations.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from src.config.settings import ModelSettings
from src.retrieval.embedding_generator import (
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_TOKEN_BUDGET,
    EmbeddingGenerator,
)


class _RecordingBackend:
    """Fake backend that reports the length of each text it was given."""

    def __init__(self, max_seq_length: int | None = None) -> None:
        self.batches: list[list[str]] = []
        self.max_seq_length = max_seq_length

    def encode(self, texts, **kwargs):  # noqa: ANN001, ANN003 - protocol shape
        self.batches.append(list(texts))
        return np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float32)

    @property
    def estimated_tokens(self) -> list[int]:
        """Token estimate per batch, mirroring the generator's fallback rule."""

        return [sum(max(1, len(text) // 4) for text in batch) for batch in self.batches]


def _generator(backend, **overrides) -> EmbeddingGenerator:
    options = {"batch_size": 4, "max_seq_length": 64, "token_budget": 12}
    options.update(overrides)
    return EmbeddingGenerator(backend=backend, **options)


def test_batches_pack_similar_lengths_under_the_token_budget() -> None:
    backend = _RecordingBackend()
    generator = _generator(backend)
    texts = ["a" * 4, "b" * 44, "c" * 8, "d" * 40, "e" * 100]

    vectors = generator.embed_texts(texts)

    # Ascending length, and 10 + 11 estimated tokens exceeds the 12-token budget.
    assert backend.batches == [["a" * 4, "c" * 8], ["d" * 40], ["b" * 44], ["e" * 100]]
    # A lone text longer than the budget gets a batch of its own; nothing else may exceed it.
    assert all(
        tokens <= generator.token_budget or len(batch) == 1
        for batch, tokens in zip(backend.batches, backend.estimated_tokens)
    )
    assert vectors.shape == (5, 2)


def test_vectors_keep_the_caller_order_after_length_sorted_batching() -> None:
    backend = _RecordingBackend()
    generator = _generator(backend, batch_size=2, token_budget=1000)
    texts = ["z" * 4, "y" * 40, "x" * 12, "w" * 80]

    vectors = generator.embed_texts(texts)

    assert backend.batches == [["z" * 4, "x" * 12], ["y" * 40, "w" * 80]]
    expected = [length / math.sqrt(length**2 + 1) for length in (4, 40, 12, 80)]
    assert all(
        math.isclose(float(row[0]), value, rel_tol=1e-6) for row, value in zip(vectors, expected)
    )


def test_sequence_length_cap_is_applied_to_the_backend() -> None:
    capped = _RecordingBackend(max_seq_length=32768)
    _generator(capped).embed_texts(["a" * 8])
    assert capped.max_seq_length == 64

    explicit = _RecordingBackend(max_seq_length=32768)
    _generator(explicit, max_seq_length=256).embed_texts(["a" * 8])
    assert explicit.max_seq_length == 256


def test_long_element_gets_a_batch_of_its_own() -> None:
    backend = _RecordingBackend()
    generator = _generator(backend, batch_size=8, token_budget=12)
    texts = ["a" * 4, "b" * 4, "c" * 4, "d" * 400]

    generator.embed_texts(texts)

    assert backend.batches == [["a" * 4, "b" * 4, "c" * 4], ["d" * 400]]


def test_from_settings_carries_environment_values() -> None:
    saved = {name: os.environ.get(name) for name in
             ("EMBEDDING_MODEL", "EMBEDDING_BATCH_SIZE", "EMBEDDING_MAX_SEQ_LENGTH", "EMBEDDING_TOKEN_BUDGET")}
    os.environ["EMBEDDING_MODEL"] = "test/model"
    os.environ["EMBEDDING_BATCH_SIZE"] = "8"
    os.environ["EMBEDDING_MAX_SEQ_LENGTH"] = "1024"
    os.environ["EMBEDDING_TOKEN_BUDGET"] = "4096"
    try:
        generator = EmbeddingGenerator.from_settings(ModelSettings.from_env())
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert generator.model_name == "test/model"
    assert generator.batch_size == 8
    assert generator.max_seq_length == 1024
    assert generator.token_budget == 4096


def test_defaults_stay_bounded() -> None:
    generator = EmbeddingGenerator()

    assert generator.batch_size > 1
    assert generator.max_seq_length == DEFAULT_EMBEDDING_MAX_SEQ_LENGTH
    assert generator.token_budget == DEFAULT_EMBEDDING_TOKEN_BUDGET
    assert generator.max_seq_length < 32768
    assert generator.runtime_config() == {
        "batch_size": generator.batch_size,
        "max_seq_length": generator.max_seq_length,
        "token_budget": generator.token_budget,
    }


def test_no_source_call_site_bypasses_settings() -> None:
    """A bare ``EmbeddingGenerator()`` silently ignores every embedding env var."""

    src_root = Path(__file__).resolve().parents[1] / "src"
    offenders = [
        str(path.relative_to(src_root))
        for path in sorted(src_root.rglob("*.py"))
        if "EmbeddingGenerator(" in path.read_text(encoding="utf-8")
        and "from_settings" not in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
