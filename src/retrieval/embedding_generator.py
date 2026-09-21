"""Embedding adapters with model-consistency and vector-safety checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from src.config.settings import ModelSettings


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
# Measured on the RTX 4060 (bf16 Qwen3-Embedding-0.6B): activations cost about
# 0.135 MiB per token in a batch, on top of ~1.1 GiB for the weights.  A batch of
# 32 long elements therefore needs well over 6 GiB and died with a CUDA OOM, so
# both a per-sequence cap and a per-batch token budget bound the peak.
DEFAULT_EMBEDDING_MAX_SEQ_LENGTH = 2048
DEFAULT_EMBEDDING_TOKEN_BUDGET = 8192
DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QUERY_PREFIX_TEMPLATE = "Instruct: {instruction}\nQuery: {query}"


class EmbeddingBackend(Protocol):
    """Small protocol that lets tests replace the heavyweight local model."""

    def encode(self, texts: Sequence[str], **kwargs: object) -> np.ndarray: ...


class EmbeddingGenerator:
    """Generate normalized, finite vectors from one declared embedding model."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        max_seq_length: int = DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
        token_budget: int = DEFAULT_EMBEDDING_TOKEN_BUDGET,
        backend: EmbeddingBackend | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.token_budget = token_budget
        self._backend = backend
        self._dimension: int | None = None

    @classmethod
    def from_settings(cls, settings: ModelSettings | None = None) -> EmbeddingGenerator:
        """Build from application settings so environment values always apply.

        Required at every call site: constructing ``EmbeddingGenerator()``
        directly silently used the code defaults (batch 32) while
        ``EMBEDDING_BATCH_SIZE`` said 8, and those oversized batches exhausted
        the GPU during a rebuild.
        """

        if settings is None:
            from src.config.settings import get_model_settings

            settings = get_model_settings()
        return cls(
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_seq_length=settings.embedding_max_seq_length,
            token_budget=settings.embedding_token_budget,
        )

    @property
    def dimension(self) -> int | None:
        """Return the known vector dimension after the first embedding call."""
        return self._dimension

    def runtime_config(self) -> dict[str, int]:
        """Batching limits that decide peak GPU memory, for the index manifest."""

        return {
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "token_budget": self.token_budget,
        }

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode document-side texts (chunk contents) without any query prefix."""
        normalized_texts = self._validate_inputs(texts)
        return self._encode(normalized_texts)

    def embed_query(
        self,
        texts: Sequence[str],
        *,
        instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> np.ndarray:
        """Encode user queries with the optional Instruct/Query prefix.

        Qwen3-Embedding was trained with an instruction-prefixed query format;
        documents are always encoded without the prefix. An empty or blank
        instruction falls back to plain encoding so other models and fake test
        backends keep working unchanged.
        """
        normalized_texts = self._validate_inputs(texts)
        stripped_instruction = instruction.strip() if isinstance(instruction, str) else ""
        if not stripped_instruction:
            return self._encode(normalized_texts)
        prefixed_texts = [
            QUERY_PREFIX_TEMPLATE.format(instruction=stripped_instruction, query=text)
            for text in normalized_texts
        ]
        return self._encode(prefixed_texts)

    def _encode(self, normalized_texts: list[str]) -> np.ndarray:
        if not normalized_texts:
            return np.empty((0, self._dimension or 0), dtype=np.float32)

        backend = self._get_backend()
        lengths = [self._estimate_tokens(backend, text) for text in normalized_texts]
        vectors: np.ndarray | None = None
        for indices in self._batch_indices(lengths):
            batch = [normalized_texts[index] for index in indices]
            encoded = self._encode_batch(backend, batch)
            if vectors is None:
                vectors = np.empty((len(normalized_texts), encoded.shape[1]), dtype=np.float32)
            vectors[indices] = encoded

        if vectors is None:  # pragma: no cover - _batch_indices never yields nothing
            raise ValueError("embedding produced no vectors")
        self._validate_vectors(vectors, expected_count=len(normalized_texts))
        self._dimension = int(vectors.shape[1])
        return self._normalize(vectors)

    def _encode_batch(self, backend: EmbeddingBackend, batch: list[str]) -> np.ndarray:
        try:
            encoded = backend.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except TypeError:
            # Minimal fake backends often expose only ``encode(texts)``.
            encoded = backend.encode(batch)
        array = np.asarray(encoded, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(batch):
            raise ValueError("embedding backend returned an invalid vector shape")
        return array

    def _batch_indices(self, lengths: list[int]) -> list[list[int]]:
        """Group texts into batches of similar length under both ceilings.

        Padding makes every sample in a batch cost as much as the longest one,
        so one long table padded alongside seven short paragraphs wastes seven
        times the activation memory.  Sorting by length packs comparable texts
        together, and the token budget then bounds the real cost of a batch;
        results are written back by original index, so callers keep their order.
        """

        order = sorted(range(len(lengths)), key=lambda index: lengths[index])
        batches: list[list[int]] = []
        current: list[int] = []
        used = 0
        for index in order:
            cost = lengths[index]
            if current and (len(current) >= self.batch_size or used + cost > self.token_budget):
                batches.append(current)
                current = []
                used = 0
            current.append(index)
            used += cost
        if current:
            batches.append(current)
        return batches

    def _estimate_tokens(self, backend: EmbeddingBackend, text: str) -> int:
        """Cheap token estimate used only to size batches."""

        tokenizer = getattr(backend, "tokenizer", None)
        if tokenizer is not None:
            try:
                return max(1, len(tokenizer.encode(text, truncation=True, max_length=self.max_seq_length)))
            except Exception:  # noqa: BLE001 - any tokenizer quirk falls back to a ratio
                pass
        return max(1, len(text) // 4)

    def _validate_inputs(self, texts: Sequence[str]) -> list[str]:
        normalized_texts = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in normalized_texts):
            raise ValueError("embedding input contains empty text")
        return normalized_texts

    def _get_backend(self) -> EmbeddingBackend:
        if self._backend is None:
            # Import here so tests and API metadata endpoints do not load the model.
            from sentence_transformers import SentenceTransformer

            self._backend = SentenceTransformer(self.model_name)
        self._apply_length_cap(self._backend)
        return self._backend

    def _apply_length_cap(self, backend: EmbeddingBackend) -> None:
        """Bound the longest sequence the model may be asked to embed.

        The model default is its full 32768-token window, where a single long
        element already costs gigabytes of activations.  Real chunks stay far
        below this cap, so it only trims pathological outliers.
        """

        if hasattr(backend, "max_seq_length"):
            backend.max_seq_length = self.max_seq_length

    def _validate_vectors(self, vectors: np.ndarray, *, expected_count: int) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != expected_count or vectors.shape[1] < 1:
            raise ValueError("embedding backend returned an invalid vector shape")
        if not np.isfinite(vectors).all():
            raise ValueError("embedding backend returned non-finite values")
        if self._dimension is not None and vectors.shape[1] != self._dimension:
            raise ValueError("embedding vector dimension changed during this index build")

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("embedding backend returned a zero vector")
        return vectors / norms
