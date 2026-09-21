"""Local cross-encoder reranker for post-fusion candidate reordering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class RerankBackend(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]], **kwargs: object) -> Sequence[float]: ...


@dataclass(frozen=True)
class RankedCandidate:
    chunk_id: str
    content: str
    rerank_score: float


class LocalCrossEncoderReranker:
    """Score (query, passage) pairs and return descending chunk order.

    The evidence gate still consumes dense cosine scores from retrieval; this
    component only decides *which* candidates survive into the final window.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        max_length: int = 512,
        backend: RerankBackend | None = None,
    ) -> None:
        if max_length < 8:
            raise ValueError("max_length must be >= 8")
        self.model_name = model_name
        self.max_length = max_length
        self._backend = backend

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        *,
        top_k: int | None = None,
    ) -> list[RankedCandidate]:
        """Reorder ``(chunk_id, content)`` pairs by cross-encoder score."""

        query_text = query.strip() if isinstance(query, str) else ""
        if not query_text:
            raise ValueError("rerank query must be non-empty")
        cleaned: list[tuple[str, str]] = []
        for chunk_id, content in candidates:
            text = content if isinstance(content, str) else ""
            if not str(chunk_id).strip() or not text.strip():
                continue
            cleaned.append((str(chunk_id), text))
        if not cleaned:
            return []

        scores = self._score_pairs(query_text, [content for _, content in cleaned])
        ranked = [
            RankedCandidate(chunk_id=chunk_id, content=content, rerank_score=float(score))
            for (chunk_id, content), score in zip(cleaned, scores)
        ]
        ranked.sort(key=lambda item: item.rerank_score, reverse=True)
        if top_k is not None:
            ranked = ranked[: max(0, int(top_k))]
        return ranked

    def _score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        backend = self._get_backend()
        pairs = [(query, document) for document in documents]
        try:
            raw = backend.predict(pairs, batch_size=min(16, len(pairs)), show_progress_bar=False)
        except TypeError:
            raw = backend.predict(pairs)
        return [float(score) for score in list(raw)]

    def _get_backend(self) -> RerankBackend:
        if self._backend is None:
            from sentence_transformers import CrossEncoder

            model_path = self._resolve_model_path(self.model_name)
            self._backend = CrossEncoder(model_path, max_length=self.max_length)
        return self._backend

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """Resolve a local snapshot first so offline containers avoid hub roundtrips."""

        from pathlib import Path

        candidates: list[Path] = []
        raw = Path(model_name)
        if raw.exists():
            return str(raw)
        # Project-local ModelScope / manual downloads (compose mounts ./data).
        project_root = Path(__file__).resolve().parents[2]
        candidates.extend(
            [
                project_root
                / "data"
                / "models"
                / "modelscope"
                / "models"
                / "BAAI--bge-reranker-v2-m3"
                / "snapshots"
                / "master",
                project_root / "data" / "models" / "bge-reranker-v2-m3",
            ]
        )
        for candidate in candidates:
            if (candidate / "config.json").is_file() and (
                (candidate / "model.safetensors").is_file()
                or (candidate / "pytorch_model.bin").is_file()
            ):
                return str(candidate)

        try:
            from huggingface_hub import snapshot_download

            # Prefer an already-complete hub cache; only then allow online fetch.
            try:
                return snapshot_download(model_name, local_files_only=True)
            except Exception:
                return snapshot_download(model_name, local_files_only=False)
        except Exception:
            return model_name
