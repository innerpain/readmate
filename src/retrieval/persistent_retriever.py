"""Query the current versioned Chroma collection.

Two entry points, one contract:

``retrieve(question)``          one query, one channel (the original behaviour)
``retrieve_multi(texts, ...)``  several queries, several channels, fused

Both return the same ``RetrievedChunk`` shape, and both report a chunk's score
as the best **cosine similarity** it earned from a dense channel.  That is
deliberate: the evidence gate is a cosine threshold, so fusion is allowed to
change *which* chunks are selected and in what order, but not to smuggle in
chunks the embedder considers irrelevant.  Chunks only the keyword channel
found are reported in ``last_diagnostics`` instead, where they can be measured
before anyone decides to loosen the gate.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.config.settings import (
    ModelSettings,
    RetrievalSettings,
    get_model_settings,
    get_retrieval_settings,
)
from src.retrieval.neighbor_expand import build_presentation, neighbor_payload
from src.retrieval.embedding_generator import EmbeddingGenerator
from src.retrieval.chroma_store import ChromaVectorStore
from src.retrieval.result_diversify import diversify_hits, drops_unwanted_table, ensure_hit_present, query_wants_tables
from src.retrieval.result_fusion import (
    Channel,
    best_dense_score,
    reciprocal_rank_fusion,
    select_with_quota,
)

logger = logging.getLogger(__name__)

# D27 (2026-09-20): the window / query limits moved into RetrievalSettings
# (``max_query_chars`` / ``max_query_texts`` / ``filtered_window_multiplier`` /
# ``max_filtered_window``) so they are tunable without editing code.


class InvalidQuestionError(ValueError):
    """Raised when a user question cannot safely enter the retrieval pipeline."""


class KnowledgeBaseUnavailableError(RuntimeError):
    """Raised when there is no valid, published local snapshot to query."""


class EmbeddingCompatibilityError(RuntimeError):
    """Raised when the query embedding configuration differs from the snapshot."""


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved Chroma result with traceable chunk metadata.

    ``content_kind``, ``atomic`` and the object ids are carried through so the
    evidence layer can tell a table row from a sentence and a degraded page from
    a clean one -- parsing improvements are useless if the answer stage cannot
    see what it was given.
    """

    chunk_id: str
    score: float
    content: str
    document_id: str
    filename: str
    page: int
    section: str
    # A3: last page the chunk covers (0 = same as ``page`` / unknown) so a
    # citation can show the real span instead of only the start page.
    page_end: int = 0
    content_kind: str = "text"
    atomic: bool = False
    heading_path: list[str] = field(default_factory=list)
    table_id: str | None = None
    figure_id: str | None = None
    figure_type: str | None = None
    image_path: str | None = None
    degraded_structure: bool = False
    structure_pages: list[int] = field(default_factory=list)
    fusion_score: float | None = None
    channels: list[str] = field(default_factory=list)
    chunk_type: str = ""
    parent_id: str | None = None
    neighbor_prev_chunk_ids: list[str] = field(default_factory=list)
    neighbor_next_chunk_ids: list[str] = field(default_factory=list)
    expanded_neighbors: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    presentation_content: str | None = None


@dataclass(frozen=True)
class PresentationBudget:
    """D26 (2026-09-20): the presentation window is data, not an agent import.

    The retriever used to reach into ``get_agent_settings()`` mid-retrieval to
    size the table/prose window, which coupled the retrieval layer to the agent
    layer.  The composition root (``model_cache.retrieval_model_kwargs``) now
    passes the agent's numbers in; these defaults keep the retriever usable on
    its own (tests, probes).
    """

    budget: int = 1200
    table_budget: int = 3000
    primary_min_ratio: float = 0.6
    neighbor_min_chars: int = 120


class PersistentRetriever:
    """Loads the current snapshot per request so restarts need no in-memory state."""

    def __init__(
        self,
        *,
        index_dir: Path | str | None = None,
        embedder: EmbeddingGenerator | None = None,
        model_settings: ModelSettings | None = None,
        retrieval_settings: RetrievalSettings | None = None,
        reranker: object | None = None,
        presentation: PresentationBudget | None = None,
    ) -> None:
        self.model_settings = model_settings or get_model_settings()
        self.retrieval_settings = retrieval_settings or get_retrieval_settings()
        self.presentation = presentation or PresentationBudget()
        project_root = Path(__file__).resolve().parents[2]
        self.index_dir = Path(index_dir) if index_dir is not None else project_root / "data" / "chroma"
        self.embedder = embedder or EmbeddingGenerator.from_settings(self.model_settings)
        self._reranker = reranker
        # Filled by the last retrieve_multi call: channels used, keyword-only
        # hits that no dense channel returned.  Diagnostics only.
        self.last_diagnostics: dict[str, object] = {}

    # --------------------------------------------------------------- public API
    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Single-query retrieval; widens the pool when rerank is enabled."""

        query = self._validate_text(question)
        settings = self.retrieval_settings
        candidate = settings.candidate_k if settings.rerank_enabled else settings.top_k
        return self.retrieve_multi(
            [query],
            candidate_k=candidate,
            final_k=settings.top_k,
        )

    def retrieve_multi(
        self,
        query_texts: Sequence[str],
        *,
        candidate_k: int | None = None,
        final_k: int | None = None,
        quota_per_query: int | None = None,
        eligible_ids: set[str] | None = None,
        document_ids: set[str] | None = None,
        gate_query: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve with several queries, fuse the channels, keep the best slots.

        ``eligible_ids`` restricts the final selection to a caller-provided set
        (the probe uses it to measure, without changing behaviour, what a wider
        gate would have admitted).

        ``document_ids`` is the collection scope: an Agent collection is a view
        over the single published snapshot, so out-of-scope chunks are dropped
        right after fusion, before the quota pick and the dense-top insurance
        read the pool.  The raw window widens first, because filtering a small
        collection can otherwise leave fewer hits than the final window wants.

        ``gate_query`` is the **user's original wording** and drives the table
        gate (``query_wants_tables``) and the B/C noise filter.  It defaults to
        the first retrieval text, but the Agent passes the raw question so that
        a model-rewritten English query cannot silently drop every table for a
        question the user phrased with "表格/Table" (D5 / A4).
        """

        texts = self._validate_texts(query_texts)
        settings = self.retrieval_settings
        gate_text = (gate_query or "").strip() or texts[0]
        candidate_k = int(candidate_k or settings.candidate_k)
        final_k = int(final_k or settings.final_k)
        quota = settings.quota_per_query if quota_per_query is None else int(quota_per_query)

        store, manifest = self._load_snapshot()
        if int(manifest.get("chunk_count", 0) or 0) == 0:
            raise KnowledgeBaseUnavailableError("knowledge_base_empty")
        dimension = int(manifest.get("embedding_dimension") or 0)
        self._validate_manifest(manifest, dimension)

        vectors = self.embedder.embed_query(
            texts,
            instruction=self.model_settings.embedding_query_instruction,
        )
        if dimension and vectors.shape[1] != dimension:
            raise EmbeddingCompatibilityError("query_embedding_dimension_mismatch")

        window = max(candidate_k, final_k)
        if document_ids is not None:
            window = min(window * settings.filtered_window_multiplier, settings.max_filtered_window)
        channels: list[Channel] = []
        content_by_id: dict[str, str] = {}
        metadata_by_id: dict[str, dict] = {}
        for index, vector in enumerate(vectors):
            result = store.query(vector, k=window)
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            channel = Channel(name=f"dense:{index}", dense_scores={})
            for content, metadata, distance in zip(documents, metadatas, distances):
                metadata = metadata or {}
                chunk_id = str(metadata.get("chunk_id"))
                if not chunk_id:
                    raise KnowledgeBaseUnavailableError("snapshot_chunk_metadata_invalid")
                content_by_id.setdefault(chunk_id, str(content))
                metadata_by_id.setdefault(chunk_id, metadata)
                channel.chunk_ids.append(chunk_id)
                channel.dense_scores[chunk_id] = 1.0 - float(distance)
            channels.append(channel)

        fused = reciprocal_rank_fusion(channels, k=settings.rrf_k)
        dropped_out_of_scope = 0
        if document_ids is not None:
            scoped: list = []
            for hit in fused:
                scope_id = str((metadata_by_id.get(hit.chunk_id) or {}).get("document_id") or "")
                if scope_id in document_ids:
                    scoped.append(hit)
                else:
                    dropped_out_of_scope += 1
            fused = scoped
        if eligible_ids is not None:
            fused = [hit for hit in fused if hit.chunk_id in eligible_ids]
        # Only chunks the embedder itself retrieved may hold a slot, so the
        # cosine gate keeps meaning what it meant before fusion existed.
        eligible = [hit for hit in fused if best_dense_score(channels, hit.chunk_id) is not None]
        rerank_enabled = bool(settings.rerank_enabled)
        # Always keep a wide pool so diversity collapse can refill emptied slots.
        pool_k = max(final_k, candidate_k)
        pool = select_with_quota(eligible, channels, top_k=pool_k, per_channel_slots=quota)

        missing = [hit.chunk_id for hit in pool if hit.chunk_id not in content_by_id]
        if missing:
            fetched = store.get_by_ids(missing)
            for chunk_id, payload in fetched.items():
                content_by_id[chunk_id] = str(payload.get("document") or "")
                metadata_by_id[chunk_id] = dict(payload.get("metadata") or {})

        ordered = pool
        rerank_applied = False
        rerank_model = None
        if rerank_enabled and pool:
            ordered, rerank_applied, rerank_model = self._apply_rerank(
                query_text=texts[0],
                pool=pool,
                content_by_id=content_by_id,
            )

        selected = diversify_hits(
            ordered,
            metadata_by_id,
            final_k=final_k,
            query=gate_text,
            score_fn=lambda hit: float(best_dense_score(channels, hit.chunk_id) or -1.0),
            score_floor_ratio=settings.score_floor_ratio,
        )
        dense_top = None
        if pool:
            allow_tables = query_wants_tables(gate_text)
            ranked_dense = sorted(
                pool,
                key=lambda hit: float(best_dense_score(channels, hit.chunk_id) or -1.0),
                reverse=True,
            )
            for hit in ranked_dense:
                meta = metadata_by_id.get(hit.chunk_id) or {}
                if drops_unwanted_table(meta, allow_tables=allow_tables):
                    continue
                dense_top = hit
                break
        selected = ensure_hit_present(selected, dense_top, final_k=final_k)
        # D19-a (2026-09-20): the A8 literal-anchor insurance was removed together
        # with the keyword channel.  Measured before removal: it was net-negative
        # -- the anchor was inserted first and then deleted by the min_score gate
        # below, while the chunk it displaced never came back (2 of 4 probe
        # questions lost a slot, 2 were unchanged).  The dense-top insurance above
        # stays: it needs no keyword channel.

        results: list[RetrievedChunk] = []
        for hit in selected:
            metadata = metadata_by_id.get(hit.chunk_id)
            if not metadata:
                continue
            score = best_dense_score(channels, hit.chunk_id)
            if score is None:
                continue
            results.append(
                _to_retrieved_chunk(
                    chunk_id=hit.chunk_id,
                    content=content_by_id.get(hit.chunk_id, ""),
                    metadata=metadata,
                    score=score,
                    fusion_score=hit.score,
                    channels=[channels[index].name for index in hit.channels],
                )
            )
        # A4 / D-4a: the retriever's exit is the single place ``min_score`` is
        # enforced, so every consumer (Agent tools, L1 probe, legacy paths)
        # shares the sufficiency gate.  ``score`` is the dense cosine, so the
        # threshold keeps its calibrated meaning.  Setting RETRIEVAL_MIN_SCORE=0
        # turns the gate off without a code change.
        # A8 layer 2 (2026-09-19): the absolute gate may *demote*, never *erase*.
        # Emptying the window turned "weak recall" into "the collection has
        # nothing", which the model then reported as "not found" for evidence that
        # is in the corpus (measured: a Chinese question whose only literal anchor
        # is a section number -> 0 hits at 0.45, while the four chunks containing
        # it scored 0.35-0.41).  When the absolute gate would empty the window,
        # keep what cleared the same *relative* floor diversify already uses and
        # mark it, so the answer layer can call the evidence weak instead of
        # calling it absent.
        min_score = settings.min_score
        dropped_by_min_score = 0
        low_confidence_floor = 0.0
        if min_score > 0 and results:
            kept = [item for item in results if item.score >= min_score]
            dropped_by_min_score = len(results) - len(kept)
            if kept:
                results = kept
            elif settings.score_floor_ratio and settings.score_floor_ratio > 0:
                # NOTE: deliberately not named ``dense_top`` -- that name belongs to
                # the dense-top-insurance hit further down, and shadowing it with a
                # float turned the diagnostics build into
                # ``'float' object has no attribute 'chunk_id'``.
                top_score = max(item.score for item in results)
                low_confidence_floor = round(float(settings.score_floor_ratio) * top_score, 4)
                results = [item for item in results if item.score >= low_confidence_floor]
        low_confidence_ids = sorted(item.chunk_id for item in results) if low_confidence_floor else []
        results = self._expand_structure_neighbors(store, results)
        self.last_diagnostics = {
            "channels": [channel.name for channel in channels],
            "channel_sizes": [len(channel.chunk_ids) for channel in channels],
            "candidate_k": window,
            "final_k": final_k,
            "document_ids_filter": len(document_ids) if document_ids is not None else None,
            "dropped_out_of_scope": dropped_out_of_scope,
            "dropped_by_min_score": dropped_by_min_score,
            "min_score": min_score,
            "low_confidence_applied": bool(low_confidence_floor),
            "low_confidence_floor": low_confidence_floor,
            "low_confidence_count": len(low_confidence_ids),
            "low_confidence_ids": low_confidence_ids[:20],
            "quota_per_query": quota,
            "fused_count": len(fused),
            "expanded_structure_hits": sum(1 for item in results if item.expanded_neighbors),
            "rerank_enabled": rerank_enabled,
            "rerank_applied": rerank_applied,
            "rerank_model": rerank_model,
            "rerank_pool": len(pool),
            # D20 (2026-09-20): the reranker scores the rewritten query while the
            # table gate and the noise checks read the user's original wording.
            # Record both so a disagreement can be attributed instead of guessed.
            "rerank_query": texts[0][:120],
            "gate_query": gate_text[:120],
            "judge_mismatch": texts[0].strip() != gate_text.strip(),
            "diversify_applied": True,
            "dense_top_insured": bool(
                dense_top is not None
                and any(item.chunk_id == dense_top.chunk_id for item in selected)
            ),
        }
        return results

    # ------------------------------------------------------------- validation
    def _validate_text(self, question: object) -> str:
        query = question.strip() if isinstance(question, str) else ""
        if not query:
            raise InvalidQuestionError("question_empty")
        if len(query) > self.retrieval_settings.max_query_chars:
            raise InvalidQuestionError("question_too_long")
        return query

    def _validate_texts(self, query_texts: Sequence[str]) -> list[str]:
        texts: list[str] = []
        for item in query_texts:
            query = self._validate_text(item)
            if query not in texts:
                texts.append(query)
            if len(texts) >= self.retrieval_settings.max_query_texts:
                break
        if not texts:
            raise InvalidQuestionError("question_empty")
        return texts

    def _load_snapshot(self) -> tuple[ChromaVectorStore, dict[str, object]]:
        try:
            return ChromaVectorStore.load(self.index_dir)
        except FileNotFoundError as error:
            raise KnowledgeBaseUnavailableError("knowledge_base_empty") from error
        except (KeyError, ValueError, OSError) as error:
            raise KnowledgeBaseUnavailableError("knowledge_base_snapshot_invalid") from error

    def _validate_manifest(self, manifest: dict[str, object], dimension: int) -> None:
        snapshot_model = manifest.get("embedding_model")
        snapshot_dimension = manifest.get("embedding_dimension")
        normalized = manifest.get("normalize_embeddings")
        if snapshot_model != self.model_settings.embedding_model:
            raise EmbeddingCompatibilityError("embedding_model_mismatch")
        if snapshot_dimension != dimension:
            raise EmbeddingCompatibilityError("snapshot_dimension_mismatch")
        if self.model_settings.embedding_dimension not in (None, dimension):
            raise EmbeddingCompatibilityError("configured_embedding_dimension_mismatch")
        if normalized is not True or not self.model_settings.normalize_embeddings:
            raise EmbeddingCompatibilityError("embedding_normalization_mismatch")

    @staticmethod
    def _merged_table_primary(store: ChromaVectorStore, result: RetrievedChunk) -> str:
        """Reassemble every shard of one table into a single primary body (D4)."""

        own = result.content or ""
        if not result.table_id:
            return own
        # Any store shape must survive this: the test doubles and pre-upgrade
        # stores do not implement the extra lookup, and a missing shard list only
        # means "no reassembly", never a failed search.
        lookup = getattr(store, "siblings_by_table", None)
        siblings = lookup(result.table_id) if callable(lookup) else []
        parts: list[str] = []
        for entry in siblings:
            text = str(entry.get("document") or "")
            if text and text not in parts:
                parts.append(text)
        if not parts:
            return own
        if own and own not in parts:
            parts.append(own)
        return "\n\n".join(parts)

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        from src.retrieval.reranker import LocalCrossEncoderReranker

        settings = self.retrieval_settings
        self._reranker = LocalCrossEncoderReranker(
            settings.rerank_model,
            max_length=settings.rerank_max_length,
        )
        return self._reranker

    def _apply_rerank(
        self,
        *,
        query_text: str,
        pool: list,
        content_by_id: dict[str, str],
    ) -> tuple[list, bool, str | None]:
        """Reorder the fused pool with a cross-encoder (no final_k cut here)."""

        try:
            reranker = self._get_reranker()
            candidates = [
                (hit.chunk_id, content_by_id.get(hit.chunk_id, ""))
                for hit in pool
            ]
            ranked = reranker.rerank(query_text, candidates, top_k=None)
        except Exception as error:
            # One warning line is enough; full traceback floods multi-question probes.
            logger.warning("rerank failed; falling back to fused order: %s", error)
            return list(pool), False, None

        by_id = {hit.chunk_id: hit for hit in pool}
        ordered = [by_id[item.chunk_id] for item in ranked if item.chunk_id in by_id]
        # Keep any pool members the reranker skipped (empty content, etc.) at the end.
        seen = {hit.chunk_id for hit in ordered}
        ordered.extend(hit for hit in pool if hit.chunk_id not in seen)
        model_name = getattr(reranker, "model_name", None)
        return ordered, True, str(model_name) if model_name else None

    def _expand_structure_neighbors(
        self,
        store: ChromaVectorStore,
        results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Attach prev/next prose windows for table/figure hits (no top_k cost)."""

        expanded: list[RetrievedChunk] = []
        seen_table_expand: set[str] = set()
        window_kwargs: dict[str, object] | None = None
        for result in results:
            if result.chunk_type not in {"table_summary", "table_pack", "figure"}:
                expanded.append(result)
                continue
            if window_kwargs is None:
                # D26: injected by the composition root, never read from the agent
                # layer here.
                window_kwargs = {
                    "budget": self.presentation.budget,
                    "table_budget": self.presentation.table_budget,
                    "primary_min_ratio": self.presentation.primary_min_ratio,
                    "neighbor_min_chars": self.presentation.neighbor_min_chars,
                }
            budget = int(window_kwargs["budget"])
            ratio = float(window_kwargs["primary_min_ratio"])
            min_chars = int(window_kwargs["neighbor_min_chars"])
            if result.table_id:
                # D6: a table's rows are its evidence -- give it the table budget
                # instead of the prose one, and put every shard of the table back
                # together (D4) so one slot carries the whole table.
                budget = int(window_kwargs.get("table_budget") or budget)
                primary = self._merged_table_primary(store, result)
            else:
                primary = result.content
            # One neighbor expansion per table_id (summary+pack may both hit).
            dedupe_key = result.table_id or result.figure_id or result.chunk_id
            if result.table_id and dedupe_key in seen_table_expand:
                # Still return the pack/summary hit, but reuse empty neighbors marker.
                expanded.append(
                    _replace_chunk(
                        result,
                        expanded_neighbors={"prev": [], "next": []},
                        presentation_content=build_presentation(
                            primary, [], [], budget,
                            primary_min_ratio=ratio,
                            neighbor_min_chars=min_chars,
                        ),
                    )
                )
                continue

            prev_ids = list(result.neighbor_prev_chunk_ids)[:2]
            next_ids = list(result.neighbor_next_chunk_ids)[:2]
            wanted = [*prev_ids, *next_ids]
            fetched = store.get_by_ids(wanted) if wanted else {}
            prev_parts = [neighbor_payload(chunk_id, fetched.get(chunk_id)) for chunk_id in prev_ids if chunk_id in fetched]
            next_parts = [neighbor_payload(chunk_id, fetched.get(chunk_id)) for chunk_id in next_ids if chunk_id in fetched]
            window = build_presentation(
                primary, prev_parts, next_parts, budget,
                primary_min_ratio=ratio,
                neighbor_min_chars=min_chars,
            )
            if result.table_id:
                seen_table_expand.add(dedupe_key)
            expanded.append(
                _replace_chunk(
                    result,
                    expanded_neighbors={"prev": prev_parts, "next": next_parts},
                    presentation_content=window,
                )
            )
        return expanded


def _replace_chunk(
    chunk: RetrievedChunk,
    *,
    expanded_neighbors: dict[str, list[dict[str, str]]],
    presentation_content: str | None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        score=chunk.score,
        content=chunk.content,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page=chunk.page,
        page_end=chunk.page_end,
        section=chunk.section,
        content_kind=chunk.content_kind,
        atomic=chunk.atomic,
        heading_path=list(chunk.heading_path),
        table_id=chunk.table_id,
        figure_id=chunk.figure_id,
        figure_type=chunk.figure_type,
        image_path=chunk.image_path,
        degraded_structure=chunk.degraded_structure,
        structure_pages=list(chunk.structure_pages),
        fusion_score=chunk.fusion_score,
        channels=list(chunk.channels),
        chunk_type=chunk.chunk_type,
        parent_id=chunk.parent_id,
        neighbor_prev_chunk_ids=list(chunk.neighbor_prev_chunk_ids),
        neighbor_next_chunk_ids=list(chunk.neighbor_next_chunk_ids),
        expanded_neighbors=expanded_neighbors,
        presentation_content=presentation_content,
    )


def _to_retrieved_chunk(
    *,
    chunk_id: str,
    content: str,
    metadata: dict,
    score: float,
    fusion_score: float | None = None,
    channels: Sequence[str] = (),
) -> RetrievedChunk:
    """Project stored chunk metadata onto the retrieval result contract."""

    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        content=content,
        document_id=str(metadata.get("document_id") or ""),
        filename=str(metadata.get("filename") or ""),
        page=int(metadata.get("page") or 0) or 1,
        # 0 for snapshots published before the span was recorded; callers fall
        # back to ``page`` so an older index keeps working.
        page_end=int(metadata.get("page_end") or 0),
        section=str(metadata.get("section") or "Unknown"),
        content_kind=str(metadata.get("content_kind") or _kind_from_type(metadata)),
        atomic=bool(int(metadata.get("atomic") or 0)) if not isinstance(metadata.get("atomic"), bool) else bool(metadata.get("atomic")),
        heading_path=_json_list(metadata.get("heading_path")),
        table_id=_optional_str(metadata.get("table_id")),
        figure_id=_optional_str(metadata.get("figure_id")),
        figure_type=_optional_str(metadata.get("figure_type")),
        image_path=_optional_str(metadata.get("image_path")),
        degraded_structure=bool(metadata.get("degraded_structure") or metadata.get("degraded")),
        structure_pages=_json_int_list(metadata.get("structure_pages")),
        fusion_score=fusion_score,
        channels=list(channels),
        chunk_type=str(metadata.get("chunk_type") or ""),
        parent_id=_optional_str(metadata.get("parent_id")),
        neighbor_prev_chunk_ids=_json_list(metadata.get("neighbor_prev_chunk_ids")),
        neighbor_next_chunk_ids=_json_list(metadata.get("neighbor_next_chunk_ids")),
    )


def _kind_from_type(metadata: dict) -> str:
    """Derive content kind from the chunk type for pre-upgrade snapshots."""

    chunk_type = str(metadata.get("chunk_type") or "")
    if chunk_type in {"table", "table_summary", "table_pack"}:
        return "table"
    if chunk_type in {"figure", "chart"}:
        return chunk_type
    if chunk_type == "formula":
        return "formula"
    if metadata.get("table_id"):
        return "table"
    if metadata.get("figure_id"):
        return "chart" if metadata.get("figure_type") == "chart" else "figure"
    return "text"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _json_int_list(value: object) -> list[int]:
    pages: list[int] = []
    for item in _json_list(value):
        try:
            pages.append(int(item))
        except (TypeError, ValueError):
            continue
    return pages
