"""Fuse several retrieval channels into one ranked list.

Fusion here is rank-based (Reciprocal Rank Fusion) rather than score-based: the
channels do not share a scale (cosine similarity versus BM25), and RRF needs no
calibration, which matters because a single global similarity threshold is
already one of the things that was mis-calibrated in this project.

Two rules make the fused list usable for multi-requirement questions:

* RRF decides the order, so a chunk several channels agree on rises.
* Argument quota guarantees every query its own slot, so a requirement that only
  one sub-query retrieved cannot be crowded out by another requirement that
  many sub-queries retrieved.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    """One chunk in the fused ranking, with the channels that voted for it."""

    chunk_id: str
    score: float
    channels: tuple[int, ...] = ()
    ranks: tuple[tuple[int, int], ...] = ()


@dataclass
class Channel:
    """One ranked result list. ``key`` is what quota is applied per list."""

    name: str
    chunk_ids: list[str] = field(default_factory=list)
    requires_own_slot: bool = True
    dense_scores: dict[str, float] = field(default_factory=dict)

    def top(self) -> str | None:
        return self.chunk_ids[0] if self.chunk_ids else None

    def without(self, used: Iterable[str]) -> "Channel":
        blocked = set(used)
        return Channel(
            name=self.name,
            chunk_ids=[chunk_id for chunk_id in self.chunk_ids if chunk_id not in blocked],
            requires_own_slot=self.requires_own_slot,
            dense_scores=dict(self.dense_scores),
        )


def reciprocal_rank_fusion(
    channels: Sequence[Channel],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[FusedHit]:
    """Merge ranked channel lists; higher score = better, ties broken stably.

    D28 (2026-09-20): the ``weights`` parameter was deleted -- it had no caller
    (every call site passed ``k`` only), and RRF's whole point is that the
    channel scales never have to be calibrated.
    """

    if k < 1:
        raise ValueError("rrf k must be positive")

    scores: dict[str, float] = {}
    channel_votes: dict[str, list[int]] = {}
    ranks: dict[str, list[tuple[int, int]]] = {}
    first_seen: dict[str, int] = {}
    counter = 0
    for index, channel in enumerate(channels):
        for rank, chunk_id in enumerate(channel.chunk_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            channel_votes.setdefault(chunk_id, []).append(index)
            ranks.setdefault(chunk_id, []).append((index, rank))
            if chunk_id not in first_seen:
                first_seen[chunk_id] = counter
                counter += 1

    hits = [
        FusedHit(
            chunk_id=chunk_id,
            score=score,
            channels=tuple(channel_votes[chunk_id]),
            ranks=tuple(ranks[chunk_id]),
        )
        for chunk_id, score in scores.items()
    ]
    hits.sort(key=lambda hit: (-hit.score, first_seen[hit.chunk_id]))
    return hits


def select_with_quota(
    fused: Sequence[FusedHit],
    channels: Sequence[Channel],
    *,
    top_k: int,
    per_channel_slots: int = 1,
) -> list[FusedHit]:
    """Reserve slots for each contributing query, then fill by fused order.

    A sub-query that is the only source of a requirement keeps one slot; the
    remaining slots go to the globally best chunks.  Quota never overrides the
    caller's ``top_k``.
    """

    if top_k < 1:
        return []
    by_id = {hit.chunk_id: hit for hit in fused}
    selected: list[FusedHit] = []
    used: set[str] = set()

    for channel in channels:
        if not channel.requires_own_slot or per_channel_slots < 1:
            continue
        taken = 0
        for chunk_id in channel.chunk_ids:
            if taken >= per_channel_slots or len(selected) >= top_k:
                break
            if chunk_id in used or chunk_id not in by_id:
                continue
            selected.append(by_id[chunk_id])
            used.add(chunk_id)
            taken += 1

    for hit in fused:
        if len(selected) >= top_k:
            break
        if hit.chunk_id in used:
            continue
        selected.append(hit)
        used.add(hit.chunk_id)

    return selected[:top_k]


def best_dense_score(channels: Sequence[Channel], chunk_id: str) -> float | None:
    """Highest cosine similarity this chunk got from any dense channel.

    The evidence gate is a cosine threshold, so a fused hit keeps the best
    cosine it actually earned; chunks no dense channel retrieved are reported as
    ``None`` and are treated as below the gate by the caller instead of being
    given an invented score.
    """

    scores = [
        channel.dense_scores[chunk_id]
        for channel in channels
        if chunk_id in channel.dense_scores
    ]
    return max(scores) if scores else None
