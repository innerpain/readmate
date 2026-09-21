"""Evaluation targets: which page has to answer which part of a question.

Kept separate from the probe so the answer-level evaluation can import the same
target model without importing the probe's CLI module (and without a duplicated
module import when the probe is run with ``python -m``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProbeTarget:
    """One required piece of evidence: a page, and what it has to answer."""

    page: int
    requirement: str = ""


@dataclass(frozen=True)
class ProbeSample:
    question: str
    answerable: bool = True
    documents: tuple[str, ...] = ()
    targets: tuple[ProbeTarget, ...] = ()
    reference_answer: str = ""

    @property
    def target_pages(self) -> set[int]:
        return {target.page for target in self.targets}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProbeSample":
        raw_targets = payload.get("targets")
        targets: list[ProbeTarget] = []
        if isinstance(raw_targets, list):
            for item in raw_targets:
                if isinstance(item, dict) and item.get("page") is not None:
                    targets.append(ProbeTarget(int(item["page"]), str(item.get("requirement", ""))))
        if not targets:
            targets = [
                ProbeTarget(int(page))
                for page in (payload.get("expected_pages") or [])
                if isinstance(page, int) or str(page).isdigit()
            ]
        return cls(
            question=str(payload["question"]),
            answerable=bool(payload.get("answerable", True)),
            documents=tuple(str(item) for item in (payload.get("expected_documents") or [])),
            targets=tuple(targets),
            reference_answer=str(payload.get("reference_answer", "")),
        )


@dataclass
class TargetMatch:
    requirement: str
    page: int
    rank: int | None = None
    score: float | None = None


def load_probe_samples(path: Path | str) -> list[ProbeSample]:
    samples: list[ProbeSample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError("record must be an object")
            samples.append(ProbeSample.from_dict(payload))
        except (TypeError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid probe record at line {line_number}") from error
    if not samples:
        raise ValueError("probe set cannot be empty")
    return samples


def rank_targets(results: Sequence[object], sample: ProbeSample) -> list[TargetMatch]:
    """Where each required page landed in the retrieved list (1-based rank)."""

    matches: list[TargetMatch] = []
    for target in sample.targets:
        rank: int | None = None
        score: float | None = None
        for position, result in enumerate(results, start=1):
            filename = str(getattr(result, "filename", ""))
            page = int(getattr(result, "page", 0) or 0)
            if page != target.page:
                continue
            if sample.documents and filename not in sample.documents:
                continue
            rank = position
            score = float(getattr(result, "score", 0.0))
            break
        matches.append(TargetMatch(target.requirement, target.page, rank, score))
    return matches


def first_target_rank(results: Sequence[object], sample: ProbeSample) -> int | None:
    """Best rank among a sample's targets (shared with the answer-level eval)."""

    ranks = [match.rank for match in rank_targets(results, sample) if match.rank is not None]
    return min(ranks) if ranks else None
