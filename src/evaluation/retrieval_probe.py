"""Measure retrieval alone, so a change can be attributed before it is believed.

The end-to-end evaluation is dominated by the answer model (8-12s of chain of
thought per question), which makes it useless for iterating on retrieval.  This
probe runs the *retrieval* half only -- no answer model, no gate -- and reports
where each required piece of evidence actually landed:

    rank of every required page       (a page at rank 52 is a ranking failure,
                                       a page that never appears is a recall
                                       failure -- the fix is different)
    recall@k for growing k
    top-5 noise                       (slots spent on pages the question never
                                       asked about)
    a reranker verdict                (derived from the two numbers above, so
                                       the decision is data, not opinion)

It also reports the top-5 noise ratio, which is the
evidence needed before loosening the cosine gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.targets import (
    ProbeSample,
    ProbeTarget,
    TargetMatch,
    first_target_rank,
    load_probe_samples,
    rank_targets,
)
from src.rag.query_planner import QueryPlanner

RETRIEVAL_PROBE_VERSION = "retrieval-probe-1"
# Verdict thresholds agreed for this project: a target that sits beyond rank 10,
# or a top-5 that spends a fifth of its slots on unrelated pages, means ordering
# is still the bottleneck.
RERANK_MEDIAN_RANK_THRESHOLD = 10
RERANK_NOISE_RATIO_THRESHOLD = 0.20
VARIANT_BASELINE = "baseline"
VARIANT_CLAUSES = "clauses"
VARIANT_PLANNED = "planned"
# The agent's own path: RetrievalRouter picks the track (rules here, the model
# in production), then the same retriever runs with the collection scope. The
# rigid variants above measure the retriever; this one measures what the Agent
# actually calls, so an L3 regression can be attributed to a layer.
VARIANT_ROUTED = "routed"

__all__ = [
    "ProbeSample",
    "ProbeTarget",
    "SampleOutcome",
    "TargetMatch",
    "first_target_rank",
    "format_details",
    "format_report",
    "load_probe_samples",
    "main",
    "rank_targets",
    "reranker_verdict",
    "run_variant",
    "summarize",
]


@dataclass
class SampleOutcome:
    question: str
    answerable: bool
    matches: list[TargetMatch] = field(default_factory=list)
    top_hits: list[tuple[str, int, float]] = field(default_factory=list)
    noise_ratio: float = 0.0
    used_ids: int = 0

    @property
    def ranks(self) -> list[int]:
        return [match.rank for match in self.matches if match.rank is not None]


def summarize(outcomes: Sequence[SampleOutcome], *, k: int, top_k: int = 5) -> dict[str, object]:
    answerable = [outcome for outcome in outcomes if outcome.answerable]
    recall: dict[str, float] = {}
    for window in (5, 10, 50, k):
        if window > k:
            continue
        hits = sum(
            1 for outcome in answerable
            if any(rank <= window for rank in outcome.ranks)
        )
        recall[f"recall@{window}"] = round(hits / len(answerable), 4) if answerable else 0.0
    all_ranks = [rank for outcome in answerable for rank in outcome.ranks]
    found = [rank for rank in all_ranks if rank is not None]
    reciprocal = [
        1.0 / min(outcome.ranks) for outcome in answerable if outcome.ranks
    ]
    median_rank = statistics.median(found) if found else float(k + 1)
    noise_values = [outcome.noise_ratio for outcome in answerable]
    noise = statistics.mean(noise_values) if noise_values else 0.0
    return {
        "probe_version": RETRIEVAL_PROBE_VERSION,
        "questions": len(outcomes),
        "answerable": len(answerable),
        "k": k,
        **recall,
        "mrr@%d" % top_k: round(sum(reciprocal) / len(answerable), 4) if answerable else 0.0,
        "targets_total": len(all_ranks),
        "targets_found": len(found),
        "median_target_rank": median_rank,
        "top%d_noise_ratio" % top_k: round(noise, 4),
        "verdict": reranker_verdict(median_rank, noise, k=k),
    }


def reranker_verdict(median_rank: float, noise_ratio: float, *, k: int) -> dict[str, object]:
    """Turn the two measurements into the agreed reranker decision."""

    reasons: list[str] = []
    if median_rank > RERANK_MEDIAN_RANK_THRESHOLD:
        reasons.append(f"median target rank {median_rank:.1f} > {RERANK_MEDIAN_RANK_THRESHOLD}")
    if noise_ratio >= RERANK_NOISE_RATIO_THRESHOLD:
        reasons.append(f"top-5 noise {noise_ratio:.0%} >= {RERANK_NOISE_RATIO_THRESHOLD:.0%}")
    missing = median_rank > k
    if missing:
        reasons.append(f"targets still missing beyond k={k}: ranking cannot fix this alone")
    return {
        "needs_reranker": bool(reasons) and not missing,
        "reasons": reasons,
        "thresholds": {
            "median_target_rank": RERANK_MEDIAN_RANK_THRESHOLD,
            "top5_noise_ratio": RERANK_NOISE_RATIO_THRESHOLD,
        },
    }


def run_variant(
    name: str,
    samples: Sequence[ProbeSample],
    *,
    retriever_factory: Callable[[], object],
    k: int,
    top_k: int = 5,
    planner: QueryPlanner | None = None,
    with_llm_planning: bool = False,
    document_ids: set[str] | None = None,
) -> tuple[list[SampleOutcome], dict[str, object]]:
    """Run one retrieval variant over every sample and summarize it."""

    retriever = retriever_factory()
    outcomes: list[SampleOutcome] = []
    diagnostics: list[dict[str, object]] = []
    for sample in samples:
        results = _retrieve_for_variant(
            retriever,
            sample,
            name=name,
            k=k,
            planner=planner,
            with_llm_planning=with_llm_planning,
            document_ids=document_ids,
        )
        matches = rank_targets(results, sample)
        top_hits = [
            (str(getattr(result, "filename", "")), int(getattr(result, "page", 0) or 0),
             round(float(getattr(result, "score", 0.0)), 4))
            for result in results[:top_k]
        ]
        noise = 0.0
        if sample.answerable and top_hits:
            unrelated = sum(1 for _, page, _ in top_hits if page not in sample.target_pages)
            noise = unrelated / len(top_hits)
        outcomes.append(SampleOutcome(
            question=sample.question,
            answerable=sample.answerable,
            matches=matches,
            top_hits=top_hits,
            noise_ratio=noise,
            used_ids=len(results),
        ))
        diagnostics.append(dict(getattr(retriever, "last_diagnostics", {}) or {}))
    summary = summarize(outcomes, k=k, top_k=top_k)
    summary["variant"] = name
    return outcomes, summary


def _retrieve_for_variant(
    retriever: object,
    sample: ProbeSample,
    *,
    name: str,
    k: int,
    planner: QueryPlanner | None,
    with_llm_planning: bool,
    document_ids: set[str] | None = None,
) -> list:
    retrieve_multi = getattr(retriever, "retrieve_multi")
    if name == VARIANT_ROUTED:
        return _retrieve_routed(retriever, sample, k=k, document_ids=document_ids)
    if name == VARIANT_BASELINE:
        return retrieve_multi(
            [sample.question], candidate_k=k, final_k=k, document_ids=document_ids
        )
    if name == VARIANT_PLANNED and planner is not None and with_llm_planning:
        plan = planner.plan(sample.question)
    else:  # clause plan: sub-queries, no LLM call
        plan = (planner or _terms_only_planner()).plan(sample.question)
    return retrieve_multi(
        plan.dense_texts(),
        candidate_k=k,
        final_k=k,
        document_ids=document_ids,
    )


def _retrieve_routed(
    retriever: object,
    sample: ProbeSample,
    *,
    k: int,
    document_ids: set[str] | None,
) -> list:
    """The agent's retrieval call: route first, then retrieve.

    Routing here is the deterministic rule path (``llm=None``), so this is a
    lower bound on the agent's real behaviour -- production lets the router
    model choose the track, which is visible per step in the agent eval's trace.
    """

    from src.adapter.route_planner import RetrievalRouter, compute_signals

    router = RetrievalRouter(llm=None)
    decision = router.decide(
        sample.question,
        compute_signals(sample.question, collection_size=len(document_ids or ())),
    )
    return getattr(retriever, "retrieve_multi")(
        decision.dense_texts(sample.question),
        candidate_k=k,
        final_k=k,
        document_ids=document_ids,
    )


def collection_document_ids(agent_db: str, collection_id: str) -> set[str]:
    """Adapter-equivalent scope: the collection's document ids, read from SQLite.

    The agent always searches inside a collection scope, which widens the raw
    window before fusion; a probe without the scope measures a different call.
    """

    import sqlite3
    from pathlib import Path

    path = Path(agent_db)
    if not path.exists():
        return set()
    connection = sqlite3.connect(str(path))
    try:
        rows = connection.execute(
            "SELECT document_id FROM collection_documents WHERE collection_id = ?",
            (collection_id,),
        ).fetchall()
    finally:
        connection.close()
    return {str(row[0]) for row in rows}


def _terms_only_planner() -> QueryPlanner:
    """Planner without an LLM: clause sub-queries only."""

    from src.config.settings import QueryPlanningSettings

    return QueryPlanner(settings=QueryPlanningSettings(llm_enabled=False))


def format_report(summaries: Sequence[dict[str, object]]) -> str:
    """One comparison table per variant, plus the verdict lines."""

    lines: list[str] = []
    header = f"{'variant':<10} {'recall@5':>9} {'recall@10':>10} {'recall@50':>10} {'MRR@5':>7} {'medRank':>8} {'noise':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for summary in summaries:
        lines.append(
            f"{str(summary.get('variant')):<10} "
            f"{float(summary.get('recall@5', 0)):>9.3f} "
            f"{float(summary.get('recall@10', 0)):>10.3f} "
            f"{float(summary.get('recall@50', 0)):>10.3f} "
            f"{float(summary.get('mrr@5', 0)):>7.3f} "
            f"{float(summary.get('median_target_rank', 0)):>8.1f} "
            f"{float(summary.get('top5_noise_ratio', 0)):>7.2f}"
        )
    lines.append("")
    for summary in summaries:
        verdict = summary.get("verdict") or {}
        lines.append(
            f"[{summary.get('variant')}] reranker needed: {verdict.get('needs_reranker')} "
            f"({'; '.join(verdict.get('reasons') or []) or 'within thresholds'})"
        )
    return "\n".join(lines)


def format_details(outcomes: Sequence[SampleOutcome], variant: str) -> str:
    lines = [f"--- {variant} per-question ---"]
    for outcome in outcomes:
        if not outcome.answerable:
            continue
        ranks = ", ".join(
            f"p{match.page}={'rank ' + str(match.rank) if match.rank else 'missing'}"
            for match in outcome.matches
        )
        lines.append(f"- {outcome.question[:60]} | {ranks}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval-only probe (no answer model).")
    parser.add_argument("--questions", default="test_data/evaluation_questions.jsonl")
    parser.add_argument("--k", type=int, default=50, help="retrieval window measured per query")
    parser.add_argument("--top-k", type=int, default=5, help="window used for the noise metric")
    parser.add_argument(
        "--variants",
        default=f"{VARIANT_BASELINE},{VARIANT_CLAUSES}",
        help="comma separated: baseline, clauses, planned, routed (routed = the agent's route-then-retrieve call)",
    )
    parser.add_argument("--with-llm-planning", action="store_true", help="allow the planner to call the LLM")
    parser.add_argument(
        "--scope-from-collection",
        default="",
        help="agent collection id: restrict retrieval to its documents, as the adapter does",
    )
    parser.add_argument("--agent-db", default="data/app.db", help="agent SQLite path used to resolve the collection scope")
    parser.add_argument("--dump", default="", help="write per-question outcomes to this JSONL file")
    parser.add_argument("--details", action="store_true", help="print per-question ranks")
    args = parser.parse_args(argv)

    from src.retrieval.persistent_retriever import PersistentRetriever

    samples = load_probe_samples(args.questions)
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    planner = QueryPlanner() if args.with_llm_planning else _terms_only_planner()
    document_ids = collection_document_ids(args.agent_db, args.scope_from_collection) if args.scope_from_collection else None
    if document_ids:
        print(f"scope: collection {args.scope_from_collection} -> {len(document_ids)} documents")

    summaries: list[dict[str, object]] = []
    for variant in variants:
        outcomes, summary = run_variant(
            variant,
            samples,
            retriever_factory=PersistentRetriever,
            k=args.k,
            top_k=args.top_k,
            planner=planner,
            with_llm_planning=args.with_llm_planning,
            document_ids=document_ids,
        )
        summary["scope_document_ids"] = sorted(document_ids) if document_ids else None
        summary["questions_file"] = str(args.questions)
        summaries.append(summary)
        if args.details:
            print(format_details(outcomes, variant))
        if args.dump:
            target = Path(args.dump)
            with target.open("a", encoding="utf-8", newline="\n") as stream:
                for outcome in outcomes:
                    stream.write(json.dumps({
                        "variant": variant,
                        "question": outcome.question,
                        "answerable": outcome.answerable,
                        "ranks": [
                            {"page": match.page, "requirement": match.requirement, "rank": match.rank}
                            for match in outcome.matches
                        ],
                        "top_hits": outcome.top_hits,
                        "noise_ratio": outcome.noise_ratio,
                    }, ensure_ascii=False) + "\n")

    print(format_report(summaries))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
