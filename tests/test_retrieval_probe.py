from types import SimpleNamespace

from src.evaluation.retrieval_probe import (
    ProbeSample,
    ProbeTarget,
    RERANK_MEDIAN_RANK_THRESHOLD,
    first_target_rank,
    format_report,
    load_probe_samples,
    rank_targets,
    reranker_verdict,
    summarize,
    SampleOutcome,
    TargetMatch,
)


def _results(*pairs):
    return [SimpleNamespace(filename=pair[0], page=pair[1], score=0.5) for pair in pairs]


def test_rank_targets_reports_the_position_of_each_required_page():
    sample = ProbeSample(
        question="训练数据、硬件和训练时间分别是什么？",
        documents=("attention.pdf",),
        targets=(ProbeTarget(7, "训练数据"), ProbeTarget(8, "硬件")),
    )
    results = _results(("attention.pdf", 3), ("attention.pdf", 7), ("attention.pdf", 12))
    matches = rank_targets(results, sample)
    assert [(match.page, match.rank) for match in matches] == [(7, 2), (8, None)]
    assert [match.requirement for match in matches] == ["训练数据", "硬件"]
    assert first_target_rank(results, sample) == 2


def test_rank_targets_ignores_other_documents_claiming_the_same_page():
    sample = ProbeSample(question="q", documents=("rag.pdf",), targets=(ProbeTarget(6),))
    results = _results(("attention.pdf", 6), ("rag.pdf", 6))
    assert rank_targets(results, sample)[0].rank == 2


def test_summarize_computes_recall_mrr_and_noise():
    outcomes = [
        SampleOutcome(question="a", answerable=True, matches=[TargetMatch("r", 7, rank=3)],
                      top_hits=[("attention.pdf", 7, 0.5), ("attention.pdf", 7, 0.4)],
                      noise_ratio=0.0),
        SampleOutcome(question="b", answerable=True, matches=[TargetMatch("r", 6, rank=9)],
                      top_hits=[("rag.pdf", 6, 0.5), ("rag.pdf", 99, 0.4)],
                      noise_ratio=0.5),
        SampleOutcome(question="c", answerable=False, matches=[], top_hits=[], noise_ratio=0.0),
    ]
    summary = summarize(outcomes, k=50, top_k=5)
    assert summary["questions"] == 3 and summary["answerable"] == 2
    assert summary["recall@5"] == 0.5
    assert summary["recall@10"] == 1.0
    assert summary["targets_found"] == 2
    assert summary["median_target_rank"] == 6.0
    assert summary["top5_noise_ratio"] == 0.25
    verdict = summary["verdict"]
    # median 6 <= 10 but noise 25% >= 20% -> ordering is still a problem.
    assert verdict["needs_reranker"] is True
    assert any("noise" in reason for reason in verdict["reasons"])


def test_verdict_prefers_recall_work_when_targets_are_missing_entirely():
    verdict = reranker_verdict(51.0, 0.4, k=50)
    assert verdict["needs_reranker"] is False
    assert any("missing beyond" in reason for reason in verdict["reasons"])


def test_verdict_is_negative_when_ranking_is_already_good():
    verdict = reranker_verdict(2.0, 0.05, k=50)
    assert verdict["needs_reranker"] is False
    assert verdict["reasons"] == []
    assert verdict["thresholds"]["median_target_rank"] == RERANK_MEDIAN_RANK_THRESHOLD


def test_probe_samples_accept_authored_targets_and_legacy_pages():
    authored = ProbeSample.from_dict({
        "question": "q1",
        "answerable": True,
        "expected_documents": ["a.pdf"],
        "expected_pages": [7, 8],
        "targets": [{"page": 7, "requirement": "data"}, {"page": 8, "requirement": "hardware"}],
    })
    assert [target.page for target in authored.targets] == [7, 8]
    assert authored.target_pages == {7, 8}
    legacy = ProbeSample.from_dict({
        "question": "q2", "answerable": True, "expected_documents": ["a.pdf"], "expected_pages": [3],
    })
    assert [target.page for target in legacy.targets] == [3]


def test_load_probe_samples_reads_the_project_evaluation_set():
    samples = load_probe_samples("test_data/evaluation_questions.jsonl")
    assert len(samples) >= 20
    answerable = [sample for sample in samples if sample.answerable]
    assert answerable and all(sample.targets for sample in answerable)


def test_format_report_prints_the_comparison_row_and_verdict():
    summaries = [
        {"variant": "baseline", "recall@5": 0.41, "recall@10": 0.5, "recall@50": 0.8,
         "mrr@5": 0.3, "median_target_rank": 12.0, "top5_noise_ratio": 0.3,
         "verdict": {"needs_reranker": True, "reasons": ["median target rank 12.0 > 10"]}},
        {"variant": "clauses", "recall@5": 0.6, "recall@10": 0.7, "recall@50": 0.9,
         "mrr@5": 0.5, "median_target_rank": 4.0, "top5_noise_ratio": 0.1,
         "verdict": {"needs_reranker": False, "reasons": []}},
    ]
    report = format_report(summaries)
    assert "baseline" in report and "clauses" in report
    assert "reranker needed: True" in report and "reranker needed: False" in report
    # D19: the keyword channel is gone, so the report has no keyword-only column.
    assert "lexical" not in report.lower()
    assert "0.410" in report and "0.600" in report
