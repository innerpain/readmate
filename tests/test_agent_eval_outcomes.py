"""Rules and denominators of the agent evaluation (no model, no services).

These tests exist because the 2026-09-15 run published three wrong numbers, and
each wrong number was a rule, not a typo.  Every test below pins one of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import outcomes as O


def _answer_row(index: int, *, expect: str = O.EXPECT_ANSWER, **overrides) -> dict:
    row = {
        "index": index,
        "question": f"q{index}",
        "expect": expect,
        "answerable": expect == O.EXPECT_ANSWER,
        "status": "ok",
        "refused": False,
        "failure": None,
        "answer": "an answer",
        "citations": [{"chunk_id": "c1", "filename": "doc.pdf", "page": 3}],
        "citation_hit": True,
        "gate": {"blocks": 0, "searched": True, "read": False},
        "number_coverage": None,
        "citation_dropped": 0,
        "rounds_total": 2,
        "tool_calls_total": 1,
        "max_steps": 6,
        "elapsed_s": 10.0,
        "trace": [{"tool": "search", "track": "direct"}],
    }
    row.update(overrides)
    return row


# ------------------------------------------------------------------ numerics
def test_number_coverage_returns_none_without_numbers_in_the_reference():
    """The 14 > 12 bug: a missing denominator was silently scored as 1.0."""

    coverage, missing = O.number_coverage("没有数字的参考答案", "随便什么回答")
    assert coverage is None
    assert missing == []


def test_number_coverage_counts_only_reference_numbers():
    coverage, missing = O.number_coverage("N=6 层、d_model=512、8 个注意力头", "它有 6 层，d_model 是 512")
    assert coverage == 0.667          # the metric rounds to three decimals
    assert missing == ["8"]


def test_summary_denominator_excludes_rows_without_numbers():
    rows = [
        _answer_row(1, number_coverage=1.0),
        _answer_row(2, number_coverage=0.0),
        _answer_row(3, number_coverage=None),   # no numbers in the reference at all
    ]
    summary = O.summarize(rows)
    assert summary["numbers"]["coverage_ge_half"] == {
        "n": 1, "of": 2, "denom": "expect=answer 且参考答案含数字",
    }
    assert summary["numbers"]["rows_without_numbers"]["n"] == 1
    assert summary["numbers"]["rows_without_numbers"]["rows"] == [3]


# ------------------------------------------------------------- classification
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, O.OUTCOME_OK),
        ({"citation_hit": False}, O.OUTCOME_ADJACENT_PAGE),
        ({"citations": []}, O.OUTCOME_NO_CITATION),
        ({"answer": "  "}, O.OUTCOME_NO_ANSWER),
        ({"failure": "max_steps_without_answer", "answer": "本轮没有取到资料证据，无法给出有出处的回答。"},
         O.OUTCOME_BUDGET_EXHAUSTED),
        ({"failure": "no_evidence_after_gate", "refused": True, "citations": []}, O.OUTCOME_GATE_FAILED),
        ({"status": "transport_error"}, O.OUTCOME_TRANSPORT_ERROR),
    ],
)
def test_one_primary_outcome_per_row(overrides, expected):
    assert O.classify(_answer_row(1, **overrides)) == expected


def test_budget_exhaustion_is_not_counted_as_a_refusal_or_as_an_empty_citation():
    """One root cause used to be counted twice: false-refusal and empty-citation."""

    row = _answer_row(
        7,
        answer="本轮没有取到资料证据，无法给出有出处的回答。",
        citations=[],
        refused=True,
        failure="max_steps_without_answer",
        gate={"blocks": 0, "searched": True, "read": True},
    )
    summary = O.summarize([row])
    assert summary["outcome_counts"] == {O.OUTCOME_BUDGET_EXHAUSTED: 1}
    assert summary["answerable"]["budget_exhausted"]["n"] == 1
    assert summary["expect_deny"]["n"] == 0


def test_denial_answer_is_flagged_for_audit_not_counted_as_correct():
    row = _answer_row(4, expect=O.EXPECT_DENY, answer="论文没有报告中文到英文的 BLEU 分数。")
    assert O.classify(row) == O.OUTCOME_DENY_UNVERIFIED
    summary = O.summarize([row])
    assert summary["expect_deny"]["deny_unverified"] == 1
    assert summary["expect_deny"]["refuse_correct"] == 0
    assert summary["audit"]["rows"] == [4]


def test_off_domain_question_answered_anyway_is_flagged_as_unfounded():
    row = _answer_row(21, expect=O.EXPECT_REFUSE, answer="2024 年诺贝尔物理学奖授予了 Hopfield 与 Hinton。")
    assert O.classify(row) == O.OUTCOME_UNFOUNDED_ANSWER
    summary = O.summarize([row])
    assert summary["expect_refuse"]["unfounded_answer"] == 1
    assert summary["audit"]["rows"] == [21]


def test_structural_refusal_counts_as_correct_for_unanswerable_expectations():
    rows = [
        _answer_row(4, expect=O.EXPECT_DENY, refused=True, failure="no_evidence_after_gate", citations=[]),
        _answer_row(21, expect=O.EXPECT_REFUSE, refused=True, failure="no_evidence_after_gate", citations=[]),
    ]
    summary = O.summarize(rows)
    assert summary["expect_deny"]["refuse_correct"] == 1
    assert summary["expect_refuse"]["refuse_correct"] == 1
    assert summary["audit"]["count"] == 0


def test_legacy_rows_without_expect_fall_back_to_answerable():
    assert O.resolve_expect({"answerable": False}) == O.EXPECT_DENY
    assert O.resolve_expect({"answerable": True}) == O.EXPECT_ANSWER
    assert O.resolve_expect({"answerable": True, "expect": "deny"}) == O.EXPECT_DENY


# ------------------------------------------------------------------ summary
def test_summary_recomputes_from_rows_and_never_trusts_the_declared_outcome():
    rows = [_answer_row(1), _answer_row(2, citation_hit=False)]
    rows[0]["outcome"] = O.OUTCOME_ADJACENT_PAGE        # a stale/incorrect declaration
    summary = O.summarize(rows)
    assert summary["recompute_ok"] is False
    assert summary["outcome_counts"] == {O.OUTCOME_OK: 1, O.OUTCOME_ADJACENT_PAGE: 1}


def test_citation_hit_reports_both_denominators():
    rows = [
        _answer_row(1),
        _answer_row(2, citation_hit=False),
        _answer_row(3, citations=[], citation_hit=False, outcome=None),
        _answer_row(4, failure="max_steps_without_answer", citations=[], citation_hit=False),
    ]
    summary = O.summarize(rows)
    assert summary["answerable"]["citation_hit_all"] == {"n": 1, "of": 4, "denom": "expect=answer"}
    assert summary["answerable"]["citation_hit_answered"] == {
        "n": 1, "of": 3, "denom": "expect=answer 且本轮作答",
    }
    assert summary["answerable"]["answered"] == 3


def test_route_tracks_ignore_non_search_steps():
    rows = [
        _answer_row(1, trace=[
            {"tool": "search", "track": "direct"},
            {"tool": "read", "track": None},          # read has no route: not a track
            {"tool": "search", "track": "planned"},
        ]),
    ]
    tracks = O.summarize(rows)["trajectory"]["route_tracks_search_only"]
    assert tracks == {"direct": 1, "planned": 1}


# --------------------------------------------------------------- assertions
def test_trace_assertions_flag_the_contract_holes_we_know_about():
    rows = [
        _answer_row(7, failure="max_steps_without_answer", rounds_total=6, tool_calls_total=9),
        # D57: the hole is a *successful* page read that contributed nothing citable.
        # The bare attempt is not one -- see
        # test_page_read_is_only_a_hole_when_it_makes_nothing_citable.
        _answer_row(8, read_by_page=True, trace=[{
            "round": 2,
            "tool": "read",
            "arguments": {"document_id": "d1", "page": 3},
            "ok": True,
            "observed": [],
        }]),
        _answer_row(9, citation_dropped=2),
        _answer_row(10, gate={"blocks": 0, "searched": False, "read": False}),
    ]
    checks = {check["name"]: check for check in O.trace_assertions(rows)}
    assert checks["step_budget_exhausted_rows"]["detail"]["rows"] == [7]
    assert checks["read_by_page_contract_hole"]["detail"]["rows"] == [8]
    assert checks["citations_dropped_as_unobserved"]["detail"]["rows"] == [9]
    assert checks["answered_without_a_successful_search"]["detail"]["rows"] == [10]
    assert checks["answered_without_a_successful_search"]["ok"] is False


def test_failed_tool_calls_are_flagged_with_the_error_code():
    """A failed read still burns the step budget: seen as document_not_found on 2026-09-16."""

    rows = [
        _answer_row(9, trace=[
            {"tool": "search", "track": "direct", "ok": True},
            {"tool": "read", "ok": False, "error": "document_not_found",
             "arguments": {"page": 5, "document_id": "sentence-bert.pdf"}},
        ]),
    ]
    checks = {check["name"]: check for check in O.trace_assertions(rows)}
    assert checks["failed_tool_calls"]["ok"] is False
    assert checks["failed_tool_calls"]["detail"]["rows"] == [9]
    assert checks["failed_tool_calls"]["detail"]["calls"] == ["read:document_not_found"]


def test_repeated_identical_calls_are_flagged_but_legacy_rows_without_arguments_are_not():
    same_read = {"tool": "read", "ok": True, "arguments": {"chunk_id": "chk_1"}}
    repeated = _answer_row(17, trace=[dict(same_read) for _ in range(5)])
    legacy = _answer_row(7, trace=[{"tool": "read", "ok": True, "arguments": {}} for _ in range(6)])

    checks = {check["name"]: check for check in O.trace_assertions([repeated])}
    assert checks["repeated_identical_calls"]["ok"] is False
    assert checks["repeated_identical_calls"]["detail"]["detail"] == [{"index": 17, "tool": "read", "repeats": 5}]

    legacy_checks = {check["name"]: check for check in O.trace_assertions([legacy])}
    assert legacy_checks["repeated_identical_calls"]["ok"] is True


def test_audit_rows_are_unverified_until_a_verdict_file_says_otherwise():
    rows = [
        _answer_row(4, expect=O.EXPECT_DENY, answer="论文没有报告该结果。"),
        _answer_row(21, expect=O.EXPECT_REFUSE, answer="资料中没有该内容。"),
        _answer_row(22, expect=O.EXPECT_REFUSE, answer="2024 年诺贝尔物理学奖得主是 Hopfield 与 Hinton。"),
    ]
    before = O.summarize(rows)["audit"]
    assert before["count"] == 3
    assert before["verdicts"] == {"correct": 0, "hallucinated": 0, "other": 0, "unverified": 3}

    verdicts = {4: {"verdict": "correct", "note": "人工核对：正确否证"},
                21: {"verdict": "correct", "note": ""},
                22: {"verdict": "hallucinated", "note": "答了资料外的事实"}}
    after = O.summarize(rows, verdicts=verdicts)["audit"]
    assert after["verdicts"] == {"correct": 2, "hallucinated": 1, "other": 0, "unverified": 0}
    assert after["detail"][0] == {"index": 4, "expect": "deny", "outcome": "deny_unverified",
                                  "verdict": "correct", "note": "人工核对：正确否证"}
    # a verdict for a row outside the audit list is surfaced, never silently dropped
    stray = O.summarize(rows, verdicts={99: {"verdict": "correct"}})["audit"]["stray_verdict_indices"]
    assert stray == [99]


def test_verdict_file_rejects_an_unknown_verdict(tmp_path: Path):
    path = tmp_path / "verdicts.json"
    path.write_text(json.dumps({"verdicts": {"4": {"verdict": "probably-fine"}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        O.load_verdicts(path)


def test_verdict_file_accepts_a_flat_mapping_and_string_verdicts(tmp_path: Path):
    path = tmp_path / "verdicts.json"
    path.write_text(json.dumps({"4": "correct"}), encoding="utf-8")
    assert O.load_verdicts(path) == {4: {"verdict": "correct", "note": ""}}


def test_audit_markdown_prints_the_recorded_verdict():
    from eval import report as R

    rows = [_answer_row(4, expect=O.EXPECT_DENY, answer="论文没有报告该结果。")]
    summary = O.summarize(rows, verdicts={4: {"verdict": "correct", "note": "人工核对通过"}})
    text = R.audit_markdown(rows, summary)
    assert "-> correct" in text
    assert "人工核对通过" in text


# ------------------------------------------------------------------ loading
def test_load_samples_assigns_stable_indices_across_files(tmp_path: Path):
    core = tmp_path / "core.jsonl"
    extra = tmp_path / "extra.jsonl"
    core.write_text("\n".join(json.dumps({"question": f"q{i}", "answerable": True}) for i in (1, 2)) + "\n",
                    encoding="utf-8")
    extra.write_text(json.dumps({"question": "q3", "answerable": False, "expect": "refuse"}) + "\n", encoding="utf-8")
    samples = O.load_samples([core, extra])
    assert [sample["index"] for sample in samples] == [1, 2, 3]
    assert O.resolve_expect(samples[2]) == O.EXPECT_REFUSE


def test_load_samples_rejects_a_contradictory_expect(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"question": "q", "answerable": False, "expect": "answer"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        O.load_samples([path])


def test_unknown_expect_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"question": "q", "expect": "maybe"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        O.load_samples([path])


# ------------------------------------------------------------------- guard
def test_output_inside_data_is_refused(tmp_path: Path):
    repo = tmp_path
    (repo / "data").mkdir()
    with pytest.raises(ValueError):
        O.assert_writable_output(repo / "data" / "eval.jsonl", repo_root=repo)
    assert O.assert_writable_output(repo / "tmp" / "eval.jsonl", repo_root=repo).name == "eval.jsonl"


# --------------------------------------------------------------- A12 refusal
def test_declared_refusal_counts_as_correct_for_refuse_and_deny():
    """A12: the model's own refusal flag is the native signal the harness wanted.

    Measured before the fix (2026-09-19 baseline): three `expect=refuse` rows were
    textually correct refusals ("资料库中没有…") yet scored `unfounded_answer`,
    because the runtime never set `refused` for a self-declined turn.
    """

    refused = _answer_row(21, expect=O.EXPECT_REFUSE, refused=True, answer="资料库中没有该信息。", citations=[])
    assert O.classify(refused) == O.OUTCOME_REFUSE_CORRECT

    denied = _answer_row(4, expect=O.EXPECT_DENY, refused=True, citations=[])
    assert O.classify(denied) == O.OUTCOME_REFUSE_CORRECT


def test_undeclared_refusal_is_still_audited():
    """The old behaviour must survive for turns that do not declare anything --
    dropping them from the audit would hide hallucinated denials."""

    prose_only = _answer_row(21, expect=O.EXPECT_REFUSE, refused=False, answer="资料库中没有该信息。", citations=[])
    assert O.classify(prose_only) == O.OUTCOME_UNFOUNDED_ANSWER

    denial = _answer_row(4, expect=O.EXPECT_DENY, refused=False, citations=[])
    assert O.classify(denial) == O.OUTCOME_DENY_UNVERIFIED


def test_false_refusal_on_an_answerable_question_is_no_answer():
    """A refusal where an answer was possible is a *false* refusal: filing it as
    `no_citation` (answered, uncited) understated it."""

    row = _answer_row(5, refused=True, answer="无法回答。", citations=[], citation_hit=False)
    assert O.classify(row) == O.OUTCOME_NO_ANSWER

    uncited = _answer_row(6, refused=False, citations=[], citation_hit=False)
    assert O.classify(uncited) == O.OUTCOME_NO_CITATION

def test_page_read_is_only_a_hole_when_it_makes_nothing_citable():
    """D57: the check used to flag *any* page-based read, quoting a hole A3 had closed.

    A page read returns its rows' ``chunk_ids`` (measured on a live adapter: three
    citable ids), so the attempt itself is fine.  The real hole is a page read that
    SUCCEEDED while contributing nothing citable -- the model then holds evidence it
    cannot cite.  A *failed* page read is a trajectory problem, counted by
    ``failed_tool_calls``, not a citation-contract hole.
    """

    def page_read(**overrides) -> list[dict]:
        step = {
            "round": 2,
            "tool": "read",
            "arguments": {"document_id": "d1", "page": 3},
            "ok": True,
            "error": None,
            "observed": ["c1", "c2"],
        }
        step.update(overrides)
        return [step]

    rows = [
        _answer_row(1, read_by_page=True, trace=page_read()),                                  # citable
        _answer_row(2, read_by_page=True, trace=page_read(observed=[])),                        # the hole
        _answer_row(3, read_by_page=True, trace=page_read(ok=False, error="document_not_found", observed=[])),
        _answer_row(4, trace=[{"round": 2, "tool": "read", "arguments": {"chunk_id": "c1"},
                               "ok": True, "observed": ["c1"]}]),                               # not a page read
    ]
    check = {item["name"]: item for item in O.trace_assertions(rows)}["read_by_page_contract_hole"]

    assert check["ok"] is False
    assert check["detail"]["rows"] == [2], "only the page read that made nothing citable"
    assert check["detail"]["page_read_rows"] == [1, 2, 3], "attempts stay visible for context"


def test_page_fidelity_accepts_any_page_a_chunk_spans():
    """D59: a chunk can span pages, so a citation naming any of them is faithful.

    The check compared the model's page against the chunk's *start* page only, which
    flagged a legitimate page-5 citation of a 4-5 chunk (row #12, sentence-bert.pdf).
    """

    spanning = {"page": 4, "page_end": 5, "model_page": 5, "page_fidelity": False}  # stale flag
    assert O._page_fidelity_mismatch(spanning) is False
    assert O._page_fidelity_mismatch({"page": 4, "page_end": 5, "model_page": 4}) is False
    assert O._page_fidelity_mismatch({"page": 4, "page_end": 5, "model_page": 6}) is True
    # single-page chunks still behave as before
    assert O._page_fidelity_mismatch({"page": 7, "page_end": 7, "model_page": 7}) is False
    assert O._page_fidelity_mismatch({"page": 7, "page_end": 7, "model_page": 3}) is True
    # rows written before those fields existed fall back to their stored flag
    assert O._page_fidelity_mismatch({"page_fidelity": True}) is False
    assert O._page_fidelity_mismatch({"page_fidelity": False}) is True
