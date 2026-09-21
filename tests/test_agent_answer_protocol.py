"""The answer protocol must never hand the wire format back as an answer.

Regression origin (2026-09-19): the model streams ``{"answer": ..., "citations":
[...], "non_source": []}``; when that envelope did not parse (truncated output,
or prose followed by a malformed object) ``parse_agent_answer`` returned the raw
text, so the user's answer bubble *and* the stored row showed the JSON keys --
and the table the model had written inside ``answer`` disappeared at turn end.
"""

from __future__ import annotations

from src.agent.answer_protocol import parse_agent_answer, validate_citations


def test_well_formed_envelope_parses_without_warnings() -> None:
    raw = (
        '{"answer": "六个相同层。", "citations": '
        '[{"chunk_id": "c1", "page": 3, "quote": "six"}], "non_source": []}'
    )
    answer = parse_agent_answer(raw)
    assert answer.text == "六个相同层。"
    assert answer.warnings == []
    assert [c.chunk_id for c in answer.citations] == ["c1"]
    assert answer.citations[0].page == 3
    assert answer.refused is False, "an envelope without the flag is not a refusal"


# ---------------------------------------------------------------- A12 refusal
def test_declared_refusal_is_read_from_the_envelope() -> None:
    """A12: a polite "资料库中没有…" answer used to be indistinguishable from a
    sourced one; the model now declares the refusal in the envelope."""

    raw = '{"answer": "资料库中没有该信息。", "citations": [], "non_source": [], "refused": true}'
    answer = parse_agent_answer(raw)
    assert answer.refused is True
    assert answer.text == "资料库中没有该信息。"


def test_refusal_flag_accepts_the_sloppy_spellings_a_model_produces() -> None:
    # Every variant below is valid JSON -- a bare  yes  /  True  would break the
    # whole envelope, which is a different (already covered) case.
    for value in ("true", '"true"', "1", '"1"', '"yes"', '" True "', '"true "'):
        raw = f'{{"answer": "x", "citations": [], "non_source": [], "refused": {value}}}'
        assert parse_agent_answer(raw).refused is True, value
    for value in ("false", '"false"', "0", '"0"', '"no"', '"null"', "null"):
        raw = f'{{"answer": "x", "citations": [], "non_source": [], "refused": {value}}}'
        assert parse_agent_answer(raw).refused is False, value


def test_refusal_survives_citation_validation() -> None:
    raw = '{"answer": "没有 Mamba 的资料。", "citations": [{"chunk_id": "c1", "page": 1}], "refused": true}'
    validated = validate_citations(parse_agent_answer(raw), {"c1": {"document_id": "d1", "filename": "a.pdf", "page": 1}})
    assert validated.refused is True
    assert [c.chunk_id for c in validated.citations] == ["c1"]


def test_unparsable_envelope_is_not_a_refusal() -> None:
    assert parse_agent_answer("just prose, no envelope").refused is False
    assert parse_agent_answer('{"answer": "cut off", "citations":').refused is False


def test_system_prompt_asks_for_the_flag() -> None:
    """The contract is part of the fix -- a prompt without the key would silently
    drop the signal for every future turn."""

    from src.agent.runtime import _system_prompt

    assert '"refused": false' in _system_prompt("deep")
    assert "refused" in _system_prompt("chat")


def test_truncated_envelope_recovers_the_answer_text() -> None:
    """A response cut off mid-envelope must yield the answer, not the envelope."""

    raw = '{"answer": "Table 2 有 5 列，ByteNet 23.75", "citations": [{"chunk_id": "c1"'
    answer = parse_agent_answer(raw)
    assert answer.text == "Table 2 有 5 列，ByteNet 23.75"
    assert answer.warnings == ["answer_recovered_partial"]
    assert answer.citations == []
    assert '"citations"' not in answer.text
    assert '"non_source"' not in answer.text


def test_prose_then_malformed_envelope_keeps_only_the_answer_field() -> None:
    """Prose + a broken object: the recovered field is the answer; neither the
    preamble nor the envelope keys leak into it."""

    raw = '找到了。\n{"answer": "这是答案", "citations": [bad'
    answer = parse_agent_answer(raw)
    assert answer.text == "这是答案"
    assert answer.warnings == ["answer_recovered_partial"]


def test_escaped_newlines_are_unescaped() -> None:
    """A streamed envelope escapes newlines; the recovered answer must carry real
    ones, or a markdown table inside ``answer`` renders as literal ``\\n``."""

    raw = '{"answer": "line1\\nline2\\ttab", "citations": [bad'
    answer = parse_agent_answer(raw)
    assert answer.text == "line1\nline2\ttab"


def test_incomplete_escape_at_the_tail_is_dropped() -> None:
    """The stream can stop inside an escape sequence; that fragment is not text."""

    raw = '{"answer": "完整部分\\u51'
    answer = parse_agent_answer(raw)
    assert answer.text == "完整部分"
    assert answer.warnings == ["answer_recovered_partial"]


def test_plain_prose_is_untouched() -> None:
    """Chat-mode answers arrive as plain prose -- nothing to recover, nothing to
    change (this is the pre-existing contract)."""

    answer = parse_agent_answer("你好！我是 ReadMate。")
    assert answer.text == "你好！我是 ReadMate。"
    assert answer.warnings == ["answer_not_json"]


def test_object_without_an_answer_key_keeps_the_prose() -> None:
    """An object that parses but has no ``answer`` is not a response: keep the
    prose around it rather than returning an empty bubble."""

    raw = '总结如下：{"citations": [], "non_source": []}'
    answer = parse_agent_answer(raw)
    assert answer.text == "总结如下："
    assert answer.warnings == ["answer_missing_in_envelope"]


def test_envelope_with_no_answer_never_leaks_the_wire_format() -> None:
    """Degenerate case: an envelope-only response yields no text at all -- the
    keys must not reach the bubble."""

    answer = parse_agent_answer('{"citations": [], "non_source": []}')
    assert answer.text == ""
    assert '"citations"' not in answer.text
    assert answer.warnings == ["answer_missing_in_envelope"]


def test_recovered_partial_answer_still_goes_through_citation_validation() -> None:
    """Recovery must not bypass the observed-chunk gate (no citations were parsed,
    and nothing invented may survive)."""

    from src.agent.answer_protocol import AgentAnswer, CitationOut

    answer = parse_agent_answer('{"answer": "正文", "citations": [{"chunk_id": "ghost"')
    assert answer.citations == []
    invented = AgentAnswer(text="x", citations=[CitationOut(chunk_id="ghost")])
    validated = validate_citations(invented, observed={})
    assert validated.citations == []
    assert "citations_dropped_unobserved" in validated.warnings