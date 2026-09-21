"""Outcome classification and metrics: pure rules first, judge only as a hook.

Why this module exists in the shape it does
-------------------------------------------
The 2026-09-15 run of the agent path (20 questions) produced a summary that was
wrong in three separate ways, and each one is a class of mistake, not a typo:

1. ``unanswerable_refused = 0`` read as "all three hallucinated", while all
   three had given a correct denial ("the paper does not report MMLU").  The
   harness matched text copied from other paths instead of reading the native
   signal (``refused`` / ``failure`` / ``gate``).  -> ``refused`` is a field, not
   a string search.

2. One root cause was counted twice: the three ``max_steps_without_answer``
   turns were reported both as ``answerable_false_refusal`` and as
   ``answerable_empty_citations``.  -> each row gets exactly one ``outcome``.

3. ``answerable_number_coverage_ge_half = 14`` while only 12 questions even had
   numbers in the reference answer: the five questions without numbers defaulted
   to coverage 1.0, so the metric could exceed its own population.  -> a ratio
   without an explicit denominator is a bug; ``number_coverage`` returns
   ``None``, never a silent 1.0.

Nothing here calls a model.  ``judge_hook`` documents where an LLM judge would
go; the residual cases are reported as *unaudited* rather than counted as
correct, because a number nobody verified is worse than a missing number.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

HARNESS_VERSION = "agent-eval-2"

# ---------------------------------------------------------------- expectations
EXPECT_ANSWER = "answer"      # the corpus has the answer
EXPECT_DENY = "deny"          # the corpus has no such fact; a confirmed denial is right
EXPECT_REFUSE = "refuse"      # the question is off-domain; the right move is to decline
EXPLICIT_EXPECTS = (EXPECT_ANSWER, EXPECT_DENY, EXPECT_REFUSE)

# -------------------------------------------------------------------- outcomes
OUTCOME_OK = "ok"
OUTCOME_ADJACENT_PAGE = "adjacent_page"
OUTCOME_NO_CITATION = "no_citation"
OUTCOME_NO_ANSWER = "no_answer"
OUTCOME_BUDGET_EXHAUSTED = "budget_exhausted"
OUTCOME_GATE_FAILED = "gate_failed"
OUTCOME_TRANSPORT_ERROR = "transport_error"
OUTCOME_REFUSE_CORRECT = "refuse_correct"
OUTCOME_DENY_UNVERIFIED = "deny_unverified"
OUTCOME_UNFOUNDED_ANSWER = "unfounded_answer"

OUTCOMES = (
    OUTCOME_OK,
    OUTCOME_ADJACENT_PAGE,
    OUTCOME_NO_CITATION,
    OUTCOME_NO_ANSWER,
    OUTCOME_BUDGET_EXHAUSTED,
    OUTCOME_GATE_FAILED,
    OUTCOME_TRANSPORT_ERROR,
    OUTCOME_REFUSE_CORRECT,
    OUTCOME_DENY_UNVERIFIED,
    OUTCOME_UNFOUNDED_ANSWER,
)

# Outcomes that mean "the turn produced an answer the user can read".
ANSWERED_OUTCOMES = frozenset(
    {OUTCOME_OK, OUTCOME_ADJACENT_PAGE, OUTCOME_NO_CITATION, OUTCOME_DENY_UNVERIFIED, OUTCOME_UNFOUNDED_ANSWER}
)
# Outcomes whose correctness a deterministic rule cannot establish on its own.
AUDIT_OUTCOMES = frozenset({OUTCOME_DENY_UNVERIFIED, OUTCOME_UNFOUNDED_ANSWER})

# Failure codes from src/agent/runtime.py -- asserted names, not guessed wording.
FAILURE_BUDGET = "max_steps_without_answer"
FAILURE_GATE = "no_evidence_after_gate"

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# A tool called with identical arguments this many times in one turn is a
# degenerate retry (seen on 2026-09-16: one question read the same chunk 5x and
# then died on the step budget), not exploration.
REPEAT_CALL_THRESHOLD = 3

# An audit row is a *flag*, not a verdict: rules can see that a turn answered
# when a denial was expected, but not whether the denial was real.  A verdict
# file (``--audit-verdicts``) records the human/judge conclusion so the row stops
# being "unverified" instead of being silently promoted to correct.
VERDICT_CORRECT = "correct"
VERDICT_HALLUCINATED = "hallucinated"
VERDICT_OTHER = "other"
VERDICT_UNVERIFIED = "unverified"
VERDICTS = (VERDICT_CORRECT, VERDICT_HALLUCINATED, VERDICT_OTHER)

# judge_hook: an LLM judge would be called here for AUDIT_OUTCOMES only.  It is
# intentionally not implemented: rules decide everything else for free, and an
# unverified number must be published as "unaudited", never as "correct".
judge_hook = None


# ------------------------------------------------------------------- numerics
def numbers(text: str) -> set[str]:
    return {match.group(0).replace(",", "").rstrip(".") for match in NUMBER_RE.finditer(text or "")}


def number_coverage(reference: str, answer: str) -> tuple[float | None, list[str]]:
    """Fraction of the reference's numbers present in the answer.

    ``None`` when the reference has no numbers at all -- the caller must decide
    what that means, instead of inheriting a free 1.0 (the 14 > 12 bug).
    """

    ref = numbers(reference)
    if not ref:
        return None, []
    found = {value for value in ref if value in numbers(answer)}
    return round(len(found) / len(ref), 3), sorted(ref - found)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[position])


# ------------------------------------------------------------------ sample I/O
def load_samples(paths: Sequence[str | Path]) -> list[dict]:
    """Load one or more JSONL question files, assigning a stable index each.

    The index is assigned across the concatenated list, so passing the frozen
    core set first keeps indices 1..N comparable with earlier baselines; extra
    probe files continue after it.
    """

    samples: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise ValueError(f"question file not found: {path}")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is not valid JSON") from error
            if not isinstance(payload, dict) or not str(payload.get("question") or "").strip():
                raise ValueError(f"{path}:{number} needs a non-empty question")
            expect = payload.get("expect")
            if expect is not None and expect not in EXPLICIT_EXPECTS:
                raise ValueError(f"{path}:{number} has unknown expect={expect!r}")
            if expect == EXPECT_ANSWER and not payload.get("answerable", True):
                raise ValueError(f"{path}:{number} expect=answer contradicts answerable=false")
            samples.append(dict(payload))
    if not samples:
        raise ValueError("question set is empty")
    for position, sample in enumerate(samples, start=1):
        sample["index"] = position
    return samples


def load_verdicts(path: str | Path) -> dict[int, dict]:
    """Load the audit verdict file: index -> {"verdict", "note"}.

    Accepts either ``{"verdicts": {...}}`` or a flat mapping.  Anything that is
    not one of the known verdicts is rejected rather than defaulted, so a typo
    cannot quietly become "correct".
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("verdicts") if isinstance(payload, dict) and "verdicts" in payload else payload
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: verdict file must be an object keyed by question index")
    verdicts: dict[int, dict] = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: key {key!r} is not a question index") from error
        entry = value if isinstance(value, dict) else {"verdict": value}
        verdict = str(entry.get("verdict") or "").strip()
        if verdict not in VERDICTS:
            raise ValueError(f"{path}: index {index} has unknown verdict {verdict!r}")
        verdicts[index] = {"verdict": verdict, "note": str(entry.get("note") or "")}
    return verdicts


def assert_writable_output(path: str | Path, *, repo_root: Path) -> Path:
    """Evaluation output must never land inside ``data/`` (it holds the index)."""

    resolved = Path(path).resolve()
    data_root = Path(repo_root).resolve() / "data"
    if resolved == data_root or data_root in resolved.parents:
        raise ValueError(f"refusing to write evaluation output inside data/: {resolved}")
    return resolved


# ---------------------------------------------------------------- classification
def resolve_expect(row: dict) -> str:
    value = str(row.get("expect") or "").strip()
    if value in EXPLICIT_EXPECTS:
        return value
    # Legacy rows: answerable=false covers both "should deny" and "should refuse".
    return EXPECT_ANSWER if row.get("answerable", True) else EXPECT_DENY


def classify(row: dict) -> str:
    """Exactly one primary outcome per row, from native signals only."""

    if str(row.get("status") or "ok") != "ok":
        return OUTCOME_TRANSPORT_ERROR
    expect = resolve_expect(row)
    failure = row.get("failure")
    if failure == FAILURE_BUDGET:
        # Burning the whole budget is a trajectory defect whatever was expected:
        # an unanswerable question should decline quickly, not spin.
        return OUTCOME_BUDGET_EXHAUSTED
    if failure:
        # Declining is exactly what a deny/refuse sample wants; against an
        # answerable sample the same failure is a false refusal.
        return OUTCOME_GATE_FAILED if expect == EXPECT_ANSWER else OUTCOME_REFUSE_CORRECT

    answered = bool(str(row.get("answer") or "").strip())

    if expect == EXPECT_ANSWER:
        if not answered or row.get("refused"):
            # A12: on an answerable question a declared refusal is a *false*
            # refusal -- closer to "no answer" than to "answered without a
            # citation", which is what it used to be filed as.
            return OUTCOME_NO_ANSWER
        if not (row.get("citations") or []):
            return OUTCOME_NO_CITATION
        return OUTCOME_OK if row.get("citation_hit") else OUTCOME_ADJACENT_PAGE

    if row.get("refused") or not answered:
        # Declining is correct for deny/refuse, even when a fuller denial was possible.
        return OUTCOME_REFUSE_CORRECT
    return OUTCOME_UNFOUNDED_ANSWER if expect == EXPECT_REFUSE else OUTCOME_DENY_UNVERIFIED


def trace_assertions(rows: Sequence[dict]) -> list[dict]:
    """L2 invariants, evaluated over archived traces instead of re-run blindly.

    Only assertions with discriminating power are kept: "citations come from this
    turn" is enforced by ``validate_citations`` in code, so asserting it live can
    never fail and costs a full run to learn nothing.
    """

    checks: list[dict] = []

    answered_ok = [row for row in rows if classify(row) in ANSWERED_OUTCOMES]
    unsearched = [row["index"] for row in answered_ok if not (row.get("gate") or {}).get("searched")]
    checks.append({
        "name": "answered_without_a_successful_search",
        "ok": not unsearched,
        "detail": {"rows": unsearched, "of": len(answered_ok), "note": "deep 模式闸门要求答前有成功 search/read"},
    })

    over_budget = [
        row["index"] for row in rows
        if int(row.get("rounds_total") or 0) > int(row.get("max_steps") or 0) > 0
    ]
    checks.append({
        "name": "rounds_within_max_steps",
        "ok": not over_budget,
        "detail": {"rows": over_budget},
    })

    budget = [row["index"] for row in rows if row.get("failure") == FAILURE_BUDGET]
    checks.append({
        "name": "step_budget_exhausted_rows",
        "ok": not budget,
        "detail": {"rows": budget, "note": "证据已取到但收尾轮没预留 —— 归类为轨迹问题，不是召回问题"},
    })

    # D57 (2026-09-21): the criterion used to be "any page-based read at all", with a
    # note describing a hole A3 had already closed -- ``_read_page`` returns the page's
    # rows' ``chunk_ids`` and every one of them reaches ``observed_chunks`` (measured on
    # a live adapter: a page read yields 3 citable ids).  Flagging the *attempt* cried
    # wolf on every run, and it counted failed reads too.  The real hole is a page read
    # that SUCCEEDED while contributing nothing citable: the model then holds evidence
    # it cannot cite.  ``observed`` per step is recorded by ``AgentRuntime._record``.
    def _page_read_hole(row: dict) -> bool:
        for step in row.get("trace") or []:
            if step.get("tool") != "read":
                continue
            if (step.get("arguments") or {}).get("chunk_id"):
                continue  # a read by chunk id, not a page read
            if step.get("ok") is False:
                continue  # a failed read belongs to failed_tool_calls, not here
            if not (step.get("observed") or []):
                return True  # succeeded, yet made nothing citable
        return False

    by_page_reads = [row["index"] for row in rows if row.get("read_by_page")]
    holes = [row["index"] for row in rows if _page_read_hole(row)]
    checks.append({
        "name": "read_by_page_contract_hole",
        "ok": not holes,
        "detail": {
            "rows": holes,
            "page_read_rows": by_page_reads,
            "note": "按页 read 成功却没产出任何可引用 chunk → 该页证据无法被引用（这才是真正的洞）",
        },
    })

    failed = [
        (row["index"], str(step.get("tool")), str(step.get("error")))
        for row in rows for step in (row.get("trace") or []) if step.get("ok") is False
    ]
    checks.append({
        "name": "failed_tool_calls",
        "ok": not failed,
        "detail": {
            "rows": sorted({index for index, _, _ in failed}),
            "calls": [f"{tool}:{error}" for _, tool, error in failed],
            "note": "失败调用照样消耗步数预算；document_not_found 出现过一次『把文件名当 document_id 传给 read』",
        },
    })

    repeated: list[dict] = []
    for row in rows:
        signatures: Counter[tuple[str, str]] = Counter()
        for step in row.get("trace") or []:
            arguments = step.get("arguments") or {}
            # Legacy rows keep no tool arguments, so repetition is unjudgeable --
            # skip instead of counting every call as an identical repeat.
            if not arguments:
                continue
            key = (str(step.get("tool")), json.dumps(arguments, sort_keys=True, ensure_ascii=False))
            signatures[key] += 1
        if signatures:
            (tool, _), count = signatures.most_common(1)[0]
            if count >= REPEAT_CALL_THRESHOLD:
                repeated.append({"index": row["index"], "tool": tool, "repeats": count})
    checks.append({
        "name": "repeated_identical_calls",
        "ok": not repeated,
        "detail": {
            "rows": [item["index"] for item in repeated],
            "detail": repeated,
            "note": f"同一工具同一参数重复 >= {REPEAT_CALL_THRESHOLD} 次是退化重试，不是探索",
        },
    })

    dropped = [row["index"] for row in rows if int(row.get("citation_dropped") or 0) > 0]
    checks.append({
        "name": "citations_dropped_as_unobserved",
        "ok": not dropped,
        "detail": {"rows": dropped, "note": "模型引用了本轮未观察到的 chunk —— 幻觉引用的结构化信号"},
    })

    timing = any(row.get("timing_available") for row in rows)
    checks.append({
        "name": "per_round_timing_recorded",
        "ok": bool(timing),
        "detail": {"note": "没有轮级时间戳就无法把延迟拆成「轮数 × 每轮耗时」"},
    })
    return checks


# -------------------------------------------------------------------- summary
def _ratio(numerator: int, denominator: int, label: str) -> dict:
    return {"n": int(numerator), "of": int(denominator), "denom": label}


def _page_fidelity_mismatch(item: dict) -> bool:
    """D59 (2026-09-21): is this citation's page wrong?

    Recomputed from the row's own ``model_page`` / ``page`` / ``page_end`` rather than
    trusting a stored boolean, for two reasons: a chunk can span pages (A3 added
    ``page_end`` for that), and rows written before this fix carry a flag computed
    against the start page only -- which flagged a legitimate page-5 citation of a 4-5
    chunk (row #12, sentence-bert.pdf).  Falls back to the stored flag when the row
    predates the fields.
    """

    model_page = item.get("model_page")
    page = item.get("page")
    page_end = item.get("page_end") or page
    if model_page is not None and page is not None:
        try:
            return not (int(page) <= int(model_page) <= int(page_end))
        except (TypeError, ValueError):
            return False
    return item.get("page_fidelity") is False


def summarize(rows: Sequence[dict], *, verdicts: dict[int, dict] | None = None) -> dict:
    """Recompute everything from the rows: the summary must never be the source."""

    total = len(rows)
    outcomes = [classify(row) for row in rows]
    declared = [str(row.get("outcome") or "") for row in rows]
    counts = Counter(outcomes)

    answerable = [row for row, outcome in zip(rows, outcomes) if resolve_expect(row) == EXPECT_ANSWER]
    answered = [
        row for row, outcome in zip(rows, outcomes)
        if resolve_expect(row) == EXPECT_ANSWER and outcome in ANSWERED_OUTCOMES
    ]
    deny_rows = [row for row, outcome in zip(rows, outcomes) if resolve_expect(row) == EXPECT_DENY]
    refuse_rows = [row for row, outcome in zip(rows, outcomes) if resolve_expect(row) == EXPECT_REFUSE]

    with_numbers = [row for row in answerable if row.get("number_coverage") is not None]
    covered = [row for row in with_numbers if float(row.get("number_coverage") or 0) >= 0.5]
    without_numbers = [row for row in answerable if row.get("number_coverage") is None]

    latencies = [float(row.get("elapsed_s") or 0) for row in rows if row.get("status") == "ok"]
    rounds = [int(row.get("rounds_total") or 0) for row in rows if row.get("status") == "ok"]
    calls = [int(row.get("tool_calls_total") or 0) for row in rows if row.get("status") == "ok"]

    tracks: Counter[str] = Counter()
    for row in rows:
        for step in row.get("trace") or []:
            if step.get("tool") == "search":
                tracks[str(step.get("track") or "unknown")] += 1

    audit = [
        row["index"] for row, outcome in zip(rows, outcomes)
        if outcome in AUDIT_OUTCOMES or (outcome == OUTCOME_NO_CITATION and int(row.get("citation_dropped") or 0) > 0)
    ]
    verdict_source = verdicts or {}
    audit_detail: list[dict] = []
    verdict_counts: Counter[str] = Counter()
    for row, outcome in zip(rows, outcomes):
        if row["index"] not in audit:
            continue
        verdict = verdict_source.get(row["index"]) or {}
        decided = str(verdict.get("verdict") or VERDICT_UNVERIFIED)
        verdict_counts[decided] += 1
        entry = {
            "index": row["index"],
            "expect": resolve_expect(row),
            "outcome": outcome,
            "verdict": decided,
        }
        if verdict.get("note"):
            entry["note"] = verdict["note"]
        audit_detail.append(entry)
    stray = sorted(set(verdict_source) - {row["index"] for row in rows})

    return {
        "harness_version": HARNESS_VERSION,
        "chain": "agent",
        "total": total,
        "transport_errors": counts[OUTCOME_TRANSPORT_ERROR],
        "outcome_counts": {outcome: counts[outcome] for outcome in OUTCOMES if counts[outcome]},
        "recompute_ok": (not any(declared)) or all(d == o for d, o in zip(declared, outcomes)),
        "recompute_note": "outcome_counts 一律由明细重算；row.outcome 只用于交叉校验",
        "answerable": {
            "n": len(answerable),
            "answered": len(answered),
            "citation_hit_all": _ratio(sum(1 for row in answerable if row.get("citation_hit")), len(answerable), "expect=answer"),
            "citation_hit_answered": _ratio(sum(1 for row in answered if row.get("citation_hit")), len(answered), "expect=answer 且本轮作答"),
            "budget_exhausted": _ratio(counts[OUTCOME_BUDGET_EXHAUSTED], len(answerable), "expect=answer"),
            "gate_failed": _ratio(counts[OUTCOME_GATE_FAILED], len(answerable), "expect=answer"),
        },
        "numbers": {
            "coverage_ge_half": _ratio(len(covered), len(with_numbers), "expect=answer 且参考答案含数字"),
            "rows_without_numbers": {
                "n": len(without_numbers),
                "rows": [row["index"] for row in without_numbers],
                "note": "旧口径把它们按 coverage=1.0 计入，于是 14 看起来大于 12 —— 分母必须写明",
            },
        },
        "expect_deny": {
            "n": len(deny_rows),
            "refuse_correct": sum(1 for row in deny_rows if classify(row) == OUTCOME_REFUSE_CORRECT),
            "deny_unverified": sum(1 for row in deny_rows if classify(row) == OUTCOME_DENY_UNVERIFIED),
            "note": "规则无法区分「正确否证」与「幻觉式否证」，这些行进入 audit，不计为正确",
        },
        "expect_refuse": {
            "n": len(refuse_rows),
            "refuse_correct": sum(1 for row in refuse_rows if classify(row) == OUTCOME_REFUSE_CORRECT),
            "unfounded_answer": sum(1 for row in refuse_rows if classify(row) == OUTCOME_UNFOUNDED_ANSWER),
        },
        "citations": {
            "dropped_rows": len([row for row in rows if int(row.get("citation_dropped") or 0) > 0]),
            "page_fidelity_mismatch_rows": len([
                row for row in rows if any(_page_fidelity_mismatch(item) for item in (row.get("citations") or []))
            ]),
        },
        "latency_s": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 2),
            "min": round(min(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "trajectory": {
            "rounds_mean": round(sum(rounds) / len(rounds), 2) if rounds else 0.0,
            "rounds_max": max(rounds) if rounds else 0,
            "tool_calls_mean": round(sum(calls) / len(calls), 2) if calls else 0.0,
            "tool_calls_max": max(calls) if calls else 0,
            "route_tracks_search_only": dict(tracks),
            "note": "工具调用数 ≠ 轮数：一轮可以并行多个调用；None 轨来自非 search 步，不计入",
        },
        "audit": {
            "rows": audit,
            "count": len(audit),
            "verdicts": {
                VERDICT_CORRECT: verdict_counts[VERDICT_CORRECT],
                VERDICT_HALLUCINATED: verdict_counts[VERDICT_HALLUCINATED],
                VERDICT_OTHER: verdict_counts[VERDICT_OTHER],
                VERDICT_UNVERIFIED: verdict_counts[VERDICT_UNVERIFIED],
            },
            "detail": audit_detail,
            "stray_verdict_indices": stray,
            "note": "verdict 只来自人工/判官的结论文件；规则本身不升级任何行为正确",
        },
        "assertions": trace_assertions(rows),
    }