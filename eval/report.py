"""Rendering for the agent evaluation: text report, audit list, baseline delta.

Pure stdlib.  Every printed ratio shows its denominator, so a reader cannot
mistake "13/14 answered" for "13/17 asked".
"""

from __future__ import annotations

from collections.abc import Sequence

from eval import outcomes as O


def _ratio_line(label: str, ratio: dict) -> str:
    return f"  {label:<34} {ratio['n']}/{ratio['of']}  ({ratio['denom']})"


def render_text(summary: dict, rows: Sequence[dict]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"agent eval — {summary.get('harness_version')} — chain={summary.get('chain')}")
    lines.append("=" * 72)
    lines.append(f"questions={summary['total']}  transport_errors={summary['transport_errors']}  "
                 f"recompute_ok={summary['recompute_ok']}")

    lines.append("\n[outcomes] 每行只有一个 primary outcome（不重复计数）")
    for outcome, count in sorted((summary.get("outcome_counts") or {}).items(), key=lambda item: -item[1]):
        lines.append(f"  {outcome:<20} {count}")

    answerable = summary.get("answerable") or {}
    lines.append(f"\n[L3 answer] expect=answer 共 {answerable.get('n')}，其中本轮作答 {answerable.get('answered')}")
    for key in ("citation_hit_all", "citation_hit_answered", "budget_exhausted", "gate_failed"):
        if answerable.get(key):
            lines.append(_ratio_line(key, answerable[key]))

    numbers = summary.get("numbers") or {}
    if numbers.get("coverage_ge_half"):
        lines.append("\n[L3 numbers]")
        lines.append(_ratio_line("coverage>=0.5", numbers["coverage_ge_half"]))
        no_numbers = numbers.get("rows_without_numbers") or {}
        lines.append(f"  参考答案无数字的行 {no_numbers.get('n')} 条（{no_numbers.get('rows')}）—— 不计入分母")

    deny = summary.get("expect_deny") or {}
    refuse = summary.get("expect_refuse") or {}
    lines.append(f"\n[L4 refusal] expect=deny {deny.get('n')} / expect=refuse {refuse.get('n')}")
    lines.append(f"  refuse_correct(deny)      {deny.get('refuse_correct')}")
    lines.append(f"  deny_unverified(deny)     {deny.get('deny_unverified')}   <- 规则判不了，进 audit")
    lines.append(f"  refuse_correct(refuse)    {refuse.get('refuse_correct')}")
    lines.append(f"  unfounded_answer(refuse)  {refuse.get('unfounded_answer')}   <- 该拒未拒")

    trajectory = summary.get("trajectory") or {}
    latency = summary.get("latency_s") or {}
    lines.append("\n[L2 trajectory]")
    lines.append(f"  rounds mean={trajectory.get('rounds_mean')} max={trajectory.get('rounds_max')}  "
                 f"tool_calls mean={trajectory.get('tool_calls_mean')} max={trajectory.get('tool_calls_max')}")
    lines.append(f"  search tracks {trajectory.get('route_tracks_search_only')}")
    lines.append(f"  latency mean={latency.get('mean')}s p95={latency.get('p95')}s "
                 f"min={latency.get('min')}s max={latency.get('max')}s")

    lines.append("\n[L2 assertions]")
    for check in summary.get("assertions") or []:
        mark = "OK  " if check["ok"] else "FAIL"
        lines.append(f"  {mark} {check['name']}  {_detail_brief(check['detail'])}")

    audit = summary.get("audit") or {}
    verdicts = audit.get("verdicts") or {}
    lines.append(
        f"\n[audit] {audit.get('count')} 行需要人工/判官（{audit.get('rows')}）"
        f" — 已验证：正确 {verdicts.get('correct', 0)} / 幻觉 {verdicts.get('hallucinated', 0)}"
        f" / 其他 {verdicts.get('other', 0)} / 未核 {verdicts.get('unverified', 0)}"
    )
    for entry in audit.get("detail") or []:
        note = f" — {entry['note']}" if entry.get("note") else ""
        lines.append(f"  #{entry['index']:<3} [{entry['expect']}] {entry['outcome']:<17} -> {entry['verdict']}{note}")
    if audit.get("stray_verdict_indices"):
        lines.append(f"  ! verdict 文件里有明细中不存在的题号：{audit['stray_verdict_indices']}")
    return "\n".join(lines)


def _detail_brief(detail: dict) -> str:
    parts: list[str] = []
    if "rows" in detail and detail["rows"]:
        parts.append(f"rows={detail['rows']}")
    if "note" in detail:
        parts.append(str(detail["note"]))
    return " | ".join(parts)


def comparison(baseline: dict, current: dict) -> str:
    """Baseline delta. Both summaries must come from the same chain and harness."""

    lines = ["\n[regression vs baseline]"]
    if baseline.get("harness_version") != current.get("harness_version"):
        lines.append(f"  ! harness differs: {baseline.get('harness_version')} -> {current.get('harness_version')}")
    if baseline.get("chain") != current.get("chain"):
        lines.append(f"  ! chain differs: {baseline.get('chain')} -> {current.get('chain')}")

    def metric(summary: dict, path: tuple[str, ...]) -> float | None:
        node: object = summary
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return float(node) if isinstance(node, (int, float)) else None

    paths = (
        ("answerable", "citation_hit_all", "n"),
        ("answerable", "citation_hit_answered", "n"),
        ("answerable", "budget_exhausted", "n"),
        ("numbers", "coverage_ge_half", "n"),
        ("expect_deny", "deny_unverified"),
        ("expect_refuse", "unfounded_answer"),
        ("citations", "dropped_rows"),
        ("audit", "count"),
    )
    header = f"  {'metric':<34} {'baseline':>10} {'current':>10} {'delta':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for path in paths:
        before = metric(baseline, path)
        after = metric(current, path)
        if before is None and after is None:
            continue
        delta = "" if before is None or after is None else f"{after - before:+.0f}"
        lines.append(f"  {'.'.join(path):<34} {'' if before is None else f'{before:.0f}':>10} "
                     f"{'' if after is None else f'{after:.0f}':>10} {delta:>8}")
    return "\n".join(lines)


def audit_markdown(rows: Sequence[dict], summary: dict) -> str:
    """The rows a rule cannot settle, printed for a human or a future judge."""

    audit = summary.get("audit") or {}
    wanted = set(audit.get("rows") or [])
    decided = {entry["index"]: entry for entry in (audit.get("detail") or [])}
    lines = [
        f"# agent eval audit — {len(wanted)} rows",
        "",
        "规则无法判定这些行（正确否证 vs 幻觉否证 / 该拒未拒）。",
        "把结论写进 --audit-verdicts 的 JSON，让汇总把『未核』变成明确的『正确/幻觉』；不要直接改本文件（下次运行会重写）。",
        "",
    ]
    for row in rows:
        if row.get("index") not in wanted:
            continue
        verdict = decided.get(row["index"]) or {}
        citations = ", ".join(
            f"{item.get('filename')}p{item.get('page')}" for item in (row.get("citations") or [])
        ) or "(no citations)"
        lines.extend([
            f"## #{row['index']} [{row.get('expect')}] {row.get('outcome')} -> {verdict.get('verdict', 'unverified')}",
            "",
            f"- verdict note: {verdict.get('note') or '(none yet)'}",
            f"- question: {row.get('question')}",
            f"- failure: {row.get('failure')} | gate: {row.get('gate')} | citations: {citations}",
            f"- reference: {row.get('reference_answer') or '(none)'}",
            "",
            "```",
            str(row.get("answer") or "(no answer)")[:1500],
            "```",
            "",
        ])
    return "\n".join(lines)