import type { SessionRow } from "./types";

/**
 * 批 D 阶段 3 (D32): say it *before* the user presses the button.
 *
 * ``src/tasks/summary.py`` stops calling the summariser after 3 consecutive
 * failures (``MAX_SUMMARY_FAILURES``); only the manual "压缩本会话" forces one more
 * attempt.  Until now the breaker was invisible: the backend counted the failures,
 * and the only place that ever mentioned it was ``compactNoteFor`` — which the user
 * had to *trigger* to read.  A conversation whose automatic summary quietly gave up
 * looks exactly like a healthy one, so this note exists.
 *
 * ``summary_paused`` is derived server-side (the threshold stays in one place); we
 * only decide the wording.
 */
export function summaryPauseNoteFor(
  row: Pick<SessionRow, "summary_paused" | "summary_failures"> | null | undefined,
): string | null {
  if (!row?.summary_paused) return null;
  const failures = row.summary_failures ?? 0;
  const counted = failures > 0 ? `连续 ${failures} 次` : "多次";
  return `自动摘要已暂停：本会话${counted}摘要失败，超出窗口的旧轮次不再自动折叠（可点「压缩本会话」手动重试）。`;
}
