// FE-4: 导出会话为 markdown.
//
// Builds the transcript from the same persisted rows the UI renders
// (``GET /agent/sessions/{id}/messages``), so an export can never disagree with what
// the user saw.  Citations are rendered as a plain list — the point of the export is
// to be readable outside the app, not to round-trip.
//
// D45: the transcript builder is a pure function (``buildSessionMarkdown``) taking
// the rows + a clock; only the download itself touches the DOM.  That split is what
// makes the export unit-testable without a jsdom/testing-library dependency.

import { api } from "./api";
import { restoreTurn } from "./components/ChatPanel";
import type { MessageRow } from "./types";

/** Turn rows that carry a conversation (tool rows are skipped). */
function isConversationRow(row: MessageRow): boolean {
  return row.role === "user" || row.role === "assistant";
}

/**
 * Render the persisted rows as the exported markdown document.
 *
 * Pure: no clock read, no DOM, no network — `now` is injected so a test can pin
 * the timestamp.  Row order is preserved as the server returned it.
 */
export function buildSessionMarkdown(sessionId: string, title: string, rows: MessageRow[], now: Date = new Date()): string {
  const lines: string[] = [`# ${title || "会话"}`, "", `会话 ID：\`${sessionId}\``, `导出时间：${now.toLocaleString()}`, ""];
  let questionNo = 0;

  for (const row of rows) {
    if (!isConversationRow(row)) continue;
    const turn = restoreTurn(row);
    if (turn.role === "user") {
      questionNo += 1;
      lines.push(`## 问 ${questionNo}`, "", turn.text, "");
      continue;
    }
    lines.push("### 答", "", turn.text || "（本轮无答案文本）", "");
    const result = turn.result;
    if (result) {
      if (result.citations.length) {
        lines.push("引用：", "");
        result.citations.forEach((citation, index) => {
          const page = citation.page != null ? ` · p${citation.page}${citation.page_end && citation.page_end !== citation.page ? `–${citation.page_end}` : ""}` : "";
          lines.push(`${index + 1}. ${citation.filename || citation.document_id.slice(0, 8)}${page}`);
        });
        lines.push("");
      }
      if (result.warnings.length) lines.push(`> 警告：${result.warnings.join(", ")}`, "");
      if (result.failure) lines.push(`> 本轮失败：${result.failure}`, "");
    }
  }

  return lines.join("\n");
}

/** File name for the downloaded blob (kept next to the builder so the naming
 *  rule is visible in one place). */
export function sessionExportFilename(sessionId: string, now: Date = new Date()): string {
  return `readmate-session-${sessionId.slice(4, 10)}-${now.toISOString().slice(0, 10)}.md`;
}

export async function exportSessionMarkdown(sessionId: string, title: string): Promise<void> {
  const data = await api.sessionMessages(sessionId);
  const markdown = buildSessionMarkdown(sessionId, title, data.messages);
  const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = sessionExportFilename(sessionId);
  anchor.click();
  URL.revokeObjectURL(url);
}
