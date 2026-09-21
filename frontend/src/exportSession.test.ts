import { describe, expect, it } from "vitest";
import { buildSessionMarkdown, sessionExportFilename } from "./exportSession";
import type { MessageRow } from "./types";

/** Minimal persisted row (only the fields restoreTurn reads). */
function row(over: Partial<MessageRow>): MessageRow {
  return {
    id: 1,
    session_id: "sess-123456",
    role: "user",
    content: "",
    tool_name: null,
    payload_json: null,
    created_at: "2026-09-20T10:00:00",
    ...over,
  };
}

const FIXED_NOW = new Date("2026-09-20T10:00:00Z");

describe("buildSessionMarkdown", () => {
  it("numbers the questions and keeps the answer text", () => {
    const md = buildSessionMarkdown("sess-123456", "我的会话", [
      row({ id: 1, role: "user", content: "第一个问题" }),
      row({ id: 2, role: "assistant", content: "第一个答案" }),
      row({ id: 3, role: "user", content: "第二个问题" }),
      row({ id: 4, role: "assistant", content: "第二个答案" }),
    ], FIXED_NOW);
    expect(md).toContain("## 问 1");
    expect(md).toContain("## 问 2");
    expect(md).toContain("第一个问题");
    expect(md).toContain("第一个答案");
    expect(md).toContain("第二个答案");
    expect(md.match(/## 问 /g)).toHaveLength(2);
  });

  it("writes the title, session id and an injected timestamp into the header", () => {
    const md = buildSessionMarkdown("sess-123456", "标题", [], FIXED_NOW);
    expect(md.split("\n")[0]).toBe("# 标题");
    expect(md).toContain("会话 ID：`sess-123456`");
    expect(md).toContain(`导出时间：${FIXED_NOW.toLocaleString()}`);
  });

  it("falls back to 「会话」 when the title is empty", () => {
    expect(buildSessionMarkdown("sess-123456", "", [], FIXED_NOW).split("\n")[0]).toBe("# 会话");
  });

  it("marks an assistant row with no text instead of emitting an empty section", () => {
    const md = buildSessionMarkdown("sess-123456", "t", [row({ role: "assistant", content: "" })], FIXED_NOW);
    expect(md).toContain("（本轮无答案文本）");
  });

  it("skips non-conversation roles (tool rows)", () => {
    const md = buildSessionMarkdown("sess-123456", "t", [
      row({ id: 1, role: "user", content: "问" }),
      row({ id: 2, role: "tool", content: "search(...) -> 5 hits" }),
    ], FIXED_NOW);
    expect(md).not.toContain("search(...)");
    expect(md.match(/## 问 /g)).toHaveLength(1);
  });

  it("renders citations as a numbered list with page labels", () => {
    const payload = {
      citations: [
        { chunk_id: "c1", document_id: "abcdef1234567890", filename: "论文.pdf", page: 3, page_end: 4, quote: "q" },
        { chunk_id: "c2", document_id: "fedcba0987654321", filename: "", page: null, quote: "q" },
      ],
    };
    const md = buildSessionMarkdown("sess-123456", "t", [row({ role: "assistant", content: "答案", payload })], FIXED_NOW);
    expect(md).toContain("引用：");
    expect(md).toContain("1. 论文.pdf · p3–4");
    // No filename -> the short document id; no page -> no page suffix.
    expect(md).toContain("2. fedcba09");
    expect(md).not.toContain("2. fedcba09 · p");
  });

  it("collapses a citation whose page_end equals page", () => {
    const payload = {
      citations: [{ chunk_id: "c1", document_id: "abcdef1234567890", filename: "a.pdf", page: 3, page_end: 3, quote: "q" }],
    };
    const md = buildSessionMarkdown("sess-123456", "t", [row({ role: "assistant", content: "答案", payload })], FIXED_NOW);
    expect(md).toContain("1. a.pdf · p3");
    expect(md).not.toContain("p3–3");
  });

  it("renders warnings and failure as blockquotes", () => {
    const payload = { warnings: ["低置信", "证据不足"], failure: "route_fallback" };
    const md = buildSessionMarkdown("sess-123456", "t", [row({ role: "assistant", content: "答案", payload })], FIXED_NOW);
    expect(md).toContain("> 警告：低置信, 证据不足");
    expect(md).toContain("> 本轮失败：route_fallback");
  });

  it("parses payload_json when the server did not decode it", () => {
    const payload_json = JSON.stringify({ warnings: ["来自 payload_json"], citations: [] });
    const md = buildSessionMarkdown("sess-123456", "t", [row({ role: "assistant", content: "答案", payload_json })], FIXED_NOW);
    expect(md).toContain("> 警告：来自 payload_json");
  });

  it("does not throw on a corrupt payload_json", () => {
    const md = buildSessionMarkdown("sess-123456", "t", [row({ role: "assistant", content: "答案", payload_json: "{broken" })], FIXED_NOW);
    expect(md).toContain("答案");
  });

  it("produces a header-only document for an empty session", () => {
    const md = buildSessionMarkdown("sess-123456", "空会话", [], FIXED_NOW);
    expect(md).toContain("# 空会话");
    expect(md).not.toContain("## 问 ");
  });

  it("is deterministic for the same rows and clock", () => {
    const rows = [row({ id: 1, role: "user", content: "问" }), row({ id: 2, role: "assistant", content: "答" })];
    expect(buildSessionMarkdown("sess-123456", "t", rows, FIXED_NOW)).toBe(buildSessionMarkdown("sess-123456", "t", rows, FIXED_NOW));
  });
});

describe("sessionExportFilename", () => {
  it("derives the name from the session id and the date", () => {
    // Real ids are `ses_<12 hex>` (agent_db.py:247); the name drops the prefix.
    expect(sessionExportFilename("ses_abcdef123456", FIXED_NOW)).toBe("readmate-session-abcdef-2026-09-20.md");
  });

  it("always ends in .md", () => {
    expect(sessionExportFilename("short", FIXED_NOW).endsWith(".md")).toBe(true);
  });
});
