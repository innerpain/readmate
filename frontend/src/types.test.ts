import { describe, expect, it } from "vitest";
import { parseQualityNotice } from "./types";
import type { ParseQuality } from "./types";

const quality = (over: Partial<ParseQuality> = {}): ParseQuality => ({
  parse_quality: "ok",
  table_packs_over_cap: 0,
  tables_truncated: false,
  dropped_elements: null,
  ...over,
});

describe("parseQualityNotice (D8/D11 展示规则)", () => {
  it("returns null when quality is null", () => {
    expect(parseQualityNotice({ degraded: false, user_notice: null, quality: null })).toBeNull();
  });

  it("returns null when the quality fields are absent entirely", () => {
    // An older backend response must not produce a badge (nor throw).
    expect(parseQualityNotice({})).toBeNull();
    expect(parseQualityNotice(undefined)).toBeNull();
    expect(parseQualityNotice(null)).toBeNull();
  });

  it("returns null for a clean document", () => {
    expect(parseQualityNotice({ degraded: false, user_notice: null, quality: quality() })).toBeNull();
  });

  it("shows a yellow badge with user_notice when degraded", () => {
    const notice = parseQualityNotice({
      degraded: true,
      user_notice: "未能可靠解析该表结构，保留原始 grid/markdown",
      quality: quality({ parse_quality: "degraded" }),
    });
    expect(notice).toEqual({ tone: "yellow", text: "未能可靠解析该表结构，保留原始 grid/markdown" });
  });

  it("still warns when degraded but the backend sent no notice", () => {
    // A silent degradation is exactly the bug D8 exists to kill.
    const notice = parseQualityNotice({ degraded: true, user_notice: null, quality: quality() });
    expect(notice?.tone).toBe("yellow");
    expect(notice?.text).toBeTruthy();
  });

  it("ignores a blank user_notice", () => {
    const notice = parseQualityNotice({ degraded: true, user_notice: "   ", quality: quality() });
    expect(notice?.text).toBeTruthy();
    expect(notice?.text.trim()).not.toBe("");
  });

  it("flags truncated tables by table_packs_over_cap", () => {
    const notice = parseQualityNotice({ degraded: false, quality: quality({ table_packs_over_cap: 3 }) });
    expect(notice?.tone).toBe("warn");
    expect(notice?.text).toContain("有表格被截断");
    expect(notice?.text).toContain("3");
  });

  it("flags truncated tables by tables_truncated", () => {
    const notice = parseQualityNotice({ degraded: false, quality: quality({ tables_truncated: true }) });
    expect(notice?.tone).toBe("warn");
    expect(notice?.text).toContain("有表格被截断");
  });

  it("prefers the degraded badge when both signals are present", () => {
    const notice = parseQualityNotice({
      degraded: true,
      user_notice: "降级说明",
      quality: quality({ table_packs_over_cap: 2, tables_truncated: true }),
    });
    expect(notice?.tone).toBe("yellow");
    expect(notice?.text).toBe("降级说明");
  });

  it("tolerates a malformed quality payload without throwing", () => {
    // A hand-edited / partially written report: the counter is a string.
    const malformed = { table_packs_over_cap: "3" } as unknown as ParseQuality;
    expect(parseQualityNotice({ degraded: false, quality: malformed })).toBeNull();
  });

  it("does not treat a negative or zero cap as truncation", () => {
    expect(parseQualityNotice({ quality: quality({ table_packs_over_cap: 0 }) })).toBeNull();
    expect(parseQualityNotice({ quality: quality({ table_packs_over_cap: -1 }) })).toBeNull();
  });
});
