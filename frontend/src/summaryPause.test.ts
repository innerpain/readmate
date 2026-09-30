import { describe, expect, it } from "vitest";

import { summaryPauseNoteFor } from "./summaryPause";

describe("summaryPauseNoteFor (批 D 阶段 3 / D32)", () => {
  it("stays silent while the breaker is closed", () => {
    expect(summaryPauseNoteFor({ summary_paused: false, summary_failures: 0 })).toBeNull();
    expect(summaryPauseNoteFor({ summary_paused: false, summary_failures: 2 })).toBeNull();
    expect(summaryPauseNoteFor(null)).toBeNull();
    expect(summaryPauseNoteFor(undefined)).toBeNull();
    expect(summaryPauseNoteFor({})).toBeNull();
  });

  it("names the failure count and the way out once the breaker is open", () => {
    const note = summaryPauseNoteFor({ summary_paused: true, summary_failures: 3 });
    expect(note).toContain("自动摘要已暂停");
    expect(note).toContain("连续 3 次");
    expect(note).toContain("压缩本会话");
  });

  it("still reads correctly without a count (older rows)", () => {
    const note = summaryPauseNoteFor({ summary_paused: true });
    expect(note).toContain("多次");
  });
});
