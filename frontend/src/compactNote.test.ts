import { describe, expect, it } from "vitest";
import { compactNoteFor } from "./compactNote";

describe("compactNoteFor (批 D 阶段 2 / D32)", () => {
  it("reports the folded size when the summary was stored", () => {
    expect(compactNoteFor({ stored: true, reason: "ok", upto_message_id: 191, chars: 125 })).toContain("125");
  });

  it("does not claim success when there was nothing to fold in", () => {
    const note = compactNoteFor({ stored: false, reason: "nothing_to_summarize", upto_message_id: null });
    expect(note).toContain("无需压缩");
    expect(note).not.toContain("已压缩");
  });

  it("says the conversation was left untouched on every failure reason", () => {
    for (const reason of ["empty_summary", "generation_failed"]) {
      expect(compactNoteFor({ stored: false, reason, upto_message_id: 42 })).toContain("保持原样");
    }
  });

  it("explains the open breaker instead of showing a raw code", () => {
    const note = compactNoteFor({ stored: false, reason: "breaker_open", upto_message_id: null, failures: 3 });
    expect(note).toContain("已暂停");
  });

  it("falls back to the server's own reason for an unknown one", () => {
    expect(compactNoteFor({ stored: false, reason: "who_knows", upto_message_id: null })).toContain("who_knows");
  });
});
