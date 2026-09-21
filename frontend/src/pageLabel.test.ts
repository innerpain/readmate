import { describe, expect, it } from "vitest";
import { pageLabel, pageLabelZh } from "./pageLabel";

describe("pageLabel", () => {
  it("renders a single page", () => {
    expect(pageLabel(3)).toBe("p3");
  });

  it("renders a span when the passage crosses a page break", () => {
    expect(pageLabel(3, 4)).toBe("p3–4");
  });

  it("falls back to the single-page form when pageEnd is absent", () => {
    // Responses recorded before A3 have no page_end at all.
    expect(pageLabel(3, undefined)).toBe("p3");
    expect(pageLabel(3, null)).toBe("p3");
  });

  it("falls back when pageEnd does not extend past page", () => {
    expect(pageLabel(3, 3)).toBe("p3");
    expect(pageLabel(3, 2)).toBe("p3");
  });

  it("renders nothing without a page", () => {
    expect(pageLabel(null)).toBe("");
    expect(pageLabel(undefined)).toBe("");
    expect(pageLabel(null, 5)).toBe("");
  });

  it("keeps page 0 distinct from 'no page'", () => {
    // 0 is a real (if unusual) page number; only null/undefined mean "unknown".
    expect(pageLabel(0)).toBe("p0");
  });
});

describe("pageLabelZh", () => {
  it("renders the Chinese single-page and span forms", () => {
    expect(pageLabelZh(7)).toBe("第 7 页");
    expect(pageLabelZh(7, 9)).toBe("第 7–9 页");
  });

  it("degrades to the single-page form without a usable pageEnd", () => {
    expect(pageLabelZh(7, null)).toBe("第 7 页");
    expect(pageLabelZh(7, 7)).toBe("第 7 页");
  });

  it("renders nothing without a page", () => {
    expect(pageLabelZh(null, 9)).toBe("");
  });
});
