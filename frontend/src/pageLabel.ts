/**
 * A3: one passage can run over a page break, so a citation (or a chunk preview)
 * must be able to say "p3–4" instead of naming only the page it starts on.
 *
 * ``pageEnd`` is absent on responses produced before the span was recorded, in
 * which case the label degrades to the single-page form.
 *
 * D61: ``kind`` says what the number *means*.  Multi-format documents number
 * their own containers: a slide in a deck, a worksheet in a workbook, or -- when
 * the format has no page concept at all -- the top-level heading section, which
 * is marked §N so a reader is never sent hunting for a page that does not exist.
 * The default ("page") keeps every pre-multi-format response byte-identical.
 */
export type PageKind = "page" | "slide" | "sheet" | "section";

export function pageLabel(
  page: number | null | undefined,
  pageEnd?: number | null,
  kind?: PageKind | null,
): string {
  if (page == null) return "";
  if (kind === "slide") return `slide ${page}`;
  if (kind === "sheet") return `sheet ${page}`;
  if (kind === "section") return `§${page}`;
  if (pageEnd != null && pageEnd > page) return `p${page}–${pageEnd}`;
  return `p${page}`;
}

export function pageLabelZh(
  page: number | null | undefined,
  pageEnd?: number | null,
  kind?: PageKind | null,
): string {
  if (page == null) return "";
  if (kind === "slide") return `第 ${page} 张幻灯片`;
  if (kind === "sheet") return `第 ${page} 个工作表`;
  if (kind === "section") return `§${page}`;
  if (pageEnd != null && pageEnd > page) return `第 ${page}–${pageEnd} 页`;
  return `第 ${page} 页`;
}

/** D61: the unit for a document's element count (``page_count``).
 *
 *  A Word file has no physical pages here and a deck has slides, so the card
 *  must not print "12 页" for either -- the user can check that in one click. */
export function countUnit(kind?: string | null): string {
  if (kind === "slide") return "张幻灯片";
  if (kind === "sheet") return "个工作表";
  if (kind === "section") return "节";
  return "页";
}
