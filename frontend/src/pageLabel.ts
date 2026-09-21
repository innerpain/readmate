/**
 * A3: one passage can run over a page break, so a citation (or a chunk preview)
 * must be able to say "p3–4" instead of naming only the page it starts on.
 *
 * ``pageEnd`` is absent on responses produced before the span was recorded, in
 * which case the label degrades to the single-page form.
 */
export function pageLabel(page: number | null | undefined, pageEnd?: number | null): string {
  if (page == null) return "";
  if (pageEnd != null && pageEnd > page) return `p${page}–${pageEnd}`;
  return `p${page}`;
}

export function pageLabelZh(page: number | null | undefined, pageEnd?: number | null): string {
  if (page == null) return "";
  if (pageEnd != null && pageEnd > page) return `第 ${page}–${pageEnd} 页`;
  return `第 ${page} 页`;
}
